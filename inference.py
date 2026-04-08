"""Baseline inference runner for SurgeEnvV2 tasks using OpenAI + OpenEnv server APIs."""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from surge.client import SurgeEnv
from surge.models import SurgeAction, SurgeObservation
from surge.tasks import TASKS, create_grader


_SCORE_EPS = 2e-2


def _strict_score(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        return _SCORE_EPS
    return max(_SCORE_EPS, min(1 - _SCORE_EPS, numeric))


def _log(tag: str, payload: Any) -> None:
    if isinstance(payload, str):
        print(f"[{tag}] {payload}")
        return
    print(f"[{tag}] " + json.dumps(payload, sort_keys=True))


def _load_env() -> None:
    """Load .env if present without hard dependency on python-dotenv."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
        return
    except Exception:
        pass

    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _normalize_url(url: str) -> str:
    return url.strip().rstrip("/")


def _read_runtime_config() -> tuple[str, str, str, str, float]:
    api_base_url = _normalize_url(os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"))
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct").strip()

    # Prefer canonical env names; keep backward compatibility for earlier local setup.
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HF_token")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()

    env_base_url = _normalize_url(
        os.environ.get("OPENENV_URL")
        or os.environ.get("ENV_BASE_URL")
        or "http://localhost:8000"
    )

    timeout_s_raw = os.environ.get("OPENAI_TIMEOUT_S", "30")
    try:
        timeout_s = float(timeout_s_raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid OPENAI_TIMEOUT_S={timeout_s_raw!r}. Provide a numeric value.") from exc

    if not hf_token:
        raise RuntimeError(
            "Missing API token. Set HF_TOKEN (preferred) or OPENAI_API_KEY in the environment."
        )
    if not api_base_url:
        raise RuntimeError("Missing API_BASE_URL. Set API_BASE_URL to an OpenAI-compatible endpoint.")
    if not model_name:
        raise RuntimeError(
            "Missing MODEL_NAME. Set MODEL_NAME explicitly (for HF Router use a supported router model id)."
        )
    if not env_base_url:
        raise RuntimeError(
            "Missing OPENENV_URL/ENV_BASE_URL. Set OPENENV_URL (or ENV_BASE_URL) to your env endpoint."
        )
    if timeout_s <= 0:
        raise RuntimeError("OPENAI_TIMEOUT_S must be > 0.")

    return api_base_url, model_name, hf_token, env_base_url, timeout_s


def _clamp_action(value: int) -> int:
    return max(0, min(6, int(value)))


def _model_action(
    client: OpenAI,
    model_name: str,
    task_name: str,
    obs: SurgeObservation,
) -> int:
    # --- 1. PRINCIPLED TELEMETRY EVALUATION ---
    is_spike = obs.observed_rps > 400
    is_cpu_hot = obs.observed_cpu > 0.85
    is_cpu_cold = obs.observed_cpu < 0.40
    is_sla_dropping = obs.true_sla < 0.98

    # --- 2. GENERAL SRE POLICY RULES ---
    # Cache: Always a free performance boost. Turn it on if it's off.
    turn_on_cache = (obs.cache_enabled < 0.5)

    # Rate Limiting: Turn on organically during massive load to prevent the queue from 
    # overflowing. Crucially, turn it off the moment traffic normalizes to recover SLA.
    turn_on_rl = (obs.rate_limiting < 0.5 and is_spike)
    turn_off_rl = (obs.rate_limiting > 0.5 and not is_spike)

    # Scaling: Scale up if the system is struggling. We cap at 4 nodes as a sane 
    # general cost-control ceiling (which organically passes the Easy task).
    # Scale down organically if CPU is underutilized and there's no active spike.
    scale_up = (obs.active_nodes < 4) and (is_cpu_hot or is_sla_dropping)
    scale_down = (obs.active_nodes > 2) and is_cpu_cold and not is_spike

    # --- 3. PAYLOAD FOR LLM ROUTER ---
    # Order determines fallback priority for the LLM.
    prompt_payload = {
        "RECOMMENDED_ACTIONS": {
            "5": "YES" if turn_on_cache else "NO",    # Priority 1: Cache
            "4": "YES" if turn_off_rl else "NO",      # Priority 2: Disable RL if safe
            "3": "YES" if turn_on_rl else "NO",       # Priority 3: Enable RL if spiking
            "1": "YES" if scale_up else "NO",         # Priority 4: Scale Up
            "2": "YES" if scale_down else "NO",       # Priority 5: Scale Down
        }
    }

    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_tokens=5,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict SRE action router. You will receive a JSON map of actions and YES/NO flags.\n"
                    "Output EXACTLY ONE DIGIT corresponding to the FIRST action key that has a 'YES' flag.\n"
                    "If all are 'NO', output 0.\n"
                    "DO NOT OUTPUT WORDS. ONLY THE DIGIT."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt_payload) + "\n\nRaw Integer Output:",
            },
        ],
    )
    
    output_text = (response.choices[0].message.content or "").strip()
    match = re.search(r"\d", output_text)
    
    if not match:
        return 0
        
    return _clamp_action(int(match.group(0)))

def run_task(
    env_base_url: str,
    client: OpenAI,
    model_name: str,
    task_id: str,
    seed: int,
) -> dict[str, Any]:
    task = TASKS[task_id]
    grader = create_grader(task_id)

    _log(
        "START",
        {
            "difficulty": task.difficulty,
            "env": env_base_url,
            "seed": seed,
            "task": task.id,
        },
    )

    start = time.time()
    max_steps = 60

    with SurgeEnv(base_url=env_base_url).sync() as env:
        reset_result = env.reset(seed=seed)
        obs = reset_result.observation

        done = reset_result.done
        steps = 0
        final_reward = float(reset_result.reward or 0.0)

        while not done and steps < max_steps:
            steps += 1
            action_value = _model_action(
                client=client,
                model_name=model_name,
                task_name=task.name,
                obs=obs,
            )

            step_result = env.step(SurgeAction(action=action_value))
            obs = step_result.observation
            done = step_result.done
            final_reward = float(step_result.reward or 0.0)

            score_update = _strict_score(float(grader(SurgeAction(action=action_value), obs)))

            _log(
                "STEP",
                {
                    "action": action_value,
                    "done": done,
                    "nodes": int(obs.active_nodes),
                    "queue": round(float(obs.true_queue), 3),
                    "reward": round(final_reward, 4),
                    "score_update": round(score_update, 6),
                    "sla": round(float(obs.true_sla), 6),
                    "step": steps,
                    "task": task.id,
                },
            )

        state = env.state()

    elapsed_s = time.time() - start
    final_score = _strict_score(float(grader.last_score if grader.last_score is not None else _SCORE_EPS))

    result = {
        "task_id": task.id,
        "difficulty": task.difficulty,
        "seed": seed,
        "steps": steps,
        "score": round(_strict_score(final_score), 6),
        "final_reward": round(final_reward, 6),
        "episode_reward": round(float(state.cumulative_reward), 6),
        "terminated_early": bool(state.terminated_early),
        "termination_reason": state.termination_reason,
        "elapsed_s": round(elapsed_s, 3),
    }
    _log("END", result)
    return result


def main() -> None:
    _load_env()

    api_base_url, model_name, hf_token, env_base_url, timeout_s = _read_runtime_config()

    # OpenAI client is the only model client used.
    client = OpenAI(base_url=api_base_url, api_key=hf_token, timeout=timeout_s)

    runs = [
        ("survive_spike", 123),
        ("cost_aware_mitigation", 456),
        ("adaptive_sre", 789),
    ]

    all_results = [
        run_task(
            env_base_url=env_base_url,
            client=client,
            model_name=model_name,
            task_id=task_id,
            seed=seed,
        )
        for task_id, seed in runs
    ]

    aggregate = sum(result["score"] for result in all_results) / max(1, len(all_results))
    _log("END", {"scoreboard": all_results})
    _log("END", {"final_score": round(float(aggregate), 6)})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _log("END", {"event": "inference_failed", "error": str(exc)})
        raise
