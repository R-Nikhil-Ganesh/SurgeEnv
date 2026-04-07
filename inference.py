"""Baseline inference runner for SurgeEnvV2 tasks using OpenAI + OpenEnv server APIs."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from surge.client import SurgeEnv
from surge.models import SurgeAction, SurgeObservation
from surge.tasks import TASKS, create_grader


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
    model_name = os.environ.get("MODEL_NAME", "openai/gpt-4o-mini").strip()

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
        raise RuntimeError("Missing MODEL_NAME. Set MODEL_NAME to the model identifier to use.")
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
    prompt_payload = {
        "task": task_name,
        "observation": {
            "timestep": obs.timestep,
            "active_nodes": obs.active_nodes,
            "provisioning_nodes": obs.provisioning_nodes,
            "observed_rps": obs.observed_rps,
            "observed_cpu": obs.observed_cpu,
            "observed_db_latency": obs.observed_db_latency,
            "rate_limiting": obs.rate_limiting,
            "cache_enabled": obs.cache_enabled,
            "true_sla": obs.true_sla,
            "episode_reward": obs.episode_reward,
        },
        "action_legend": {
            0: "No-Op",
            1: "Scale Up",
            2: "Scale Down",
            3: "RateLimit ON",
            4: "RateLimit OFF",
            5: "Cache ON",
            6: "Cache OFF",
        },
        "instruction": "Return ONLY one integer in [0,6].",
    }

    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            max_tokens=8,
            messages=[
                {
                    "role": "system",
                    "content": "You are an SRE control policy for a simulator. Output only a single integer action in [0,6].",
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt_payload),
                },
            ],
        )
        output_text = (response.choices[0].message.content or "").strip()
        match = re.search(r"-?\d+", output_text)
        if not match:
            finish_reason = response.choices[0].finish_reason
            _log(
                "END",
                {
                    "event": "model_parse_fallback",
                    "finish_reason": finish_reason,
                    "model": model_name,
                    "reason": "non_integer_output",
                    "task": task_name,
                },
            )
            return 0
        return _clamp_action(int(match.group(0)))
    except Exception as exc:
        _log(
            "END",
            {
                "event": "model_action_fallback",
                "model": model_name,
                "reason": str(exc),
                "task": task_name,
            },
        )
        return 0


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

            score_update = float(grader(SurgeAction(action=action_value), obs))

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
    final_score = float(grader.last_score if grader.last_score is not None else 0.0)

    result = {
        "task_id": task.id,
        "difficulty": task.difficulty,
        "seed": seed,
        "steps": steps,
        "score": round(final_score, 6),
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
