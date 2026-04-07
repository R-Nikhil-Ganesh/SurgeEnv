"""Smoke test for Surge OpenEnv service health and basic API responsiveness."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

from surge.client import SurgeEnv
from surge.models import SurgeAction


def _http_get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = response.read().decode("utf-8")
        return json.loads(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Surge OpenEnv endpoints")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for the OpenEnv server")
    parser.add_argument("--steps", type=int, default=5, help="Number of step() calls to execute")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    try:
        health = _http_get_json(f"{base_url}/health")
        print(f"[HEALTH] status={health.get('status', 'unknown')}")
    except urllib.error.URLError as exc:
        print(f"[HEALTH] failed: {exc}")
        return 1

    try:
        with SurgeEnv(base_url=base_url).sync() as env:
            reset_result = env.reset(seed=42)
            print(
                "[RESET] "
                + json.dumps(
                    {
                        "done": reset_result.done,
                        "reward": reset_result.reward,
                        "obs_len": len(reset_result.observation.vector),
                    },
                    sort_keys=True,
                )
            )

            for i in range(args.steps):
                action = SurgeAction(action=i % 7)
                step_result = env.step(action)
                print(
                    "[STEP] "
                    + json.dumps(
                        {
                            "i": i + 1,
                            "action": action.action,
                            "reward": step_result.reward,
                            "done": step_result.done,
                            "nodes": step_result.observation.active_nodes,
                            "sla": step_result.observation.true_sla,
                        },
                        sort_keys=True,
                    )
                )
                if step_result.done:
                    break

            state = env.state()
            print(
                "[STATE] "
                + json.dumps(
                    {
                        "episode_id": state.episode_id,
                        "step_count": state.step_count,
                        "active_nodes": state.active_nodes,
                        "queue_size": state.queue_size,
                        "termination_reason": state.termination_reason,
                    },
                    sort_keys=True,
                )
            )

    except Exception as exc:
        print(f"[ERROR] smoke test failed: {exc}")
        return 1

    print("[OK] service is healthy and responsive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
