from __future__ import annotations

from types import SimpleNamespace

import pytest

from surge.inference import _strict_score
from surge.tasks import TASKS, create_grader


SCORE_MIN = 0.002
SCORE_MAX = 0.998


def _observation(**overrides):
    base = {
        "done": True,
        "active_nodes": 1,
        "rate_limiting": 0.0,
        "cache_enabled": 0.0,
        "true_sla": 1.0,
        "termination_reason": "none",
        "episode_reward": 0.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _assert_in_bounds(value: float) -> None:
    assert SCORE_MIN <= value <= SCORE_MAX
    assert value != 0.0
    assert value != 1.0


@pytest.mark.parametrize("value", [float("-inf"), -1.0, 0.0, 0.5, 1.0, float("inf"), float("nan")])
def test_strict_score_helper_is_bounded(value: float) -> None:
    _assert_in_bounds(_strict_score(value))


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_task_grader_scores_are_bounded(task_id: str) -> None:
    grader = create_grader(task_id)

    # Intermediate-step output must never escape the strict bounds.
    intermediate = grader.forward(None, _observation(done=False))
    _assert_in_bounds(intermediate)

    # Exercise the raw final-score path with representative high- and low-end states.
    if task_id == "survive_spike":
        low_state = _observation(active_nodes=5, true_sla=0.0, termination_reason="catastrophic_failure")
        high_state = _observation(active_nodes=4, true_sla=0.999999, termination_reason="none")
    elif task_id == "cost_aware_mitigation":
        low_state = _observation(active_nodes=5, true_sla=0.0, rate_limiting=0.0, cache_enabled=0.0)
        high_state = _observation(active_nodes=1, true_sla=0.999999, rate_limiting=1.0, cache_enabled=1.0)
    else:
        low_state = _observation(episode_reward=-100.0, termination_reason="catastrophic_failure")
        high_state = _observation(episode_reward=100.0, termination_reason="none")

    low_raw = grader._final_score(low_state)
    high_raw = grader._final_score(high_state)
    _assert_in_bounds(low_raw)
    _assert_in_bounds(high_raw)

    # Forward() should clamp the final score as well.
    grader.reset()
    final_score = grader.forward(None, high_state)
    _assert_in_bounds(final_score)


def test_all_task_metadata_exists() -> None:
    assert sorted(TASKS) == ["adaptive_sre", "cost_aware_mitigation", "survive_spike"]
