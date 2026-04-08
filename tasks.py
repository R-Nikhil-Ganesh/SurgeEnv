"""Task and grader definitions for SurgeEnvV2 hackathon submission."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Literal, Type

from openenv.core.rubrics.base import Rubric
from pydantic import BaseModel, Field


Difficulty = Literal["easy", "medium", "hard"]
_SCORE_EPS = 2e-2
_SCORE_ONE = 1 - _SCORE_EPS


class TaskDefinition(BaseModel):
    """Declarative task configuration used by inference and manifests."""

    id: str
    name: str
    difficulty: Difficulty
    description: str
    grader_class: str = Field(description="Python class path for the grader")


class SurgeTaskRubric(Rubric):
    """Base class for episode-level Surge graders."""

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self) -> None:
        self.last_score = _SCORE_EPS
        self._steps = 0
        self._min_sla = _SCORE_ONE
        self._sum_nodes = 0.0
        self._max_nodes = 0
        self._used_rate_limit = False
        self._used_cache = False
        self._terminated_early = False
        self._termination_reason = "none"
        self._episode_reward = 0.0

    def _track(self, observation: Any) -> None:
        self._steps += 1
        active_nodes = int(getattr(observation, "active_nodes", 0))
        self._sum_nodes += float(active_nodes)
        self._max_nodes = max(self._max_nodes, active_nodes)
        self._used_rate_limit = self._used_rate_limit or float(getattr(observation, "rate_limiting", 0.0)) > 0.5
        self._used_cache = self._used_cache or float(getattr(observation, "cache_enabled", 0.0)) > 0.5

        metadata = getattr(observation, "metadata", {}) or {}
        sla = float(getattr(observation, "true_sla", metadata.get("true_sla", 0.0)))
        self._min_sla = min(self._min_sla, sla)
        self._termination_reason = str(
            getattr(observation, "termination_reason", metadata.get("termination_reason", "none"))
        )
        self._terminated_early = self._termination_reason in {
            "catastrophic_failure",
            "prolonged_instability",
        }
        self._episode_reward = float(
            getattr(observation, "episode_reward", metadata.get("episode_reward", self._episode_reward))
        )

    @staticmethod
    def _clamp(value: float) -> float:
        # Strictly bound between > 0.0 and < 1.0, including non-finite safety.
        numeric = float(value)
        if not math.isfinite(numeric):
            return _SCORE_EPS
        return max(_SCORE_EPS, min(1 - _SCORE_EPS, numeric))

    def forward(self, action: Any, observation: Any) -> float:
        del action
        self._track(observation)
        if not bool(getattr(observation, "done", False)):
            # Never return exact 0.0 on intermediate steps
            self.last_score = _SCORE_EPS
            return _SCORE_EPS

        score = self._final_score(observation)
        score = self._clamp(score)
        self.last_score = score
        return score

    def _final_score(self, observation: Any) -> float:
        del observation
        raise NotImplementedError


TASKS: Dict[str, TaskDefinition] = {}
GRADERS: Dict[str, Type[SurgeTaskRubric]] = {}


def register_task(
    task_id: str,
    name: str,
    difficulty: Difficulty,
    description: str,
) -> Callable[[Type[SurgeTaskRubric]], Type[SurgeTaskRubric]]:
    """Decorator for registering a task with its rubric class."""

    def _decorator(grader_cls: Type[SurgeTaskRubric]) -> Type[SurgeTaskRubric]:
        GRADERS[task_id] = grader_cls
        TASKS[task_id] = TaskDefinition(
            id=task_id,
            name=name,
            difficulty=difficulty,
            description=description,
            grader_class=f"{grader_cls.__module__}:{grader_cls.__name__}",
        )
        return grader_cls

    return _decorator


@register_task(
    task_id="survive_spike",
    name="Survive the spike",
    difficulty="easy",
    description="Maintain SLA > 0.95 for the full episode and never exceed 4 billed nodes.",
)
class SurviveSpikeGrader(SurgeTaskRubric):
    """Easy task grader."""

    def _final_score(self, observation: Any) -> float:
        del observation
        within_sla = self._min_sla > 0.95
        within_nodes = self._max_nodes <= 4
        no_early_termination = not self._terminated_early
        return self._clamp(_SCORE_ONE if (within_sla and within_nodes and no_early_termination) else _SCORE_EPS)


@register_task(
    task_id="cost_aware_mitigation",
    name="Cost-aware mitigation",
    difficulty="medium",
    description="Keep SLA > 0.95, average active nodes <= 3, and actively use both cache and rate-limiting mitigations.",
)
class CostAwareMitigationGrader(SurgeTaskRubric):
    """Medium task grader."""

    def _final_score(self, observation: Any) -> float:
        del observation
        avg_nodes = self._sum_nodes / max(1, self._steps)

        sla_score = self._clamp(_SCORE_ONE if self._min_sla > 0.95 else max(_SCORE_EPS, self._min_sla / 0.95))
        cost_score = self._clamp(_SCORE_ONE if avg_nodes <= 3.0 else max(_SCORE_EPS, _SCORE_ONE - ((avg_nodes - 3.0) / 2.0)))
        tool_score = self._clamp(_SCORE_ONE if (self._used_cache and self._used_rate_limit) else _SCORE_EPS)
        stability_score = self._clamp(_SCORE_ONE if not self._terminated_early else _SCORE_EPS)

        # SLA and cost are primary; tool usage is mandatory for full credit.
        return self._clamp(0.40 * sla_score + 0.35 * cost_score + 0.20 * tool_score + 0.05 * stability_score)


@register_task(
    task_id="adaptive_sre",
    name="Adaptive SRE",
    difficulty="hard",
    description="Achieve cumulative reward > 30 under randomized spike timing/intensity and avoid early termination.",
)
class AdaptiveSREGrader(SurgeTaskRubric):
    """Hard task grader."""

    def _final_score(self, observation: Any) -> float:
        del observation
        reward_score = self._clamp(_SCORE_ONE if self._episode_reward > 30.0 else max(_SCORE_EPS, (self._episode_reward + 30.0) / 60.0))
        no_early_termination = self._clamp(_SCORE_ONE if not self._terminated_early else _SCORE_EPS)
        return self._clamp(0.8 * reward_score + 0.2 * no_early_termination)


def create_grader(task_id: str) -> SurgeTaskRubric:
    """Factory helper for constructing graders by task id."""
    if task_id not in GRADERS:
        available = ", ".join(sorted(GRADERS))
        raise KeyError(f"Unknown task_id '{task_id}'. Available: {available}")
    return GRADERS[task_id]()
