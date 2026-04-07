# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the SurgeEnvV2 OpenEnv environment."""

from typing import Dict, List, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class SurgeAction(Action):
    """Discrete control action for the SRE simulator."""

    action: int = Field(
        ...,
        ge=0,
        le=6,
        description="Discrete action in [0,6] (0=NoOp, 1=ScaleUp, 2=ScaleDown, 3=RateLimitOn, 4=RateLimitOff, 5=CacheOn, 6=CacheOff)",
    )


class SurgeObservation(Observation):
    """Lagged dashboard observation returned to the agent."""

    vector: List[float] = Field(
        default_factory=list,
        description="Observation vector [timestep, active_nodes, provisioning_nodes, observed_rps, observed_cpu, observed_db_latency, rate_limiting, cache_enabled]",
    )
    timestep: int = Field(default=0, ge=0)
    active_nodes: int = Field(default=1, ge=1)
    provisioning_nodes: int = Field(default=0, ge=0)
    observed_rps: float = Field(default=50.0, ge=0.0)
    observed_cpu: float = Field(default=0.0, ge=0.0)
    observed_db_latency: float = Field(default=10.0, ge=0.0)
    rate_limiting: float = Field(default=0.0, ge=0.0, le=1.0)
    cache_enabled: float = Field(default=0.0, ge=0.0, le=1.0)
    true_sla: float = Field(default=1.0, ge=0.0)
    true_queue: float = Field(default=0.0, ge=0.0)
    true_rps: float = Field(default=50.0, ge=0.0)
    true_processed_rps: float = Field(default=50.0, ge=0.0)
    termination_reason: Literal["none", "catastrophic_failure", "prolonged_instability", "max_steps"] = Field(default="none")
    action_taken: int = Field(default=0, ge=0, le=6)
    episode_reward: float = Field(default=0.0)


class SurgeState(State):
    """Full internal simulator state."""

    max_steps: int = Field(default=50, ge=1)
    timestep: int = Field(default=0, ge=0)
    active_nodes: int = Field(default=2, ge=1)
    provisioning_nodes: int = Field(default=0, ge=0)
    provisioning_queue: List[int] = Field(default_factory=list)
    true_rps: float = Field(default=50.0, ge=0.0)
    true_processed_rps: float = Field(default=50.0, ge=0.0)
    current_sla: float = Field(default=1.0, ge=0.0)
    queue_size: float = Field(default=0.0, ge=0.0)
    max_queue: float = Field(default=5000.0, ge=1.0)
    rate_limiting: float = Field(default=0.0, ge=0.0, le=1.0)
    cache_enabled: float = Field(default=0.0, ge=0.0, le=1.0)
    degraded_steps: int = Field(default=0, ge=0)
    terminated_early: bool = Field(default=False)
    termination_reason: Literal["none", "catastrophic_failure", "prolonged_instability", "max_steps"] = Field(default="none")
    cumulative_reward: float = Field(default=0.0)
    action_counts: Dict[str, int] = Field(default_factory=dict)
    metric_history_rps: List[float] = Field(default_factory=list)
    metric_history_cpu: List[float] = Field(default_factory=list)
    metric_history_db_latency: List[float] = Field(default_factory=list)


class SurgeResetResponse(BaseModel):
    """Typed reset response (mirrors OpenEnv reset payload shape)."""

    observation: SurgeObservation
    reward: float | None = Field(default=None)
    done: bool = Field(default=False)


class SurgeStepResponse(BaseModel):
    """Typed step response (mirrors OpenEnv step payload shape)."""

    observation: SurgeObservation
    reward: float
    done: bool
