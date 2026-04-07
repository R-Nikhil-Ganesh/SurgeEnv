# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Surge Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import SurgeAction, SurgeObservation, SurgeState


class SurgeEnv(EnvClient[SurgeAction, SurgeObservation, SurgeState]):
    """
    Client for the Surge Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with SurgeEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.vector)
        ...
        ...     result = client.step(SurgeAction(action=1))
        ...     print(result.observation.active_nodes)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = SurgeEnv.from_docker_image("surge-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(SurgeAction(action=0))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SurgeAction) -> Dict:
        """
        Convert SurgeAction to JSON payload for step message.

        Args:
            action: SurgeAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[SurgeObservation]:
        """
        Parse server response into StepResult[SurgeObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with SurgeObservation
        """
        obs_data = payload.get("observation", {})
        reward = payload.get("reward")
        done = payload.get("done", False)
        observation = SurgeObservation(
            vector=obs_data.get("vector", []),
            timestep=obs_data.get("timestep", 0),
            active_nodes=obs_data.get("active_nodes", 1),
            provisioning_nodes=obs_data.get("provisioning_nodes", 0),
            observed_rps=obs_data.get("observed_rps", 50.0),
            observed_cpu=obs_data.get("observed_cpu", 0.0),
            observed_db_latency=obs_data.get("observed_db_latency", 10.0),
            rate_limiting=obs_data.get("rate_limiting", 0.0),
            cache_enabled=obs_data.get("cache_enabled", 0.0),
            true_sla=obs_data.get("true_sla", 1.0),
            true_queue=obs_data.get("true_queue", 0.0),
            true_rps=obs_data.get("true_rps", 50.0),
            true_processed_rps=obs_data.get("true_processed_rps", 50.0),
            termination_reason=obs_data.get("termination_reason", "none"),
            action_taken=obs_data.get("action_taken", 0),
            episode_reward=obs_data.get("episode_reward", 0.0),
            done=done,
            reward=reward,
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict) -> SurgeState:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return SurgeState(**payload)
