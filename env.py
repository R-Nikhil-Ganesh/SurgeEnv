"""Core SurgeEnvV2 simulator logic with typed OpenEnv-compatible responses."""

from __future__ import annotations

from collections import deque
from typing import Any
from uuid import uuid4

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from .models import (
        SurgeAction,
        SurgeObservation,
        SurgeResetResponse,
        SurgeState,
        SurgeStepResponse,
    )
except ImportError:
    from models import (
        SurgeAction,
        SurgeObservation,
        SurgeResetResponse,
        SurgeState,
        SurgeStepResponse,
    )


# ==========================================
# MODULE 1: TRAFFIC GENERATOR
# ==========================================
class TrafficGenerator:
    """Handles randomized, non-deterministic workload generation."""

    def __init__(self, rng: np.random.Generator, base_rps: float = 50.0):
        self.rng = rng
        self.base_rps = base_rps
        # Randomize spike timing and intensity so the agent cannot memorize patterns.
        self.spike_start = int(self.rng.integers(10, 30))
        self.spike_duration = int(self.rng.integers(5, 15))
        self.spike_intensity = float(self.rng.uniform(500, 3000))

    def get_rps(self, timestep: int) -> float:
        noise = float(self.rng.normal(0, 15))
        if self.spike_start <= timestep <= self.spike_start + self.spike_duration:
            return float(np.clip(self.base_rps + self.spike_intensity + noise, 50, 5000))
        return float(np.clip(self.base_rps + noise, 50, 5000))


# ==========================================
# MODULE 2: SYSTEM PHYSICS (Imperfect Tools)
# ==========================================
class SystemModel:
    """Simulates thread exhaustion, connection pool contention, and imperfect mitigations."""

    def __init__(self, max_queue: float = 5000.0, rng: np.random.Generator | None = None):
        self.queue_size = 0.0
        self.max_queue = max_queue
        self.node_thread_pool = 250
        self.base_db_latency = 10.0
        self.rng = rng if rng else np.random.default_rng()

    def step(
        self,
        active_nodes: int,
        incoming_rps: float,
        cache_enabled: float,
        rate_limiting_enabled: float,
    ) -> dict[str, float]:
        # --- 1. IMPERFECT RATE LIMITING ---
        if rate_limiting_enabled == 1.0:
            # Sheds randomly between 30% and 60% of traffic.
            shed_fraction = float(self.rng.uniform(0.3, 0.6))
            effective_rps = incoming_rps * (1.0 - shed_fraction)
        else:
            effective_rps = incoming_rps

        # --- 2. PROBABILISTIC CACHE ---
        if cache_enabled == 1.0:
            # Base hit rate is noisy (75% to 95%).
            base_hit_rate = float(self.rng.uniform(0.75, 0.95))
            # Stampede penalty: massive RPS spikes degrade the hit rate.
            stampede_penalty = min(0.4, effective_rps / 5000.0)
            actual_hit_rate = max(0.1, base_hit_rate - stampede_penalty)
            db_hit_rate = 1.0 - actual_hit_rate
        else:
            db_hit_rate = 1.0
            actual_hit_rate = 0.0

        db_load_rps = effective_rps * db_hit_rate

        # --- 3. SCALING DIMINISHING RETURNS (DB Connection Pool) ---
        # 10 nodes fighting for locks is exponentially worse than 2 nodes.
        node_overhead_multiplier = 1.0 + (active_nodes / 5.0) ** 1.5
        effective_db_load = db_load_rps * node_overhead_multiplier

        latency_multiplier = 2 ** (effective_db_load / 300.0)
        true_db_latency = min(3000.0, self.base_db_latency + latency_multiplier)

        if cache_enabled == 1.0:
            true_db_latency = (true_db_latency * db_hit_rate) + (5.0 * actual_hit_rate)

        # --- 4. THREAD EXHAUSTION (Little's Law: L = lambda*W) ---
        required_threads = effective_rps * (true_db_latency / 1000.0)
        total_cluster_threads = max(1, active_nodes) * self.node_thread_pool

        if required_threads <= total_cluster_threads:
            # Healthy.
            true_cpu = required_threads / total_cluster_threads
            processed_rps = effective_rps
        else:
            # Degraded: Thread pool exhausted.
            true_cpu = 1.0
            processed_rps = total_cluster_threads / (true_db_latency / 1000.0)

        # --- 5. QUEUE DYNAMICS ---
        if effective_rps > processed_rps:
            spillover = effective_rps - processed_rps
            self.queue_size = min(self.max_queue, self.queue_size + spillover)
        else:
            spare_capacity = processed_rps - effective_rps
            if self.queue_size > 0:
                self.queue_size = max(0.0, self.queue_size - spare_capacity)
                true_cpu = 1.0
                processed_rps += min(spare_capacity, self.queue_size)

        return {
            "cpu": float(true_cpu),
            "db_latency": float(true_db_latency),
            "queue_size": float(self.queue_size),
            "processed_rps": float(processed_rps),
        }


# ==========================================
# MODULE 3: GYMNASIUM ENVIRONMENT
# ==========================================
class SurgeEnvV2(gym.Env):
    """SRE Simulator with delay, lagging metrics, cascading failures, and constrained optimization rewards."""

    metadata = {"render_modes": ["console"]}

    def __init__(self):
        super().__init__()

        # Actions: 0: No-Op, 1: Scale Up, 2: Scale Down,
        # 3: Rate Limit ON, 4: Rate Limit OFF, 5: Cache ON, 6: Cache OFF
        self.action_space = spaces.Discrete(7)

        # Obs: [timestep, active_nodes, provisioning_nodes, observed_rps, observed_cpu,
        # observed_db_latency, rate_limiting, cache_enabled]
        self.observation_space = spaces.Box(
            low=np.array([0, 1, 0, 0, 0.0, 0, 0.0, 0.0], dtype=np.float32),
            high=np.array([100, 10, 10, 5000, 1.0, 5000, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.np_random = np.random.default_rng()
        self.max_steps = 50
        self.node_boot_time = 3  # Action delay

        self._episode_id = str(uuid4())
        self._cumulative_reward = 0.0
        self._terminated_early = False
        self._termination_reason = "none"
        self._action_counts = {str(i): 0 for i in range(7)}

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.traffic = TrafficGenerator(self.np_random)
        self.system_model = SystemModel(rng=self.np_random)
        self.timestep = 0

        # Ground truth state
        self.active_nodes = 2
        self.provisioning_queue: list[int] = []
        self.true_rps = 50.0
        self.true_processed_rps = 50.0

        # SRE toggles
        self.rate_limiting = 0.0
        self.cache_enabled = 0.0

        # Telemetry (lagged dashboards) - 3 tick delay
        self.metric_history = {
            "rps": deque([50.0] * 3, maxlen=3),
            "cpu": deque([0.0] * 3, maxlen=3),
            "db_latency": deque([10.0] * 3, maxlen=3),
        }

        self.degraded_steps = 0
        self._episode_id = str(uuid4())
        self._cumulative_reward = 0.0
        self._terminated_early = False
        self._termination_reason = "none"
        self._action_counts = {str(i): 0 for i in range(7)}

        observation = self._build_observation(reward=0.0, done=False)
        return np.array(observation.vector, dtype=np.float32), self._observation_info(observation)

    def reset_typed(self, seed: int | None = None, options: dict[str, Any] | None = None) -> SurgeResetResponse:
        """Typed reset helper for OpenEnv-style callers."""
        self.reset(seed=seed, options=options)
        observation = self._build_observation(reward=0.0, done=False)
        return SurgeResetResponse(observation=observation, reward=0.0, done=False)

    def step(self, action: int | SurgeAction) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if isinstance(action, SurgeAction):
            selected_action = action.action
        else:
            selected_action = int(action)

        if selected_action < 0 or selected_action > 6:
            raise ValueError("Action must be an integer in [0, 6].")

        self._action_counts[str(selected_action)] += 1
        self.timestep += 1

        # 1. Process actions (with delay for scaling up)
        if selected_action == 1 and (self.active_nodes + len(self.provisioning_queue) < 10):
            self.provisioning_queue.append(self.timestep + self.node_boot_time)
        elif selected_action == 2 and self.active_nodes > 1:
            self.active_nodes -= 1  # Scale down is immediate
        elif selected_action == 3:
            self.rate_limiting = 1.0
        elif selected_action == 4:
            self.rate_limiting = 0.0
        elif selected_action == 5:
            self.cache_enabled = 1.0
        elif selected_action == 6:
            self.cache_enabled = 0.0

        # Boot provisioned nodes
        while self.provisioning_queue and self.provisioning_queue[0] <= self.timestep:
            self.provisioning_queue.pop(0)
            self.active_nodes += 1

        # 2. Generate workload
        self.true_rps = self.traffic.get_rps(self.timestep)

        # 3. Calculate true system physics (rate limiting happens inside)
        system_metrics = self.system_model.step(
            active_nodes=self.active_nodes,
            incoming_rps=self.true_rps,
            cache_enabled=self.cache_enabled,
            rate_limiting_enabled=self.rate_limiting,
        )

        self.true_processed_rps = system_metrics["processed_rps"]

        # 4. Update lagged dashboards
        self.metric_history["rps"].append(self.true_rps)
        self.metric_history["cpu"].append(system_metrics["cpu"])
        self.metric_history["db_latency"].append(system_metrics["db_latency"])

        # 5. Calculate SLA, reward & termination
        # Rate limiting intentionally destroys SLA because it drops traffic.
        current_sla = self.true_processed_rps / max(1.0, self.true_rps)

        step_reward = self._calculate_reward(current_sla)
        terminated, termination_penalty, reason = self._check_termination(current_sla)

        total_reward = float(step_reward + termination_penalty)
        self._cumulative_reward += total_reward
        if terminated and reason != "max_steps":
            self._terminated_early = True
        self._termination_reason = reason

        observation = self._build_observation(
            reward=total_reward,
            done=terminated,
            extra_metadata={
                "true_sla": float(current_sla),
                "true_queue": float(self.system_model.queue_size),
                "true_rps": float(self.true_rps),
                "true_processed_rps": float(self.true_processed_rps),
                "termination_reason": reason,
                "action": selected_action,
                "episode_reward": float(self._cumulative_reward),
            },
        )
        truncated = False
        return (
            np.array(observation.vector, dtype=np.float32),
            total_reward,
            terminated,
            truncated,
            self._observation_info(observation),
        )

    def step_typed(self, action: int | SurgeAction) -> SurgeStepResponse:
        """Typed step helper for OpenEnv-style callers."""
        _, reward, terminated, _, info = self.step(action)
        observation = self._build_observation(
            reward=reward,
            done=terminated,
            extra_metadata={
                "true_sla": float(info.get("sla", 0.0)),
                "true_queue": float(info.get("true_queue", 0.0)),
                "true_rps": float(info.get("true_rps", 0.0)),
                "true_processed_rps": float(info.get("true_processed_rps", 0.0)),
                "termination_reason": str(info.get("termination_reason", "none")),
                "action": int(info.get("action", 0)),
                "episode_reward": float(info.get("episode_reward", self._cumulative_reward)),
            },
        )
        return SurgeStepResponse(observation=observation, reward=reward, done=terminated)

    def state(self) -> SurgeState:
        """Return full typed simulator state."""
        current_sla = float(self.true_processed_rps / max(1.0, self.true_rps))
        return SurgeState(
            episode_id=self._episode_id,
            step_count=self.timestep,
            max_steps=self.max_steps,
            timestep=self.timestep,
            active_nodes=self.active_nodes,
            provisioning_nodes=len(self.provisioning_queue),
            provisioning_queue=list(self.provisioning_queue),
            true_rps=float(self.true_rps),
            true_processed_rps=float(self.true_processed_rps),
            current_sla=current_sla,
            queue_size=float(self.system_model.queue_size),
            max_queue=float(self.system_model.max_queue),
            rate_limiting=float(self.rate_limiting),
            cache_enabled=float(self.cache_enabled),
            degraded_steps=self.degraded_steps,
            terminated_early=self._terminated_early,
            termination_reason=self._termination_reason,
            cumulative_reward=float(self._cumulative_reward),
            action_counts=dict(self._action_counts),
            metric_history_rps=[float(v) for v in self.metric_history["rps"]],
            metric_history_cpu=[float(v) for v in self.metric_history["cpu"]],
            metric_history_db_latency=[float(v) for v in self.metric_history["db_latency"]],
        )

    def _build_observation(
        self,
        reward: float | None,
        done: bool,
        extra_metadata: dict[str, Any] | None = None,
    ) -> SurgeObservation:
        vector = [
            float(self.timestep),
            float(self.active_nodes),
            float(len(self.provisioning_queue)),
            float(self.metric_history["rps"][0]),
            float(self.metric_history["cpu"][0]),
            float(self.metric_history["db_latency"][0]),
            float(self.rate_limiting),
            float(self.cache_enabled),
        ]
        metadata = {
            "observation_space": [
                "timestep",
                "active_nodes",
                "provisioning_nodes",
                "observed_rps",
                "observed_cpu",
                "observed_db_latency",
                "rate_limiting",
                "cache_enabled",
            ],
            "action_space": {
                "0": "No-Op",
                "1": "Scale Up",
                "2": "Scale Down",
                "3": "Rate Limit ON",
                "4": "Rate Limit OFF",
                "5": "Cache ON",
                "6": "Cache OFF",
            },
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return SurgeObservation(
            vector=vector,
            timestep=int(vector[0]),
            active_nodes=int(vector[1]),
            provisioning_nodes=int(vector[2]),
            observed_rps=float(vector[3]),
            observed_cpu=float(vector[4]),
            observed_db_latency=float(vector[5]),
            rate_limiting=float(vector[6]),
            cache_enabled=float(vector[7]),
            true_sla=float(metadata.get("true_sla", 1.0)),
            true_queue=float(metadata.get("true_queue", 0.0)),
            true_rps=float(metadata.get("true_rps", 50.0)),
            true_processed_rps=float(metadata.get("true_processed_rps", 50.0)),
            termination_reason=str(metadata.get("termination_reason", "none")),
            action_taken=int(metadata.get("action", 0)),
            episode_reward=float(metadata.get("episode_reward", self._cumulative_reward)),
            reward=reward,
            done=done,
            metadata=metadata,
        )

    @staticmethod
    def _observation_info(observation: SurgeObservation) -> dict[str, Any]:
        """Convert typed observation metadata into Gymnasium info."""
        return {
            "sla": float(observation.metadata.get("true_sla", 0.0)),
            "true_queue": float(observation.metadata.get("true_queue", 0.0)),
            "true_rps": float(observation.metadata.get("true_rps", 0.0)),
            "true_processed_rps": float(observation.metadata.get("true_processed_rps", 0.0)),
            "termination_reason": str(observation.metadata.get("termination_reason", "none")),
            "action": int(observation.metadata.get("action", 0)),
            "episode_reward": float(observation.metadata.get("episode_reward", 0.0)),
            "observation": observation.model_dump(),
        }

    def _calculate_reward(self, current_sla: float) -> float:
        sla_threshold = 0.95
        total_billed_nodes = self.active_nodes + len(self.provisioning_queue)
        cost_penalty = total_billed_nodes / 10.0

        if current_sla >= sla_threshold:
            # SAFE ZONE: Optimize for cost.
            return 1.0 - cost_penalty

        # DANGER ZONE: Piecewise penalty (cost ignored).
        cliff_penalty = -2.0
        violation_severity = sla_threshold - current_sla
        severity_penalty = -10.0 * violation_severity
        queue_penalty = -5.0 if self.system_model.queue_size >= self.system_model.max_queue else 0.0

        return cliff_penalty + severity_penalty + queue_penalty

    def _check_termination(self, current_sla: float) -> tuple[bool, float, str]:
        if self.timestep >= self.max_steps:
            return True, 0.0, "max_steps"

        # 1. Catastrophic failure (unrecoverable).
        is_queue_full = self.system_model.queue_size >= self.system_model.max_queue
        is_sla_dead = current_sla < 0.10

        if is_queue_full and is_sla_dead:
            return True, -50.0, "catastrophic_failure"

        # 2. Prolonged instability (failing to mitigate).
        if current_sla < 0.95:
            self.degraded_steps += 1
        else:
            self.degraded_steps = 0

        if self.degraded_steps >= 10:  # 10 ticks in danger zone
            return True, -20.0, "prolonged_instability"

        # 3. Keep playing.
        return False, 0.0, "none"


if __name__ == "__main__":
    env = SurgeEnvV2()
    obs_vec, reset_info = env.reset(seed=42)
    _ = reset_info

    print("Environment initialized successfully.")
    print("Running a random agent baseline for 50 steps...")
    print("-" * 60)

    total_reward = 0.0
    for i in range(1, 51):
        action = int(env.action_space.sample())
        obs_vec, reward, terminated, truncated, info = env.step(action)
        _ = truncated
        obs = SurgeObservation(**info["observation"])
        total_reward += reward

        if i % 5 == 0 or terminated:
            print(
                f"Step: {i:02d} | Action: {action} | Nodes: {obs.active_nodes} (+{obs.provisioning_nodes} booting) | "
                f"Obs RPS: {obs.observed_rps:.0f} | Obs CPU: {obs.observed_cpu:.2f} | Obs Latency: {obs.observed_db_latency:.0f}ms | "
                f"True SLA: {obs.metadata.get('true_sla', 0.0):.2%} | Reward: {reward:.2f}"
            )

        if terminated:
            print("-" * 60)
            print(f"EPISODE TERMINATED AT STEP {i}")
            print(f"Reason: {obs.metadata.get('termination_reason', 'unknown')}")
            break

    final_state = env.state()
    print("-" * 60)
    print(f"Final Episode Reward: {total_reward:.2f}")
    print(f"Final state: {final_state.model_dump()}")
