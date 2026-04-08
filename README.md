---
title: SurgeEnvV2
emoji: ⚡
colorFrom: blue
colorTo: orange
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - sre
---

# SurgeEnvV2 OpenEnv Submission

SurgeEnvV2 simulates an SRE traffic surge incident where an agent must stabilize service under random spikes while balancing reliability and infrastructure cost. The environment includes delayed scaling effects, lagged dashboards, queue dynamics, and imperfect mitigation tools (cache + rate limiting).

## What This Environment Simulates

- Random traffic spike timing and intensity on every episode.
- Imperfect operational controls: rate limiting and cache behavior are stochastic.
- Non-linear failure dynamics: DB contention, thread exhaustion, and queue overflow.
- Constrained optimization objective: maintain SLA while minimizing node cost.

## Observation Space

The observation vector is:

`[timestep, active_nodes, provisioning_nodes, observed_rps, observed_cpu, observed_db_latency, rate_limiting, cache_enabled]`

- `timestep`: current tick in episode.
- `active_nodes`: currently serving nodes.
- `provisioning_nodes`: nodes still booting (action delay).
- `observed_rps`: lagged requests/second metric.
- `observed_cpu`: lagged CPU utilization.
- `observed_db_latency`: lagged DB latency (ms).
- `rate_limiting`: binary toggle (`0/1`).
- `cache_enabled`: binary toggle (`0/1`).

## Action Space

Discrete(7) control actions:

- `0`: No-Op
- `1`: Scale Up
- `2`: Scale Down
- `3`: RateLimit ON
- `4`: RateLimit OFF
- `5`: Cache ON
- `6`: Cache OFF

## Reward Function

SurgeEnvV2 uses a constrained reward objective:

- If `SLA >= 0.95` (safe zone): reward is `1.0 - cost_penalty`, where cost tracks billed nodes.
- If `SLA < 0.95` (danger zone): apply cliff penalty + severity penalty + optional queue-overflow penalty.

This creates the intended tradeoff:

- Stay above SLA threshold to avoid severe penalties.
- Once safe, optimize for lower infrastructure cost.

## Graded Tasks

Three submission tasks are implemented in `tasks.py`:

- `easy`: **Survive the spike**
  - Maintain `SLA > 0.95` for full episode.
  - Keep node usage constrained (target <= 4).
- `medium`: **Cost-aware mitigation**
  - Maintain `SLA > 0.95`.
  - Keep average active nodes <= 3.
  - Use both cache and rate-limiting mitigations.
- `hard`: **Adaptive SRE**
  - Achieve final cumulative reward > 30.
  - Handle randomized spike timing/intensity.
  - Avoid early termination.

Each grader returns a float score in `[0.02, 0.98]` using OpenEnv rubric-style grading classes.

## Local Setup

```bash
cd surge
pip install -e .
```

Run the OpenEnv server locally:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Useful endpoints:

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `GET /docs`

## Baseline Inference

`inference.py` runs all three tasks and logs structured lines:

- `[START]`
- `[STEP]`
- `[END]`

Environment variables:

- `API_BASE_URL`: OpenAI-compatible endpoint URL.
- `MODEL_NAME`: model identifier.
- `HF_TOKEN`: API token.
- `ENV_BASE_URL`: OpenEnv server base URL (local or Space URL).

Run:

```bash
API_BASE_URL=https://router.huggingface.co/v1 MODEL_NAME=openai/gpt-4o-mini HF_TOKEN=<token> ENV_BASE_URL=http://localhost:8000 python inference.py
```

The script prints per-task scores and a final reproducible aggregate score to stdout.

## Docker Build and Run

```bash
docker build -t surge-env:latest -f Dockerfile .
docker run --rm -d -p 8000:8000 --name surge-env surge-env:latest
```

Health check:

```bash
curl http://localhost:8000/health
```

Run a full smoke test (`health`, `reset`, multiple `step`, and `state`):

```bash
python smoke_test.py --base-url http://localhost:8000 --steps 5
```

Stop container when done:

```bash
docker stop surge-env
```

## Hugging Face Spaces Deployment

From the environment directory (`surge/`):

```bash
openenv push
```

Or with options:

```bash
openenv push --repo-id <namespace>/<space-name> --private
```

OpenEnv will use `openenv.yaml` and your Docker configuration to deploy the environment server.

### Space Compatibility Notes

- `Dockerfile` reads `PORT` at runtime and binds `0.0.0.0:${PORT}` (HF-compatible).
- `inference.py` supports both local and Space endpoints via `ENV_BASE_URL`.
- No hardcoded local filesystem path is required for inference or server runtime.
- Configure Space secrets/variables:
  - `API_BASE_URL`
  - `MODEL_NAME`
  - `HF_TOKEN`

## Project Layout

```text
surge/
├── env.py                     # SurgeEnvV2 simulator (typed reset/step/state)
├── models.py                  # Action/Observation/State and response models
├── tasks.py                   # 3 graded tasks + rubric graders
├── inference.py               # Baseline task runner using OpenAI client
├── openenv.yaml               # OpenEnv manifest with task definitions
├── Dockerfile                 # Container image for deployment
├── client.py                  # Typed OpenEnv client
└── server/
    ├── app.py                 # FastAPI/OpenEnv app
    └── surge_environment.py   # Adapter from OpenEnv server API to SurgeEnvV2
```
