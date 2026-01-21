# Pallas-Flash Implementation Roadmap

## 1. Project Overview & Directory Structure

```text
pallas-flash-tpu/
├── pyproject.toml              # Dependency management (jax[tpu], flax, chex).

```

## 2. Implementation Phases

### Phase 1: The "Lab Bench" (Infrastructure & Baselines)

**Goal:** Establish the mathematical control group and the measurement instruments. We cannot optimize what we cannot measure.

1. `pyproject.toml`

* **Why:** Sets up the environment with `jax[tpu]`. We need the runtime before anything else.