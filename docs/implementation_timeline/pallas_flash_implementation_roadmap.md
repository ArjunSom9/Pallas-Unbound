# Pallas-Flash Implementation Roadmap

## 1. Project Overview & Directory Structure

```text
pallas-flash-tpu/
├── pyproject.toml              # Dependency management (jax[tpu], flax, chex).
├── src/
│   └── pallas_flash/
│       ├── config.py           # Hardware constants (TPU v5e-4 HBM BW, VPU FLOPs)

```

## 2. Implementation Phases

### Phase 1: The "Lab Bench" (Infrastructure & Baselines)

**Goal:** Establish the mathematical control group and the measurement instruments. We cannot optimize what we cannot measure.

1. `pyproject.toml`

* **Why:** Sets up the environment with `jax[tpu]`. We need the runtime before anything else.

2. `src/pallas_flash/config.py`

* **Why:** Defines the hardware constants (HBM Bandwidth = 1.2 TB/s, Matrix Unit dimensions = 128x128). These constants are required for the roofline model.