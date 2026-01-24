# Pallas-Flash-v5e: Exhaustive Implementation Roadmap

This document outlines the implementation order for every file in the `pallas-flash-v5e` directory.

## Phase 0: Project Scaffold & Baselines
**Goal:** Initialize the package structure and prove the "Memory Wall" problem.

1. **scripts/setup_vm.sh**
    * **Action:** Script to install python dependencies, zsh, htop, and configure the TPU environment variables.

2. **pyproject.toml** <!-- Completed -->
    * **Action:** Define dependencies (`jax[tpu]`, `flax`, `libtpu-nightly`).

3. **Makefile**
    * **Action:** Create shortcuts for `make test` and `make benchmark`.

4. **.gitignore** <!-- Completed -->
    * **Action:** Ignore `/tmp/xla_dump`, `__pycache__`, `*.rtrace`, and `.venv` to keep the repo clean.

5. **Package Initialization (`__init__.py` files)** - do actual implementation later <!-- Completed -->
    * **Action:** Create empty `__init__.py` files to make directories importable:
        * `src/pallas_flash/__init__.py`
        * `src/pallas_flash/ops/__init__.py`
        * `src/pallas_flash/kernels/__init__.py`
        * `src/pallas_flash/distributed/__init__.py`
        * `src/pallas_flash/low_level/__init__.py`

6. **src/pallas_flash/config.py** <!-- Completed -->
    * **Action:** Define constants: HBM_BW (BW=819e9), VMEM=128MB.