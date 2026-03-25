# Pallas-Flash-v5e: Exhaustive Implementation Roadmap

This document outlines the implementation order for every file in the `pallas-flash-v5e` directory.

## Phase 0: Project Scaffold & Baselines
**Goal:** Initialize the package structure and prove the "Memory Wall" problem.

1. **scripts/setup_vm.sh**
    * **Action:** Script to install python dependencies, zsh, htop, and configure the TPU environment variables.

2. **pyproject.toml**
    * **Action:** Define dependencies (`jax[tpu]`, `flax`, `libtpu-nightly`).

3. **Makefile**
    * **Action:** Create shortcuts for `make test` and `make benchmark`.

4. **.gitignore**
    * **Action:** Ignore `/tmp/xla_dump`, `__pycache__`, `*.rtrace`, and `.venv` to keep the repo clean.

5. **Package Initialization (`__init__.py` files)** - do actual implementation later
    * **Action:** Create empty `__init__.py` files to make directories importable:
        * `src/pallas_flash/__init__.py`
        * `src/pallas_flash/ops/__init__.py`
        * `src/pallas_flash/kernels/__init__.py`
        * `src/pallas_flash/distributed/__init__.py`
        * `src/pallas_flash/low_level/__init__.py`

6. **src/pallas_flash/config.py**
    * **Action:** Define constants: HBM_BW (BW=819e9), VMEM=128MB.

7. **benchmarks/baseline_mha.py**
    * **Action:** Implement pure JAX Attention to trigger OOM at ~12k seq length.

8. **benchmarks/roofline.py**
    * **Action:** Plot Arithmetic Intensity vs. Peak TFLOPS (197 TF).

---

## Phase 1: Low-Level Foundations
**Goal:** Define the memory layout and debugging flags before writing complex logic.

7. **src/pallas_flash/kernels/layout.py**
    * **Action:** Implement padding functions for 128-byte alignment.

8. **src/pallas_flash/kernels/tiling.py**
    * **Action:** Implement BlockSpec logic. Hardcode Block_Q (Q=1024) for v5e.

9. **src/pallas_flash/low_level/intrinsics.py**
    * **Action:** Wrap `jax.lax.dot_general` with bfloat16 precision flags.

10. **profiling/xla_flags.py**
    * **Action:** Define XLA flags to dump HLO (needed for early debugging).

---

## Phase 2: Core Kernel & Rigorous Testing
**Goal:** A working, numerically stable training kernel.

11. **src/pallas_flash/kernels/pipeline.py**
    * **Action:** Implement `pl.pipeline` to prefetch K_{j+1} blocks.

12. **src/pallas_flash/kernels/attention.py**
    * **Action:** Implement the fused Pallas attention loop.

13. **src/pallas_flash/ops/interface.py**
    * **Action:** Wrap kernel in `jax.custom_partitioning`.

14. **tests/test_correctness.py**
    * **Action:** Compare Pallas output vs. Baseline output.

15. **tests/test_numerics.py**
    * **Action:** Verify bfloat16 stability (check for NaNs in Softmax).

16. **tests/test_memory.py**
    * **Action:** Assert that VMEM usage stays under 128MB limits.

---

## Phase 3: The "Ceiling" (Distributed Inference)
**Goal:** Unlock the 4-chip slice for FlashDecoding.

17. **src/pallas_flash/distributed/mesh.py**
    * **Action:** Configure `jax.sharding.Mesh` for 2x2 topology.

18. **src/pallas_flash/distributed/collectives.py**
    * **Action:** Implement `jax.lax.psum` over ICI.

19. **src/pallas_flash/kernels/decoding.py**
    * **Action:** Implement Split-KV Decoding (parallel sequence processing).

20. **tests/test_distributed.py**
    * **Action:** A unit test using `jax.sharding.Mesh` to ensure the 4 chips return the same result as a single-chip reference.

---

## Phase 4: Profiling, Automation & Documentation
**Goal:** Gather evidence and automated artifacts for the final report.

20. **scripts/dump_hlo.sh**
    * **Action:** Script to run tests with XLA_FLAGS active.

21. **profiling/hlo_analyzer.py**
    * **Action:** Python script to parse HLO dumps for "Copy" instructions.

22. **scripts/capture_trace.sh**
    * **Action:** Script to capture JAX profiles.

23. **profiling/trace_viewer.py**
    * **Action:** Tool to programmatically detect pipeline bubbles in traces.

24. **docs/assembly_analysis.md**
    * **Action:** Write-up of the HLO and Trace findings ("Before vs. After").

---

## Phase 5: Final Benchmarks & Report
**Goal:** Prove the economic and performance value.

25. **benchmarks/scaling.py**
    * **Action:** Generate Linear vs. Quadratic scaling graph.

26. **benchmarks/inference_decode.py**
    * **Action:** Benchmark loop simulating token generation on the 4-chip mesh.

27. **benchmarks/economics.py**
    * **Action:** Calculate Tokens/Sec/$ based on v5e pricing.

28. **scripts/run_benchmark.sh**
    * **Action:** Master script to run all benchmarks in sequence.

29. **README.md**
    * **Action:** Final executive summary linking to all artifacts.