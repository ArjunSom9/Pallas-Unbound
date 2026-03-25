# Pallas-Unbound: High-Performance Transformer Kernel for TPU v5e Architecture

## Executive Summary

The computational landscape for Large Language Models (LLMs) has undergone a paradigm shift, moving from a regime constrained by arithmetic throughput to one dominated by the limitations of memory bandwidth. This transition, often referred to as the "Memory Wall," necessitates a fundamental rethinking of how deep learning kernels are architected, particularly for the Attention mechanism which lies at the heart of the Transformer architecture.

The Google Cloud TPU v5e (Efficiency) represents a distinct class of domain-specific accelerators compared to its performance-focused predecessors. While architectures like the v4 and v5p optimize for maximum Floating Point Operations Per Second (FLOPS), the v5e is architected to maximize performance-per-dollar and energy efficiency, primarily for inference and mid-scale training workloads. This shift in hardware imposes new constraints: significantly reduced High Bandwidth Memory (HBM) capacity (16 GiB per chip versus 32 GiB on v4) and lower aggregate memory bandwidth (819 GB/s versus 1.2 TB/s).

**Pallas-Unbound** (internally known as "Pallas-Flash") evolves from a strategy of "supercomputer utilization" to one of "extreme resource efficiency". This project bypasses the high-level heuristics of the XLA (Accelerated Linear Algebra) compiler to achieve direct, manual control over the TPU memory hierarchy using JAX Pallas. 

## Key Deliverables

* **Pallas-Flash Training Kernel**: A highly optimized, IO-aware attention kernel that leverages the v5e's generous 128 MiB VMEM to maximize arithmetic intensity.
* **FlashDecoding Inference Kernel**: A distributed decoding kernel utilizing Split-KV parallelism across the 2x2 mesh to minimize single-batch latency.
* **v5e Benchmarking Suite**: A rigorous validation framework measuring "tokens per second per dollar," verifying the economic and performance advantages of the custom kernel on efficiency-tier hardware.

## The Hardware Context: TPU v5e Architecture

To engineer a kernel that approaches the theoretical roofline of the TPU v5e, it is imperative to understand its distinct microarchitecture.

### Compute Hierarchy
* **Single TensorCore**: The TPU v5e features a single TensorCore per chip. The mapping becomes a strict 1:1 relationship between the physical chip and the Pallas program instance. Each TensorCore has exclusive access to its 16 GiB of HBM and its ICI links, eliminating resource contention.
* **Matrix Multiply Unit (MXU)**: The v5e employs a systolic array architecture. Crucially, the v5e retains the standard 128x128 systolic array dimensions found in previous generations. Any matrix dimension in the kernel must be padded to a multiple of 128. 
* **Vector Processing Unit (VPU)**: The VPU is responsible for the element-wise operations critical to the Attention mechanism: Softmax exponentiation, scaling, and masking. The v5e design requires careful pipelining to overlap VPU activation calculations with MXU matrix multiplications.

### Memory Hierarchy: Exploiting the 128 MiB VMEM
* The v5e possesses approximately 128 MiB of Vector Memory (VMEM) per core.
* Instead of small 128x128 blocks, we can load huge chunks of the Query matrix (e.g., 1024x128) into VMEM and keep them resident.
* By reusing these resident Query blocks against streaming Key/Value blocks from HBM, we dramatically increase Arithmetic Intensity (FLOPs per byte transferred).

### Topology and Interconnects
* The deployment is a TPU v5e-4 slice consisting of 4 physical chips connected in a 2x2 Torus topology via Inter-Chip Interconnects (ICI).
* For slices of 1, 4, or 8 chips, the TPU v5e operates as a single-host device.
* The v5e features an ICI bandwidth of 400 GB/s per chip.

## Technical Deep Dive

### 1. The "Copy" Trap and Memory Alignment
The TPU MXU is a 128x128 systolic array. All inner dimensions of matrix multiplications must be multiples of 128. If inputs provided to the Pallas kernel are not padded, the XLA compiler will silently inject massive data-copy operations to pad them in HBM before the kernel runs. This doubles memory usage and kills performance. The data pipeline must utilize `jax.numpy.pad` to ensure all tensor dimensions passed to the kernel are strictly aligned to 128 bytes. 

### 2. Split-KV Decoding on a 4-Chip Mesh
A single query token attending to a 32k KV cache is purely memory-bound. On a single v5e chip, scanning a 32k KV cache (float16) involves reading ~130MB of data, taking ~158μs at 819 GB/s. To solve this, we implement Split-KV Decoding:
* **Shard the KV Cache**: The sequence length N is divided equally across the 4 chips. For N=32k, each chip holds 8k tokens.
* **Local Attention**: Each chip effectively runs a "mini" attention operation on its local 8k KV shard.
* **Global Reduction**: We use `jax.lax.psum` over the ICI network to combine results. We perform a stable LogSumExp reduction across the 4 chips to correctly normalize the partial attention scores.
* The aggregate HBM bandwidth of the 4-chip slice is 4 x 819 GB/s ≈ 3.2 TB/s. 

## Project Structure

```text
pallas-unbound/
├── benchmarks/
│   ├── baseline_mha.py      # Pure JAX Control Group 
│   ├── economics.py         # "Tokens Per Second Per Dollar" calculator
│   ├── roofline.py          # Arithmetic Intensity vs. Peak FLOPs plotter
│   └── scaling.py           # Linear vs. Quadratic scaling analysis
├── docs/
│   ├── assembly_analysis.md # "Before vs. After" trace evidence
│   ├── implementation_roadmap.md
│   └── project_plan.md      # Full strategic blueprint
├── profiling/
│   ├── hlo_analyzer.py      # Parses XLA dumps for "Copy Trap" instructions
│   ├── trace_viewer.py      # Detects pipeline bubbles in execution traces
│   └── xla_flags.py         # Configures '--xla_dump_to' for debugging
├── scripts/
│   ├── capture_trace.sh     # Trace collection wrapper
│   ├── dump_hlo.sh          # Helper to extract compilation artifacts
│   └── run_benchmark.sh     # E2E Latency/Throughput sweep
├── src/pallas_flash/
│   ├── config.py            # Hardware configs (VMEM=128MB, BW=819e9)
│   ├── distributed/
│   │   ├── collectives.py   # Custom 'jax.lax.psum' over ICI
│   │   └── mesh.py          # v5e-4 Topology: 2x2 Mesh setup
│   ├── kernels/
│   │   ├── attention.py     # Training Kernel: "Online Softmax" loop
│   │   ├── decoding.py      # Inference Kernel: Split-KV logic
│   │   ├── layout.py        # Padding logic for 128-byte alignment
│   │   ├── pipeline.py      # 'pl.pipeline' to prefetch K/V blocks
│   │   └── tiling.py        # BlockSpec logic for 1024-size tiles
│   ├── low_level/
│   │   └── intrinsics.py    # MXU wrappers (dot_general, bf16 conversion)
│   └── ops/
│       └── interface.py     # High-level `jax.custom_partitioning` wrapper
├── tests/
│   ├── test_correctness.py  # Validates output vs. baseline_mha.py
│   ├── test_memory.py       # Verifies VMEM usage < 128 MiB
│   └── test_numerics.py     # bfloat16 stability & NaN checks
├── Makefile                 # Automation panel
└── pyproject.toml           # JAX/Flax dependencies
```

## Getting Started

### Installation
Ensure you are operating on a `ct5lp-hightpu-4t` VM.

```bash
# Clone the repository
git clone [https://github.com/ArjunSom9/Pallas-Unbound.git](https://github.com/ArjunSom9/Pallas-Unbound.git)
cd Pallas-Unbound

# Run the environment setup script
bash completed_files/setup_vm_raw.sh

# Alternatively, use the Makefile
make install
```

### Automation & Makefile Commands
The project includes a robust `Makefile` for streamlined development.

* `make format`: Auto-format code using Black and Isort.
* `make check`: Verify code quality (linting).
* `make test`: Run the correctness and numerics suite.
* `make debug`: Run tests with XLA HLO dumping enabled for assembly analysis.
* `make analyze`: Parse HLO dumps to detect copy instructions.
* `make profile`: Capture JAX traces for timeline analysis.
* `make roofline`: Generate the Arithmetic Intensity vs TFLOPS plot.
* `make benchmark`: Run the E2E performance and economics sweep.
* `make clean`: Remove cache and compilation artifacts.

## Performance Benchmarking & Debugging

To scientifically validate performance, this framework measures:
* **Latency**: Wall-clock time per step.
* **HBM Bandwidth Utilization**: Using `jax.profiler` to monitor the memory bus.
* **Model FLOPs Utilization (MFU)**: The ratio of achieved FLOPs to the theoretical peak of 197 TFLOPS.

If you detect pipeline bubbles via the JAX Profiler Trace Viewer, increase the `Block_Q` size to increase compute duration in the inner loop, providing the DMA engine more time to complete the fetch.

## License
This project is licensed under the Apache License, Version 2.0. See the `LICENSE` file for full details.