# Assembly-Level Debugging & The "Copy Trap" on TPU v5e

This document outlines the methodology for profiling and debugging Pallas-Flash kernels at the XLA assembly level on the TPU v5e architecture.

## 1. The TPU v5e Architecture and The "Copy Trap"

The Matrix Multiply Unit (MXU) on the TPU v5e is a strict $128 \times 128$ systolic array. To achieve maximum Model FLOPs Utilization (MFU), memory fed into the MXU must perfectly align with these boundaries. 

When tensors of arbitrary shapes (e.g., $N=1000, D=64$) are passed into standard `jax.numpy` operations, the XLA compiler silently injects padding instructions into the High-Level Optimizer (HLO) graph to make the math work on the hardware. 

We call this the **"Copy Trap"**. 

These implicit `copy`, `copy-start`, and `copy-done` instructions force the TPU to duplicate data in High Bandwidth Memory (HBM). Because the v5e is heavily constrained by its 819 GB/s HBM bandwidth, triggering the Copy Trap immediately throttles your kernel and can cause Out-Of-Memory (OOM) errors on the 16 GiB chip.

---

## 2. Generating HLO Assembly Dumps

To verify our kernel is zero-copy and respects memory limits, we must inspect the compiled assembly.

### Step 1: Trigger the Dump
Use the project automation to clear previous artifacts and generate new ones:
* **Command**: `make debug` (or `bash scripts/dump_hlo.sh`).
* **Action**: This sets the `XLA_FLAGS` to `--xla_dump_to=/tmp/xla_dump` and runs the correctness tests.

### Step 2: Locate the Final Graph
The compiler dumps dozens of `.txt` files representing different optimization passes. For accurate performance analysis, you want the file generated *last*, typically representing the graph after all optimizations and buffer assignments:
* **Filename**: `module_xxxx.after_optimizations_after_buffer_assignment.txt`

---

## 3. Reading the HLO Dump

HLO assembly is a static single assignment (SSA) representation of the hardware execution graph.

### ❌ Fatal: The Synchronous Copy Trap
Synchronous `copy` instructions are the primary indicator of failure. They signify the hardware is stalling to physically reshuffle memory in HBM.
```hlo
%copy.1 = f32[1024,128]{1,0} copy(f32[1024,128]{1,0} %custom-call.2)
```

### ⚠️ Warning: Standard DMA Setups
Asynchronous `copy-start` and `copy-done` pairs are used for DMA transfers. While excessive async copies indicate stalls, small counts (typically $\le 4$) are standard for initial parameter loading and parameter setup.

### ✅ Success: Zero-Copy Compute
A perfectly optimized kernel feeds directly into MXU `dot` operations without intermediate memory moves.
```hlo
%dot.2 = f32[1024,128]{1,0} dot(f32[1024,128]{1,0} %custom-call.1, ...), lhs_contracting_dims={1}
```

---

## 4. Hardware-Specific Debugging Cases

### The `swapaxes` Physical Copy
Using `jnp.swapaxes` on VMEM-resident blocks can trigger physical memory copies within the vector registers. The `pallas_flash` implementation avoids this by using `trans_rhs=True` in the low-level `mxu_matmul` intrinsic. This utilizes `dot_general` contracting dimensions to perform the transpose logically rather than physically.

### The 16 MiB Scoped VMEM Limit
The TPU v5e compiler has a strict **16 MiB limit** for any single VMEM allocation managed via a `BlockSpec`. For massive sequence lengths (e.g., $N=65536$), requesting the entire KV sequence in one block will cause an `XlaRuntimeError`. The solution is to utilize the `pipeline_kv_loop` to stream small chunks from HBM manually, keeping the active VMEM footprint minimal.

---

## 5. Automated Analysis

Use the provided tools to scan and verify your kernel artifacts:
* **HLO Analyzer**: `make analyze` parses dumps for "Copy Trap" instructions like `copy` and `dynamic-slice`.
* **Trace Viewer**: `python3 profiling/trace_viewer.py --dir=/tmp/pallas_trace` identifies "pipeline bubbles" where compute stalls for DMA loads.