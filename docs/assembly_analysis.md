# Assembly-Level Debugging & The "Copy Trap"

This document outlines the methodology for profiling and debugging Pallas-Flash kernels at the XLA assembly level on the TPU v5e.

## 1. The TPU v5e Architecture and The "Copy Trap"

The Matrix Multiply Unit (MXU) on the TPU v5e is a strict 128x128 systolic array. To achieve maximum Model FLOPs Utilization (MFU), memory fed into the MXU must perfectly align with these boundaries.

When you pass tensors of arbitrary shapes (e.g., Sequence Length = 1000, Head Dimension = 64) into standard `jax.numpy` operations, the XLA compiler attempts to be helpful. It silently injects padding instructions into the High-Level Optimizer (HLO) graph to make the math work on the hardware. 

We call this the **"Copy Trap"**. 

These implicit `copy`, `copy-start`, and `copy-done` instructions force the TPU to duplicate data in High Bandwidth Memory (HBM). Because the v5e is heavily constrained by its 819 GB/s HBM bandwidth, triggering the Copy Trap will immediately throttle your kernel and can easily cause Out-Of-Memory (OOM) errors.

### The Pallas-Flash Solution
Our kernel relies on the `pallas_flash.kernels.layout` module to do **explicit padding** in pure JAX *before* dispatching to the custom Pallas kernel. This allows us to guarantee to the XLA compiler that the inputs are perfectly aligned, forcing it to generate a "zero-copy" inner loop.

---

## 2. Generating HLO Assembly Dumps

To verify our kernel is zero-copy, we must inspect the compiled assembly. JAX provides an interface to dump the XLA HLO graphs.

You can automatically generate these dumps using the provided shell script:
`bash scripts/dump_hlo.sh benchmarks/baseline_mha.py`

Alternatively, you can enable it programmatically in your Python code:
```python
from pallas_flash.profiling.xla_flags import enable_hlo_dump

enable_hlo_dump("/tmp/xla_dump")
# Your JIT-compiled function call here
```

### Where to Look
The compiler will dump dozens of `.txt` files representing different optimization passes. You want to look at the file generated *last*, typically named something like:
`module_xxxx.after_optimizations.txt`

---

## 3. Reading the HLO Dump

HLO (High-Level Optimizer) assembly is a static single assignment (SSA) representation. Here is what you are looking for.

### ❌ Bad Assembly (The Copy Trap Triggered)
If your `layout.py` logic failed or your `BlockSpec` in `tiling.py` is misaligned, you will see operations like this injected into your graph:

```hlo
%copy.1 = f32[1024,128]{1,0} copy(f32[1024,128]{1,0} %custom-call.2)
%copy-start.3 = (f32[1024,128]{1,0}, u32[0]{0}) copy-start(f32[1024,128]{1,0} %copy.1)
%copy-done.4 = f32[1024,128]{1,0} copy-done((f32[1024,128]{1,0}, u32[0]{0}) %copy-start.3)
```
*Diagnosis:* The compiler is wasting precious HBM bandwidth duplicating memory asynchronously. 

### ✅ Good Assembly (Compute Bound)
A perfectly aligned Pallas kernel will compile down to native instructions without implicit data movement:

```hlo
%custom-call.1 = f32[1024,128]{1,0} custom-call(f32[1024,128]{1,0} %prm.1, ...), custom_call_target="pallas_call"
%dot.2 = f32[1024,128]{1,0} dot(f32[1024,128]{1,0} %custom-call.1, f32[128,128]{1,0} %prm.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
```
*Diagnosis:* The kernel is feeding directly into the MXU (`dot` operations) without intermediate copies.

---

## 4. Automated Analysis

Manually reading 10,000 lines of HLO assembly is tedious. Use the provided analyzer script to scan the dumps automatically:

`python3 profiling/hlo_analyzer.py --dir=/tmp/xla_dump`

The script will tally all instances of `copy`, `copy-start`, and `dynamic-slice`. If it reports `0`, you have successfully achieved a zero-copy layout.