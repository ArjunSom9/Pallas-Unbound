"""
Performance Scaling and TFLOPS Benchmark (scaling.py)

This module systematically measures the performance of the Pallas-Flash kernel 
across varying sequence lengths and compares it against a standard XLA baseline.

Context (Phase IV - Hardware Utilization Profiling):
The goal is to prove two things:
1. The "Memory Wall": The baseline will Out-Of-Memory (OOM) at large N (e.g., 32K).
2. The "Compute Bound": The Pallas-Flash kernel will not only survive large N 
   but will achieve significantly higher TFLOPS (approaching the 197 TFLOPS limit) 
   by overlapping memory I/O with MXU compute.

If `PALLAS_PROFILE_DIR` is present in the environment (set by `capture_trace.sh`), 
this script will also capture a hardware trace for the largest successful sequence.
"""

import os
import time
import argparse
import traceback
import jax
import jax.numpy as jnp

# Import our custom kernel
from pallas_flash.ops.interface import pallas_flash_attention

# -----------------------------------------------------------------------------
# 1. Baseline Implementation
# -----------------------------------------------------------------------------

@jax.jit
def baseline_mha(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """Standard JAX attention. Will OOM at large sequence lengths."""
    scale = jnp.sqrt(jnp.array(q.shape[-1], dtype=q.dtype))
    logits = jnp.matmul(q, k.swapaxes(-1, -2)) / scale
    attn_weights = jax.nn.softmax(logits, axis=-1)
    return jnp.matmul(attn_weights, v)

# -----------------------------------------------------------------------------
# 2. Benchmarking Engine
# -----------------------------------------------------------------------------

def calculate_tflops(batch: int, heads: int, seq_len: int, head_dim: int, latency_s: float) -> float:
    """
    Calculates the achieved TFLOPS.
    For standard Attention forward pass:
    - Q * K^T: 2 * B * H * N * N * D FLOPs
    - P * V  : 2 * B * H * N * N * D FLOPs
    Total FLOPs = 4 * B * H * N^2 * D
    """
    flops = 4 * batch * heads * (seq_len ** 2) * head_dim
    tflops = flops / (1e12)
    return tflops / latency_s if latency_s > 0 else 0.0

def benchmark_kernel(kernel_fn, q, k, v, num_warmup=2, num_steps=5):
    """
    Times a kernel execution and returns the average latency in seconds.
    Gracefully handles OOM errors.
    """
    try:
        # Warmup and Compilation
        for _ in range(num_warmup):
            _ = kernel_fn(q, k, v).block_until_ready()
            
        # Benchmarking
        start_time = time.perf_counter()
        for _ in range(num_steps):
            _ = kernel_fn(q, k, v).block_until_ready()
        end_time = time.perf_counter()
        
        avg_latency = (end_time - start_time) / num_steps
        return avg_latency, None
        
    except Exception as e:
        error_type = type(e).__name__
        return None, error_type

# -----------------------------------------------------------------------------
# 3. Main Scaling Execution
# -----------------------------------------------------------------------------

def run_scaling_sweep(batch: int, heads: int, head_dim: int, seq_lengths: list):
    print("=========================================================================================")
    print(f" Pallas-Flash TPU v5e Scaling Benchmark")
    print(f" Fixed Dimensions -> Batch: {batch}, Heads: {heads}, HeadDim: {head_dim}")
    print(f" Target Hardware  -> Peak: 197 TFLOPS | Bandwidth: 819 GB/s")
    print("=========================================================================================")
    print(f"{'Seq Len (N)':<12} | {'Baseline Latency':<18} | {'Pallas Latency':<18} | {'Pallas TFLOPS':<15}")
    print("-" * 89)

    trace_dir = os.environ.get("PALLAS_PROFILE_DIR")
    captured_trace = False

    for seq_len in seq_lengths:
        # Memory cleanup between runs
        key = jax.random.PRNGKey(seq_len)
        k_q, k_k, k_v = jax.random.split(key, 3)
        
        shape = (batch, heads, seq_len, head_dim)
        
        # NOTE: TPU MXU operates optimally on bfloat16
        q = jax.random.normal(k_q, shape, dtype=jnp.bfloat16)
        k = jax.random.normal(k_k, shape, dtype=jnp.bfloat16)
        v = jax.random.normal(k_v, shape, dtype=jnp.bfloat16)
        
        # 1. Benchmark Baseline
        base_lat, base_err = benchmark_kernel(baseline_mha, q, k, v)
        
        # Type narrowing: explicitly check if base_lat is None
        if base_err or base_lat is None:
            base_str = f"FAILED ({base_err})"
        else:
            base_str = f"{base_lat * 1000:.2f} ms"
            
        # 2. Benchmark Pallas-Flash
        # Optional: Capture trace on the highest sequence length before we exit
        is_tracing = False
        if trace_dir and not captured_trace and seq_len >= 8192:
            print(f"\n[Profiler] Capturing JAX trace for N={seq_len} at {trace_dir}...")
            jax.profiler.start_trace(trace_dir)
            is_tracing = True
            
        pallas_lat, pallas_err = benchmark_kernel(pallas_flash_attention, q, k, v)
        
        if is_tracing:
            jax.profiler.stop_trace()
            captured_trace = True
            print("-" * 89)
            
        # Type narrowing: explicitly check if pallas_lat is None
        if pallas_err or pallas_lat is None:
            pallas_str = f"FAILED ({pallas_err})"
            tflops_str = "N/A"
        else:
            pallas_str = f"{pallas_lat * 1000:.2f} ms"
            tflops = calculate_tflops(batch, heads, seq_len, head_dim, pallas_lat)
            tflops_str = f"{tflops:.1f} TFLOPS"

        print(f"{seq_len:<12} | {base_str:<18} | {pallas_str:<18} | {tflops_str:<15}")
        
        # Explicitly delete massive tensors to prevent host CPU OOMs during the loop
        del q, k, v

    print("=========================================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pallas-Flash Scaling Benchmark")
    parser.add_argument("--batch", type=int, default=1, help="Batch Size (B)")
    parser.add_argument("--heads", type=int, default=16, help="Number of Attention Heads (H)")
    parser.add_argument("--dim", type=int, default=128, help="Head Dimension (D)")
    args = parser.parse_args()

    # Standard progression of context windows
    sequence_lengths = [1024, 2048, 4096, 8192, 16384]
    
    run_scaling_sweep(
        batch=args.batch, 
        heads=args.heads, 
        head_dim=args.dim, 
        seq_lengths=sequence_lengths
    )