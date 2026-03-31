"""
Performance Scaling and TFLOPS Benchmark (scaling.py)
"""
import os
import time
import argparse
import traceback
import jax
import jax.numpy as jnp
from pallas_flash.ops.interface import pallas_flash_attention

@jax.jit
def baseline_mha(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    scale = jnp.sqrt(jnp.array(q.shape[-1], dtype=q.dtype))
    logits = jnp.matmul(q, k.swapaxes(-1, -2)) / scale
    attn_weights = jax.nn.softmax(logits, axis=-1)
    return jnp.matmul(attn_weights, v)

def calculate_tflops(batch: int, heads: int, seq_len: int, head_dim: int, latency_s: float) -> float:
    flops = 4 * batch * heads * (seq_len ** 2) * head_dim
    tflops = flops / (1e12)
    return tflops / latency_s if latency_s > 0 else 0.0

def benchmark_kernel(kernel_fn, q, k, v, num_warmup=2, num_steps=5):
    try:
        for _ in range(num_warmup):
            _ = kernel_fn(q, k, v).block_until_ready()
        start_time = time.perf_counter()
        for _ in range(num_steps):
            _ = kernel_fn(q, k, v).block_until_ready()
        end_time = time.perf_counter()
        return (end_time - start_time) / num_steps, None
    except Exception as e:
        return None, type(e).__name__

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
        key = jax.random.PRNGKey(seq_len)
        k_q, k_k, k_v = jax.random.split(key, 3)
        shape = (batch, heads, seq_len, head_dim)
        
        q = jax.random.normal(k_q, shape, dtype=jnp.bfloat16)
        k = jax.random.normal(k_k, shape, dtype=jnp.bfloat16)
        v = jax.random.normal(k_v, shape, dtype=jnp.bfloat16)
        
        base_lat, base_err = benchmark_kernel(baseline_mha, q, k, v)
        base_str = f"FAILED ({base_err})" if base_err else f"{base_lat * 1000:.2f} ms"
            
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
            
        if pallas_err:
            pallas_str = f"FAILED ({pallas_err})"
            tflops_str = "N/A"
        else:
            pallas_str = f"{pallas_lat * 1000:.2f} ms"
            tflops = calculate_tflops(batch, heads, seq_len, head_dim, pallas_lat)
            tflops_str = f"{tflops:.1f} TFLOPS"

        print(f"{seq_len:<12} | {base_str:<18} | {pallas_str:<18} | {tflops_str:<15}")
        del q, k, v
    print("=========================================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pallas-Flash Scaling Benchmark")
    parser.add_argument("--batch", type=int, default=1, help="Batch Size (B)")
    parser.add_argument("--heads", type=int, default=16, help="Number of Attention Heads (H)")
    parser.add_argument("--dim", type=int, default=128, help="Head Dimension (D)")
    args = parser.parse_args()

    # Capped at 16K to prevent hitting the 16MB scoped XLA VMEM compiler limit
    sequence_lengths = [1024, 2048, 4096, 8192, 16384]
    run_scaling_sweep(args.batch, args.heads, args.dim, sequence_lengths)