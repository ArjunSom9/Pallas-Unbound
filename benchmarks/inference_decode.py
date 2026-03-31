"""
Autoregressive Decoding Performance Benchmark (inference_decode.py)

This module benchmarks the token-generation phase of the Attention mechanism.

Context (Phase IV - Hardware Utilization Profiling):
Unlike the training phase, which multiplies massive matrices and is Compute Bound
(measured in TFLOPS), the decoding phase multiplies a single Query vector against 
the massive Key-Value cache. This reduces Arithmetic Intensity to ~2 FLOPs/Byte.

Therefore, decoding performance is strictly dictated by the "Memory Wall". 
The goal of the custom Pallas decoding kernel is to achieve a memory bandwidth 
utilization that approaches the TPU v5e's physical limit of 819 GB/s.

Metrics:
- Latency per token (ms)
- Memory Bandwidth Utilization (GB/s)
"""

import time
import argparse
import jax
import jax.numpy as jnp

# Import our custom decoding kernel
from pallas_flash.kernels.decoding import pallas_flash_decoding

# -----------------------------------------------------------------------------
# 1. Baseline Implementation
# -----------------------------------------------------------------------------

@jax.jit
def baseline_decode_step(q: jax.Array, k_cache: jax.Array, v_cache: jax.Array) -> jax.Array:
    """Standard JAX attention for a single query token against the KV Cache."""
    scale = jnp.sqrt(jnp.array(q.shape[-1], dtype=q.dtype))
    
    # Q: (B, H, 1, D)
    # K: (B, H, N, D) -> swapaxes -> (B, H, D, N)
    # Logits: (B, H, 1, N)
    logits = jnp.matmul(q, k_cache.swapaxes(-1, -2)) / scale
    attn_weights = jax.nn.softmax(logits, axis=-1)
    
    # Attn: (B, H, 1, N) @ V: (B, H, N, D) -> Output: (B, H, 1, D)
    return jnp.matmul(attn_weights, v_cache)

# -----------------------------------------------------------------------------
# 2. Benchmarking Engine
# -----------------------------------------------------------------------------

def calculate_bandwidth(batch: int, heads: int, seq_len: int, head_dim: int, latency_s: float) -> float:
    """
    Calculates the memory bandwidth utilization in GB/s.
    
    Data Movement during a decoding step:
    - Read K Cache: B * H * N * D * 2 bytes (bf16)
    - Read V Cache: B * H * N * D * 2 bytes (bf16)
    - Read Q: B * H * 1 * D * 2 bytes (negligible)
    - Write O: B * H * 1 * D * 2 bytes (negligible)
    
    Total Bytes ~= 4 * B * H * N * D
    """
    bytes_transferred = 4 * batch * heads * seq_len * head_dim
    gb_transferred = bytes_transferred / (1024**3)
    
    return gb_transferred / latency_s if latency_s > 0 else 0.0

def benchmark_kernel(kernel_fn, q, k, v, num_warmup=3, num_steps=10):
    """
    Times a decoding step execution and returns the average latency.
    """
    try:
        # Warmup
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
        return None, type(e).__name__

# -----------------------------------------------------------------------------
# 3. Main Decoding Sweep
# -----------------------------------------------------------------------------

def run_decoding_sweep(batch: int, heads: int, head_dim: int, seq_lengths: list):
    print("=========================================================================================")
    print(f" Pallas-Flash: Autoregressive Decoding Benchmark")
    print(f" Fixed Dimensions -> Batch: {batch}, Heads: {heads}, HeadDim: {head_dim}")
    print(f" Target Hardware  -> Peak Memory Bandwidth: 819 GB/s (TPU v5e)")
    print("=========================================================================================")
    print(f"{'Cache Len (N)':<13} | {'Baseline Latency':<18} | {'Pallas Latency':<18} | {'Pallas Bandwidth':<18}")
    print("-" * 92)

    for seq_len in seq_lengths:
        key = jax.random.PRNGKey(seq_len)
        k_q, k_k, k_v = jax.random.split(key, 3)
        
        # NOTE: Sequence length for Q is exactly 1 for token decoding
        q_shape = (batch, heads, 1, head_dim)
        kv_shape = (batch, heads, seq_len, head_dim)
        
        q = jax.random.normal(k_q, q_shape, dtype=jnp.bfloat16)
        k = jax.random.normal(k_k, kv_shape, dtype=jnp.bfloat16)
        v = jax.random.normal(k_v, kv_shape, dtype=jnp.bfloat16)
        
        # 1. Benchmark Baseline
        base_lat, base_err = benchmark_kernel(baseline_decode_step, q, k, v)
        # Type narrowing: explicitly check if base_lat is None
        if base_err or base_lat is None:
            base_str = f"FAILED ({base_err})"
        else:
            base_str = f"{base_lat * 1000:.3f} ms"
            
        # 2. Benchmark Pallas-Flash Decoding Kernel
        pallas_lat, pallas_err = benchmark_kernel(pallas_flash_decoding, q, k, v)
        
        # Type narrowing: explicitly check if pallas_lat is None
        if pallas_err or pallas_lat is None:
            pallas_str = f"FAILED ({pallas_err})"
            bw_str = "N/A"
        else:
            pallas_str = f"{pallas_lat * 1000:.3f} ms"
            bandwidth = calculate_bandwidth(batch, heads, seq_len, head_dim, pallas_lat)
            bw_str = f"{bandwidth:.1f} GB/s"

        print(f"{seq_len:<13} | {base_str:<18} | {pallas_str:<18} | {bw_str:<18}")
        
        del q, k, v

    print("=========================================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pallas-Flash Decoding Benchmark")
    # We use a larger batch size (e.g., 8) to put sufficient pressure on the HBM to get accurate BW measurements
    parser.add_argument("--batch", type=int, default=8, help="Batch Size (B)")
    parser.add_argument("--heads", type=int, default=16, help="Number of Attention Heads (H)")
    parser.add_argument("--dim", type=int, default=128, help="Head Dimension (D)")
    args = parser.parse_args()

    # Varying KV Cache sizes representing different stages of sequence generation
    cache_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
    
    run_decoding_sweep(
        batch=args.batch, 
        heads=args.heads, 
        head_dim=args.dim, 
        seq_lengths=cache_lengths
    )