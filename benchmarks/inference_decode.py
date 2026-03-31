"""
Autoregressive Decoding Performance Benchmark (inference_decode.py)
"""
import time
import argparse
import jax
import jax.numpy as jnp
from pallas_flash.kernels.decoding import pallas_flash_decoding

@jax.jit
def baseline_decode_step(q: jax.Array, k_cache: jax.Array, v_cache: jax.Array) -> jax.Array:
    scale = jnp.sqrt(jnp.array(q.shape[-1], dtype=q.dtype))
    logits = jnp.matmul(q, k_cache.swapaxes(-1, -2)) / scale
    attn_weights = jax.nn.softmax(logits, axis=-1)
    return jnp.matmul(attn_weights, v_cache)

def calculate_bandwidth(batch: int, heads: int, seq_len: int, head_dim: int, latency_s: float) -> float:
    bytes_transferred = 4 * batch * heads * seq_len * head_dim
    gb_transferred = bytes_transferred / (1024**3)
    return gb_transferred / latency_s if latency_s > 0 else 0.0

def benchmark_kernel(kernel_fn, q, k, v, num_warmup=3, num_steps=10):
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
        
        q_shape = (batch, heads, 1, head_dim)
        kv_shape = (batch, heads, seq_len, head_dim)
        
        q = jax.random.normal(k_q, q_shape, dtype=jnp.bfloat16)
        k = jax.random.normal(k_k, kv_shape, dtype=jnp.bfloat16)
        v = jax.random.normal(k_v, kv_shape, dtype=jnp.bfloat16)
        
        base_lat, base_err = benchmark_kernel(baseline_decode_step, q, k, v)
        base_str = f"FAILED ({base_err})" if base_err else f"{base_lat * 1000:.3f} ms"
            
        pallas_lat, pallas_err = benchmark_kernel(pallas_flash_decoding, q, k, v)
        
        if pallas_err:
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
    parser.add_argument("--batch", type=int, default=8, help="Batch Size (B)")
    parser.add_argument("--heads", type=int, default=16, help="Number of Attention Heads (H)")
    parser.add_argument("--dim", type=int, default=128, help="Head Dimension (D)")
    args = parser.parse_args()

    # Capped at 16K to prevent hitting the 16MB scoped XLA VMEM compiler limit
    cache_lengths = [1024, 2048, 4096, 8192, 16384]
    run_decoding_sweep(args.batch, args.heads, args.dim, cache_lengths)