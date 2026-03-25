"""
Pure JAX Control Group: Standard Multi-Head Attention (baseline_mha.py)

This script implements the baseline Scaled Dot-Product Attention mechanism 
using standard jax.numpy primitives and compiles it with XLA via @jax.jit.

According to Phase I of the "Pallas-Flash" project plan, this script serves
to quantify the "Memory Wall" on the TPU v5e's 16GB HBM. It scales the sequence
length (N) until an Out-Of-Memory (OOM) error occurs, caused by the materialization
of the massive (B, H, N, N) attention logits tensor.
"""

import jax
import jax.numpy as jnp
import time
import argparse

# -----------------------------------------------------------------------------
# 1. Mathematical Formulation of Standard Attention
# -----------------------------------------------------------------------------

@jax.jit
def baseline_mha(q, k, v):
    """
    Computes standard Scaled Dot-Product Attention:
    Attention(Q,K,V) = softmax(Q*K^T / sqrt(D) + M) * V
    
    Args:
        q: Query tensor of shape [B, H, N, D]
        k: Key tensor of shape [B, H, N, D]
        v: Value tensor of shape [B, H, N, D]
    
    Returns:
        output: Tensor of shape [B, H, N, D]
    """
    B, H, N, D = q.shape
    
    # Scale factor
    scale = jnp.sqrt(jnp.array(D, dtype=q.dtype))
    
    # 1. Compute Logits: Q * K^T
    # q is [B, H, N, D], k.swapaxes is [B, H, D, N]
    # Resulting logits shape: [B, H, N, N]
    # NOTE: This intermediate tensor is what causes the OOM on v5e at large N.
    logits = jnp.matmul(q, k.swapaxes(-1, -2)) / scale
    
    # 2. Causal Mask
    # Create a lower triangular mask. Upper elements are set to -inf.
    mask = jnp.triu(jnp.full((N, N), -jnp.inf, dtype=q.dtype), k=1)
    logits = logits + mask
    
    # 3. Softmax
    attn_weights = jax.nn.softmax(logits, axis=-1)
    
    # 4. Attn * V
    output = jnp.matmul(attn_weights, v)
    
    return output

# -----------------------------------------------------------------------------
# 2. Benchmarking Methodology
# -----------------------------------------------------------------------------

def benchmark_sequence_length(B, H, N, D, dtype=jnp.bfloat16):
    """
    Measures the latency of the baseline_mha function and estimates its memory.
    Gracefully catches OOM exceptions to demonstrate the Memory Wall.
    """
    print(f"\n--- Testing Sequence Length (N) = {N} ---")
    
    # Memory math for the intermediate (B, H, N, N) logits tensor
    bytes_per_elem = 2 if dtype == jnp.bfloat16 else 4
    intermediate_memory_gb = (B * H * N * N * bytes_per_elem) / (1024**3)
    print(f"[Estimate] Logits Tensor Memory: {intermediate_memory_gb:.2f} GB")
    
    if intermediate_memory_gb > 15.0:
        print("[Warning] Intermediate memory approaches/exceeds 16GB limit!")
    
    # Initialize random keys and tensors
    key = jax.random.PRNGKey(0)
    k_q, k_k, k_v = jax.random.split(key, 3)
    
    shape = (B, H, N, D)
    q = jax.random.normal(k_q, shape, dtype=dtype)
    k = jax.random.normal(k_k, shape, dtype=dtype)
    v = jax.random.normal(k_v, shape, dtype=dtype)
    
    try:
        # Warmup (forces JIT compilation)
        print("Compiling XLA Graph...")
        warmup_start = time.perf_counter()
        out = baseline_mha(q, k, v).block_until_ready()
        warmup_time = time.perf_counter() - warmup_start
        print(f"Compilation finished in {warmup_time:.3f}s")
        
        # Benchmark runs
        num_runs = 5
        print(f"Running {num_runs} benchmarking steps...")
        
        start_time = time.perf_counter()
        for _ in range(num_runs):
            out = baseline_mha(q, k, v).block_until_ready()
        end_time = time.perf_counter()
        
        avg_latency_ms = ((end_time - start_time) / num_runs) * 1000
        print(f"Success! Average Latency: {avg_latency_ms:.2f} ms per step")
        
        # Free up memory immediately
        del q, k, v, out
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Exception encountered!")
        print(f"Error Type: {type(e).__name__}")
        print("This is the expected Memory Wall failure mode on TPU v5e.")
        return False

# -----------------------------------------------------------------------------
# 3. Execution Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline MHA Memory Stress Test")
    parser.add_argument("--batch", type=int, default=1, help="Batch Size (B)")
    parser.add_argument("--heads", type=int, default=16, help="Number of Heads (H)")
    parser.add_argument("--dim", type=int, default=128, help="Head Dimension (D)")
    args = parser.parse_args()

    print("======================================================")
    print(f" Pallas-Flash Phase I: Baseline Memory Wall Benchmark")
    print(f" Fixed Dims -> Batch: {args.batch}, Heads: {args.heads}, Dim: {args.dim}")
    print("======================================================")
    
    # The progression of sequence lengths to test.
    # The PDF projects failure around 8k to 12k sequences for the v5e 16GB HBM.
    sequence_lengths = [1024, 2048, 4096, 8192, 12288, 16384, 32768]
    
    for seq_len in sequence_lengths:
        success = benchmark_sequence_length(
            B=args.batch, 
            H=args.heads, 
            N=seq_len, 
            D=args.dim, 
            dtype=jnp.bfloat16
        )
        
        if not success:
            print("\n[!] Stopping benchmark sweep due to OOM limit reached.")
            break
            
    print("\nBenchmark sweep completed.")