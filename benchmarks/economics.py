"""
Economic Analysis and ROI Benchmark (economics.py)
"""
import time
import argparse
import jax
import jax.numpy as jnp
from pallas_flash.ops.interface import pallas_flash_attention

TPU_V5E_HOURLY_COST = 1.20 
COST_PER_SECOND = TPU_V5E_HOURLY_COST / 3600.0

@jax.jit
def baseline_mha(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    scale = jnp.sqrt(jnp.array(q.shape[-1], dtype=q.dtype))
    logits = jnp.matmul(q, k.swapaxes(-1, -2)) / scale
    attn_weights = jax.nn.softmax(logits, axis=-1)
    return jnp.matmul(attn_weights, v)

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

def calculate_economics(latency_s: float, batch: int, seq_len: int) -> dict:
    tokens_processed = batch * seq_len
    throughput_tps = tokens_processed / latency_s
    seconds_per_1m = 1_000_000 / throughput_tps
    cost_per_1m_tokens = seconds_per_1m * COST_PER_SECOND
    
    return {
        "throughput_tps": throughput_tps,
        "cost_per_1m": cost_per_1m_tokens
    }

def run_economic_sweep(batch: int, heads: int, head_dim: int, seq_lengths: list):
    print("=======================================================================================")
    print(" Pallas-Flash: Operating Cost & ROI Analysis")
    print(f" Assumptions: TPU v5e @ ${TPU_V5E_HOURLY_COST:.2f}/hr")
    print("=======================================================================================")
    print(f"{'Context (N)':<12} | {'Kernel':<15} | {'Throughput':<15} | {'Cost / 1M Tokens':<18}")
    print("-" * 87)

    for seq_len in seq_lengths:
        shape = (batch, heads, seq_len, head_dim)
        
        q = jax.random.normal(jax.random.PRNGKey(0), shape, dtype=jnp.bfloat16)
        k = jax.random.normal(jax.random.PRNGKey(1), shape, dtype=jnp.bfloat16)
        v = jax.random.normal(jax.random.PRNGKey(2), shape, dtype=jnp.bfloat16)
        
        base_lat, base_err = benchmark_kernel(baseline_mha, q, k, v)
        if base_err:
            base_tps_str, base_cost_str = f"OOM ({base_err})", "N/A"
            base_cost = float('inf')
        else:
            base_eco = calculate_economics(base_lat, batch, seq_len)
            base_tps_str = f"{base_eco['throughput_tps']:,.0f} Tok/s"
            base_cost_str = f"${base_eco['cost_per_1m']:.4f}"
            base_cost = base_eco['cost_per_1m']

        pallas_lat, pallas_err = benchmark_kernel(pallas_flash_attention, q, k, v)
        if pallas_err:
            pal_tps_str, pal_cost_str = f"FAILED ({pallas_err})", "N/A"
            pal_cost = float('inf')
        else:
            pal_eco = calculate_economics(pallas_lat, batch, seq_len)
            pal_tps_str = f"{pal_eco['throughput_tps']:,.0f} Tok/s"
            pal_cost_str = f"${pal_eco['cost_per_1m']:.4f}"
            pal_cost = pal_eco['cost_per_1m']

        print(f"{seq_len:<12} | {'Standard XLA':<15} | {base_tps_str:<15} | {base_cost_str:<18}")
        print(f"{'':<12} | {'Pallas-Flash':<15} | {pal_tps_str:<15} | \033[92m{pal_cost_str:<18}\033[0m")
        
        if base_cost != float('inf') and pal_cost != float('inf') and pal_cost > 0:
            savings_pct = ((base_cost - pal_cost) / base_cost) * 100
            multiplier = base_cost / pal_cost
            print(f"{'':<12} | {'--- ROI ---':<15} | \033[96m{multiplier:.2f}x Faster\033[0m  | \033[96m{savings_pct:.1f}% Cheaper\033[0m")
        elif base_cost == float('inf') and pal_cost != float('inf'):
            print(f"{'':<12} | {'--- ROI ---':<15} | \033[96mInfinite (Baseline Crashed)\033[0m")
            
        print("-" * 87)
        del q, k, v
    print("=======================================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pallas-Flash Economic Benchmark")
    parser.add_argument("--batch", type=int, default=1, help="Batch Size (B)")
    parser.add_argument("--heads", type=int, default=32, help="Number of Attention Heads (H)")
    parser.add_argument("--dim", type=int, default=128, help="Head Dimension (D)")
    args = parser.parse_args()

    # Capped at 16K to prevent hitting the 16MB scoped XLA VMEM compiler limit
    sequence_lengths = [2048, 4096, 8192, 16384]
    run_economic_sweep(args.batch, args.heads, args.dim, sequence_lengths)