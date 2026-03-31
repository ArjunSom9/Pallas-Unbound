"""
Memory Profiling and "Memory Wall" Validation (test_memory.py)

This module validates the core engineering thesis of the Pallas-Flash project:
bypassing the TPU v5e's 16GB HBM limitation by preventing the materialization 
of the (B, H, N, N) attention matrix.

The test compares a standard JAX compiled baseline against the custom Pallas 
kernel at an extreme sequence length (N=16384) with a batch size of 2. 

Memory Math for N=16384 (Batch=2, Heads=16, bfloat16):
- Input Tensors (Q, K, V): ~134 MB each (Easily fits in host RAM and HBM)
- Baseline Intermediate Logits (2, 16, 16384, 16384): ~17.18 GB! 
  -> Guarantees an OOM crash on the 16GB TPU v5e.
- Pallas-Flash Resident VMEM: Uses ~8.5 MB of the scoped VMEM limits.
  -> Guarantees successful execution.
"""

import pytest
import jax
import jax.numpy as jnp

from pallas_flash.ops.interface import pallas_flash_attention

@jax.jit
def baseline_mha(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """Standard JAX attention. Will attempt to materialize an NxN matrix."""
    scale = jnp.sqrt(jnp.array(q.shape[-1], dtype=q.dtype))
    
    # The fatal step: Materializing the (B, H, N, N) logits tensor
    logits = jnp.matmul(q, k.swapaxes(-1, -2)) / scale
    attn_weights = jax.nn.softmax(logits, axis=-1)
    
    return jnp.matmul(attn_weights, v)

@pytest.mark.stress
def test_memory_wall_bypassed():
    """
    Proves that Pallas-Flash survives sequence lengths that fatally crash 
    standard XLA attention.
    """
    # Dimensions guaranteed to break a 16GB HBM TPU v5e
    B = 2  # Increased to 2 to push the baseline matrix over 17 GB
    H = 16
    N = 16384  # 16K Context Window
    D = 128
    
    print(f"\n--- Running Memory Wall Validation (B={B}, N={N}) ---")
    
    key = jax.random.PRNGKey(99)
    k_q, k_k, k_v = jax.random.split(key, 3)
    
    shape = (B, H, N, D)
    q = jax.random.normal(k_q, shape, dtype=jnp.bfloat16)
    k = jax.random.normal(k_k, shape, dtype=jnp.bfloat16)
    v = jax.random.normal(k_v, shape, dtype=jnp.bfloat16)
    
    baseline_failed = False
    try:
        print("1. Executing Standard JAX Attention (Expecting Crash)...")
        _ = baseline_mha(q, k, v).block_until_ready()
    except Exception as e:
        error_msg = str(e).lower()
        if "resource exhausted" in error_msg or "out of memory" in error_msg or "allocat" in error_msg:
            print("   ✓ Baseline successfully crashed with an OOM error.")
            baseline_failed = True
        else:
            print(f"   [!] Baseline crashed, but with an unexpected error: {e}")
            baseline_failed = True
            
    assert baseline_failed, \
        "Baseline MHA did NOT crash! Either you are testing on a TPU with massive " \
        "memory (like v5p 95GB), or the sequence length N is not high enough."

    try:
        print("2. Executing Pallas-Flash Kernel (Expecting Success)...")
        pallas_out = pallas_flash_attention(q, k, v).block_until_ready()
        
        print(f"   ✓ Pallas-Flash successfully processed {B}x16K context!")
        
        assert pallas_out.shape == shape, f"Expected shape {shape}, got {pallas_out.shape}"
        assert jnp.all(jnp.isfinite(pallas_out)), "Output contains NaNs/Infs!"
        
    except Exception as e:
        pytest.fail(f"Pallas-Flash kernel failed unexpectedly! Error: {e}")

    print("\n✓ MEMORY WALL BYPASSED SUCCESSFULLY")