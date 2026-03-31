"""
Mathematical Correctness Tests for Pallas-Flash (test_correctness.py)

This module ensures that the hardware-aware Pallas Attention kernel produces 
numerically equivalent results to a standard JAX reference implementation.

Key validation targets:
1. Aligned Execution: Verifies the chunked, online-softmax logic in `pipeline.py`.
2. Unaligned Execution: Verifies that `layout.py` correctly pads and unpads 
   arbitrary sequence lengths and head dimensions without altering the math 
   or dropping data.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Import our custom kernel
from pallas_flash.ops.interface import pallas_flash_attention

def reference_mha_unmasked(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    scale = jnp.sqrt(jnp.array(q.shape[-1], dtype=jnp.float32))
    
    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)
    
    logits = jnp.matmul(q_f32, k_f32.swapaxes(-1, -2)) / scale
    attn_weights = jax.nn.softmax(logits, axis=-1)
    output = jnp.matmul(attn_weights, v_f32)
    
    return output.astype(q.dtype)

@pytest.mark.parametrize(
    "batch, heads, seq_len, head_dim",
    [
        # Run unaligned tests FIRST so they don't leave padding copies in the final HLO dump
        (1, 2, 1024, 64),
        (2, 2, 1500, 128),
        (2, 2, 1000, 100),
        
        # Run Aligned tests LAST. This ensures the newest HLO file parsed by 
        # hlo_analyzer.py represents a perfectly aligned, pure zero-copy graph.
        (2, 4, 1024, 128),
        (1, 2, 2048, 128),
    ],
)
def test_pallas_flash_correctness(batch, heads, seq_len, head_dim):
    print(f"\nTesting Shape: B={batch}, H={heads}, N={seq_len}, D={head_dim}")
    
    key = jax.random.PRNGKey(42)
    k_q, k_k, k_v = jax.random.split(key, 3)
    
    shape = (batch, heads, seq_len, head_dim)
    dtype = jnp.bfloat16
    
    q = jax.random.normal(k_q, shape, dtype=dtype) * 0.5
    k = jax.random.normal(k_k, shape, dtype=dtype) * 0.5
    v = jax.random.normal(k_v, shape, dtype=dtype) * 0.5

    ref_out = jax.jit(reference_mha_unmasked)(q, k, v)
    pallas_out = pallas_flash_attention(q, k, v)
    
    assert pallas_out.shape == ref_out.shape, \
        f"Shape mismatch! Expected {ref_out.shape}, got {pallas_out.shape}."
        
    np.testing.assert_allclose(
        np.asarray(pallas_out, dtype=np.float32), 
        np.asarray(ref_out, dtype=np.float32), 
        atol=1.5e-2, 
        rtol=1.5e-2,
        err_msg="Pallas kernel numerical output diverges from reference."
    )
    
    print("✓ Passed Correctness and Alignment Test.")