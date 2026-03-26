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

# -----------------------------------------------------------------------------
# 1. Reference Implementation
# -----------------------------------------------------------------------------

def reference_mha_unmasked(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """
    A pure JAX reference implementation of unmasked Scaled Dot-Product Attention.
    
    We accumulate in float32 to match the precision of the VPU online softmax 
    accumulators in the Pallas kernel.
    """
    scale = jnp.sqrt(jnp.array(q.shape[-1], dtype=jnp.float32))
    
    # Cast to fp32 for stable reference math
    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)
    
    # Q * K^T / sqrt(D)
    logits = jnp.matmul(q_f32, k_f32.swapaxes(-1, -2)) / scale
    
    # Softmax
    attn_weights = jax.nn.softmax(logits, axis=-1)
    
    # Attn * V
    output = jnp.matmul(attn_weights, v_f32)
    
    # Cast back to the original dtype (bfloat16)
    return output.astype(q.dtype)

# -----------------------------------------------------------------------------
# 2. Parameterized Test Suite
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "batch, heads, seq_len, head_dim",
    [
        # Case 1: Perfectly Aligned to MXU (128) and BLOCK_Q (1024)
        (2, 4, 1024, 128),
        (1, 2, 2048, 128),
        
        # Case 2: Unaligned Head Dimension (e.g., D=64)
        # Tests padding on the last axis.
        (1, 2, 1024, 64),
        
        # Case 3: Unaligned Sequence Length 
        # Tests padding on the sequence axis (forces layout.py to act).
        (2, 2, 1500, 128),
        
        # Case 4: Completely Unaligned
        # A stress test for the pad/unpad logic across multiple dimensions.
        (2, 2, 1000, 100),
    ],
)
def test_pallas_flash_correctness(batch, heads, seq_len, head_dim):
    """
    Compares the output of the Pallas kernel with the standard JAX reference.
    """
    print(f"\nTesting Shape: B={batch}, H={heads}, N={seq_len}, D={head_dim}")
    
    # 1. Setup Random Tensors
    key = jax.random.PRNGKey(42)
    k_q, k_k, k_v = jax.random.split(key, 3)
    
    shape = (batch, heads, seq_len, head_dim)
    
    # Use bfloat16 as it is the target precision for the v5e MXU
    dtype = jnp.bfloat16
    
    # Generate random normal data. Scale down slightly to prevent extreme
    # softmax saturation which makes floating-point differences hard to compare.
    q = jax.random.normal(k_q, shape, dtype=dtype) * 0.5
    k = jax.random.normal(k_k, shape, dtype=dtype) * 0.5
    v = jax.random.normal(k_v, shape, dtype=dtype) * 0.5

    # 2. Run Reference (JIT compiled for speed)
    ref_out = jax.jit(reference_mha_unmasked)(q, k, v)
    
    # 3. Run Custom Pallas Kernel
    pallas_out = pallas_flash_attention(q, k, v)
    
    # 4. Assert Shape Equality
    # If unpadding fails, the shape will be (B, H, padded_N, padded_D)
    assert pallas_out.shape == ref_out.shape, \
        f"Shape mismatch! Expected {ref_out.shape}, got {pallas_out.shape}. " \
        f"Unpadding logic in layout.py may be failing."
        
    # 5. Assert Numerical Equality
    # Bfloat16 has an 8-bit exponent but only a 7-bit mantissa. This inherently 
    # limits precision. We set wide tolerances (atol=1e-2, rtol=1e-2) which is 
    # standard for bf16 accumulation differences. 
    # The reference accumulates in standard matmuls, while Pallas chunks it into 
    # an online softmax which changes the order of operations, contributing to 
    # minor floating-point drift.
    np.testing.assert_allclose(
        np.asarray(pallas_out, dtype=np.float32), 
        np.asarray(ref_out, dtype=np.float32), 
        atol=1.5e-2, 
        rtol=1.5e-2,
        err_msg="Pallas kernel numerical output diverges from reference."
    )
    
    print("✓ Passed Correctness and Alignment Test.")