"""
Low-Level Hardware Intrinsics for TPU v5e Architecture (intrinsics.py)

Updated: Fixed 'Bad lhs type' by using explicit dot_general lowering for Mosaic.
"""

import jax
import jax.numpy as jnp

def mxu_matmul(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    """
    Directly targets the TPU v5e Matrix Multiplication Unit (MXU).
    
    Fix: Ensures Mosaic identifies this as a valid TPU MatMul by providing
    bfloat16 inputs and requesting float32 accumulation explicitly.
    """
    # Force inputs to bfloat16 (The only type the v5e MXU accepts for MatMul)
    lhs_bf16 = lhs.astype(jnp.bfloat16)
    rhs_bf16 = rhs.astype(jnp.bfloat16)
    
    # dimension_numbers for (M, K) @ (K, N) -> (M, N)
    # Contracting dim is last of LHS (1) and first of RHS (0)
    dn = (((len(lhs.shape) - 1,), (0,)), ((), ()))
    
    # On TPU v5e, using Precision.DEFAULT with preferred_element_type=float32
    # is the standard way to trigger the bf16 -> f32 accumulation path in Mosaic.
    return jax.lax.dot_general(
        lhs_bf16, 
        rhs_bf16, 
        dimension_numbers=dn,
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32
    )

def vpu_stable_exp(x: jax.Array, max_val: jax.Array) -> jax.Array:
    """
    Directly targets the TPU v5e Vector Processing Unit (VPU).
    Computes exp(x - max) with high numerical stability.
    """
    x_f32 = x.astype(jnp.float32)
    max_f32 = max_val.astype(jnp.float32)
    res = jnp.exp(x_f32 - max_f32)
    return res.astype(jnp.bfloat16)

def cast_to_fp32(x: jax.Array) -> jax.Array:
    """Explicitly promotes a tensor to float32."""
    return x.astype(jnp.float32)

def cast_to_bf16(x: jax.Array) -> jax.Array:
    """Downcasts to bfloat16."""
    return x.astype(jnp.bfloat16)