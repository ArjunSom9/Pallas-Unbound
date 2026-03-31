"""
Low-Level Hardware Intrinsics for TPU v5e Architecture (intrinsics.py)

Updated: Added `trans_rhs` logic to avoid physical memory copies inside 
the VMEM registers.
"""

import jax
import jax.numpy as jnp

def mxu_matmul(lhs: jax.Array, rhs: jax.Array, trans_rhs: bool = False) -> jax.Array:
    """
    Directly targets the TPU v5e Matrix Multiplication Unit (MXU).
    """
    lhs_bf16 = lhs.astype(jnp.bfloat16)
    rhs_bf16 = rhs.astype(jnp.bfloat16)
    
    if trans_rhs:
        # Contract LHS last dim with RHS last dim (equivalent to Q @ K^T)
        dn = (((len(lhs.shape) - 1,), (len(rhs.shape) - 1,)), ((), ()))
    else:
        # Contract LHS last dim with RHS first dim (equivalent to P @ V)
        dn = (((len(lhs.shape) - 1,), (0,)), ((), ()))
        
    return jax.lax.dot_general(
        lhs_bf16, 
        rhs_bf16, 
        dimension_numbers=dn,
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32
    )

def vpu_stable_exp(x: jax.Array, max_val: jax.Array) -> jax.Array:
    x_f32 = x.astype(jnp.float32)
    max_f32 = max_val.astype(jnp.float32)
    res = jnp.exp(x_f32 - max_f32)
    return res.astype(jnp.bfloat16)

def cast_to_fp32(x: jax.Array) -> jax.Array:
    return x.astype(jnp.float32)

def cast_to_bf16(x: jax.Array) -> jax.Array:
    return x.astype(jnp.bfloat16)