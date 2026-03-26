"""
Low-Level Hardware Intrinsics for TPU v5e (intrinsics.py)

This module provides explicit wrappers for operations targeting the two 
primary compute engines of the TPU v5e TensorCore:
1. The MXU (Matrix Multiply Unit) - 197 TFLOPS (bfloat16)
2. The VPU (Vector Processing Unit) - Handles element-wise ops (Softmax, masking)

By explicitly casting inputs to bfloat16 and managing precision flags, 
we ensure the Pallas kernel does not silently fall back to slower float32 
vector lanes, thereby maximizing Model FLOPs Utilization (MFU).
"""

import jax
import jax.numpy as jnp

# -----------------------------------------------------------------------------
# 1. Precision & Casting Utilities
# -----------------------------------------------------------------------------

def cast_to_bf16(x: jax.Array) -> jax.Array:
    """
    Safely downcasts a tensor to bfloat16. 
    
    The v5e MXU natively operates on bfloat16. Feeding float32 to the MXU 
    will cause the compiler to either reject the kernel or emulate the math 
    using the VPU, decimating throughput.
    """
    if x.dtype == jnp.bfloat16:
        return x
    return x.astype(jnp.bfloat16)

def cast_to_fp32(x: jax.Array) -> jax.Array:
    """
    Upcasts to float32. 
    
    Used strictly for accumulating values in VMEM (like the Output accumulator
    or LogSumExp statistics) to prevent precision loss during the inner loop.
    """
    if x.dtype == jnp.float32:
        return x
    return x.astype(jnp.float32)

# -----------------------------------------------------------------------------
# 2. MXU (Matrix Multiply Unit) Operations
# -----------------------------------------------------------------------------

def mxu_matmul(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    """
    A hardware-aware wrapper for matrix multiplication targeting the MXU.
    
    This enforces bfloat16 precision and uses jax.lax.dot_general with 
    Precision.DEFAULT to guarantee systolic array utilization.
    
    Args:
        lhs: Left-hand side tensor (e.g., Query block).
        rhs: Right-hand side tensor (e.g., Key^T block).
        
    Returns:
        The resulting matrix product.
    """
    # 1. Enforce bfloat16 for maximum MXU throughput
    lhs_bf16 = cast_to_bf16(lhs)
    rhs_bf16 = cast_to_bf16(rhs)
    
    # 2. Execute dot product. In Pallas, jnp.matmul lowers to dot_general.
    # Setting precision to DEFAULT tells the XLA compiler it is safe to use 
    # the bfloat16 MXU hardware natively without software emulation.
    return jnp.matmul(
        lhs_bf16, 
        rhs_bf16, 
        precision=jax.lax.Precision.DEFAULT
    )

# -----------------------------------------------------------------------------
# 3. VPU (Vector Processing Unit) Operations
# -----------------------------------------------------------------------------

def vpu_stable_exp(x: jax.Array, max_val: jax.Array) -> jax.Array:
    """
    Computes a numerically stable exponential for the Online Softmax.
    
    bfloat16 has a massive dynamic range but very low precision (8 bits exponent, 
    7 bits mantissa). If `x` gets too large, `exp(x)` will overflow to Inf, 
    resulting in NaNs when multiplied by V.
    
    Args:
        x: The unnormalized attention logits (Q * K^T).
        max_val: The running maximum of the logits for the current row.
        
    Returns:
        exp(x - max_val) safely computed on the VPU.
    """
    # Shift logits by the running maximum to ensure the largest value is exp(0)=1
    shifted_logits = x - max_val
    
    # Exponentials are computed element-wise on the VPU. 
    # We maintain fp32 for the exponentiation to avoid underflow to 0 
    # for strongly negative attention scores.
    shifted_logits_fp32 = cast_to_fp32(shifted_logits)
    
    return jnp.exp(shifted_logits_fp32)