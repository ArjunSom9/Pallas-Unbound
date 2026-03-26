"""
Pallas-Flash High-Level Interface (interface.py)

This module provides the user-facing API for the optimized TPU v5e Attention kernel.
It orchestrates the end-to-end execution pipeline:
1. Padding inputs to MXU boundaries (avoiding the XLA Copy Trap).
2. Scaling queries.
3. Defining the Pallas grid and memory BlockSpecs.
4. Invoking the fused custom kernel.
5. Slicing the output back to the original logical dimensions.

This corresponds to Phase II (Kernel Dispatch & Layout Handling) of the project plan.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from functools import partial

# Internal imports from the pallas_flash package
from pallas_flash.kernels.layout import pad_tensor, unpad_tensor
from pallas_flash.kernels.tiling import get_attention_specs, BLOCK_Q, BLOCK_KV
from pallas_flash.kernels.attention import flash_attention_kernel


@jax.jit
def pallas_flash_attention(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """
    Computes Scaled Dot-Product Attention utilizing the TPU v5e optimized Pallas kernel.
    
    This function acts as a drop-in replacement for standard `jax.numpy` attention,
    but executes using a fused, memory-bandwidth-bypassing inner loop.
    
    Args:
        q: Query tensor of shape [Batch, Heads, SeqLen_Q, HeadDim]
        k: Key tensor of shape [Batch, Heads, SeqLen_KV, HeadDim]
        v: Value tensor of shape [Batch, Heads, SeqLen_KV, HeadDim]
        
    Returns:
        Output tensor of shape [Batch, Heads, SeqLen_Q, HeadDim]
    """
    
    # 1. Hardware Precision Enforcement
    # Ensure inputs are bfloat16 to maximize MXU utilization.
    q = q.astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = v.astype(jnp.bfloat16)
    
    # 2. Layout Alignment (The "Anti-Copy Trap" step)
    # We pad the sequence length and head dimension to multiples of the v5e's 128x128 MXU.
    # We only need to save the original shape of Q to unpad the final output.
    q_pad, q_original_shape = pad_tensor(q)
    k_pad, _ = pad_tensor(k)
    v_pad, _ = pad_tensor(v)
    
    # Extract padded dimensions
    batch_size, num_heads, seq_len_pad, head_dim_pad = q_pad.shape
    
    # 3. Query Scaling
    # We scale the queries prior to the kernel to save VPU cycles inside the tightly 
    # pipelined inner loop.
    scale_factor = 1.0 / jnp.sqrt(head_dim_pad)
    q_scaled = q_pad * scale_factor
    
    # 4. Kernel Spec Generation
    # Retrieve the Grid mapping and the BlockSpecs defining how memory is sliced 
    # between HBM and VMEM.
    grid, in_specs, out_specs = get_attention_specs(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len_pad,
        head_dim=head_dim_pad,
        block_q=BLOCK_Q
    )
    
    # 5. Bind Kernel Hyperparameters
    # We use functools.partial to bake the static sequence and block sizes into 
    # the kernel signature expected by `pallas_call`.
    bound_kernel = partial(
        flash_attention_kernel,
        seq_len=seq_len_pad,
        block_kv=BLOCK_KV,
        head_dim=head_dim_pad
    )
    
    # 6. Pallas Dispatch
    # The output accumulator O has the exact same shape and type as the padded Q.
    out_shape_struct = jax.ShapeDtypeStruct(q_scaled.shape, jnp.bfloat16)
    
    # Instantiate the custom XLA call
    # Note: For TPU v5e, we don't need complex `compiler_params` like dimension 
    # semantics required by Megacore architectures (v4/v5p).
    pallas_op = pl.pallas_call(
        bound_kernel,
        out_shape=out_shape_struct,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid
    )
    
    # Execute the fused kernel
    o_pad = pallas_op(q_scaled, k_pad, v_pad)
    
    # 7. Unpad to Logical Dimensions
    # Strip away the hardware-aligned padding zeroes so the user receives the 
    # exact shape they originally requested.
    o_unpadded = unpad_tensor(o_pad, q_original_shape)
    
    return o_unpadded

# Aliasing for ease of use
attention = pallas_flash_attention