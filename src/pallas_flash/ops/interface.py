"""
High-Level Dispatch Interface (interface.py)

Fixed: Used explicit keyword arguments for BlockSpec to resolve Pylance 
type-checking errors and ensure cross-version JAX compatibility.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from functools import partial

from pallas_flash.kernels.attention import flash_attention_kernel
from pallas_flash.kernels.layout import pad_tensor, unpad_tensor

# v5e Optimized Block Sizes
BLOCK_Q = 1024
BLOCK_KV = 128

@jax.jit
def pallas_flash_attention(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    # 1. Capture original metadata
    orig_batch, orig_heads, orig_seq, orig_dim = q.shape
    
    # 2. Hardware Alignment (Padding)
    k_pad, _ = pad_tensor(k, axes=(-2,))
    v_pad, _ = pad_tensor(v, axes=(-2,))
    q_pad, q_orig_shape = pad_tensor(q, axes=(-2, -1))
    
    batch_size, num_heads, padded_seq_q, head_dim_pad = q_pad.shape
    _, _, padded_seq_kv, _ = k_pad.shape
    
    # 3. Kernel Configuration
    scale = 1.0 / jnp.sqrt(head_dim_pad)
    q_scaled = (q_pad * scale).astype(jnp.bfloat16)
    
    # Grid: Batch x Heads x Q_Blocks
    grid = (batch_size, num_heads, padded_seq_q // BLOCK_Q)
    
    # 4. Define BlockSpecs using Keyword Arguments
    # This solves the Pylance "Argument of type Function cannot be assigned to block_shape" error.
    in_specs = (
        pl.BlockSpec(
            index_map=lambda b, h, q_idx: (b, h, q_idx, 0), 
            block_shape=(1, 1, BLOCK_Q, head_dim_pad)
        ),
        pl.BlockSpec(
            index_map=lambda b, h, q_idx: (b, h, 0, 0), 
            block_shape=(1, 1, padded_seq_kv, head_dim_pad)
        ),
        pl.BlockSpec(
            index_map=lambda b, h, q_idx: (b, h, 0, 0), 
            block_shape=(1, 1, padded_seq_kv, head_dim_pad)
        ),
    )
    
    out_spec = pl.BlockSpec(
        index_map=lambda b, h, q_idx: (b, h, q_idx, 0), 
        block_shape=(1, 1, BLOCK_Q, head_dim_pad)
    )
    
    # 5. Invoke Pallas
    pallas_op = pl.pallas_call(
        partial(
            flash_attention_kernel,
            padded_seq_len=padded_seq_kv,
            original_seq_len=orig_seq,
            block_kv=BLOCK_KV,
            head_dim=head_dim_pad
        ),
        out_shape=jax.ShapeDtypeStruct(q_scaled.shape, jnp.bfloat16),
        in_specs=in_specs,
        out_specs=out_spec,
        grid=grid
    )
    
    o_pad = pallas_op(q_scaled, k_pad.astype(jnp.bfloat16), v_pad.astype(jnp.bfloat16))
    return unpad_tensor(o_pad, q_orig_shape)