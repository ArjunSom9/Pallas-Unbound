"""
Pallas Flash Attention for Autoregressive Decoding (decoding.py)

Updated: Used explicit keyword arguments for BlockSpec to resolve Pylance 
type-checking errors and ensure cross-version JAX compatibility.
Added sequence padding masking to prevent zeros in padding from inflating 
the softmax denominator.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from functools import partial
from typing import Tuple

from pallas_flash.low_level.intrinsics import mxu_matmul, vpu_stable_exp, cast_to_fp32
from pallas_flash.kernels.layout import pad_tensor, unpad_tensor

# Standard blocking for streaming KV from HBM (matches v5e MXU alignment)
BLOCK_KV_DECODE = 128

# -----------------------------------------------------------------------------
# 1. Inner Decoding Pipeline Loop
# -----------------------------------------------------------------------------

def decoding_kv_loop(
    q_vec: jax.Array,
    k_ref,
    v_ref,
    padded_seq_len: int,
    original_seq_len: int,
    block_kv: int,
    head_dim: int,
    b_idx: int,
    h_idx: int
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    The pipelined inner loop specifically tailored for a 1D Query.
    """
    num_kv_steps = padded_seq_len // block_kv

    # Accumulators in fp32 to prevent precision degradation
    o_acc = jnp.zeros((1, head_dim), dtype=jnp.float32)
    m_i = jnp.full((1, 1), -jnp.inf, dtype=jnp.float32)
    l_i = jnp.zeros((1, 1), dtype=jnp.float32)

    def kv_step(j: int, carry: Tuple[jax.Array, jax.Array, jax.Array]):
        o_acc_prev, m_prev, l_prev = carry

        # 1. DMA Load KV Blocks from Cache
        k_block = pl.load(k_ref, (b_idx, h_idx, pl.Slice(j * block_kv, block_kv), slice(None)))
        v_block = pl.load(v_ref, (b_idx, h_idx, pl.Slice(j * block_kv, block_kv), slice(None)))

        # 2. Vector-Matrix Multiply
        s_ij = mxu_matmul(q_vec, jnp.swapaxes(k_block, 0, 1))

        # Padding Masking: Set out-of-bounds logits to effectively -infinity
        kv_indices = j * block_kv + jnp.arange(block_kv)
        mask = kv_indices < original_seq_len
        s_ij = jnp.where(mask[None, :], s_ij, -1e10)

        # 3. 1D Online Softmax (VPU)
        m_curr = jnp.maximum(m_prev, jnp.max(cast_to_fp32(s_ij), axis=-1, keepdims=True))
        p_ij = vpu_stable_exp(s_ij, m_curr)
        
        scale_factor = jnp.exp(m_prev - m_curr)
        l_curr = l_prev * scale_factor + jnp.sum(p_ij, axis=-1, keepdims=True)
        o_acc_scaled = o_acc_prev * scale_factor

        # 4. Update Output
        o_acc_curr = o_acc_scaled + cast_to_fp32(mxu_matmul(p_ij, v_block))

        return (o_acc_curr, m_curr, l_curr)

    o_acc_final, m_final, l_final = jax.lax.fori_loop(0, num_kv_steps, kv_step, (o_acc, m_i, l_i))
    o_acc_normalized = o_acc_final / l_final

    return o_acc_normalized, m_final, l_final

# -----------------------------------------------------------------------------
# 2. Pallas Kernel Definition
# -----------------------------------------------------------------------------

def flash_decoding_kernel(
    q_ref, k_ref, v_ref, o_ref,
    *,
    padded_seq_len: int,
    original_seq_len: int,
    block_kv: int,
    head_dim: int
):
    """
    The fused decoding kernel executed on the v5e TensorCore.
    """
    q_vec = pl.load(q_ref, (0, 0, 0, slice(None)))
    q_vec_2d = q_vec.reshape(1, head_dim)
    
    o_acc_2d, _, _ = decoding_kv_loop(
        q_vec=q_vec_2d,
        k_ref=k_ref,
        v_ref=v_ref,
        padded_seq_len=padded_seq_len,
        original_seq_len=original_seq_len,
        block_kv=block_kv,
        head_dim=head_dim,
        b_idx=0,
        h_idx=0
    )
    
    # Cast to bfloat16 and reshape to 1D to match the (0, 0, 0, slice) indexer
    o_acc_bf16 = o_acc_2d.astype(jnp.bfloat16).reshape(-1)
    pl.store(o_ref, (0, 0, 0, slice(None)), o_acc_bf16)

# -----------------------------------------------------------------------------
# 3. High-Level JAX Dispatch API
# -----------------------------------------------------------------------------

def get_decoding_specs(seq_len: int, head_dim: int):
    # Using explicit keyword arguments to satisfy Pylance type checking
    q_spec = pl.BlockSpec(
        index_map=lambda b, h: (b, h, 0, 0), 
        block_shape=(1, 1, 1, head_dim)
    )
    kv_spec = pl.BlockSpec(
        index_map=lambda b, h: (b, h, 0, 0), 
        block_shape=(1, 1, seq_len, head_dim)
    )
    o_spec = pl.BlockSpec(
        index_map=lambda b, h: (b, h, 0, 0), 
        block_shape=(1, 1, 1, head_dim)
    )
    
    return (q_spec, kv_spec, kv_spec), o_spec

@jax.jit
def pallas_flash_decoding(q: jax.Array, k_cache: jax.Array, v_cache: jax.Array) -> jax.Array:
    q = q.astype(jnp.bfloat16)
    k_cache = k_cache.astype(jnp.bfloat16)
    v_cache = v_cache.astype(jnp.bfloat16)
    
    orig_batch, orig_heads, orig_seq, orig_dim = k_cache.shape

    k_pad, _ = pad_tensor(k_cache, axes=(-2, -1))
    v_pad, _ = pad_tensor(v_cache, axes=(-2, -1))
    q_pad, q_orig_shape = pad_tensor(q, axes=(-1,))
    
    batch_size, num_heads, _, head_dim_pad = q_pad.shape
    _, _, seq_len_pad, _ = k_pad.shape
    
    scale_factor = 1.0 / jnp.sqrt(head_dim_pad)
    q_scaled = q_pad * scale_factor
    
    grid = (batch_size, num_heads)
    in_specs, out_specs = get_decoding_specs(seq_len_pad, head_dim_pad)
    
    bound_kernel = partial(
        flash_decoding_kernel,
        padded_seq_len=seq_len_pad,
        original_seq_len=orig_seq,
        block_kv=BLOCK_KV_DECODE,
        head_dim=head_dim_pad
    )
    
    out_shape_struct = jax.ShapeDtypeStruct(q_scaled.shape, jnp.bfloat16)
    
    pallas_op = pl.pallas_call(
        bound_kernel,
        out_shape=out_shape_struct,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid
    )
    
    o_pad = pallas_op(q_scaled, k_pad, v_pad)
    return unpad_tensor(o_pad, q_orig_shape)