"""
Pipelined Key-Value Execution Loop (pipeline.py)

Fixed: Removed global program_id indexing. BlockSpec now handles 
batch and head slicing automatically.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from typing import Tuple
from pallas_flash.low_level.intrinsics import mxu_matmul, vpu_stable_exp, cast_to_fp32

def pipeline_kv_loop(
    q_block: jax.Array,
    k_ref, 
    v_ref, 
    padded_seq_len: int,
    original_seq_len: int,
    block_kv: int,
    head_dim: int
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    block_q = q_block.shape[0]
    num_kv_steps = padded_seq_len // block_kv

    # Accumulators in FP32
    o_acc = jnp.zeros((block_q, head_dim), dtype=jnp.float32)
    m_i = jnp.full((block_q, 1), -jnp.inf, dtype=jnp.float32)
    l_i = jnp.zeros((block_q, 1), dtype=jnp.float32)

    def kv_step(j: int, carry: Tuple[jax.Array, jax.Array, jax.Array]):
        o_acc_prev, m_prev, l_prev = carry

        # Load blocks (batch and head are already isolated by BlockSpec, so we use 0)
        k_block = pl.load(k_ref, (0, 0, pl.Slice(j * block_kv, block_kv), slice(None)))
        v_block = pl.load(v_ref, (0, 0, pl.Slice(j * block_kv, block_kv), slice(None)))

        # Matrix Multiply
        s_ij = mxu_matmul(q_block, jnp.swapaxes(k_block, 0, 1))

        # Padding Masking: Set out-of-bounds logits to effectively -infinity
        kv_indices = j * block_kv + jnp.arange(block_kv)
        mask = kv_indices < original_seq_len
        s_ij = jnp.where(mask[None, :], s_ij, -1e10)

        # Online Softmax Update (VPU)
        m_curr = jnp.maximum(m_prev, jnp.max(cast_to_fp32(s_ij), axis=-1, keepdims=True))
        p_ij = vpu_stable_exp(s_ij, m_curr)
        
        scale_factor = jnp.exp(m_prev - m_curr)
        l_curr = l_prev * scale_factor + jnp.sum(p_ij, axis=-1, keepdims=True)
        o_acc_scaled = o_acc_prev * scale_factor

        # Accumulate Output
        o_acc_curr = o_acc_scaled + cast_to_fp32(mxu_matmul(p_ij, v_block))

        return (o_acc_curr, m_curr, l_curr)

    o_acc_final, m_final, l_final = jax.lax.fori_loop(0, num_kv_steps, kv_step, (o_acc, m_i, l_i))
    return o_acc_final / l_final, m_final, l_final