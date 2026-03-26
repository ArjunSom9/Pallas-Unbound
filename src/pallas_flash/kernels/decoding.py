"""
Pallas Flash Attention for Autoregressive Decoding (decoding.py)

This module implements the specialized attention kernel used during the 
generation phase (token-by-token decoding) on the TPU v5e.

Context & Engineering Challenge:
During prefill/training, Arithmetic Intensity is high because we multiply 
matrices (Q x K^T). During decoding, the sequence length of the Query is exactly 1. 
This means we are doing a Vector-Matrix multiplication against the massive KV cache.
Arithmetic Intensity drops to ~2 FLOPs/Byte.

We cannot be compute-bound here. Our goal shifts entirely to saturating the 
v5e's 819 GB/s HBM bandwidth. This kernel keeps the 1D Query vector resident 
in fast VMEM and aggressively streams the KV cache through the MXU, using an 
optimized 1D online softmax.
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
    k_ref: pl.Ref,
    v_ref: pl.Ref,
    seq_len: int,
    block_kv: int,
    head_dim: int,
    b_idx: int,
    h_idx: int
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    The pipelined inner loop specifically tailored for a 1D Query.
    
    Args:
        q_vec: Resident Query vector in VMEM. Shape: (1, D).
        k_ref: HBM Reference to KV Cache Keys. Shape: (1, 1, N, D).
        v_ref: HBM Reference to KV Cache Values. Shape: (1, 1, N, D).
        seq_len: Current length of the KV cache.
        block_kv: The block size for streaming (128).
        head_dim: Feature dimension.
    """
    num_kv_steps = seq_len // block_kv

    # Accumulators in fp32 to prevent precision degradation
    o_acc = jnp.zeros((1, head_dim), dtype=jnp.float32)
    m_i = jnp.full((1, 1), -jnp.inf, dtype=jnp.float32)
    l_i = jnp.zeros((1, 1), dtype=jnp.float32)

    def kv_step(j: int, carry: Tuple[jax.Array, jax.Array, jax.Array]):
        o_acc_prev, m_prev, l_prev = carry

        # 1. DMA Load KV Blocks from Cache
        k_block = pl.load(k_ref, (b_idx, h_idx, pl.slice(j * block_kv, block_kv), pl.slice(0, head_dim)))
        v_block = pl.load(v_ref, (b_idx, h_idx, pl.slice(j * block_kv, block_kv), pl.slice(0, head_dim)))

        # 2. Vector-Matrix Multiply
        # q_vec (1, D) @ K_block^T (D, BLOCK_KV) -> S_ij (1, BLOCK_KV)
        s_ij = mxu_matmul(q_vec, jnp.swapaxes(k_block, 0, 1))

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
    seq_len: int,
    block_kv: int,
    head_dim: int
):
    """
    The fused decoding kernel executed on the v5e TensorCore.
    Mapped across a 2D grid: (Batch, Heads).
    """
    # Load the 1D Query for this specific batch and head.
    # q_ref shape is (1, 1, 1, D). Squeeze out spatial dims to get (1, D).
    q_vec_4d = pl.load(q_ref, (0, 0, 0, slice(None)))
    q_vec_2d = jnp.squeeze(q_vec_4d, axis=(0, 1))
    
    # Execute pipelined cache streaming
    o_acc_2d, _, _ = decoding_kv_loop(
        q_vec=q_vec_2d,
        k_ref=k_ref,
        v_ref=v_ref,
        seq_len=seq_len,
        block_kv=block_kv,
        head_dim=head_dim,
        b_idx=0,
        h_idx=0
    )
    
    # Store back to HBM
    o_acc_4d = jnp.expand_dims(o_acc_2d, axis=(0, 1))
    pl.store(o_ref, (0, 0, 0, slice(None)), o_acc_4d)

# -----------------------------------------------------------------------------
# 3. High-Level JAX Dispatch API
# -----------------------------------------------------------------------------

def get_decoding_specs(seq_len: int, head_dim: int):
    """
    Generates BlockSpecs for the Decoding Phase.
    Notice the grid is only 2D (Batch, Heads) because Sequence Length for Q is 1.
    """
    q_spec = pl.BlockSpec(lambda b, h: (b, h, 0, 0), (1, 1, 1, head_dim))
    kv_spec = pl.BlockSpec(lambda b, h: (b, h, 0, 0), (1, 1, seq_len, head_dim))
    o_spec = pl.BlockSpec(lambda b, h: (b, h, 0, 0), (1, 1, 1, head_dim))
    
    return (q_spec, kv_spec, kv_spec), o_spec

@jax.jit
def pallas_flash_decoding(q: jax.Array, k_cache: jax.Array, v_cache: jax.Array) -> jax.Array:
    """
    Autoregressive attention step.
    
    Args:
        q: The current query token. Shape: [Batch, Heads, 1, HeadDim]
        k_cache: The Key Cache. Shape: [Batch, Heads, SeqLen, HeadDim]
        v_cache: The Value Cache. Shape: [Batch, Heads, SeqLen, HeadDim]
        
    Returns:
        The updated representation for the current token. Shape: [B, H, 1, D]
    """
    # 1. Enforce MXU Precision
    q = q.astype(jnp.bfloat16)
    k_cache = k_cache.astype(jnp.bfloat16)
    v_cache = v_cache.astype(jnp.bfloat16)
    
    # 2. Pad to MXU limits to avoid Copy Traps
    # K/V Caches must be padded on the seq_len axis if it's not a multiple of 128
    k_pad, _ = pad_tensor(k_cache, axes=(-2, -1))
    v_pad, _ = pad_tensor(v_cache, axes=(-2, -1))
    
    # Q only needs padding on the HeadDim axis, Sequence axis is 1 (ignored by layout block logic mostly, 
    # but strictly speaking, MXU expects 128 multiples. For pure 1D vector matrix mults, 
    # JAX can lower to VPU if not strictly MXU aligned, but we pad D to keep it hardware-happy).
    q_pad, q_orig_shape = pad_tensor(q, axes=(-1,))
    
    batch_size, num_heads, _, head_dim_pad = q_pad.shape
    _, _, seq_len_pad, _ = k_pad.shape
    
    # 3. Query Scaling
    scale_factor = 1.0 / jnp.sqrt(head_dim_pad)
    q_scaled = q_pad * scale_factor
    
    # 4. Kernel Specs
    grid = (batch_size, num_heads)
    in_specs, out_specs = get_decoding_specs(seq_len_pad, head_dim_pad)
    
    bound_kernel = partial(
        flash_decoding_kernel,
        seq_len=seq_len_pad,
        block_kv=BLOCK_KV_DECODE,
        head_dim=head_dim_pad
    )
    
    out_shape_struct = jax.ShapeDtypeStruct(q_scaled.shape, jnp.bfloat16)
    
    # 5. Dispatch
    pallas_op = pl.pallas_call(
        bound_kernel,
        out_shape=out_shape_struct,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid
    )
    
    o_pad = pallas_op(q_scaled, k_pad, v_pad)
    
    # 6. Restore logical dimensions
    return unpad_tensor(o_pad, q_orig_shape)