"""
Pipelined Key-Value Execution Loop for TPU v5e (pipeline.py)

This module contains the inner compute loop for the Pallas-Flash kernel.
It iterates over the Sequence Length (N) in chunks of BLOCK_KV.

The Goal (Phase II: Overlapping I/O and Compute):
By keeping a massive Query block resident in the 128 MiB VMEM, this loop 
streams Key and Value blocks from HBM. We rely on the XLA compiler's loop 
unrolling and software pipelining to issue asynchronous DMA loads for 
K_{j+1} and V_{j+1} while the MXU is actively computing Q * K_j^T.

This ensures a "solid wall" of MXU activity on the profiler trace, 
effectively bypassing the TPU v5e's 819 GB/s memory bandwidth limit.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from typing import Tuple

# Import low-level hardware intrinsics for MXU and VPU targeting
from pallas_flash.low_level.intrinsics import mxu_matmul, vpu_stable_exp, cast_to_fp32

def pipeline_kv_loop(
    q_block: jax.Array,
    k_ref: pl.Ref,
    v_ref: pl.Ref,
    seq_len: int,
    block_kv: int,
    head_dim: int,
    b_idx: int,
    h_idx: int
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Executes the pipelined inner loop over Key and Value blocks.
    
    This function implements the "Online Softmax" algorithm, maintaining 
    running statistics (max and sum) to compute the softmax incrementally 
    without materializing the full (N, N) attention matrix in HBM.

    Args:
        q_block: The resident Query block loaded into VMEM. Shape: (BLOCK_Q, D).
        k_ref: HBM Reference to the Key tensor. Shape: (1, 1, N, D).
        v_ref: HBM Reference to the Value tensor. Shape: (1, 1, N, D).
        seq_len: Total sequence length (N).
        block_kv: The block size for streaming Keys and Values (typically 128).
        head_dim: The feature dimension (D) of the heads.
        b_idx: Current batch index in the Pallas grid.
        h_idx: Current head index in the Pallas grid.

    Returns:
        A tuple containing the final normalized Accumulator (O), 
        the running max (M), and the running sum (L).
    """
    block_q = q_block.shape[0]
    num_kv_steps = seq_len // block_kv

    # 1. Initialize VMEM Accumulators
    # We explicitly cast to fp32 to prevent underflow/overflow during accumulation.
    # o_acc: Output accumulator for the attention weighted values.
    # m_i: Running maximum of the attention logits (per query).
    # l_i: Running sum of the exponentials (per query).
    o_acc = jnp.zeros((block_q, head_dim), dtype=jnp.float32)
    m_i = jnp.full((block_q, 1), -jnp.inf, dtype=jnp.float32)
    l_i = jnp.zeros((block_q, 1), dtype=jnp.float32)

    # 2. Define the Pipeline Step
    def kv_step(j: int, carry: Tuple[jax.Array, jax.Array, jax.Array]):
        o_acc_prev, m_prev, l_prev = carry

        # --- Pipeline Stage 1: Load (DMA Prefetch) ---
        # The XLA compiler overlaps these pl.load calls with the MXU compute 
        # of the *previous* iteration.
        k_block = pl.load(k_ref, (b_idx, h_idx, pl.slice(j * block_kv, block_kv), pl.slice(0, head_dim)))
        v_block = pl.load(v_ref, (b_idx, h_idx, pl.slice(j * block_kv, block_kv), pl.slice(0, head_dim)))

        # --- Pipeline Stage 2: Compute Logits (MXU) ---
        # Q * K^T. Uses the explicitly bfloat16-cast hardware intrinsic.
        # k_block must be transposed: (BLOCK_KV, D) -> (D, BLOCK_KV)
        s_ij = mxu_matmul(q_block, jnp.swapaxes(k_block, 0, 1))

        # --- Pipeline Stage 3: Online Softmax Stats (VPU) ---
        # Calculate the new maximum for stability
        m_curr = jnp.maximum(m_prev, jnp.max(cast_to_fp32(s_ij), axis=-1, keepdims=True))
        
        # Calculate exponential of current scores relative to the NEW max
        p_ij = vpu_stable_exp(s_ij, m_curr)
        
        # Scale previous running sum and accumulator down based on the new max
        scale_factor = jnp.exp(m_prev - m_curr)
        l_curr = l_prev * scale_factor + jnp.sum(p_ij, axis=-1, keepdims=True)
        
        # Scale previous output accumulator
        o_acc_scaled = o_acc_prev * scale_factor

        # --- Pipeline Stage 4: Update Output (MXU) ---
        # Acc += P * V
        o_acc_curr = o_acc_scaled + cast_to_fp32(mxu_matmul(p_ij, v_block))

        return (o_acc_curr, m_curr, l_curr)

    # 3. Execute the Loop
    # Using jax.lax.fori_loop ensures XLA lowers this into a tight hardware loop
    # rather than unrolling it infinitely (which would blow up compilation time).
    o_acc_final, m_final, l_final = jax.lax.fori_loop(
        0, 
        num_kv_steps, 
        kv_step, 
        (o_acc, m_i, l_i)
    )

    # 4. Final Normalization
    # Divide the weighted sum by the running sum of the exponentials to 
    # finalize the softmax operation.
    o_acc_normalized = o_acc_final / l_final

    return o_acc_normalized, m_final, l_final