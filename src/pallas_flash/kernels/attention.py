"""
Pallas FlashAttention Training Kernel (attention.py)

This module defines the core Pallas kernel for the TPU v5e. 

Context (Phase II - Kernel Architecture):
The v5e architecture features 128 MiB of on-chip Vector Memory (VMEM). 
Unlike GPU implementations that use small (128x128) tiles for both Q and K/V, 
this kernel loads a massive Query block (e.g., 1024x128) and keeps it resident 
in VMEM. It then delegates the Key/Value streaming to `pipeline_kv_loop`.

This maximizes Arithmetic Intensity (FLOPs/Byte) to prevent the MXU from stalling 
on the v5e's restricted 819 GB/s HBM bandwidth.
"""

import jax.numpy as jnp
from jax.experimental import pallas as pl

# Import the inner hardware-pipelined KV loop
from pallas_flash.kernels.pipeline import pipeline_kv_loop

def flash_attention_kernel(
    q_ref, k_ref, v_ref, o_ref,
    *,
    seq_len: int,
    block_kv: int,
    head_dim: int
):
    """
    The fused Pallas attention kernel for a single TensorCore.
    
    This kernel is mapped across the 3D grid: (Batch, Heads, Num_Q_Blocks).
    Because of the BlockSpec defined in `tiling.py`, the references provided 
    to this kernel are already sliced for the specific Batch and Head index.
    
    Args:
        q_ref: HBM Reference to the Query block. 
               Shape: (1, 1, BLOCK_Q, D).
        k_ref: HBM Reference to the full Sequence Keys for this batch/head. 
               Shape: (1, 1, N, D).
        v_ref: HBM Reference to the full Sequence Values for this batch/head. 
               Shape: (1, 1, N, D).
        o_ref: HBM Reference to the Output accumulator block. 
               Shape: (1, 1, BLOCK_Q, D).
        seq_len: Total sequence length (N).
        block_kv: The block size for streaming Keys and Values (e.g., 128).
        head_dim: The feature dimension of the heads (D).
    """
    
    # 1. Load Resident Query Block
    # We load the entire BLOCK_Q into the massive 128 MiB VMEM. 
    # Because q_ref is already sliced to shape 1 in batch/head dimensions, 
    # we index those dimensions with 0.
    q_block_4d = pl.load(q_ref, (0, 0, slice(None), slice(None)))
    
    # Squeeze out the batch and head dimensions to perform clean 2D 
    # matrix multiplications inside the pipeline loop.
    # Resulting shape: (BLOCK_Q, D)
    q_block_2d = jnp.squeeze(q_block_4d, axis=(0, 1))
    
    # 2. Execute Pipelined KV Streaming
    # K and V refs are passed directly to the inner loop which handles 
    # the asynchronous `pl.load` fetching.
    # We pass b_idx=0 and h_idx=0 because k_ref/v_ref are already narrowed 
    # to the current batch and head by the Pallas Grid index_map.
    o_acc_2d, m_final, l_final = pipeline_kv_loop(
        q_block=q_block_2d,
        k_ref=k_ref,
        v_ref=v_ref,
        seq_len=seq_len,
        block_kv=block_kv,
        head_dim=head_dim,
        b_idx=0, 
        h_idx=0  
    )
    
    # NOTE: `pipeline.py` uses swapaxes(0, 1). Ensure that in production,
    # if `k_block` is loaded as 4D inside `pipeline.py`, it is squeezed to 2D
    # prior to the transpose, or swapaxes(-1, -2) is used, to avoid rank mismatches 
    # with `q_block_2d`.
    
    # 3. Store the Output
    # Re-expand dimensions to match the 4D o_ref structure: (1, 1, BLOCK_Q, D)
    o_acc_4d = jnp.expand_dims(o_acc_2d, axis=(0, 1))
    
    # DMA Write back to High Bandwidth Memory (HBM)
    pl.store(o_ref, (0, 0, slice(None), slice(None)), o_acc_4d)