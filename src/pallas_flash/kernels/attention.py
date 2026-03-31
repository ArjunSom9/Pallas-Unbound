"""
Pallas FlashAttention Training Kernel (attention.py)

Fixed: Now explicitly queries the program Grid ID to manually index into the
global HBM Key and Value References.
"""

import jax.numpy as jnp
from jax.experimental import pallas as pl
from pallas_flash.kernels.pipeline import pipeline_kv_loop

def flash_attention_kernel(
    q_ref, k_ref, v_ref, o_ref,
    *,
    padded_seq_len: int,
    original_seq_len: int,
    block_kv: int,
    head_dim: int
):
    # Retrieve the current grid coordinates to index into the global HBM K/V arrays
    b_idx = pl.program_id(0)
    h_idx = pl.program_id(1)
    
    # Load Q block using standard Python slices (Q is still mapped to VMEM via BlockSpec)
    q_block = pl.load(q_ref, (0, 0, slice(None), slice(None)))
    # Ensure rank is normalized for the pipeline
    q_block_2d = q_block.reshape(-1, head_dim)
    
    # Execute hardware-pipelined loop
    o_acc_2d, _, _ = pipeline_kv_loop(
        q_block=q_block_2d,
        k_ref=k_ref,
        v_ref=v_ref,
        padded_seq_len=padded_seq_len,
        original_seq_len=original_seq_len,
        block_kv=block_kv,
        head_dim=head_dim,
        b_idx=b_idx, 
        h_idx=h_idx  
    )
    
    # Cast to match HBM ref dtype and store
    pl.store(o_ref, (0, 0, slice(None), slice(None)), o_acc_2d.astype(jnp.bfloat16))