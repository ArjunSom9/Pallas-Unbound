"""
Pallas Tiling Strategies and BlockSpec Definitions for TPU v5e (tiling.py)

This module defines the grid and memory layout specifications for the 
custom Pallas Attention kernel.

The Optimization Thesis (from Phase II of the Project Plan):
Standard GPU FlashAttention uses small blocks (128x64 or 128x128) to fit into 
tiny SRAM. The TPU v5e has ~128 MiB of VMEM. We exploit this by loading 
massive blocks of Queries (e.g., Block_Q = 1024) into VMEM and keeping them 
resident. By streaming K and V blocks (Block_KV = 128) against this large 
resident Q block, we maximize Arithmetic Intensity (FLOPs/Byte) to saturate 
the 197 TFLOPS compute ceiling without stalling on the 819 GB/s HBM bandwidth.
"""

import jax
from jax.experimental import pallas as pl
from typing import Tuple, Callable, Any

# -----------------------------------------------------------------------------
# 1. Tile Size Configurations
# -----------------------------------------------------------------------------

# Massive Query block leveraging the v5e's 128 MiB VMEM.
# A 1024x128 block in bf16 is only ~256 KB, easily fitting into VMEM
# with plenty of room left over for accumulators, masks, and pipelined K/V blocks.
BLOCK_Q = 1024

# Streaming block size for Keys and Values. 
# Matches the natural 128x128 systolic array (MXU) dimension.
BLOCK_KV = 128

# -----------------------------------------------------------------------------
# 2. Grid and BlockSpec Generator
# -----------------------------------------------------------------------------

def get_attention_specs(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    block_q: int = BLOCK_Q
) -> Tuple[Tuple[int, int, int], Tuple[pl.BlockSpec, pl.BlockSpec, pl.BlockSpec], pl.BlockSpec]:
    """
    Generates the Pallas Grid and BlockSpecs for the v5e Attention Kernel.

    Target Grid: (Batch_Size, Num_Heads, N_Blocks_Output)
    Unlike TPU v4/v5p which require complex dimension mappings for "Megacores", 
    the v5e's single TensorCore design allows a straightforward 3D grid mapping.

    Args:
        batch_size: Batch size dimension (B).
        num_heads: Number of attention heads (H).
        seq_len: Sequence length (N), padded to a multiple of 128.
        head_dim: Head dimension (D).
        block_q: Size of the Query block (defaults to 1024).

    Returns:
        grid: A tuple defining the 3D execution grid.
        in_specs: BlockSpecs for Q, K, and V inputs.
        out_specs: BlockSpec for the Output tensor.
    """
    
    # Calculate grid dimensions. 
    # Assumes seq_len is perfectly divisible by block_q (guaranteed by layout.py).
    num_blocks_q = seq_len // block_q
    
    # 3D Grid mapping to (B, H, num_blocks_q)
    grid = (batch_size, num_heads, num_blocks_q)

    # --- Query Spec ---
    # Sliced along the sequence dimension. The kernel instance gets a specific 
    # chunk of queries of shape (1, 1, BLOCK_Q, D).
    q_spec = pl.BlockSpec(
        index_map=lambda b, h, q_idx: (b, h, q_idx, 0),
        block_shape=(1, 1, block_q, head_dim)
    )

    # --- Key and Value Specs ---
    # K and V are NOT sliced along the sequence dimension by the grid.
    # We pass the full sequence (1, 1, seq_len, D) as a reference to the kernel.
    # The kernel will manually stream this from HBM using an inner loop 
    # and pl.load() calls with chunks of size BLOCK_KV.
    kv_spec = pl.BlockSpec(
        index_map=lambda b, h, q_idx: (b, h, 0, 0),
        block_shape=(1, 1, seq_len, head_dim)
    )

    # --- Output Spec ---
    # The output accumulator O has the exact same blocking pattern as Q.
    o_spec = pl.BlockSpec(
        index_map=lambda b, h, q_idx: (b, h, q_idx, 0),
        block_shape=(1, 1, block_q, head_dim)
    )

    in_specs = (q_spec, kv_spec, kv_spec)
    out_specs = o_spec

    return grid, in_specs, out_specs