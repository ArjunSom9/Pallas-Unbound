"""
Pallas Tiling Strategies and BlockSpec Definitions for TPU v5e (tiling.py)

This module defines the grid and memory layout specifications for the 
custom Pallas Attention kernel.
"""

import jax
from jax.experimental import pallas as pl
from typing import Tuple, Callable, Any

BLOCK_Q = 1024
BLOCK_KV = 128

def get_attention_specs(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    block_q: int = BLOCK_Q
) -> Tuple[Tuple[int, int, int], Tuple[pl.BlockSpec, pl.BlockSpec, pl.BlockSpec], pl.BlockSpec]:
    
    num_blocks_q = seq_len // block_q
    grid = (batch_size, num_heads, num_blocks_q)

    q_spec = pl.BlockSpec(
        index_map=lambda b, h, q_idx: (b, h, q_idx, 0),
        block_shape=(1, 1, block_q, head_dim)
    )

    o_spec = pl.BlockSpec(
        index_map=lambda b, h, q_idx: (b, h, q_idx, 0),
        block_shape=(1, 1, block_q, head_dim)
    )

    k_spec = pl.BlockSpec(
        index_map=lambda b, h, q_idx: (b, h, 0, 0),
        block_shape=(1, 1, seq_len, head_dim)
    )
    
    v_spec = pl.BlockSpec(
        index_map=lambda b, h, q_idx: (b, h, 0, 0),
        block_shape=(1, 1, seq_len, head_dim)
    )

    in_specs = (q_spec, k_spec, v_spec)

    return grid, in_specs, o_spec