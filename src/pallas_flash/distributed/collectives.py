"""
Distributed Collective Operations for TPU v5e (collectives.py)

This module provides hardware-aware wrappers for JAX's collective communication 
primitives. 

Context:
In our Tensor Parallelism (TP) strategy, the Attention Heads are divided across 
the 4 chips of a TPU v5e-4 slice. 
- The Attention kernel (Pallas-Flash) runs completely independently on each chip.
- However, the subsequent Output Projection (W_O dense layer) computes partial 
  matrix multiplications that must be summed across all 4 chips to get the final 
  correct activation tensor.

These utilities leverage the v5e's dedicated Inter-Core Interconnect (ICI) 
links to perform these cross-chip reductions at maximum bandwidth.
"""

import jax
import jax.numpy as jnp
from typing import Any

# -----------------------------------------------------------------------------
# 1. Tensor Parallelism (TP) Collectives
# -----------------------------------------------------------------------------

def all_reduce_tp(x: jax.Array) -> jax.Array:
    """
    Sums a tensor across the Tensor Parallel ('tp') axis.
    
    Crucial for the Output Projection layer following the Attention mechanism.
    Each TPU chip computes `Attention_Local * W_O_Local`. This function performs 
    an All-Reduce to sum those partial results over the ICI network.
    
    Args:
        x: The local tensor chunk.
        
    Returns:
        The globally summed tensor, replicated identically on all 'tp' devices.
    """
    return jax.lax.psum(x, axis_name='tp')

def all_gather_tp(x: jax.Array, tiled_axis: int = -1) -> jax.Array:
    """
    Gathers sharded tensors across the 'tp' axis into a single concatenated tensor.
    
    Useful if downstream layers are not tensor-parallelized and require the 
    full, unsharded activation tensor.
    
    Args:
        x: The local tensor chunk.
        tiled_axis: The dimension along which to concatenate the gathered chunks.
        
    Returns:
        The full tensor, gathered from all TP devices.
    """
    return jax.lax.all_gather(x, axis_name='tp', tiled=True, axis=tiled_axis)

def pmax_tp(x: jax.Array) -> jax.Array:
    """
    Finds the maximum value across the Tensor Parallel ('tp') axis.
    
    (Note: Standard multi-head attention does not require cross-head softmax, 
    but this is provided for advanced architectures like Sequence Parallelism 
    where the sequence length is sharded across devices).
    """
    return jax.lax.pmax(x, axis_name='tp')

# -----------------------------------------------------------------------------
# 2. Data Parallelism (DP) Collectives
# -----------------------------------------------------------------------------

def all_reduce_dp(x: jax.Array) -> jax.Array:
    """
    Sums a tensor across the Data Parallel ('dp') axis.
    
    Used exclusively during the backward pass to synchronize and average 
    gradients across different batches processed by different TP groups.
    
    Args:
        x: The local gradient tensor.
        
    Returns:
        The globally summed gradient tensor.
    """
    return jax.lax.psum(x, axis_name='dp')

def all_reduce_dp_mean(x: jax.Array) -> jax.Array:
    """
    Averages a tensor across the Data Parallel ('dp') axis.
    
    A convenience wrapper over `all_reduce_dp` for gradient synchronization.
    """
    return jax.lax.pmean(x, axis_name='dp')

# -----------------------------------------------------------------------------
# 3. Utility Wrappers
# -----------------------------------------------------------------------------

def cross_entropy_loss_dp(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """
    Computes a distributed categorical cross-entropy loss.
    
    Safely calculates the local loss and uses `pmean` to synchronize the 
    scalar loss value across the Data Parallel batch dimension.
    """
    # Numerically stable log-softmax
    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - max_logits
    log_probs = shifted_logits - jnp.log(jnp.sum(jnp.exp(shifted_logits), axis=-1, keepdims=True))
    
    # Gather the log probabilities of the target classes
    target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    local_loss = -jnp.mean(target_log_probs)
    
    # Synchronize across the batch (DP) dimension
    return jax.lax.pmean(local_loss, axis_name='dp')