"""
Memory Layout and Padding Utilities for TPU v5e (layout.py)

This module enforces the strict memory alignment requirements of the TPU v5e
Matrix Multiply Unit (MXU). 

The TPU v5e relies on a 128x128 systolic array. If tensors passed to a Pallas 
kernel are not perfectly aligned to multiples of 128 in their inner dimensions 
(e.g., Sequence Length, Head Dimension), the XLA compiler will silently insert 
"copy-start" and "copy-done" HLO instructions. These implicit copies duplicate 
data in the limited 16GB HBM, causing catastrophic Out-Of-Memory (OOM) errors 
and severe performance degradation (the "Copy Trap").

These utilities ensure explicit padding is done via pure JAX before reaching
the custom kernel, allowing precise control over memory layout.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Sequence

# Fallback to 128 if config cannot be imported
try:
    from pallas_flash.config import MXU_SIZE
except ImportError:
    MXU_SIZE = 128

def get_padded_dim(dim: int, alignment: int = MXU_SIZE) -> int:
    """
    Calculates the nearest ceiling multiple of the alignment size.
    
    Args:
        dim: The original dimension size.
        alignment: The required alignment boundary (defaults to 128).
        
    Returns:
        The padded dimension size.
    """
    return (dim + alignment - 1) // alignment * alignment

def pad_tensor(x: jax.Array, axes: Sequence[int] = (-2, -1), alignment: int = MXU_SIZE) -> Tuple[jax.Array, Tuple[int, ...]]:
    """
    Pads specific axes of a tensor to be multiples of the MXU alignment.
    
    This function explicitly pads the tensor with zeros so that the target 
    dimensions match the hardware boundaries, avoiding implicit XLA copies.
    
    Args:
        x: The input tensor (e.g., Query, Key, or Value).
        axes: A sequence of axis indices to pad. Defaults to the last two 
              axes (typically Sequence Length and Head Dimension).
        alignment: The padding multiple (defaults to 128).
        
    Returns:
        A tuple containing:
            - The padded jax.Array.
            - The original shape tuple (needed for unpadding later).
    """
    original_shape = x.shape
    rank = len(original_shape)
    
    # Normalize negative axes to positive indices
    normalized_axes = [ax % rank for ax in axes]
    
    pad_widths = []
    requires_padding = False
    
    for i, dim in enumerate(original_shape):
        if i in normalized_axes:
            padded_dim = get_padded_dim(dim, alignment)
            pad_amount = padded_dim - dim
            pad_widths.append((0, pad_amount))
            if pad_amount > 0:
                requires_padding = True
        else:
            pad_widths.append((0, 0))
            
    if not requires_padding:
        return x, original_shape
        
    padded_x = jnp.pad(x, pad_widths, mode='constant', constant_values=0)
    return padded_x, original_shape

def unpad_tensor(x: jax.Array, original_shape: Tuple[int, ...]) -> jax.Array:
    """
    Slices a tensor back to its original unpadded shape.
    
    Used after the Pallas kernel execution to strip away the hardware-aligned
    padding and return the expected logical tensor dimensions to the user.
    
    Args:
        x: The padded output tensor from the kernel.
        original_shape: The shape tuple of the original tensor before padding.
        
    Returns:
        The sliced (unpadded) jax.Array.
    """
    if x.shape == original_shape:
        return x
        
    # Construct a tuple of slices spanning exactly the original dimensions
    slices = tuple(slice(0, dim) for dim in original_shape)
    return x[slices]