"""
Distributed Mesh Configuration for TPU v5e-4 (mesh.py)

This module handles the multi-device distribution strategy for the Pallas-Flash
attention kernel. 

Context:
The target hardware is a Google Cloud TPU v5e-4 slice, which provides 4 individual 
TensorCore chips connected via ICI (Inter-Core Interconnect). To maximize throughput 
and memory capacity, we shard the massive (B, H, N, D) tensors across these 4 chips.

The optimal strategy for Attention is:
- Data Parallelism (DP) over the Batch dimension (B).
- Tensor Parallelism (TP) over the Heads dimension (H).

This module provides the logical Mesh and NamedSharding primitives to automatically 
distribute the inputs before they hit the Pallas kernel.
"""

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from typing import Tuple

def get_tpu_devices():
    """Returns the available JAX devices, defaulting to CPU if TPU is unavailable."""
    devices = jax.devices()
    print(f"[Mesh Setup] Found {len(devices)} device(s): {devices[0].platform.upper()}")
    return devices

def create_v5e_mesh(dp_size: int = 1, tp_size: int = 4) -> Mesh:
    """
    Creates a 2D logical device mesh from the physical TPU v5e-4 topology.
    
    Args:
        dp_size: Number of devices for Data Parallelism (Batch splitting).
        tp_size: Number of devices for Tensor Parallelism (Head splitting).
        
    Returns:
        A jax.sharding.Mesh context object mapped to ('dp', 'tp').
    """
    devices = get_tpu_devices()
    num_devices = len(devices)
    
    if num_devices < (dp_size * tp_size):
        print(f"[Warning] Requested DP({dp_size}) x TP({tp_size}) = {dp_size * tp_size} devices, "
              f"but only found {num_devices}. Defaulting to 1x{num_devices} mesh.")
        dp_size = 1
        tp_size = num_devices

    # Reshape the flat array of physical devices into a 2D logical grid
    device_grid = np.array(devices[:dp_size * tp_size]).reshape((dp_size, tp_size))
    
    # Define the mesh with named axes: 'dp' for data parallel, 'tp' for tensor parallel
    return Mesh(device_grid, axis_names=('dp', 'tp'))

def get_attention_sharding(mesh: Mesh) -> NamedSharding:
    """
    Generates the NamedSharding spec for standard Attention tensors.
    
    Target Tensor Shape: (Batch, Heads, Sequence_Length, Head_Dim)
    Partitioning Rule: ( 'dp',  'tp',  None,             None )
    
    This ensures that each TPU chip only receives its specific chunk of batches 
    and attention heads. The sequence length and head dimension are kept completely 
    intact on each chip to ensure the MXU systolic arrays have contiguous memory.
    
    Args:
        mesh: The active jax.sharding.Mesh context.
        
    Returns:
        A NamedSharding object that can be applied to Q, K, V, and Output tensors.
    """
    # P stands for PartitionSpec. 
    # None means "replicate" or "do not shard along this axis".
    spec = P('dp', 'tp', None, None)
    return NamedSharding(mesh, spec)

def shard_tensor(tensor: jax.Array, sharding: NamedSharding) -> jax.Array:
    """
    Physically distributes a tensor across the TPU mesh according to the sharding spec.
    
    Args:
        tensor: The input jax.Array (e.g., Query, Key, or Value).
        sharding: The NamedSharding specification.
        
    Returns:
        A distributed jax.Array. Operations performed on this array will now 
        automatically execute in parallel across the TPU v5e-4 slice.
    """
    # jax.device_put acts as a non-blocking scatter operation
    return jax.device_put(tensor, sharding)

# -----------------------------------------------------------------------------
# Example Usage Context Manager
# -----------------------------------------------------------------------------

def example_distributed_setup():
    """
    Demonstrates how this module integrates with the main Pallas-Flash kernel.
    (Used purely for documentation and local testing).
    """
    # 1. Initialize the 4-chip mesh
    mesh = create_v5e_mesh(dp_size=1, tp_size=4)
    sharding = get_attention_sharding(mesh)
    
    # 2. Within the mesh context, distribute inputs and run the kernel
    with mesh:
        # Assume Q, K, V are large global tensors created on the host CPU
        q_shape = (4, 16, 8192, 128)
        dummy_q = jax.numpy.zeros(q_shape, dtype=jax.numpy.bfloat16)
        
        # Distribute: B(4) is kept whole (dp=1), H(16) is split 4 ways (tp=4).
        # Each TPU chip will now physically hold a (4, 4, 8192, 128) local slice.
        q_sharded = shard_tensor(dummy_q, sharding)
        
        print(f"Global Shape: {q_sharded.shape}")
        print(f"Sharding Spec: {q_sharded.sharding}")
        
if __name__ == "__main__":
    example_distributed_setup()