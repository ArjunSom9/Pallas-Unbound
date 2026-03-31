"""
Distributed Mesh and Collectives Tests (test_distributed.py)

This module validates the multi-device sharding and communication primitives 
defined in `mesh.py` and `collectives.py`.

Context:
The target architecture is a TPU v5e-4 slice (4 interconnected chips). To test
this logic locally or in CI pipelines without an actual TPU attached, we force 
JAX to emulate 4 distinct CPU devices before initializing the runtime.

Validation Targets:
1. Mesh Generation: Ensuring the 2D logical grid is correctly formed.
2. Tensor Sharding: Verifying that standard (B, H, N, D) attention tensors 
   are correctly partitioned across the Data Parallel (DP) and Tensor Parallel 
   (TP) axes.
3. Collectives: Proving that `psum` and `pmean` operations properly synchronize 
   data across the simulated Inter-Core Interconnect (ICI) links.
"""

import os
import pytest

# -----------------------------------------------------------------------------
# 1. Hardware Simulation Setup
# -----------------------------------------------------------------------------
# Force JAX to emulate 4 CPU devices. This MUST be set before JAX is imported.
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map

# Import our custom distributed modules
from pallas_flash.distributed.mesh import create_v5e_mesh, get_attention_sharding, shard_tensor
from pallas_flash.distributed.collectives import all_reduce_tp, all_reduce_dp_mean

# -----------------------------------------------------------------------------
# 2. Topology and Sharding Tests
# -----------------------------------------------------------------------------

def test_device_emulation_active():
    """Validates that our 4-device simulation successfully initialized."""
    num_devices = len(jax.devices())
    assert num_devices >= 4, \
        f"Expected at least 4 devices for v5e-4 simulation, found {num_devices}. " \
        "Ensure XLA_FLAGS='--xla_force_host_platform_device_count=4' is set."
    print(f"\n✓ Found {num_devices} JAX devices.")

def test_mesh_and_sharding():
    """
    Tests the creation of the 2D (DP, TP) logical mesh and verifies that 
    the Attention tensors are partitioned correctly.
    """
    # 1. Create a 1x4 logical mesh (1 Batch chunk, 4 Head chunks)
    mesh = create_v5e_mesh(dp_size=1, tp_size=4)
    sharding = get_attention_sharding(mesh)
    
    # 2. Validate the Sharding Spec
    # Expected: ('dp', 'tp', None, None)
    # Meaning: Shard batch across DP, Shard heads across TP, replicate sequence and features.
    assert sharding.spec == P('dp', 'tp', None, None), \
        f"Incorrect PartitionSpec. Got {sharding.spec}"
        
    # 3. Apply Sharding to a Dummy Tensor
    # B=2, H=16, N=128, D=64
    global_shape = (2, 16, 128, 64)
    dummy_tensor = jnp.zeros(global_shape)
    
    with mesh:
        sharded_tensor = shard_tensor(dummy_tensor, sharding)
        
    # 4. Verify physical distribution
    # The tensor should now be backed by 4 underlying device buffers (addressable shards).
    num_buffers = len(sharded_tensor.addressable_shards)
    assert num_buffers == 4, \
        f"Expected tensor to be physically split into 4 buffers, found {num_buffers}."
        
    # Each physical chunk on the devices should have 1/4th of the heads (16 / 4 = 4)
    local_shape = sharded_tensor.addressable_shards[0].data.shape
    assert local_shape == (2, 4, 128, 64), \
        f"Incorrect local chunk shape. Expected (2, 4, 128, 64), got {local_shape}"
        
    print("✓ Passed Mesh Initialization and Tensor Sharding.")

# -----------------------------------------------------------------------------
# 3. Inter-Core Communication (Collectives) Tests
# -----------------------------------------------------------------------------

def test_tp_all_reduce():
    """
    Validates that the `all_reduce_tp` wrapper correctly synchronizes and sums 
    data across the Tensor Parallel axis.
    """
    mesh = create_v5e_mesh(dp_size=1, tp_size=4)
    sharding = get_attention_sharding(mesh)
    
    # Initialize a global tensor of ones
    global_tensor = jnp.ones((2, 16, 128, 64), dtype=jnp.float32)
    
    # JIT compile the collective operation inside the mesh context.
    # We specify the output should maintain the same sharding layout.
    @jax.jit
    def distributed_sum(x):
        # shard_map binds the mesh axis names ('dp', 'tp') so jax.lax.psum can use them locally
        # It requires the PartitionSpec (sharding.spec), not the full NamedSharding object
        _sum_fn = shard_map(
            all_reduce_tp,
            mesh=mesh,
            in_specs=sharding.spec,
            out_specs=sharding.spec
        )
        return _sum_fn(x)
        
    with mesh:
        # Shard the ones across the 4 devices
        sharded_ones = shard_tensor(global_tensor, sharding)
        
        # Execute the all-reduce
        result = distributed_sum(sharded_ones)
        
    # Because each of the 4 devices held a chunk of "1"s and we summed across 
    # the TP axis (size 4), the resulting global tensor should be filled with "4"s.
    assert jnp.all(result == 4.0), \
        "TP All-Reduce failed! Expected all elements to be summed to 4.0."
        
    print("✓ Passed Tensor Parallel (TP) All-Reduce Collective.")

def test_dp_all_reduce_mean():
    """
    Validates that the `all_reduce_dp_mean` correctly averages data across 
    the Data Parallel axis. We use a 4x1 mesh for this test to force DP routing.
    """
    # Create a 4x1 logical mesh (4 Batch chunks, 1 Head chunk)
    mesh = create_v5e_mesh(dp_size=4, tp_size=1)
    sharding = get_attention_sharding(mesh)
    
    # Initialize a global tensor of values
    global_tensor = jnp.full((8, 16, 128, 64), 10.0, dtype=jnp.float32)
    
    @jax.jit
    def distributed_mean(x):
        # shard_map exposes the 'dp' axis name context 
        _mean_fn = shard_map(
            all_reduce_dp_mean,
            mesh=mesh,
            in_specs=sharding.spec,
            out_specs=sharding.spec
        )
        return _mean_fn(x)
        
    with mesh:
        sharded_tens = shard_tensor(global_tensor, sharding)
        result = distributed_mean(sharded_tens)
        
    # Averaging across identical values should result in the same value,
    # proving the pmean collective resolved correctly across the DP mesh.
    assert jnp.all(result == 10.0), \
        "DP All-Reduce Mean failed to synchronize correctly."

    print("✓ Passed Data Parallel (DP) All-Reduce Mean Collective.")