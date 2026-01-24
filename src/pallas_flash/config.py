"""
TPU v5e Hardware Configuration and Constants.

This file serves as the Single Source of Truth (SSOT) for all hardware-specific
parameters used in the Pallas-Flash kernel.

Why this matters:
1. Roofline Modeling: Performance benchmarks (MFU) are calculated against these
   theoretical peaks.
2. Tiling Logic: The `tiling.py` module imports `VMEM_CAPACITY` to calculate
   optimal block sizes (e.g., Block_Q=1024).
3. Portability: To target TPU v4 or v5p in the future, we simply swap the
   constants here rather than hunting through kernel code.

References:
- TPU v5e HBM Bandwidth: 819 GB/s (per chip)
- TPU v5e Peak FLOPS (bf16): 197 TFLOPS
- TPU v5e VMEM: ~128 MiB (Unified Vector Memory)
"""

# Code Analysis: https://docs.google.com/document/d/1XLzEYV69WIB1D52nOx8WCfihbsdaok1jceqCsyY7gWM/edit?tab=t.0

# Import dataclass from dataclasses
from dataclasses import dataclass

# Use dataclass decorator with frozen parameter set to True to make it immutable.
@dataclass(frozen=True)
# Create a class called TPUv5eSpecs
class TPUv5eSpecs:
    """Immutable hardware specifications for the Google Cloud TPU v5e"""
    
    # Implement the memory hierarchy
    # Set HBM Bandwidth to 819 GBps in Bps
    HBM_BANDWIDTH_BYTES_PER_SEC: float = 819 * 1e9

    # Set VMEM Capacity as 128 MiB in B
    VMEM_CAPACITY_BYTES: int = 128 * 1024 * 1024

    # Standard precision size for memory math (bfloat16 = 2 bytes)
    BYTES_PER_BF16: int = 2

    

    # Implement the compute capability
    # Set peak throughput (bfloat16) to 197 TFLOPs but in FLOPs
    PEAK_FLOPS_BF16: float = 197 * 1e12

    # Set MXU size to 128
    MXU_SIZE: int = 128


    # Implement the Topology
    # Set ICI to 400 GBps but in Bps
    ICI_BANDWIDTH_BYTES_PER_SEC: float = 400 * 1e9

    # v5e-4 Slice Topology (2x2 Mesh)
    MESH_SHAPE: tuple = (2, 2)
    NUM_CHIPS: int = 4

# Instantiate singleton configuration
v5e_specs = TPUv5eSpecs()

# Implement helper accessors for cleaner imports in tiling logic
VMEM_LIMIT = v5e_specs.VMEM_CAPACITY_BYTES
MXU_SIZE = v5e_specs.MXU_SIZE