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
- TPU v5e HBM Bandwidth: 819 GB/s
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
    # Set HBM Bandwith to 819 GBps in Bps
    HBM_BANDWITH_BYTES_PER_SEC: float = 819 * 1e9

    # Set VMEM Capacity as 128 MiB in B
    VMEM_CAPACITY_BYTES: int = 128 * 1024 * 1024


    # Implement the compute capability
    # Set peak throughput (bfloat16) to 197 TFLOPs but in FLOPs
    PEAK_FLOPS_BF16: float = 197 * 1e12

    # Set MXU size to 128
    MXU_SIZE: int = 128


    # Implement the Topology
    # Set ICI to 400 GBps but in Bps

# Instantiate singleton configuration

# Implement helper accessors for cleaner imports in tiling logic