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

# Use dataclass decorator with frozen parameter set to True to make it immutable.
# Create a class called TPUv5eSpecs
    
    # Implement the memory hierarchy
    
    # Set HBM Bandwith to 819 GBps in Bps

    # Set VMEM Capacity as 128 MiB in B



    # Implement the compute capability
    
    # Set peak throughput (bfloat16) to 197 TFLOPs but in FLOPs

    # Set MXU size to 128



    # Implement the Topology
    
    # Set ICI to 400 GBps but in Bps

# Instantiate singleton configuration

# Implement helper accessors for cleaner imports in tiling logic