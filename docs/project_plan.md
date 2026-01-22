# Project Plan: The "Pallas-Flash" High-Performance Transformer Kernel for TPU v5e Architecture 

## 1. Executive Summary and Strategic Objectives

The paradigm shift in Large Language Model (LLM) computations is from arithmetic throughput to memory bandwith, known as the memory wall. This means deep learning kernels need to be rethought, especially for the Attention mechanism of the Transformer architecture.

The initial scope of this project focused on high-powered TPU v4 and v5p training environments but due to resource constraints TPU v5e must be used. With the TPU v5e-4 slice, the engineering philosophy must adapt strategically to meet requirements that differ greatly from those of the TPU v4 and v5p. The Google Cloud TPU v5e is a vastly different class of domain-specific accelerators than the TPU v4 and v5p, which were geared toward high FLOPs and massive interconnect scaling. Instead, the TPU v5e optimizes for performance-per-dollar and energy efficiency, intended primarily for inference and mid-scaled training workloads.

The move to this new hardware will create some negative impacts: reduced High Bandwith Memory (HBM) and less total memory bandwith. The new "Pallas-Flash" product will transition from a model focused on "supercomputer utilization" to one that is primarily concerned with "extreme resource efficiency".

This project plan representes the update technical architecture of a purpose-built, IO-aware Attention kernel, built specifically for TPU v5e-4. The core technical focus of the project remains the same as stated above, which is to bypass the XLA (Accelerated Linear Algebra) compiler's high-level heuristics and work directly with the TPU memory structure via JAX Pallas. By controlling the movement of data explicitly between the HBM and on-chip VMEM, we expect to complete the implementation of a tiled, fused FlashAttention-2 operation and avoid using off-chip memory entirely. Thus, we anticipate substantially fewer memory-related bandwith constraints on the v5e system than previously encountered with our earlier v5e nodes.

The project's continued focus on overcoming specific obstacles associated with the v5e-4 topology (a 4-chip single-host slice connected by a 2x2 Inter-Chip Interconnect mesh) also encourages FlashDecoding, a new type of Specialized Inference Kernel. In this added capability, the FlashDecoding process can be performed in parallel over the entire width of the mesh by loading the Key-Value cache across multiple chips in parallel while utilizing the total memory bandwith of all the chips within the slice as one large memory bandwith resource to minimize inference latency as compared to larger more expensive memory systems.

Completing this plan will result in producing three major items:

1. **Pallas-Flash Training Kernel:** Highly efficient, I/O aware attention kernels utilizing a maximum arithmetic intensity for the v5e platform through available VMEM (128MiB).

2. **FlashDecoding Inference Kernel:** Distributed kernel decoding uses Split-KV parallelism across a 2x2 mesh for improved latency of single-batch processes.

3. **v5e Benchmarking Suite:** A formal methodology for validating "tokens per second per dollar" that confirms that the custome kernel provides legitimate advantages over traditionally used efficiency tier hardware from an economic and performance standpoint.

---