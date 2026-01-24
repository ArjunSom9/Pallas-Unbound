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

## 2. Technical Prerequisites: The TPU v5e Hardware Landscape 

To create a kernel that achieves near-maximum performance on the TPU v5e, it is necessary to move away from any prior assumptions about the v4 or v5p architecture. The TPU v5e is not just an under-clocked version of the TPU v4; rather, it is an entirely new microarchitecture built around desing decisions optimized to maximizie compute to power. To effectively program for the TPU v5e using Pallas, we must first understand how these differences affect programming for the TPU v5e.

### 2.1 The Compute Hierarchy: Single-Core TensorCore Architecture 

The core architectural varaition between the v5e and v4/v5p modles is the difference in the core configuration. The TPU v4 and v5p use a dual core model, where there are two TensorCores on one chip, which is often referred to as a Megacore by the software stack and allows co-use of the HBM bandwith. The TPU v5e, on the other hand, has one TensorCore per chip. This simplified architecture has very far-reaching consequences for the way that Pallas kernels are developed.

* **Simplified Grid Mapping:** The developers will not have to deal with the complications of intra-chip communication amongst the paired cores or manage the complexities of the dimension_map for sub-cores within a Megacore. The mapping will be a 1 to 1 relationship between the physical chip and the Pallas program instance.

* **Dedicated Resources:** Each TensorCore is able to access its own 16 GiB of HBM and its own ICI (inter-chip interconnect) link. This removes any contention for resources between cores on the same die.

#### 2.1.1 The Matrix Multiply Unit (MXU) 

The TPU is powered by the Matrix Multiply Unit that continues to serve as the core computational component for this technology.

* **Systolic Array Design:** The latest version is using the classic systolic array design style; therefore, it allows for the data to be processed in a predictable manner through a 2D grid of arithmetic logic elements. This design allows for increased power savings and material efficiency, since there are fewer accesses to the registers of the compute unit.

* **Dimensions:** The v5e retains the original $128 \times 128$ dimensions found in previous versions of the TPU (v4 and v4). This dimension is what's called an "atomic" tile size for the Pallas programming environment. It essentially means that any kernel that runs on the TPU must be configured to use matrix dimensions that are multiples of 128. Matrices that have a dimension of 129 for example, will be padded to a dimension of 256, which causes the hardware to lose 50% of its throughput due to the double padding. The programming environment requires that the Pallas flash kernel enfore strict alignment to 128 for all tile sizes ($B_c, B_r$).

* **Throughput:** The v5e has a peak throughput of 197 TFLOPS per chip when using bfloat16 precision. This is consiberably less than the v4's 275 TFLOPS, but the benefit of the v5e is that is it able to maintain relatively high utilization across smaller batches that are designed to be used for inference purposes.

* **MXU Count:** Each v5e TensorCore has four MXUs, providing the potential for very high levels of parallel matrix multipy capability. By properly tiling the kernel, multiple attention head blocks can be processed simultaneously.

#### 2.1.2 The Vector Processing Unit (VPU) and Scalar Unit

Because MXUs take care of most of the computational power involved in computing matrix multiplications of ($Q \cdot K^T$ and $A \cdot V$), the Vector Processing Unit (VPU) takes care of all the other calculations that form the basis of the Attention mechanism such as calculating the Softmax exponentials with respect to the values in the Q and K matrices, scaling them appropriately, and applying the masking operations.

* **Throughput Asymmetry:** Many naive kernel implementations face a bottleneck because of the fact that VPUs are the most time-consuming part of the kernel, with MXUs providing far more FLOP speed than VPUs. If the Pallas kernel is designed so that the MXU is idle while the VPU is computing the Softmax exponential calculations, the performance of the kernel can be significantly impeded. In particular, the v5e design requires a large amount of coordination between the VPU's activation calculations and the MXU's matrix multiplication calculations via pipelining.

* **Scalar Unit:** The Scalar Unit is responsible for controlling the flow of execution through the kernel, including looping and generating addresses for accessing arrays. In Pallas, the Scalar Unit is programmed using control strucutures in Python, which are then lower-level reduced to scalar instructions by the Pallas compiler.