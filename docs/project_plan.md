# Project Plan: The "Pallas-Flash" High-Performance Transformer Kernel for TPU v5e Architecture 

## 1. Executive Summary and Strategic Objectives

The paradigm shift in Large Language Model (LLM) computations is from arithmetic throughput to memory bandwith, known as the memory wall. This means deep learning kernels need to be rethought, especially for the Attention mechanism of the Transformer architecture.

The initial scope of this project focused on high-powered TPU v4 and v5p training environments but due to resource constraints TPU v5e must be used. With the TPU v5e-4 slice, the engineering philosophy must adapt strategically to meet requirements that differ greatly from those of the TPU v4 and v5p. The Google Cloud TPU v5e is a vastly different class of domain-specific accelerators than the TPU v4 and v5p, which were geared toward high FLOPs and massive interconnect scaling. Instead, the TPU v5e optimizes for performance-per-dollar and energy efficiency, intended primarily for inference and mid-scaled training workloads.

The move to this new hardware will create some negative impacts: reduced High Bandwith Memory (HBM) and less total memory bandwith. The new "Pallas-Flash" product will transition from a model focused on "supercomputer utilization" to one that is primarily concerned with "extreme resource efficiency".

This project plan representes the update technical architecture of a purpose-built, IO-aware Attention kernel, built specifically for TPU v5e-4. The core technical focus of the project remains the same as stated above, which is to bypass the XLA (Accelerated Linear Algebra) compiler's high-level heuristics and work directly with the TPU memory structure via JAX Pallas. By controlling the movement of data explicitly between the HBM and on-chip Vector Memory (VMEM), we expect to complete the implementation of a tiled, fused FlashAttention-2 operation and avoid using off-chip memory entirely. Thus, we anticipate substantially fewer memory-related bandwith constraints on the v5e system than previously encountered with our earlier v5e nodes.

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

* **Dedicated Resources:** Each TensorCore is able to access its own 16 GiB of HBM and its own inter-chip interconnect (ICI) link. This removes any contention for resources between cores on the same die.

#### 2.1.1 The Matrix Multiply Unit (MXU) 

The TPU is powered by the Matrix Multiply Unit that continues to serve as the core computational component for this technology.

* **Systolic Array Design:** The latest version is using the classic systolic array design style; therefore, it allows for the data to be processed in a predictable manner through a 2D grid of arithmetic logic elements. This design allows for increased power savings and material efficiency, since there are fewer accesses to the registers of the compute unit.

* **Dimensions:** The v5e retains the original $128 \times 128$ dimensions found in previous versions of the TPU (v4 and v4). This dimension is what's called an "atomic" tile size for the Pallas programming environment. It essentially means that any kernel that runs on the TPU must be configured to use matrix dimensions that are multiples of 128. Matrices that have a dimension of 129 for example, will be padded to a dimension of 256, which causes the hardware to lose 50% of its throughput due to the double padding. The programming environment requires that the Pallas flash kernel enfore strict alignment to 128 for all tile sizes ($B_c, B_r$).

* **Throughput:** The v5e has a peak throughput of 197 TFLOPs per chip when using bfloat16 precision. This is consiberably less than the v4's 275 TFLOPs, but the benefit of the v5e is that is it able to maintain relatively high utilization across smaller batches that are designed to be used for inference purposes.

* **MXU Count:** Each v5e TensorCore has four MXUs, providing the potential for very high levels of parallel matrix multipy capability. By properly tiling the kernel, multiple attention head blocks can be processed simultaneously.

#### 2.1.2 The Vector Processing Unit (VPU) and Scalar Unit

Because MXUs take care of most of the computational power involved in computing matrix multiplications of ($Q \cdot K^T$ and $A \cdot V$), the Vector Processing Unit (VPU) takes care of all the other calculations that form the basis of the Attention mechanism such as calculating the Softmax exponentials with respect to the values in the Q and K matrices, scaling them appropriately, and applying the masking operations.

* **Throughput Asymmetry:** Many naive kernel implementations face a bottleneck because of the fact that VPUs are the most time-consuming part of the kernel, with MXUs providing far more FLOP speed than VPUs. If the Pallas kernel is designed so that the MXU is idle while the VPU is computing the Softmax exponential calculations, the performance of the kernel can be significantly impeded. In particular, the v5e design requires a large amount of coordination between the VPU's activation calculations and the MXU's matrix multiplication calculations via pipelining.

* **Scalar Unit:** The Scalar Unit is responsible for controlling the flow of execution through the kernel, including looping and generating addresses for accessing arrays. In Pallas, the Scalar Unit is programmed using control strucutures in Python, which are then lower-level reduced to scalar instructions by the Pallas compiler.

### 2.2 The Memory Hierarchy: The 16GB Constraint 

The most critical constraint for this project - and the driver for the pivot in strategy - is the memory hierarchy of the v5e.

**Table 1: Comparative Memory Specifications** 

| Feature | TPU v4 (Reference) | TPU v5e (Target) | Implication for Pallas-Flash |
| :--- | :--- | :--- | :--- |
| **HBM Capacity** | 32 GiB per chip | 16 GiB per chip | **Critical constraint.** Standard $O(N^2)$ attention will trigger OOM errors at much shorter sequence lengths. Memory efficiency is paramount. |
| **HBM Bandwidth** | $\sim1.2~TB/s$ | 819 GB/s | **Throughput Bottleneck.** Reduced bandwidth demands higher arithmetic intensity to avoid stalling the MXUs. |
| **VMEM Size** | ~16-32 MiB (plus CMEM) | ~128 MiB | **Strategic Advantage.** The v5e has a massive on-chip scratchpad relative to its compute. This allows storing extremely large tiles. |
| **Core Architecture** | 2 Cores/Chip (Megacore) | 1 Core/Chip | **Simplified programming model;** no shared HBM contention between cores. |

#### The Optimization Thesis: Exploiting VMEM 

The most important focus of the current project is the discovery that the TPU v5e has approximately 128 MiB of VMEM available per core. When designing v4 of this TPU architecture, the memory was divided into a smaller sized VMEM and a larger size Common Memory (CMEM), thus introducing an additional complexity in managing VMEM. In contrast, the TPU v5e allows for a much greater volume of VMEM in a single, large Unified Vector Scratchpad.

Therefore, the available 128 MiB of VMEM is an enormous amount of VMEM in comparison to L1/Shared Memory of equivalent GPUs (which typically have 128 KB to 256 KB). As a result, the "Pallas-Flash" kernel has an entirely different tiling strategy than those of existing GPU based tiling implementations:

* The tiling strategy allows for larger $1024 \times 128$ Query blocks to be loaded into VMEM and always be kept resident within the TPU v5e, rather than smaller $128 \times 128$ blocks used within a GPU.

* The ability to reuse these preloaded resident Query blocks against streaming Key/Value blocks from being loaded into HBM allows for the increase of Arithmetic Intensity (FLOPs per byte transferred).

* This increase is essential to allow the TPU v5e to reach the maximum level of compute performance (197 TFLOPs) while staying constrained by the limited amount of HBM bandwidth (819 GB/s).

### 2.3 Topology: The $2 \times 2$ Mesh and Interconnects 

The user uses a TPU v5e-4 slice, which is composed of four separate physical chips connected to each other in a 2D Torus topology through ICIs.

* **Single-Host Architecture:** The TPU v5e architecture is designed to be used as a single-host device when used as a slice of either one, four, or eight chips. This means that all four chips are connected to one virutal machine (VM) instance, such as `ct5lp-hightpu-4`. As a result of being attached to the same VM, the software architecture is much less complex as there are no remote prodecure calls (RPCs) or orchestration of multiple processes acorss distinct physical hosts. The data load and preprocessing of the input into the TPU happen within the same Python process.

* **ICI Bandwidth:** The ICI has a bandwidth of 400 GB/s per chip. For the 2D mesh topology, each chip has direct connection to two neighbouring chips (in the torus topology, the connection wraps around forming a fully connected ring in each dimension). The high bandwith of the ICI boosts the efficiency of the collective operations (all-gather and reduce-scatter) that are used in the FlashDecoding inference approach.

---

## 3. Phase I: The Baseline Construction

In order to compare the performance of a custom kernel to the performance of a standard software stack, one must first create a baseline. To do this, we would use the default XLA compiler to establish a performance benchmark that will allow for a comparison against a custom-engineered kernel. By establishing a performance benchmark using the default software stack, we are able to determine how to proceed with the design of our kernel based upon limitations within the design of v5e due to limited memory within HBM.

### 3.1 Mathematical Formulation of Standard Attention 

The baseline for attention would use standard `jax.numpy` primitives for the formulation of Scaled Dot Product Attention:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V $$

Where:
* $Q, K, V \in \mathbb{R}^{B \times H \times N \times D}$ (Batch, Heads, Sequence Length, Head Dimension). 
* $M$ is the causal mask (lower triangular matrix of zeros, upper triangular of $-\infty$). 

### 3.2 Implementation Strategy: Pure JAX with XLA

Pure JAX can be applied in conjuntion with XLA by decorating the code `baseline_mha.py` with the `@jax.jit` decorator to compile the graph using XLA.

**Projected Failure Modes on v5e:** 

1.  **Memory Explosion (OOM):** The standard implementation requires computing the `logits = jnp.matmul(Q, K.T)` which generates an intermediatte tensor of shape $(B, H, N, N)$ as it relates to shape $(B, H)$ from the original input tensor. For example, given a sequence length of $N=32,768$, a batch size of $B=1$, and heads $H=16, the equivalent float32 memory required is approximately:

    $$ 1 \times 16 \times 32768^2 \times 4 \text{ bytes} \approx 68 \text{ GB} $$ 

It should be noted that this is significantly greater than the maximum available on a single v5e chip, which is a maximum of 16 GB HBM. Even when using bfloat16 (34GB), there is not enough memory for this sequence length to be processed successfully; this, the baseline should fail with OOM at approximately 8k-12k in this case (Needs to be tested).

2.  **Bandwidth Saturation:** The standard implementation runs out of memory (OOM) at lower sequence lengths (less than approximately 12k), requiring the repeated loading and unloading of matrices of shape $N \times N$ (calculation, masking, softmax, and multiplication). The I/O generated will consume the vast amount of available 819 GB/s bandwidth on the v5e and will therefore create a bottleneck in execution time and enable the MXUs to run idle while waiting for I/O operations to complete.

### 3.3 Benchmarking Methodology

In order to demonstrate scientifically the basis of our belief in "Why this wins" we will measure the following:

* **Latency** as wall-clock time per step after calling the method `block_until_ready()`. Latency will be measure using the method `time.perf_counter()`.
* **Throughput** as tokens processed in one second.
* **HBM Bandwidth Utilization**, by monitoring the memory bus using `jax.profiler`, we will know if we have a memory-bound workload by looking at the relationship between a saturated memory bus and low MXU.
* **Model FLOPs Utilization (MFU):** the ratio of FLOPs that were achieved divided by the theoretical peak of 197 TFLOPs.

---

## 4. Phase II: The Pallas Kernel Architecture 

This is the primary engineering focus of Pallas: switch out the memory inefficient XLA graph and double the original output size by manually tiling Pallas kernels for efficiency on v5e.

### 4.1 The Pallas Programming Model for v5e 

Pallas serves as a conduit to low-level Mosaic/TPU assembly. The v5e programming model is much easier to work with because it uses a single core.

* **Grid Definition:** Pallas executes the kernel as independent bin blocks of the output matrix, where the number of program instances is defined by the grid.
    * Target Grid: `(Batch_Size, Num_Heads, N_Blocks_Output)`
    * In the case of the v5e, each point in the grid becomes a scheduled task on the single TensorCore - as opposed to Megacore configurations where you need to map specific dimensions to sub-cores on v4/v5p systems.

* **Memory Spaces:** Pallas separates Refs in HBM (Global) from Refs in VMEM/SMEM (Local).
    * **Input/Output:** Located in HBM
    * **Scratchpad:** All memory buffers must be allocated explicitly, either in VMEM or SMEM.

### 4.2 The Tiling Strategy: Exploiting the 128 MIB VMEM 

The "Pallas-Flash" has the greatest advantage of using the 128 MiB VMEM aggressively.

**The Block Size Calculation:** 
In standard FlashAttention utilizing GPUs, standard block sizes ($128 \times 64$ and $128 \times 128$) were designed around the 100KB-200KB SRAM limitation on these systems; in contrast, the TPU v5e has many times more physical memory available than this.

* **Query Block ($B_q$):** Multiple Qs could be loaded at once with very large blocks. For example, for $D=128$, a single block of $1024 \times 128$ floats takes up only 0.5 MB worth of space. Such blocks can easily fit many blocks of size $1024$, and even $2048$, into the VMEM.

* **Key/Value Block ($B_k$):** Smaller chunks of K and V are streamed through HBM. For example, $B_k = 512$

**Why Larger Tiles Win on v5e:** 
* **Arithmetic Intensity:** The arithmetic intensity of $B_k$ increases with the size of $Q_{blocks}$ used because it computes $Q_{block} \times K_{block}^T$. So larger $Q_{block}$ sizes will result in a greater number of FLOPs computed per byte of K/V loaded from HBM.

* **Hiding Latency:** The v5e's HBM bandwidth of 819 GB/s is not as high as that of other recent TPUs. By using larger tile sizes, longer periods of time exist between when the last compute phase completes and when the next block can be fetched via DMA, thereby effectively hiding the HBM memory latency.

**The Algorithm Flow (Per Program Instance):** 
1.  **Initialization:** Allocate Accumulator ($O_{acc}$) and Stats ($M, L$) in VMEM. 
2.  **Load Q:** DMA a large block of Queries ($Q_i$) from HBM to VMEM. This stays resident. 
3.  **The KV Loop:** Iterate $j$ from $0$ to $N/B_k$: 
    * **Pipeline Load:** Issue DMA request for $K_{j+1}, V_{j+1}$ (Prefetch). 
    * **Compute Scores:** $S_{ij} = Q_i \cdot K_j^T$ (MXU). 
    * **Online Softmax:** Update max/sum stats using VPU. 
    * **Update Output:** $O_{acc} += P_{ij} \cdot V_j$ (MXU). 
    * **Synchronization:** Wait for DMA of next block.
4.  **Finalize:** Normalize $O_{acc}$ and DMA write to HBM. 

### 4.3 Memory Alignment and Padding 

The TPU MXU is a $128 \times 128$ systolic array. 

* **Constraint:** All inner dimensions of matrices utilized in matrix multiplication must be multiples of 128.
* **The "Copy" Trap:** The XLA library will insert large operations into the input process to support padding in HBM before executing the kernel if input size does not meet padding requirements (i.e., incorrect input sizes, such as a head dimension of 96, etc.). Such added memory usage is significant and can negatively affect overall performance.
* **Mandatory Requirement:** `jax.numpy.pad` must be used in the data pipeline to pad all tensor dimensions (and other tensor dimensionality) passed to the kernel to 128-byte aligned formats. Correspondingly aligned tiles should be requested in the Pallas `BlockSpec` of the kernel.

---

## 5. Phase III: Assembly Debugging and Low-Level Optimization 

In order to create high-performance kernels, it is important to validate that the hardware is functioning properly.

### 5.1 Artifact Analysis 

When analyzing the compiled binary we will use the command `XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"`.

* **HLO (High-Level Optimizer) Analysis:** We will look for the `copy-start` and `copy-done` instructions around the `custom-call` in the HLO text. If they are present, it indicates that we have not aligned the memory correctly, causing the "Copy Trap" error to occur.
* **Vector Register Spilling:** If we have tiled too aggressively, or gone over the 128 MiB VMEM and/or fragmented the 128 MiB VMEM, the compiler may put some of the registers onto HBM instead of keeping them all in the vector registers. We will be able to identify this by checking for unexpected HBM traffic in the profiler.

### 5.2 Pipeline Bubbles 

Using the JAX Profiler Trace Viewer: 
* **Goal:** To establish a "solid wall" of periods where MXUs are busy.
* **Failure:** MXU block(s) that have empty space (a gap) means that the computation was complete prior to receiving the next DMA transfer load.
* **Fix for v5e:** If gaps exist between MXUs, increase the size of the `Block_Q`. By increasing the duration of the computation on the inside of the loop without increasing the number of bytes transferred (data loading), there will be more time for the DMA engine to complete the fetch operation. The increase in size of the `Block_Q` is the primary tuning knob for the v5e's 819 GB/s bandwidth.

---

## 6. Phase IV: The Ceiling - FlashDecoding on 4 Chips 

The "FlashDecoding" phase addresses the inference bottleneck. In decoding, we generate one token at a time. A single query token attending to a 32k KV cache is purely memory-bound.