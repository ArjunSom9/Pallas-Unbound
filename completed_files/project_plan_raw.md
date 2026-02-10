# Project Plan: The "Pallas-Flash" High-Performance Transformer Kernel for TPU v5e Architecture 

## 1. Executive Summary and Strategic Objectives 

The computational landscape for Large Language Models (LLMs) has undergone a paradigm shift, moving from a regime constrained by arithmetic throughput to one dominated by the limitations of memory bandwidth. This transition, often referred to as the "Memory Wall," necessitates a fundamental rethinking of how deep learning kernels are architected, particularly for the Attention mechanism which lies at the heart of the Transformer architecture. 

The original scope of this project targeted the training-centric, high-power TPU v4 and v5p architectures. However, the revised operational environment—a TPU v5e-4 slice—mandates a strategic pivot in engineering philosophy. The Google Cloud TPU v5e (Efficiency) represents a distinct class of domain-specific accelerators compared to its performance-focused predecessors. While the TPU v4 and v5p optimize for maximum Floating Point Operations Per Second (FLOPS) and massive interconnect scale for foundation model pre-training, the v5e is architected to maximize performance-per-dollar and energy efficiency, primarily for inference and mid-scale training workloads. 

This shift in hardware imposes new constraints: significantly reduced High Bandwidth Memory (HBM) capacity (16 GiB per chip versus 32 GiB on v4) and lower aggregate memory bandwidth (819 GB/s versus $1.2~TB/s$). Consequently, the "Pallas-Flash" project must evolve from a strategy of "supercomputer utilization" to one of "extreme resource efficiency." 

This comprehensive research report serves as the revised technical blueprint for architecting a custom, IO-aware Attention kernel specifically optimized for the TPU v5e-4. The primary technical objective remains unchanged: to bypass the high-level heuristics of the XLA (Accelerated Linear Algebra) compiler and achieve direct, manual control over the TPU memory hierarchy using JAX Pallas. By explicitly orchestrating data movement between HBM and the on-chip Vector Memory (VMEM), this project aims to implement a tiled, fused FlashAttention-2 operation that minimizes off-chip memory access, thereby mitigating the stricter bandwidth limitations of the v5e platform. 

Furthermore, the project addresses the specific challenges of the v5e-4 topology—a 4-chip, single-host slice connected via a 2x2 Inter-Chip Interconnect (ICI) mesh. This configuration presents unique opportunities for "FlashDecoding," a specialized inference kernel designed to parallelize Key-Value (KV) cache loading across the mesh. By treating the 4-chip slice as a unified logical entity, we can aggregate memory bandwidth to achieve inference latencies that rival or exceed larger, more expensive hardware configurations. 

The successful execution of this plan will yield three critical deliverables: 

1.  **Pallas-Flash Training Kernel:** A highly optimized, IO-aware attention kernel that leverages the v5e's generous 128 MiB VMEM to maximize arithmetic intensity. 
2.  **FlashDecoding Inference Kernel:** A distributed decoding kernel utilizing Split-KV parallelism across the 2x2 mesh to minimize single-batch latency. 
3.  **v5e Benchmarking Suite:** A rigorous validation framework measuring "tokens per second per dollar," verifying the economic and performance advantages of the custom kernel on efficiency-tier hardware. 

---

## 2. Technical Prerequisites: The TPU v5e Hardware Landscape 

To engineer a kernel that approaches the theoretical roofline of the TPU v5e, it is imperative to discard assumptions based on the v4 or v5p architectures. The v5e is not merely a "down-clocked" version of the v4; it is a distinct microarchitecture with specific design choices made to optimize the ratio of compute to power consumption. Understanding these nuances is the prerequisite for effective Pallas programming. 

### 2.1 The Compute Hierarchy: Single-Core TensorCore Architecture 

The most profound architectural difference between the v5e and the v4/v5p lineages is the core configuration. The TPU v4 and v5p utilize a dual-core design, where two TensorCores reside on a single chip, often presented to the software stack as a "Megacore" to share HBM bandwidth. In contrast, the TPU v5e features a single TensorCore per chip. This architectural simplification has profound implications for Pallas kernel development: 

* **Simplified Grid Mapping:** Developers no longer need to manage the complexity of intra-chip communication between paired cores or handle the `dimension_map` complexities required to address sub-cores within a Megacore. The mapping becomes a strict 1:1 relationship between the physical chip and the Pallas program instance. 
* **Dedicated Resources:** Each TensorCore has exclusive access to its 16 GiB of HBM and its ICI links, eliminating resource contention between cores sharing a die. 

#### 2.1.1 The Matrix Multiply Unit (MXU) 

The Matrix Multiply Unit (MXU) remains the computational engine of the TPU. 

* **Systolic Array Design:** The v5e employs a systolic array architecture, where data flows rhythmically through a grid of Arithmetic Logic Units (ALUs). This design maximizes density and energy efficiency by minimizing register file access. 
* **Dimensions:** Crucially, the v5e retains the standard $128 \times 128$ systolic array dimensions found in previous generations (v3, v4). This dimension is the fundamental "atomic unit" for Pallas tiling. Any matrix dimension in the kernel must be padded to a multiple of 128. A dimension of 129, for example, would force the hardware to pad to 256, resulting in a 50% loss of effective throughput. The "Pallas-Flash" kernel must enforce strict 128-alignment for all block sizes ($B_c, B_r$). 
* **Throughput:** The v5e delivers a peak performance of 197 TFLOPS in bfloat16 precision per chip. While significantly lower than the v4's 275 TFLOPS, the v5e's value proposition lies in its ability to sustain high utilization on smaller, inference-centric batch sizes. 
* **MXU Count:** Each v5e TensorCore contains four MXUs. This provides substantial parallel matrix-multiply capability, allowing the kernel to process multiple attention head blocks simultaneously if tiled correctly. 

#### 2.1.2 The Vector Processing Unit (VPU) and Scalar Unit 

While the MXU handles the heavy lifting of matrix multiplications ($Q \cdot K^T$ and $A \cdot V$), the Vector Processing Unit (VPU) is responsible for the element-wise operations critical to the Attention mechanism: Softmax exponentiation, scaling, and masking. 

* **Throughput Asymmetry:** A common bottleneck in naive kernel implementations is the "VPU bound" scenario. The MXU offers significantly higher FLOPs than the VPU. If the Pallas kernel is structured such that the MXU sits idle while the VPU computes the Softmax exponentials, performance will degrade significantly. The v5e design requires careful pipelining to overlap VPU activation calculations with MXU matrix multiplications. 
* **Scalar Unit:** The scalar unit manages control flow, loop indices, and address generation. In Pallas, this is programmed via Python control structures, which the compiler lowers to scalar instructions.

### 2.2 The Memory Hierarchy: The 16GB Constraint 

The most critical constraint for this project—and the primary driver for the pivot in strategy—is the memory hierarchy of the v5e. 

**Table 1: Comparative Memory Specifications** 

| Feature | TPU v4 (Reference) | TPU v5e (Target) | Implication for Pallas-Flash |
| :--- | :--- | :--- | :--- |
| **HBM Capacity** | 32 GiB per chip | 16 GiB per chip | **Critical constraint.** Standard $O(N^2)$ attention will trigger OOM errors at much shorter sequence lengths. Memory efficiency is paramount. |
| **HBM Bandwidth** | $\sim1.2~TB/s$ | 819 GB/s | **Throughput Bottleneck.** Reduced bandwidth demands higher arithmetic intensity to avoid stalling the MXUs. |
| **VMEM Size** | ~16-32 MiB (plus CMEM) | ~128 MiB | **Strategic Advantage.** The v5e has a massive on-chip scratchpad relative to its compute. This allows storing extremely large tiles. |
| **Core Architecture** | 2 Cores/Chip (Megacore) | 1 Core/Chip | **Simplified programming model;** no shared HBM contention between cores. |

#### The Optimization Thesis: Exploiting VMEM 

The revelation that the TPU v5e possesses approximately 128 MiB of Vector Memory (VMEM) per core is the pivotal insight for this project. In the v4 architecture, memory was split between a smaller VMEM and a larger Common Memory (CMEM), requiring complex management. The v5e simplifies this into a large, unified vector scratchpad. 

This 128 MiB capacity is massive compared to the L1/Shared Memory of equivalent GPUs (often 128KB - 256KB). This allows the "Pallas-Flash" kernel to adopt a tiling strategy that is fundamentally different from GPU implementations: 

* Instead of small $128 \times 128$ blocks, we can load huge chunks of the Query matrix (e.g., $1024 \times 128$) into VMEM and keep them resident. 
* By reusing these resident Query blocks against streaming Key/Value blocks from HBM, we dramatically increase Arithmetic Intensity (FLOPs per byte transferred). 
* This high intensity is the only way to saturate the 197 TFLOPS compute capability given the restricted 819 GB/s HBM bandwidth. 

### 2.3 Topology: The $2 \times 2$ Mesh and Interconnects 

The user's deployment is a TPU v5e-4 slice. This consists of 4 physical chips connected in a $2 \times 2$ Torus topology via Inter-Chip Interconnects (ICI). 

* **Single-Host Architecture:** Crucially, for slices of 1, 4, or 8 chips, the TPU v5e operates as a single-host device. This means all 4 chips are attached to a single VM (e.g., `ct5lp-hightpu-4t`). This simplifies the software architecture significantly, as there is no need for RPCs or multi-process orchestration across different physical hosts. Data loading and preprocessing happen in a single Python process. 
* **ICI Bandwidth:** The v5e features an ICI bandwidth of 400 GB/s per chip. In a $2 \times 2$ mesh, each chip is directly connected to two neighbors (in a torus, the connections wrap around, forming a fully connected ring in each dimension). This high bandwidth enables efficient collective operations like all-gather and reduce-scatter, which are essential for the FlashDecoding inference strategy. 

---

## 3. Phase I: The Baseline Construction 

Before engineering the custom kernel, we must establish a rigorous baseline using the standard XLA compiler. This serves to quantify the performance ceiling of the default software stack and highlights the specific limitations imposed by the v5e's 16GB HBM. 

### 3.1 Mathematical Formulation of Standard Attention 

The baseline implementation will utilize standard `jax.numpy` primitives to implement the Scaled Dot-Product Attention mechanism: 

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V $$

Where:
* $Q, K, V \in \mathbb{R}^{B \times H \times N \times D}$ (Batch, Heads, Sequence Length, Head Dimension). 
* $M$ is the causal mask (lower triangular matrix of zeros, upper triangular of $-\infty$). 

### 3.2 Implementation Strategy: Pure JAX with XLA 

The baseline code `baseline_mha.py` will be decorated with `@jax.jit` to allow XLA to compile the graph. 

**Projected Failure Modes on v5e:** 

1.  **Memory Explosion (OOM):** The standard implementation computes `logits = jnp.matmul(Q, K.T)`. This creates an intermediate tensor of shape $(B, H, N, N)$. For a sequence length $N=32,768$, batch size $B=1$, and heads $H=16$, this tensor (in float32) requires:
    $$ 1 \times 16 \times 32768^2 \times 4 \text{ bytes} \approx 68 \text{ GB} $$ 
    This far exceeds the 16 GB HBM capacity of a single v5e chip. Even with bfloat16 (34 GB), it is unfeasible. We expect the baseline to crash with OOM at sequence lengths around 8k to 12k. 
2.  **Bandwidth Saturation:** Even at smaller sequence lengths where OOM is avoided, the standard implementation requires reading and writing these $N \times N$ matrices to HBM multiple times (calculation, masking, softmax, multiplication). With the v5e's 819 GB/s bandwidth, this I/O traffic will dominate execution time, leaving the MXUs idle. 

### 3.3 Benchmarking Methodology 

To scientifically validate the "Why this wins" proposition, we will measure: 
* **Latency:** Wall-clock time per step (using `time.perf_counter` after `block_until_ready()`).
* **Throughput:** Tokens per second. 
* **HBM Bandwidth Utilization:** Using `jax.profiler`, we will monitor the memory bus. A saturated memory bus with low MXU utilization confirms a memory-bound workload. 
* **Model FLOPs Utilization (MFU):** The ratio of achieved FLOPs to the theoretical peak (197 TFLOPS). 

---

## 4. Phase II: The Pallas Kernel Architecture 

This phase constitutes the core engineering effort: replacing the memory-inefficient XLA graph with a manually tiled Pallas kernel optimized for the v5e. 

### 4.1 The Pallas Programming Model for v5e 

Pallas acts as a bridge to the low-level Mosaic/TPU assembly. For the v5e, the programming model is simplified due to the single-core architecture. 

* **Grid Definition:** The kernel is executed by dividing the output matrix into independent blocks. The grid defines the number of program instances. 
    * Target Grid: `(Batch_Size, Num_Heads, N_Blocks_Output)` 
    * On v5e, each point in the grid is scheduled onto the single TensorCore. There is no need to map specific dimensions to "sub-cores" as required in v4/v5p "Megacore" configurations. 
* **Memory Spaces:** Pallas distinguishes between Refs in HBM (Global) and Refs in VMEM/SMEM (Local). 
    * **Input/Output:** Reside in HBM. 
    * **Scratchpad:** We explicitly allocate buffers in VMEM (Vector Memory) and SMEM (Scalar Memory). 

### 4.2 The Tiling Strategy: Exploiting the 128 MIB VMEM 

The defining characteristic of the "Pallas-Flash" on v5e is the aggressive use of the 128 MiB VMEM. 

**The Block Size Calculation:** 
Standard FlashAttention on GPUs often uses block sizes of $128 \times 64$ or $128 \times 128$ to fit into the small (100KB-200KB) SRAM. On TPU v5e, we have orders of magnitude more space. 

* **Query Block ($B_q$):** We can load a very large block of Queries. For $D=128$, a block of $1024 \times 128$ floats takes only 0.5 MB. We can easily fit blocks of $B_q=1024$ or even $2048$ into VMEM. 
* **Key/Value Block ($B_k$):** We stream $K$ and $V$ in smaller chunks, e.g., $B_k = 512$. 

**Why Larger Tiles Win on v5e:** 
* **Arithmetic Intensity:** The inner loop computes $Q_{block} \times K_{block}^T$. By increasing the size of the resident $Q_{block}$, we perform more FLOPs for every byte of $K/V$ loaded from HBM. 
* **Hiding Latency:** The v5e has a lower HBM bandwidth (819 GB/s). Larger tiles create longer compute phases, providing a larger time window to fetch the next blocks via DMA, effectively hiding the memory latency. 

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

* **Constraint:** All inner dimensions of matrix multiplications must be multiples of 128.
* **The "Copy" Trap:** If inputs provided to the Pallas kernel are not padded (e.g., head dimension 96), the XLA compiler will silently inject massive data-copy operations to pad them in HBM before the kernel runs. This doubles memory usage and kills performance.
* **Mandatory Requirement:** The data pipeline must utilize `jax.numpy.pad` to ensure all tensor dimensions passed to the kernel are strictly aligned to 128 bytes. The Pallas `BlockSpec` must request aligned tiles. 

---

## 5. Phase III: Assembly Debugging and Low-Level Optimization 

Developing high-performance kernels requires verifying that the hardware is executing as intended. 

### 5.1 Artifact Analysis 

We will use `XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"` to inspect the compiled binary.

* **HLO (High-Level Optimizer) Analysis:** We search the HLO text for `copy-start` and `copy-done` instructions. Their presence surrounding our `custom_call` indicates a failure to align memory, triggering the "Copy Trap." 
* **Vector Register Spilling:** If we tile too aggressively and exceed the 128 MiB VMEM (or fragment it), the compiler may spill registers to HBM. We verify this by checking for unexpected HBM traffic in the profiler. 

### 5.2 Pipeline Bubbles 

Using the JAX Profiler Trace Viewer: 
* **Goal:** A solid "wall" of MXU activity. 
* **Failure:** Gaps between MXU blocks indicate that the compute finished before the next DMA load arrived. 
* **Fix for v5e:** If bubbles exist, we must increase the `Block_Q` size. This increases the compute duration of the inner loop without increasing data loading, giving the DMA engine more time to complete the fetch. This is the primary tuning lever for the v5e's 819 GB/s bandwidth. 

---

## 6. Phase IV: The Ceiling - FlashDecoding on 4 Chips 

The "FlashDecoding" phase addresses the inference bottleneck. In decoding, we generate one token at a time. A single query token attending to a 32k KV cache is purely memory-bound. 

### 6.1 The Inference Challenge on v5e 

On a single v5e chip, scanning a 32k KV cache (float16) involves reading ~130MB of data. At 819 GB/s, this takes:
$$ \frac{130 \text{ MB}}{819 \text{ GB/s}} \approx 158 \mu s $$ 

This is fast, but as context grows to 128k or 1M, this linear scan becomes the latency bottleneck. Furthermore, utilizing only 1 chip of the 4 available leaves 75% of the slice's bandwidth idle. 

### 6.2 The Solution: Split-KV Parallelism 

We will implement Split-KV Decoding to utilize the full 2x2 mesh.

**Algorithm:**
1.  **Shard the KV Cache:** The sequence length $N$ is divided equally across the 4 chips. For $N=32k$, each chip holds 8k tokens. Sharding Spec: `PartitionSpec('N', 'H',...)` mapped to the 4-device mesh. 
2.  **Broadcast Query:** The single query token $Q$ is broadcast to all 4 chips. 
3.  **Local Attention:** Each chip effectively runs a "mini" attention operation on its local 8k KV shard. This happens in parallel. Each chip produces: `(local_output, local_max, local_sum)`. 
4.  **Global Reduction:** 
    * We use `jax.lax.psum` (or `all_gather`) over the ICI network to combine results. The 400 GB/s ICI bandwidth is extremely fast for this small reduction (only passing output vectors, not the KV cache). 
    * **LogSumExp:** We perform a stable LogSumExp reduction across the 4 chips to correctly normalize the partial attention scores. 

### 6.3 Why v5e-4 Excels Here 

The aggregate HBM bandwidth of the 4-chip slice is $4 \times 819 \text{ GB/s} \approx 3.2 \text{ TB/s}$. By parallelizing the memory read, we reduce the memory stall time by nearly 4x compared to a single chip. This allows the v5e slice to deliver inference speeds competitive with much more expensive single GPUs (like the H100) for long-context workloads. 

---

## 7. Project Roadmap and Implementation Plan 

This 8-week timeline is calibrated for the specific challenges of the v5e architecture. 

**Week 1-2: Foundation and Baseline (v5e-4 Environment)** 
* **Objective:** Quantify the "Memory Wall" on 16GB chips. 
* **Tasks:**
    * Provision `ct5lp-hightpu-4t` (v5e-4) VM. 
    * Implement `baseline_mha.py` with `jax.jit`.
    * Stress Test: Determine the maximum Sequence Length before OOM (likely ~16k-20k). 
* **Deliverable:** Roofline plot showing HBM Bandwidth saturation at ~800 GB/s. 

**Week 3-4: Pallas Kernel V1 (Single Chip)** 
* **Objective:** Functional correctness with 128x128 tiling. 
* **Tasks:** 
    * Implement `flash_attention_kernel` in Pallas. 
    * Implement `BlockSpec` logic mapping (B, H, N) to 128-sized tiles. 
    * **v5e Specific:** Verify that VMEM usage is within 128 MiB limits using `jax.profiler`. 
    * Debug NaNs in Softmax (ensure bf16 stability). 

**Week 5-6: Optimization & Pipelining (The 128MB Pivot)** 
* **Objective:** Maximize Arithmetic Intensity. 
* **Tasks:** 
    * **Tiling Tuning:** Increase `Block_Q` from 128 to 512, then 1024. Observe impact on latency. 
    * **Pipelining:** Implement `pl.pipeline` to overlap the next KV block load with current compute. 
    * **HLO Inspection:** Verify removal of `copy-start` instructions by enforcing 128-byte alignment on inputs. 
* **Deliverable:** Kernel achieving >60% MFU on long sequences. 

**Week 7: FlashDecoding (4-Chip Distributed)** 
* **Objective:** Low-latency Inference. 
* **Tasks:** 
    * Implement `flash_decoding_kernel` designed for Batch = 1. 
    * Use `jax.shard_map` or `jax.pmap` to distribute KV cache across the 4 devices. 
    * Implement the custom LogSumExp reduction over the $2 \times 2$ ICI mesh. 
    * **Benchmark:** Compare Latency of Single-Chip vs. 4-Chip Split-KV. 

**Week 8: Documentation & Handover** 
* **Objective:** Knowledge Transfer. 
* **Tasks:** 
    * Final Report: "Optimizing for the v5e Memory Hierarchy." 
    * Release code with specific comments on v5e constraints (16GB limit, 128MB VMEM usage). 

---

## 8. Conclusion 

The shift to the TPU v5e requires a departure from "brute-force" scaling strategies. We are no longer limited by the synchronization of thousands of chips, but rather by the strict memory boundaries of a single, efficient accelerator. By leveraging the Pallas programming model, we can convert the v5e's unique architectural features—specifically its massive 128 MiB on-chip VMEM and efficient single-core design—into a decisive advantage. 

The Pallas-Flash kernel will demonstrate that by manually managing memory hierarchy, we can overcome the 16 GiB HBM constraint and the 819 GB/s bandwidth limit. Furthermore, the FlashDecoding implementation will prove that a modest 4-chip v5e slice, when orchestrated correctly via Split-KV parallelism, can deliver inference performance that punches far above its weight class. This project does not merely "port" FlashAttention; it re-engineers it to define the state-of-the-art for efficient, long-context AI compute. 

---

## 9. Appendix: Data Tables and Reference Information 

**Table 2: Hardware Specification Comparison** 

| Specification | TPU v4 (Reference) | TPU v5e (Target) | Impact on Kernel Design |
| :--- | :--- | :--- | :--- |
| **Chip Architecture** | Dual TensorCore (Megacore) | Single TensorCore | **Simplified Grid:** No need to handle intra-chip devicemap. |
| **MXU Array** | 4x (128x128) per core | 4x ($128 \times 128$) per core | **Tiling:** Must align to 128. Padding is critical. |
| **HBM Capacity** | 32 GiB / chip | 16 GiB / chip | **High Risk:** OOM is the primary threat. FlashAttention is mandatory. |
| **HBM Bandwidth** | $1.2~TB/s$ | 819 GB/s | **Bottleneck:** Requires larger resident blocks in VMEM to hide latency. |
| **VMEM Size** | ~16 MiB (+ CMEM) | ~128 MiB | **Advantage:** Use large tiles (e.g., $1024 \times 128$) to increase arithmetic intensity. |
| **Topology** | 3D Torus | 2x2 2D Torus | **Connectivity:** High-speed ICI |

**Table 3: Pallas Tiling Recommendations for TPU v5e** 

| Parameter | Recommended Value | Reasoning |
| :--- | :--- | :--- |
| **Block_Q (Query Tile)** | 512 or 1024 | Maximizes reuse of loaded Q data; leverages 128MB VMEM to hide HBM latency. |
| **Block_KV (Key/Value Tile)** | 128 | Small enough to stream efficiently; matches MXU dimension alignment. |
| **Head Dimension** | 128 or 256 | Power of 2; ensures full utilization of vector lanes. |
| **Pipeline Stages** | 2 or 3 | Necessary to saturate the DMA engine and hide the 819 GB/s bandwidth limit. enables efficient 4-way Split-KV. |

## Citations 

[1] TPU v5e - Google Cloud Documentation, accessed January 21, 2026, https://docs.cloud.google.com/tpu/docs/v5e 
[2] Performance per dollar of GPUs and TPUs for AI inference | Google Cloud Blog, accessed January 21, 2026, https://cloud.google.com/blog/products/compute/performance-per-dollar-of-gpus-and-tpus-for-ai-inference 
[3] TPU v4 - Google Cloud Documentation, accessed January 21, 2026, https://docs.cloud.google.com/tpu/docs/v4 
[4] Serve Gemma open models using TPUs on GKE with Saxml | Kubernetes Engine, accessed January 21, 2026, https://docs.cloud.google.com/kubernetes-engine/docs/deprecations/serve-gemma-tpu-saxml 
[5] How to Think About TPUs | How To Scale Your Model - GitHub Pages, accessed January 21, 2026, https://jax-ml.github.io/scaling-book/tpus/ 
[6] TPU architecture - Google Cloud Documentation, accessed January 21, 2026, https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm 
[7] Pallas-Flash Transformer Kernel Project Plan.pdf 
[8] The Rise of Pallas: Unlocking TPU Potential with Custom Kernels - Medium, accessed January 21, 2026, https://medium.com/data-science/the-rise-of-pallas-unlocking-tpu-potential-with-custom-kernels-67be10ab846a 
[9] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | OpenReview, accessed January 21, 2026, https://openreview.net/forum?id=mZn2Xyh9Ec