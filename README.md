<!-- Executive Summary, Roofline Results, & Tokens/$ Metrics -->

# Pallas-Unbound
High-performance TPU v5e kernel engineered with JAX Pallas. Implements IO-aware FlashAttention &amp; FlashDecoding via manual HBM/VMEM management. Bypasses XLA heuristics to achieve linear O(N) scaling and low-latency inference on long-context workloads.
