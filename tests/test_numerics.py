"""
Numerical Stability Tests for Pallas-Flash (test_numerics.py)

This module validates the low-level VPU (Vector Processing Unit) math 
implemented in the Pallas-Flash kernel. 

Context:
The TPU v5e MXU operates strictly in bfloat16. However, bfloat16 is highly 
susceptible to overflow during the exponentiation phase of the Softmax 
operation. We use a localized "Online Softmax" that incrementally subtracts 
the running maximum and accumulates the denominator in float32. 

These tests feed extreme, degenerate, and adversarial inputs to the kernel 
to guarantee that it never returns NaNs (Not a Number) or Infs, proving 
the robustness of the `vpu_stable_exp` intrinsic.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Import our custom kernel
from pallas_flash.ops.interface import pallas_flash_attention

# Standard dimensions for fast testing
B, H, N, D = 2, 2, 512, 128

def test_extreme_magnitude_logits():
    """
    Tests the kernel against massive input values.
    
    If the `vpu_stable_exp` does not correctly shift logits by their running 
    maximum, Q * K^T will exceed the maximum representable value for float32/bf16, 
    causing jnp.exp() to return Infinity, which then turns into NaNs when divided.
    """
    print("\n--- Testing Extreme Magnitudes ---")
    
    # 1. Initialize with massive values
    # Q * K^T will result in values in the thousands/millions.
    # jnp.exp(100) already overflows standard bf16 without shifting.
    q = jnp.ones((B, H, N, D), dtype=jnp.bfloat16) * 100.0
    k = jnp.ones((B, H, N, D), dtype=jnp.bfloat16) * 100.0
    v = jnp.ones((B, H, N, D), dtype=jnp.bfloat16)
    
    # 2. Run Kernel
    out = pallas_flash_attention(q, k, v)
    
    # 3. Assertions
    # There should be exactly zero NaNs or Infs.
    assert jnp.all(jnp.isfinite(out)), "Extreme logits caused NaN or Inf explosions!"
    
    # Because all Q and K are identical, the attention weights should be uniform.
    # Therefore, the output should just be exactly V (which is 1.0 everywhere).
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32),
        np.ones((B, H, N, D), dtype=np.float32),
        atol=1e-2,
        err_msg="Extreme uniform logits failed to average the values correctly."
    )
    print("✓ Passed Extreme Magnitude Test.")

def test_zero_variance_inputs():
    """
    Tests the kernel against zeroed inputs to ensure no division-by-zero 
    errors occur in the online softmax normalization phase.
    """
    print("\n--- Testing Zero Variance ---")
    
    q = jnp.zeros((B, H, N, D), dtype=jnp.bfloat16)
    k = jnp.zeros((B, H, N, D), dtype=jnp.bfloat16)
    
    # Random values for V
    v = jax.random.normal(jax.random.PRNGKey(0), (B, H, N, D), dtype=jnp.bfloat16)
    
    out = pallas_flash_attention(q, k, v)
    
    # No NaNs should exist
    assert jnp.all(jnp.isfinite(out)), "Zeroed inputs caused a division by zero!"
    print("✓ Passed Zero Variance Test.")

def test_highly_negative_keys():
    """
    Simulates the behavior of causal or padding masks by injecting highly 
    negative values. Ensures the float32 accumulator (l_i) does not underflow 
    to exactly 0.0, which would cause a division by zero at the end of the loop.
    """
    print("\n--- Testing Highly Negative Values (Mask Simulation) ---")
    
    q = jax.random.normal(jax.random.PRNGKey(1), (B, H, N, D), dtype=jnp.bfloat16)
    
    # Create keys that will result in massive negative logits
    k = jnp.ones((B, H, N, D), dtype=jnp.bfloat16) * -1e4
    v = jax.random.normal(jax.random.PRNGKey(2), (B, H, N, D), dtype=jnp.bfloat16)
    
    out = pallas_flash_attention(q, k, v)
    
    assert jnp.all(jnp.isfinite(out)), "Highly negative keys resulted in an underflow trap!"
    print("✓ Passed Highly Negative Keys Test.")

def test_precision_preservation():
    """
    Ensures the final output matches the requested MXU hardware precision 
    (bfloat16) despite the internal pipeline accumulating in float32.
    """
    print("\n--- Testing Hardware Precision Contracts ---")
    
    q = jax.random.normal(jax.random.PRNGKey(3), (B, H, N, D), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.PRNGKey(4), (B, H, N, D), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.PRNGKey(5), (B, H, N, D), dtype=jnp.bfloat16)
    
    out = pallas_flash_attention(q, k, v)
    
    assert out.dtype == jnp.bfloat16, \
        f"Output dtype mismatch! Expected bfloat16, got {out.dtype}. " \
        "The VPU float32 accumulators are leaking into the output."
    print("✓ Passed Precision Preservation Test.")