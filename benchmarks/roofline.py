"""
Roofline Model Generator for TPU v5e Architecture (roofline.py)

This script generates a Roofline plot demonstrating the "Memory Wall"
and the theoretical advantage of the Pallas-Flash tiled attention kernel.

The Roofline model plots:
    Y-axis: Attainable Performance (TFLOPS)
    X-axis: Arithmetic Intensity (FLOPs / Byte)

TPU v5e Hardware Limits:
    - Peak Compute (bfloat16): 197 TFLOPS
    - Peak HBM Bandwidth: 819 GB/s
    - Ridge Point (Compute / Bandwidth): ~240.5 FLOPs/Byte

Deliverable for Phase 0 of the Pallas-Flash Project.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Attempt to load from the project config, fallback to known v5e constants
try:
    from pallas_flash.config import v5e_specs
    PEAK_TFLOPS = v5e_specs.PEAK_FLOPS_BF16 / 1e12
    BANDWIDTH_GBPS = v5e_specs.HBM_BANDWIDTH_BYTES_PER_SEC / 1e9
except ImportError:
    PEAK_TFLOPS = 197.0
    BANDWIDTH_GBPS = 819.0

def generate_roofline_plot(save_path="roofline_v5e.png"):
    """
    Calculates the memory/compute bounds and plots the Roofline model.
    """
    print(f"Generating TPU v5e Roofline Model...")
    print(f"  Peak Compute: {PEAK_TFLOPS} TFLOPS")
    print(f"  HBM Bandwidth: {BANDWIDTH_GBPS} GB/s")

    # Ridge Point Calculation: Where the memory bound meets the compute bound
    # Required AI to hit peak FLOPs = Peak TFLOPS (in FLOPs) / Bandwidth (in Bytes)
    ridge_point = (PEAK_TFLOPS * 1e12) / (BANDWIDTH_GBPS * 1e9)
    print(f"  Ridge Point: {ridge_point:.2f} FLOPs/Byte")

    # Define Arithmetic Intensity (AI) axis (log scale)
    ai_vals = np.logspace(0, 4, 1000)

    # Calculate Attainable Performance bounds
    # Memory bound: Performance = AI * Bandwidth
    # Compute bound: Performance = Peak_Compute
    perf_memory_bound = (ai_vals * BANDWIDTH_GBPS * 1e9) / 1e12  # converted to TFLOPS
    perf_compute_bound = np.full_like(ai_vals, PEAK_TFLOPS)
    
    # Actual theoretical roofline is the minimum of the two
    attainable_perf = np.minimum(perf_memory_bound, perf_compute_bound)

    # --- Plotting ---
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Plot the main roofline
    plt.plot(ai_vals, attainable_perf, color='black', linewidth=2, label='TPU v5e Roofline')
    
    # Fill the attainable region
    plt.fill_between(ai_vals, 0, attainable_perf, alpha=0.1, color='gray')

    # Add theoretical projection points for Attention
    # 1. Standard XLA Attention (Memory Bound due to intermediate reads/writes)
    # Estimate: Read/Write heavy, roughly 20-40 FLOPs/byte
    std_ai = 30 
    std_perf = min((std_ai * BANDWIDTH_GBPS * 1e9) / 1e12, PEAK_TFLOPS)
    plt.scatter([std_ai], [std_perf], color='red', s=100, zorder=5, label='Standard XLA Attention (Est.)')
    plt.annotate('Memory Wall\n(Standard Attn)', xy=(std_ai, std_perf), 
                 xytext=(std_ai/3, std_perf*1.5),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6),
                 fontsize=10, color='red')

    # 2. Pallas-Flash (Compute Bound due to 128MB VMEM blocking)
    # Estimate: Large Q_blocks (1024) reused against streaming KV, ~500-1000 FLOPs/byte
    flash_ai = 800
    flash_perf = min((flash_ai * BANDWIDTH_GBPS * 1e9) / 1e12, PEAK_TFLOPS)
    plt.scatter([flash_ai], [flash_perf], color='green', s=100, zorder=5, label='Pallas-Flash (Target)')
    plt.annotate('Target Operation\n(Compute Bound)', xy=(flash_ai, flash_perf), 
                 xytext=(flash_ai*1.5, flash_perf*0.6),
                 arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=6),
                 fontsize=10, color='green')

    # Formatting
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Arithmetic Intensity (FLOPs / Byte)', fontsize=12)
    plt.ylabel('Performance (TFLOPS)', fontsize=12)
    plt.title('TPU v5e Roofline Model: Bypassing the Memory Wall', fontsize=14, fontweight='bold')
    
    # Grid and limits
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlim(1, 10000)
    plt.ylim(1, 1000)
    
    # Plot Ridge Point vertical line for clarity
    plt.axvline(x=ridge_point, color='blue', linestyle=':', alpha=0.6, label=f'Ridge Point ({ridge_point:.1f} FLOPs/B)')

    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()

    # Save and show
    plt.savefig(save_path)
    print(f"\nPlot saved successfully to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    # If run in the benchmarks folder, output to the root directory
    out_file = "v5e_roofline_analysis.png"
    if os.path.basename(os.getcwd()) == "benchmarks":
        out_file = "../" + out_file
        
    generate_roofline_plot(save_path=out_file)