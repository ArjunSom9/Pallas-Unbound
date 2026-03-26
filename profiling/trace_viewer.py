"""
JAX/XLA Trace Viewer and Analyzer (trace_viewer.py)

This script parses the raw Perfetto/Chrome Trace Format (.json.gz) files 
dumped by `jax.profiler` during the `capture_trace.sh` benchmark.

Context (Phase IV - Utilization Profiling):
To bypass the Memory Wall on the TPU v5e, our inner loop must overlap DMA 
memory loads with MXU compute. A perfectly pipelined loop will show that 
the time spent in the custom Pallas kernel dominates the trace, with almost 
zero time stalled on explicit 'copy' instructions.

This tool extracts the event durations from the trace and provides a 
command-line summary of where the TPU spent its time.
"""

import os
import json
import gzip
import glob
import argparse
from collections import defaultdict

DEFAULT_TRACE_DIR = "/tmp/pallas_trace"

def find_latest_trace(base_dir: str) -> str:
    """
    Locates the most recently generated trace.json.gz file within the 
    TensorBoard profiler directory structure.
    """
    if not os.path.exists(base_dir):
        return None
        
    # JAX profiler typically saves traces in plugins/profile/<timestamp>/...
    search_pattern = os.path.join(base_dir, "plugins", "profile", "**", "*.trace.json.gz")
    trace_files = glob.glob(search_pattern, recursive=True)
    
    if not trace_files:
        return None
        
    # Return the most recently modified trace file
    trace_files.sort(key=os.path.getmtime, reverse=True)
    return trace_files[0]

def analyze_trace(trace_file: str):
    """
    Parses the gzipped JSON trace and extracts execution time metrics.
    """
    print(f"\n[Trace Analyzer] Loading trace from:\n -> {trace_file}")
    print("[Trace Analyzer] Unzipping and parsing JSON (this may take a moment)...")
    
    try:
        with gzip.open(trace_file, 'rt', encoding='utf-8') as f:
            trace_data = json.load(f)
    except Exception as e:
        print(f"[-] Failed to parse trace file: {e}")
        return
        
    events = trace_data.get("traceEvents", [])
    
    # Dictionaries to hold aggregated metrics
    # dur: Duration of the event in microseconds
    op_durations = defaultdict(float)
    category_durations = defaultdict(float)
    total_compute_time = 0.0
    
    print("[Trace Analyzer] Aggregating hardware events...")
    
    for event in events:
        # We only care about Complete ('X') events that have a duration ('dur')
        if event.get("ph") == "X" and "dur" in event:
            name = event.get("name", "unknown")
            cat = event.get("cat", "unknown")
            dur_ms = event["dur"] / 1000.0  # Convert microseconds to milliseconds
            
            # Filter for device-side execution (ignore host python overhead)
            if "Device" in event.get("args", {}).get("name", "") or "device" in cat.lower():
                op_durations[name] += dur_ms
                category_durations[cat] += dur_ms
                total_compute_time += dur_ms

    if total_compute_time == 0:
        print("\n[!] No device execution time found. Did the trace capture properly?")
        return

    print_report(op_durations, category_durations, total_compute_time)

def print_report(op_durations, category_durations, total_compute_time):
    """
    Formats and prints the engineering trace report.
    """
    print("\n============================================================")
    print(" Pallas-Flash: Hardware Utilization Report")
    print("============================================================")
    print(f" Total Tracked Device Time : {total_compute_time:.2f} ms")
    print("------------------------------------------------------------")
    
    print(" [Top 10 Most Expensive Operations]")
    # Sort operations by descending duration
    sorted_ops = sorted(op_durations.items(), key=lambda x: x[1], reverse=True)
    
    for i, (op_name, dur) in enumerate(sorted_ops[:10]):
        percentage = (dur / total_compute_time) * 100
        # Highlight our custom Pallas kernel if we find it
        if "pallas_call" in op_name.lower() or "custom-call" in op_name.lower():
            op_name = f"\033[92m{op_name} (Target Kernel)\033[0m"
            
        print(f" {i+1:2d}. {op_name:<40} : {dur:>8.2f} ms ({percentage:>5.1f}%)")
        
    print("------------------------------------------------------------")
    print(" [Diagnostic Check]")
    
    # Look for the "Copy Trap" footprint in the trace
    copy_time = sum(dur for name, dur in op_durations.items() if "copy" in name.lower())
    copy_percentage = (copy_time / total_compute_time) * 100
    
    if copy_percentage < 5.0:
        print(" ✅ PASS: Memory I/O is sufficiently hidden.")
        print(f"    Only {copy_percentage:.1f}% of device time spent on explicit copies.")
        print("    The kernel is compute-bound, successfully overlapping DMA & MXU.")
    else:
        print(" ❌ FAIL: Memory Wall Hit!")
        print(f"    {copy_percentage:.1f}% of device time is stalled on explicit copies.")
        print("    The software pipeline loop in `pipeline.py` is failing to hide")
        print("    the v5e's 819 GB/s bandwidth limit. Review the BlockSpecs.")
        
    print("============================================================")
    print(" Note: For a visual timeline, upload the trace.json.gz file")
    print(" to https://ui.perfetto.dev/ or use TensorBoard.")
    print("============================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pallas-Flash Trace Analyzer")
    parser.add_argument("--dir", type=str, default=DEFAULT_TRACE_DIR, 
                        help="Base directory containing TensorBoard/JAX traces.")
    args = parser.parse_args()

    latest_trace = find_latest_trace(args.dir)
    
    if not latest_trace:
        print(f"[-] No trace.json.gz files found in {args.dir}")
        print("[!] Ensure you have run: ./scripts/capture_trace.sh")
    else:
        analyze_trace(latest_trace)