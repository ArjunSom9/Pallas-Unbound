"""
Hardware Timeline Trace Viewer (trace_viewer.py)

This script analyzes the JAX/XLA profiler output (`trace.json.gz`) to 
breakdown exactly where the TPU is spending its time.

Fixed: Added dynamic thread mapping and broadened the search categories 
to catch XLA 'Op' and 'custom-call' events on TPU 'compute' streams.
"""

import os
import json
import gzip
import glob
import sys
from collections import defaultdict

def analyze_trace(trace_dir):
    print(f"\n[Trace Analyzer] Loading trace from:\n -> {trace_dir}")
    
    # 1. Find the trace file
    trace_files = glob.glob(os.path.join(trace_dir, '**/*.trace.json.gz'), recursive=True)
    if not trace_files:
        print("[-] No trace file found. Check if the profiler was enabled.")
        sys.exit(1)

    trace_file = trace_files[0]
    print(f"[Trace Analyzer] Unzipping and parsing JSON (this may take a moment)...")
    
    with gzip.open(trace_file, 'rt') as f:
        trace_data = json.load(f)
        
    print("[Trace Analyzer] Aggregating hardware events...")
    
    # 2. Map Thread IDs to their names
    thread_names = {}
    for event in trace_data.get('traceEvents', []):
        if event.get('name') == 'thread_name':
            tid = event.get('tid')
            tname = event.get('args', {}).get('name', '')
            thread_names[tid] = tname.lower()

    # 3. Identify likely device (TPU) threads
    device_threads = set()
    for tid, name in thread_names.items():
        # XLA usually runs device ops on threads named "Main compute", "Stream", "TPU...", etc.
        if any(kw in name for kw in ['tpu', 'device', 'stream', 'compute', 'ops']):
            device_threads.add(tid)

    # 4. Aggregate Execution Time
    total_device_time_us = 0
    op_counts = defaultdict(int)
    
    for event in trace_data.get('traceEvents', []):
        if event.get('ph') == 'X':  # 'X' represents a Duration event
            tid = event.get('tid')
            cat = event.get('cat', '').lower()
            name = event.get('name', 'unknown')
            name_lower = name.lower()
            dur = event.get('dur', 0)
            
            # Ignore completely host-side overheads and sleep times
            if 'idle' in name_lower or 'host' in cat or 'cpu' in cat:
                continue
                
            # It's a device compute op if:
            # - It ran on a known device thread
            # - Or it's categorized as a hardware Op/Kernel
            # - Or it's a known XLA custom-call (which Pallas kernels compile into)
            is_device_compute = (
                tid in device_threads or 
                'op' in cat or 
                'kernel' in cat or 
                'device' in cat or
                'custom-call' in name_lower or
                'pallas' in name_lower
            )

            if is_device_compute:
                total_device_time_us += dur
                op_counts[name] += dur

    # 5. Fallback check
    if total_device_time_us == 0:
        print("\n[!] No device execution time found.")
        print("    Dumping found thread names for debugging:")
        for tid, name in thread_names.items():
            print(f"      - TID {tid}: {name}")
        sys.exit(1)
        
    # 6. Diagnostic Report
    print("\n============================================================")
    print(" Pallas-Flash: Hardware Timeline Profile")
    print("============================================================")
    print(f" Total Device Compute Time : {total_device_time_us / 1000:.2f} ms")
    print("------------------------------------------------------------")
    print(" [Top 5 Most Expensive Operations]")
    
    sorted_ops = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, dur in sorted_ops:
        print(f"   - {name[:40]:<40}: {dur / 1000:.2f} ms")
        
    print("============================================================\n")
    print(" ✅ TFLOPS computation is strictly bounded by these operations.")
    print("    If 'custom-call' (Pallas) dominates, you are compute-bound!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pallas Trace Viewer")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing trace.json.gz")
    args = parser.parse_args()
    analyze_trace(args.dir)