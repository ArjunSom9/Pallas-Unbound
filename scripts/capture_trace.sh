#!/bin/bash
# Pallas-Flash: JAX Performance Trace Capture Utility (scripts/capture_trace.sh)
#
# This script prepares the environment and executes a target benchmark
# while capturing a JAX/XLA execution trace.
#
# Context (Phase IV - Utilization Profiling):
# To prove our pipeline loop successfully overlaps HBM I/O with MXU compute,
# we need to capture a hardware trace. A successful Pallas-Flash trace will 
# show a "solid wall" of MXU execution without gaps waiting for memory transfers.

set -e

# Default configuration
TRACE_DIR="/tmp/pallas_trace"
DEFAULT_TARGET="benchmarks/scaling.py"

# Allow the user to pass a specific script to profile
TARGET_SCRIPT=${1:-$DEFAULT_TARGET}

echo "============================================================"
echo " Pallas-Flash Timeline Profiler"
echo "============================================================"
echo "Target Script : $TARGET_SCRIPT"
echo "Trace Output  : $TRACE_DIR"
echo ""

# 1. Clean the environment
# TensorBoard/Perfetto profiling can get corrupted if multiple runs are 
# dumped into the same unstructured directory. We clean it first.
if [ -d "$TRACE_DIR" ]; then
    echo "[-] Clearing previous trace artifacts from $TRACE_DIR..."
    rm -rf "$TRACE_DIR"
fi
mkdir -p "$TRACE_DIR"

# 2. Set Profiling Environment Variables
# Our benchmarking and testing scripts will look for this environment variable.
# If present, they will wrap their execution block in `jax.profiler.start_trace()`
export PALLAS_PROFILE_DIR="$TRACE_DIR"

# 3. Execute the target
echo "[+] Executing $TARGET_SCRIPT with JAX tracing enabled..."
echo "------------------------------------------------------------"

python3 "$TARGET_SCRIPT"

echo "------------------------------------------------------------"
echo "[+] Execution Complete."
echo "[+] Trace artifacts successfully saved to: $TRACE_DIR"
echo ""
echo "============================================================"
echo " How to View the Trace:"
echo "============================================================"
echo "1. Start the TensorBoard Profiler Server:"
echo "   pip install tensorboard-plugin-profile"
echo "   tensorboard --logdir=$TRACE_DIR --port=6006"
echo ""
echo "2. Open your browser and navigate to:"
echo "   http://localhost:6006/#profile"
echo ""
echo "3. Alternatively, parse the trace programmatically using:"
echo "   python3 profiling/trace_viewer.py --dir=$TRACE_DIR"
echo "============================================================"