#!/bin/bash
# Pallas-Flash: HLO Assembly Dump Utility (scripts/dump_hlo.sh)
#
# This script executes a target Python file while forcing the XLA compiler
# to dump its High-Level Optimizer (HLO) compilation graphs. 
#
# Context:
# If our layout padding (layout.py) fails to perfectly align with the 128x128 
# v5e MXU, the XLA compiler will silently inject 'copy-start' and 'copy-done' 
# operations to pad the memory on the fly. This script captures the assembly
# so our hlo_analyzer.py can parse it and verify those copies don't exist.

set -e

# Default configuration
DUMP_DIR="/tmp/xla_dump"
DEFAULT_TARGET="tests/test_correctness.py"

# Allow the user to pass a specific script to profile, otherwise default to the correctness tests.
TARGET_SCRIPT=${1:-$DEFAULT_TARGET}

echo "============================================================"
echo " XLA Compiler Artifact Dumper"
echo "============================================================"
echo "Target Script : $TARGET_SCRIPT"
echo "Dump Location : $DUMP_DIR"
echo ""

# 1. Clean the environment
# We must clear the dump directory before every run. If we don't, XLA will just
# add new numbered dumps alongside the old ones, making it impossible for our 
# automated analyzer to know which graph belongs to the current run.
if [ -d "$DUMP_DIR" ]; then
    echo "[-] Clearing previous XLA dumps..."
    rm -rf "$DUMP_DIR"
fi
mkdir -p "$DUMP_DIR"

# 2. Set the magical XLA_FLAGS
# --xla_dump_to: Tells XLA where to put the .txt and .pb artifacts.
# --xla_dump_hlo_pass_re=.* : Tells XLA to dump the graph after EVERY optimization pass.
export XLA_FLAGS="--xla_dump_to=$DUMP_DIR --xla_dump_hlo_pass_re=.*"

# 3. Execute the target
echo "[+] Executing $TARGET_SCRIPT with JIT HLO dumping enabled..."
echo "------------------------------------------------------------"

# Use python3 or pytest depending on the target script
if [[ "$TARGET_SCRIPT" == *"test_"* ]]; then
    pytest -v -s "$TARGET_SCRIPT"
else
    python3 "$TARGET_SCRIPT"
fi

echo "------------------------------------------------------------"
echo "[+] Execution Complete."
echo "[+] HLO graphs successfully dumped to: $DUMP_DIR"
echo ""
echo "Next Step: Run the HLO Analyzer to check for 'Copy Traps'."
echo "Command: python3 profiling/hlo_analyzer.py"
echo "============================================================"