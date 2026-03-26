#!/bin/bash
# Pallas-Flash: End-to-End Benchmark Runner (scripts/run_benchmark.sh)
#
# This script executes the complete benchmarking suite for the TPU v5e
# Pallas-Flash kernel. It systematically evaluates arithmetic intensity,
# memory bandwidth saturation, and economic ROI.
#
# Usage:
#   chmod +x scripts/run_benchmark.sh
#   ./scripts/run_benchmark.sh

set -e

# Define color codes for terminal output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}======================================================================${NC}"
echo -e "${CYAN} Pallas-Flash: Comprehensive TPU v5e Benchmark Suite${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo "Starting benchmarking pipeline..."
echo ""

# -----------------------------------------------------------------------------
# Phase 0: Roofline Model Generation
# -----------------------------------------------------------------------------
echo -e "${GREEN}[1/4] Generating Hardware Roofline Model...${NC}"
python3 benchmarks/roofline.py
echo "Roofline plot saved. Moving to hardware execution..."
echo ""

# -----------------------------------------------------------------------------
# Phase 1: Training / Prefill Scaling (Compute Bound)
# -----------------------------------------------------------------------------
echo -e "${GREEN}[2/4] Executing Forward Pass Scaling Benchmark (TFLOPS)...${NC}"
# We use a smaller batch/head size here to prevent the baseline from OOMing
# immediately, allowing us to see the scaling curve before it hits the Memory Wall.
python3 benchmarks/scaling.py --batch=1 --heads=16 --dim=128
echo ""

# -----------------------------------------------------------------------------
# Phase 2: Autoregressive Decoding (Memory Bound)
# -----------------------------------------------------------------------------
echo -e "${GREEN}[3/4] Executing Autoregressive Decoding Benchmark (GB/s)...${NC}"
# We increase the batch size here to apply maximum pressure to the 819 GB/s 
# HBM bandwidth limit, testing the 1D online softmax.
python3 benchmarks/inference_decode.py --batch=8 --heads=16 --dim=128
echo ""

# -----------------------------------------------------------------------------
# Phase 3: Economic Analysis (ROI)
# -----------------------------------------------------------------------------
echo -e "${GREEN}[4/4] Executing Economic ROI and Cost Analysis...${NC}"
# This runs the cost-per-million-tokens analysis using standard LLM dimensions.
python3 benchmarks/economics.py --batch=1 --heads=32 --dim=128
echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo -e "${CYAN}======================================================================${NC}"
echo -e "${CYAN} Benchmarking Suite Complete!${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo "To generate a timeline trace of the MXU utilization, run:"
echo "  ./scripts/capture_trace.sh"
echo "To verify the zero-copy assembly generation, run:"
echo "  ./scripts/dump_hlo.sh"
echo "======================================================================"