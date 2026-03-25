# Pallas-Unbound (Pallas-Flash) Makefile
# Acts as the control panel for the project lifecycle.

# --- Configuration ---
PYTHON := python3
PIP := pip
PYTEST := pytest
BASH := bash

# Directory definitions
SRC_DIR := src
TEST_DIR := tests
SCRIPT_DIR := scripts
BENCHMARK_DIR := benchmarks
PROFILING_DIR := profiling
DUMP_DIR := /tmp/xla_dump

# XLA Flags for low-level assembly debugging
# We dump HLO (High Level Optimizer) intermediate representations to analyze
# if implicit copies are being inserted by the compiler.
DEBUG_FLAGS := --xla_dump_to=$(DUMP_DIR) --xla_dump_hlo_pass_re=".*"

.PHONY: help setup install format check test debug analyze profile roofline benchmark clean

# Default target: List available commands
help:
	@echo "Pallas-Flash-v5e Automation"
	@echo "==========================="
	@echo "make setup      : Provision the VM environment (run setup script)"
	@echo "make install    : Install dependencies (requires manual TPU runtime setup)"
	@echo "make format     : Auto-format code using Black and Isort"
	@echo "make check      : Verify code quality (linting)"
	@echo "make test       : Run the correctness, numerics, and memory suite"
	@echo "make debug      : Run tests with XLA HLO dumping enabled (for assembly analysis)"
	@echo "make analyze    : Parse HLO dumps to detect copy instructions"
	@echo "make profile    : Capture JAX traces for timeline analysis"
	@echo "make roofline   : Generate the Arithmetic Intensity vs TFLOPS plot"
	@echo "make benchmark  : Run the E2E performance and economics sweep"
	@echo "make clean      : Remove cache, compilation artifacts, and logs"

# --- Setup ---

# Provision system dependencies and virtual environment
setup:
	@echo "Running VM setup script..."
	$(BASH) completed_files/setup_vm_raw.sh

# Install Python requirements
install:
	@echo "Installing project in editable mode..."
	$(PIP) install -e ".[dev]"
	@echo ""
	@echo "⚠️  IMPORTANT: Ensure you have the TPU runtime installed:"
	@echo "   pip install 'jax[tpu]>=0.4.23' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"

# --- Development Loop ---

# Formatting
format:
	black .
	isort .

# Linting
check:
	black --check .
	isort --check .

# Standard Testing
test:
	$(PYTEST) $(TEST_DIR)

# --- Advanced Engineering Tasks ---

# The "Microscope": Runs code with XLA flags to generate assembly dumps
debug: clean
	@echo "Running tests with XLA Dump enabled at $(DUMP_DIR)..."
	mkdir -p $(DUMP_DIR)
	XLA_FLAGS="$(DEBUG_FLAGS)" $(PYTEST) $(TEST_DIR)/test_correctness.py
	@echo "Assembly dumped. Analyze with: make analyze"

# Run the HLO analyzer on the dumped assembly (Phase 4)
analyze:
	$(PYTHON) $(PROFILING_DIR)/hlo_analyzer.py

# Capture a JAX trace for performance debugging (Phase 4)
profile:
	chmod +x $(SCRIPT_DIR)/capture_trace.sh
	$(SCRIPT_DIR)/capture_trace.sh

# Generate the Roofline plot (Phase 0)
roofline:
	$(PYTHON) $(BENCHMARK_DIR)/roofline.py

# The "Report Card": Runs the full economic analysis
benchmark:
	@echo "Running End-to-End Benchmarks..."
	chmod +x $(SCRIPT_DIR)/run_benchmark.sh
	$(SCRIPT_DIR)/run_benchmark.sh

# --- Cleanup ---

# Erase temporary files, caches, and built artifacts
clean:
	@echo "Cleaning up caches and artifacts..."
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf $(DUMP_DIR)
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete."