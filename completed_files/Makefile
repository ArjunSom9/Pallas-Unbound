# Pallas-Flash-v5e Makefile
# acts as the control panel for the project.

# --- Configuration ---
PYTHON := python3
PIP := pip
PYTEST := pytest

# Directory definitions
SRC_DIR := src
TEST_DIR := tests
SCRIPT_DIR := scripts
DUMP_DIR := /tmp/xla_dump

# XLA Flags for low-level assembly debugging
# We dump HLO (High Level Optimizer) intermediate representations to analyze
# if implicit copies are being inserted.
DEBUG_FLAGS := --xla_dump_to=$(DUMP_DIR) --xla_dump_hlo_pass_re=".*"

.PHONY: help install test format check benchmark debug clean

# Default target: List available commands
help:
	@echo "Pallas-Flash-v5e Automation"
	@echo "==========================="
	@echo "make install    : Install dependencies (requires manual TPU runtime setup)"
	@echo "make format     : Auto-format code using Black and Isort"
	@echo "make check      : Verify code quality (linting)"
	@echo "make test       : Run the correctness and numerics suite"
	@echo "make debug      : Run tests with XLA HLO dumping enabled (for assembly analysis)"
	@echo "make analyze    : Parse HLO dumps to detect copy instructions"
	@echo "make profile    : Capture JAX traces for timeline analysis"
	@echo "make roofline   : Generate the Arithmetic Intensity vs TFLOPS plot"
	@echo "make benchmark  : Run the E2E performance and economics sweep"
	@echo "make clean      : Remove cache and compilation artifacts"

# --- Setup ---
install:
	@echo "Installing project in editable mode..."
	$(PIP) install -e ".[dev]"
	@echo ""
	@echo "⚠️  IMPORTANT: Now install the TPU runtime manually:"
	@echo "   pip install 'jax[tpu]>=0.4.23' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"

# --- Development Loop ---
format:
	black .
	isort .

check:
	black --check .
	isort --check .

test:
	$(PYTEST) $(TEST_DIR)

# --- Advanced Engineering Tasks ---

# The "Microscope": Runs code with XLA flags to generate assembly dumps
debug: clean
	@echo "Running tests with XLA Dump enabled at $(DUMP_DIR)..."
	mkdir -p $(DUMP_DIR)
	XLA_FLAGS="$(DEBUG_FLAGS)" $(PYTEST) $(TEST_DIR)/test_correctness.py
	@echo "Assembly dumped. Analyze with: python3 profiling/hlo_analyzer.py"

# Run the HLO analyzer on the dumped assembly (Phase 4)
analyze:
	$(PYTHON) profiling/hlo_analyzer.py

# Capture a JAX trace for performance debugging (Phase 4)
profile:
	chmod +x $(SCRIPT_DIR)/capture_trace.sh
	$(SCRIPT_DIR)/capture_trace.sh

# Generate the Roofline plot (Phase 0)
roofline:
	$(PYTHON) benchmarks/roofline.py

# The "Report Card": Runs the full economic analysis
benchmark:
	@echo "Running End-to-End Benchmarks..."
	chmod +x $(SCRIPT_DIR)/run_benchmark.sh
	$(SCRIPT_DIR)/run_benchmark.sh

# --- Cleanup ---
clean:
	rm -rf __pycache__
	rm -rf $(SRC_DIR)/__pycache__
	rm -rf .pytest_cache
	rm -rf $(DUMP_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} +