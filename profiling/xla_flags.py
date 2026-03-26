"""
XLA Compiler Debugging Configuration (xla_flags.py)

This module provides programmatic utilities to configure JAX's underlying XLA 
compiler flags. It is primarily used to dump High-Level Optimizer (HLO) 
artifacts for assembly-level analysis.

Context (Phase III - Assembly Debugging):
If tensor dimensions do not align perfectly with the v5e's 128x128 MXU, the 
compiler silently injects data-copy operations. By dumping the HLO passes, 
we can statically analyze the compiled graph for 'copy-start' and 'copy-done' 
instructions to verify that our padding logic in `layout.py` successfully 
prevented the "Copy Trap".
"""

import os
import shutil
import contextlib
from typing import Optional

# Standard dump directory used by the Makefile and analysis scripts
DEFAULT_DUMP_DIR = "/tmp/xla_dump"

def clear_dump_dir(dump_dir: str = DEFAULT_DUMP_DIR):
    """
    Clears the specified XLA dump directory to ensure artifact analysis
    only captures the most recent compilation graph.
    """
    if os.path.exists(dump_dir):
        shutil.rmtree(dump_dir)
    os.makedirs(dump_dir, exist_ok=True)
    print(f"[XLA Profiler] Cleared and prepared dump directory: {dump_dir}")

def enable_hlo_dump(dump_dir: str = DEFAULT_DUMP_DIR):
    """
    Globally sets the XLA_FLAGS environment variable to dump HLO passes.
    Note: This must be called BEFORE jax.jit compiles the function.
    """
    # Create the directory if it doesn't exist
    os.makedirs(dump_dir, exist_ok=True)
    
    # Flag Breakdown:
    # --xla_dump_to: Where to save the artifacts.
    # --xla_dump_hlo_pass_re: Regex for which compiler passes to dump (.* means all).
    flags = f"--xla_dump_to={dump_dir} --xla_dump_hlo_pass_re=.*"
    
    # Append to existing flags if they exist, otherwise set
    existing_flags = os.environ.get("XLA_FLAGS", "")
    if flags not in existing_flags:
        os.environ["XLA_FLAGS"] = f"{existing_flags} {flags}".strip()
        
    print(f"[XLA Profiler] HLO dumping enabled globally. Dest: {dump_dir}")

@contextlib.contextmanager
def xla_dump_context(dump_dir: str = DEFAULT_DUMP_DIR, clear_first: bool = True):
    """
    A context manager to temporarily enable HLO dumping for a specific block
    of code, isolating the compilation artifacts.
    
    Usage:
        with xla_dump_context():
            # The first time this is called, JIT compiles and dumps HLO
            out = pallas_attention(q, k, v)
    """
    if clear_first:
        clear_dump_dir(dump_dir)
    else:
        os.makedirs(dump_dir, exist_ok=True)

    original_flags = os.environ.get("XLA_FLAGS")
    
    # Set the dump flags
    flags = f"--xla_dump_to={dump_dir} --xla_dump_hlo_pass_re=.*"
    os.environ["XLA_FLAGS"] = f"{original_flags or ''} {flags}".strip()
    
    try:
        yield
    finally:
        # Restore the original environment state
        if original_flags is None:
            del os.environ["XLA_FLAGS"]
        else:
            os.environ["XLA_FLAGS"] = original_flags

if __name__ == "__main__":
    # Simple self-test / initialization script
    print("XLA Profiling Utility")
    clear_dump_dir()
    enable_hlo_dump()
    print(f"Current XLA_FLAGS: {os.environ.get('XLA_FLAGS')}")