"""
XLA Environment Configuration Utility (xla_flags.py)

This module manages the environment variables required to force XLA to 
dump its compilation artifacts (HLO assembly) to disk.

Usage:
    from profiling.xla_flags import set_dump_flags
    set_dump_flags("/tmp/my_dump_dir")
"""

import os
import pathlib

def set_dump_flags(dump_dir: str = "/tmp/xla_dump"):
    """
    Sets the XLA_FLAGS environment variable to trigger HLO dumping.
    
    Args:
        dump_dir: The directory where XLA will save .txt assembly files.
    """
    # Ensure the directory exists
    path = pathlib.Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    # --xla_dump_to: Specifies the output directory
    # --xla_dump_hlo_pass_re: A regex that filters which compiler passes to dump.
    #    We use ".*" to capture everything important for analysis.
    flags = [
        f"--xla_dump_to={dump_dir}",
        "--xla_dump_hlo_pass_re=.*",
        "--xla_dump_hlo_as_text",
    ]
    
    # Append to existing flags if they exist to avoid overwriting user settings
    existing_flags = os.environ.get("XLA_FLAGS", "")
    new_flags = " ".join(flags)
    
    if new_flags not in existing_flags:
        os.environ["XLA_FLAGS"] = f"{existing_flags} {new_flags}".strip()
    
    return dump_dir

def get_current_flags() -> str:
    """Returns the current XLA_FLAGS string."""
    return os.environ.get("XLA_FLAGS", "")

if __name__ == "__main__":
    # Self-test: print the flags to be used
    target = set_dump_flags()
    print(f"\n[XLA Flags] Configuration active.")
    print(f" -> Target Directory : {target}")
    print(f" -> Current Flags    : {get_current_flags()}\n")