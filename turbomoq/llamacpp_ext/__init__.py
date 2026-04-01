"""llama.cpp C extension — topology-aware KV cache allocation at the kernel level.

This module provides:
  1. A C implementation of TurboMOQ's head scoring + bit allocation
  2. Metal kernel stubs for Apple Silicon rotation + quantize
  3. A Python ctypes wrapper for integration with the TurboMOQ pipeline

The C code compiles to a shared library (.dylib on macOS) that can be
loaded by both Python (via ctypes) and llama.cpp (as a KV cache plugin).

Build:
    cd turbomoq/llamacpp_ext
    make          # Builds libturbomoq.dylib
    make test     # Runs C-level tests

Integration with llama.cpp:
    The compiled library exposes a C API that llama.cpp can call at the
    KV cache allocation point to get per-head bit allocations. This brings
    topology-aware compression to the Metal kernel level.
"""

from turbomoq.llamacpp_ext.wrapper import (
    LlamaCppExtension,
    EXTENSION_AVAILABLE,
    compile_extension,
)

__all__ = ["LlamaCppExtension", "EXTENSION_AVAILABLE", "compile_extension"]
