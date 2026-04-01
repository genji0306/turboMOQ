"""Python ctypes wrapper for the TurboMOQ C extension.

Loads libturbomoq.dylib/.so and provides a Pythonic API for:
  - Head scoring (synthetic + real attention)
  - Bit allocation
  - Symmetric quantize/dequantize
  - Rotation (generate + apply)

Build the extension first:
    cd turbomoq/llamacpp_ext && make

Or call compile_extension() which invokes clang/gcc automatically.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import subprocess
import sys
from ctypes import (
    POINTER,
    Structure,
    c_char_p,
    c_float,
    c_int,
    c_int8,
    c_uint32,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["LlamaCppExtension", "EXTENSION_AVAILABLE", "compile_extension"]

logger = logging.getLogger("turbomoq.llamacpp_ext")

try:
    import numpy as np
    _NP = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NP = False

# ── C Struct Mirrors ─────────────────────────────────────────────────────────

TURBOMOQ_MAX_LAYERS = 128
TURBOMOQ_MAX_HEADS = 128


class _Config(Structure):
    _fields_ = [
        ("n_layers", c_int),
        ("n_heads", c_int),
        ("head_dim", c_int),
        ("default_k_bits", c_int),
        ("default_v_bits", c_int),
        ("target_avg_bits", c_float),
        ("topo_weight", c_float),
        ("grad_weight", c_float),
        ("position_weight", c_float),
    ]


class _HeadScore(Structure):
    _fields_ = [
        ("layer", c_int),
        ("head", c_int),
        ("topo_score", c_float),
        ("grad_score", c_float),
        ("position_score", c_float),
        ("combined", c_float),
        ("allocated_k_bits", c_int),
        ("allocated_v_bits", c_int),
    ]


class _Ctx(Structure):
    _fields_ = [
        ("config", _Config),
        ("scores", _HeadScore * (TURBOMOQ_MAX_LAYERS * TURBOMOQ_MAX_HEADS)),
        ("n_scores", c_int),
        ("calibrated", c_int),
    ]


class _QTensor(Structure):
    _fields_ = [
        ("data", POINTER(c_int8)),
        ("scales", POINTER(c_float)),
        ("rows", c_int),
        ("cols", c_int),
        ("bits", c_int),
    ]


# ── Library Loading ──────────────────────────────────────────────────────────

_LIB_DIR = Path(__file__).parent
_LIB_NAME = "libturbomoq.dylib" if platform.system() == "Darwin" else "libturbomoq.so"
_LIB_PATH = _LIB_DIR / _LIB_NAME

_lib: ctypes.CDLL | None = None
EXTENSION_AVAILABLE = False


def _load_lib() -> ctypes.CDLL | None:
    global _lib, EXTENSION_AVAILABLE
    if _lib is not None:
        return _lib
    if not _LIB_PATH.exists():
        return None
    try:
        _lib = ctypes.CDLL(str(_LIB_PATH))
        _setup_signatures(_lib)
        EXTENSION_AVAILABLE = True
        logger.info("Loaded C extension from %s", _LIB_PATH)
        return _lib
    except OSError as e:
        logger.warning("Failed to load C extension: %s", e)
        return None


def _setup_signatures(lib: ctypes.CDLL) -> None:
    lib.turbomoq_init.argtypes = [POINTER(_Ctx), POINTER(_Config)]
    lib.turbomoq_init.restype = c_int

    lib.turbomoq_position_importance.argtypes = [c_int, c_int]
    lib.turbomoq_position_importance.restype = c_float

    lib.turbomoq_compute_persistence.argtypes = [POINTER(c_float), c_int]
    lib.turbomoq_compute_persistence.restype = c_float

    lib.turbomoq_compute_gradient.argtypes = [POINTER(c_float), c_int]
    lib.turbomoq_compute_gradient.restype = c_float

    lib.turbomoq_score_synthetic.argtypes = [POINTER(_Ctx), c_uint32]
    lib.turbomoq_score_synthetic.restype = c_int

    lib.turbomoq_allocate_bits.argtypes = [POINTER(_Ctx)]
    lib.turbomoq_allocate_bits.restype = c_int

    lib.turbomoq_get_bits.argtypes = [POINTER(_Ctx), c_int, c_int, POINTER(c_int), POINTER(c_int)]
    lib.turbomoq_get_bits.restype = c_int

    lib.turbomoq_symmetric_quantize.argtypes = [POINTER(c_float), POINTER(_QTensor), c_int, c_int, c_int]
    lib.turbomoq_symmetric_quantize.restype = c_int

    lib.turbomoq_symmetric_dequantize.argtypes = [POINTER(_QTensor), POINTER(c_float)]
    lib.turbomoq_symmetric_dequantize.restype = c_int

    lib.turbomoq_rotate.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int]
    lib.turbomoq_rotate.restype = c_int

    lib.turbomoq_unrotate.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int]
    lib.turbomoq_unrotate.restype = c_int

    lib.turbomoq_generate_rotation.argtypes = [POINTER(c_float), c_int, c_uint32]
    lib.turbomoq_generate_rotation.restype = c_int

    lib.turbomoq_version.argtypes = []
    lib.turbomoq_version.restype = c_char_p

    lib.turbomoq_free.argtypes = [POINTER(_Ctx)]
    lib.turbomoq_free.restype = None


# ── Compilation ──────────────────────────────────────────────────────────────

def compile_extension(force: bool = False) -> bool:
    """Compile the C extension. Returns True on success."""
    if not force and _LIB_PATH.exists():
        return True

    src = _LIB_DIR / "turbomoq_kv.c"
    if not src.exists():
        logger.error("C source not found: %s", src)
        return False

    if platform.system() == "Darwin":
        cc = "clang"
        flags = ["-O2", "-shared", "-fPIC", "-o", str(_LIB_PATH), str(src), "-lm"]
    else:
        cc = "gcc"
        flags = ["-O2", "-shared", "-fPIC", "-o", str(_LIB_PATH), str(src), "-lm"]

    try:
        result = subprocess.run([cc] + flags, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.error("Compilation failed: %s", result.stderr)
            return False
        logger.info("Compiled C extension: %s", _LIB_PATH)
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error("Compilation error: %s", e)
        return False


# ── Python Wrapper ───────────────────────────────────────────────────────────


@dataclass
class CHeadScore:
    """Python-side head score from the C extension."""
    layer: int
    head: int
    topo_score: float
    grad_score: float
    position_score: float
    combined: float
    allocated_k_bits: int
    allocated_v_bits: int


class LlamaCppExtension:
    """High-level wrapper for the TurboMOQ C extension.

    Usage::

        ext = LlamaCppExtension(n_layers=32, n_heads=8, head_dim=128)

        # Score and allocate
        ext.score_synthetic(seed=42)
        ext.allocate_bits()

        # Get per-head allocation
        k_bits, v_bits = ext.get_bits(layer=5, head=3)

        # Quantize a tensor
        quantized, scales = ext.symmetric_quantize(data, bits=4)
        reconstructed = ext.symmetric_dequantize(quantized, scales, bits=4)
    """

    def __init__(self, n_layers: int = 32, n_heads: int = 8, head_dim: int = 128,
                 default_bits: int = 4, target_avg: float = 4.0,
                 weights: tuple[float, float, float] = (0.4, 0.3, 0.3)):
        lib = _load_lib()
        if lib is None:
            # Try auto-compile
            if compile_extension():
                lib = _load_lib()
            if lib is None:
                raise RuntimeError(
                    "C extension not available. Run: cd turbomoq/llamacpp_ext && make"
                )
        self._lib = lib

        config = _Config()
        config.n_layers = n_layers
        config.n_heads = n_heads
        config.head_dim = head_dim
        config.default_k_bits = default_bits
        config.default_v_bits = default_bits
        config.target_avg_bits = target_avg
        config.topo_weight = weights[0]
        config.grad_weight = weights[1]
        config.position_weight = weights[2]

        self._ctx = _Ctx()
        rc = self._lib.turbomoq_init(ctypes.byref(self._ctx), ctypes.byref(config))
        if rc != 0:
            raise RuntimeError("turbomoq_init failed")

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim

    def version(self) -> str:
        return self._lib.turbomoq_version().decode()

    def position_importance(self, layer: int) -> float:
        return float(self._lib.turbomoq_position_importance(layer, self.n_layers))

    def score_synthetic(self, seed: int = 42) -> list[CHeadScore]:
        """Score all heads using synthetic data."""
        n = self._lib.turbomoq_score_synthetic(ctypes.byref(self._ctx), c_uint32(seed))
        if n < 0:
            raise RuntimeError("score_synthetic failed")
        return self._extract_scores()

    def allocate_bits(self) -> dict[tuple[int, int], tuple[int, int]]:
        """Allocate bits per head. Must call score_* first."""
        rc = self._lib.turbomoq_allocate_bits(ctypes.byref(self._ctx))
        if rc != 0:
            raise RuntimeError("allocate_bits failed (not calibrated?)")
        scores = self._extract_scores()
        return {(s.layer, s.head): (s.allocated_k_bits, s.allocated_v_bits) for s in scores}

    def get_bits(self, layer: int, head: int) -> tuple[int, int]:
        """Get bit allocation for specific (layer, head)."""
        k = c_int(0)
        v = c_int(0)
        self._lib.turbomoq_get_bits(ctypes.byref(self._ctx), layer, head,
                                     ctypes.byref(k), ctypes.byref(v))
        return k.value, v.value

    def symmetric_quantize(self, data: Any, bits: int = 4) -> tuple[Any, Any]:
        """Quantize float32 array via C extension. Returns (quantized, scales)."""
        if not _NP:
            raise ImportError("numpy required")
        arr = np.ascontiguousarray(data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        rows, cols = arr.shape

        q_data = (c_int8 * (rows * cols))()
        q_scales = (c_float * cols)()

        qt = _QTensor()
        qt.data = q_data
        qt.scales = q_scales

        src = arr.ctypes.data_as(POINTER(c_float))
        rc = self._lib.turbomoq_symmetric_quantize(src, ctypes.byref(qt), rows, cols, bits)
        if rc != 0:
            raise RuntimeError("symmetric_quantize failed")

        q_np = np.frombuffer(q_data, dtype=np.int8).reshape(rows, cols).copy()
        s_np = np.frombuffer(q_scales, dtype=np.float32).copy()
        return q_np, s_np

    def symmetric_dequantize(self, quantized: Any, scales: Any, bits: int = 4) -> Any:
        """Dequantize back to float32."""
        if not _NP:
            raise ImportError("numpy required")
        q_np = np.ascontiguousarray(quantized, dtype=np.int8)
        s_np = np.ascontiguousarray(scales, dtype=np.float32)
        if q_np.ndim == 1:
            q_np = q_np.reshape(1, -1)
        rows, cols = q_np.shape

        qt = _QTensor()
        qt.data = q_np.ctypes.data_as(POINTER(c_int8))
        qt.scales = s_np.ctypes.data_as(POINTER(c_float))
        qt.rows = rows
        qt.cols = cols
        qt.bits = bits

        dst = np.zeros((rows, cols), dtype=np.float32)
        rc = self._lib.turbomoq_symmetric_dequantize(ctypes.byref(qt),
                                                      dst.ctypes.data_as(POINTER(c_float)))
        if rc != 0:
            raise RuntimeError("symmetric_dequantize failed")
        return dst

    def generate_rotation(self, dim: int, seed: int = 42) -> Any:
        """Generate random orthogonal matrix via C extension."""
        if not _NP:
            raise ImportError("numpy required")
        Q = np.zeros((dim, dim), dtype=np.float32, order='C')
        rc = self._lib.turbomoq_generate_rotation(
            Q.ctypes.data_as(POINTER(c_float)), dim, c_uint32(seed))
        if rc != 0:
            raise RuntimeError("generate_rotation failed")
        return Q

    def rotate(self, data: Any, Q: Any) -> Any:
        """Apply rotation: data @ Q."""
        if not _NP:
            raise ImportError("numpy required")
        src = np.ascontiguousarray(data, dtype=np.float32)
        Q_c = np.ascontiguousarray(Q, dtype=np.float32)
        if src.ndim == 1:
            src = src.reshape(1, -1)
        rows, dim = src.shape
        dst = np.zeros_like(src)

        rc = self._lib.turbomoq_rotate(
            src.ctypes.data_as(POINTER(c_float)),
            dst.ctypes.data_as(POINTER(c_float)),
            Q_c.ctypes.data_as(POINTER(c_float)),
            rows, dim)
        if rc != 0:
            raise RuntimeError("rotate failed")
        return dst

    def __del__(self):
        if hasattr(self, '_lib') and hasattr(self, '_ctx'):
            self._lib.turbomoq_free(ctypes.byref(self._ctx))

    def _extract_scores(self) -> list[CHeadScore]:
        scores = []
        for i in range(self._ctx.n_scores):
            s = self._ctx.scores[i]
            scores.append(CHeadScore(
                layer=s.layer, head=s.head,
                topo_score=s.topo_score, grad_score=s.grad_score,
                position_score=s.position_score, combined=s.combined,
                allocated_k_bits=s.allocated_k_bits,
                allocated_v_bits=s.allocated_v_bits,
            ))
        return scores


# Try loading at import time
_load_lib()
