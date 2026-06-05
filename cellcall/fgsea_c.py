"""Python wrapper for the C fgsea exact algorithm."""

from __future__ import annotations
import ctypes
import os
import subprocess
import sys
import numpy as np


def _build_lib():
    """Build the C extension if not already built."""
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    lib_path = os.path.join(src_dir, "fgsea_exact.so")
    src_path = os.path.join(src_dir, "fgsea_exact.c")

    if os.path.exists(lib_path):
        if os.path.getmtime(lib_path) >= os.path.getmtime(src_path):
            return lib_path

    # Build
    cc = "gcc"
    cmd = [cc, "-O3", "-shared", "-fPIC", "-o", lib_path, src_path, "-lm"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Try without -fPIC on Windows
        cmd = [cc, "-O3", "-shared", "-o", lib_path.replace(".so", ".dll"), src_path, "-lm"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to build C extension: {result.stderr}")
        return lib_path.replace(".so", ".dll")

    return lib_path


def _load_lib():
    """Load the compiled C library."""
    lib_path = _build_lib()
    return ctypes.CDLL(lib_path)


# Lazy load
_lib = None


def get_lib():
    global _lib
    if _lib is None:
        _lib = _load_lib()
        # Set function signatures
        _lib.fgsea_es.restype = ctypes.c_double
        _lib.fgsea_es.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # values
            ctypes.POINTER(ctypes.c_int),      # hit_mask
            ctypes.c_int,                       # n
            ctypes.c_double,                    # exponent
        ]
        _lib.fgsea_exact_pvalue.restype = ctypes.c_double
        _lib.fgsea_exact_pvalue.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # values
            ctypes.POINTER(ctypes.c_int),      # hit_mask
            ctypes.c_int,                       # n
            ctypes.c_double,                    # es
            ctypes.c_double,                    # exponent
            ctypes.c_int,                       # n_bins
        ]
    return _lib


def compute_es(values, hit_mask, exponent=1.0):
    """Compute GSEA enrichment score using C implementation."""
    lib = get_lib()
    n = len(values)
    values_c = (ctypes.c_double * n)(*values.astype(np.float64))
    mask_c = (ctypes.c_int * n)(*hit_mask.astype(np.int32))
    return lib.fgsea_es(values_c, mask_c, n, exponent)


def compute_exact_pvalue(values, hit_mask, es, exponent=1.0, n_bins=201):
    """Compute exact GSEA p-value using C DP algorithm."""
    lib = get_lib()
    n = len(values)
    values_c = (ctypes.c_double * n)(*values.astype(np.float64))
    mask_c = (ctypes.c_int * n)(*hit_mask.astype(np.int32))
    return lib.fgsea_exact_pvalue(values_c, mask_c, n, es, exponent, n_bins)
