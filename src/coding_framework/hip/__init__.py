"""
HIP/ROCm execution environment for kernel compilation and performance benchmarking.

This module provides tools for compiling HIP kernels, measuring performance,
and managing ROCm execution workflows on AMD GPUs.
"""

from .benchmarker import BenchmarkResult, HIPBenchmarker
from .compiler import CompilationResult, HIPCompiler

__all__ = [
    "HIPCompiler",
    "CompilationResult",
    "HIPBenchmarker",
    "BenchmarkResult",
]

