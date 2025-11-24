"""
Kernel execution and profiling infrastructure.

This module handles:
- Compiling HIP kernels
- Running them in a Docker sandbox
- Profiling with rocprof
- Comparing against baselines
"""

from .sandbox import KernelSandbox, ExecutionResult
from .profiler import ROCProfiler, ProfileResult
from .compiler import HIPCompiler, CompilationResult

__all__ = [
    "KernelSandbox",
    "ExecutionResult",
    "ROCProfiler",
    "ProfileResult",
    "HIPCompiler",
    "CompilationResult",
]

