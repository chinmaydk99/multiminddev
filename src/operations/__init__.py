"""
Operation definitions for kernel generation.

Each operation defines:
- Input/output tensor specifications
- Shape parameters and valid ranges
- Reference implementation (PyTorch)
- Baseline to compare against (rocBLAS/MIOpen)
"""

from .base import OperationSpec, ShapeConfig, TensorSpec
from .attention import AttentionOp, FlashAttentionOp
from .gemm import GemmOp, BatchedGemmOp
from .normalization import LayerNormOp, RMSNormOp
from .elementwise import RoPEOp, SiLUOp, GeluOp

__all__ = [
    "OperationSpec",
    "ShapeConfig",
    "TensorSpec",
    "AttentionOp",
    "FlashAttentionOp",
    "GemmOp",
    "BatchedGemmOp",
    "LayerNormOp",
    "RMSNormOp",
    "RoPEOp",
    "SiLUOp",
    "GeluOp",
]

