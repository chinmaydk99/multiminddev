"""
GEMM (General Matrix Multiplication) operation specifications.

Covers:
- Standard GEMM: C = α * A @ B + β * C
- Batched GEMM
- Strided batched GEMM
"""

from dataclasses import dataclass
from typing import Dict, List

import torch

from .base import DataType, OperationSpec, ShapeConfig, TensorSpec


@dataclass
class GemmOp(OperationSpec):
    """
    General Matrix Multiplication: C = A @ B
    
    This is the core operation in neural networks (linear layers, projections).
    """
    
    name: str = "gemm"
    description: str = "Matrix multiplication C[M,N] = A[M,K] @ B[K,N]"
    
    # GEMM variants
    transpose_a: bool = False
    transpose_b: bool = False
    alpha: float = 1.0
    beta: float = 0.0
    
    @property
    def shape_params(self) -> Dict[str, ShapeConfig]:
        return {
            "M": ShapeConfig(
                name="M",
                min_value=1,
                max_value=16384,
                typical_values=[1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
            ),
            "K": ShapeConfig(
                name="K",
                min_value=256,
                max_value=32768,
                typical_values=[1024, 2048, 4096, 8192, 14336, 16384],  # Common LLM hidden dims
                must_be_multiple_of=256,
            ),
            "N": ShapeConfig(
                name="N",
                min_value=256,
                max_value=32768,
                typical_values=[1024, 2048, 4096, 8192, 14336, 16384, 28672],
                must_be_multiple_of=256,
            ),
        }
    
    @property
    def input_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(name="A", shape=("M", "K")),
            TensorSpec(name="B", shape=("K", "N")),
        ]
    
    @property
    def output_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(name="C", shape=("M", "N"), is_output=True),
        ]
    
    def reference_impl(self, A: torch.Tensor, B: torch.Tensor) -> Dict[str, torch.Tensor]:
        """PyTorch reference implementation."""
        C = torch.matmul(A, B)
        return {"C": C}
    
    def baseline_impl(self, A: torch.Tensor, B: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Baseline using rocBLAS via PyTorch."""
        # PyTorch uses rocBLAS for matmul on AMD GPUs
        C = torch.matmul(A, B)
        return {"C": C}
    
    def get_hip_template(self) -> str:
        """Get HIP kernel template for GEMM."""
        return '''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// GEMM kernel for MI300
// Optimizations to consider:
// - Tiling with LDS (128x128 or 256x256 tiles)
// - Vectorized loads (float4 for fp32, half8 for fp16)
// - Register blocking
// - Double buffering
// - Matrix core operations (MFMA instructions)

#define TILE_M 128
#define TILE_N 128
#define TILE_K 32

__global__ void gemm_kernel(
    const half* __restrict__ A,  // [M, K]
    const half* __restrict__ B,  // [K, N]
    half* __restrict__ C,        // [M, N]
    int M, int K, int N
) {
    // Shared memory for A and B tiles
    __shared__ half As[TILE_M][TILE_K];
    __shared__ half Bs[TILE_K][TILE_N];
    
    // Thread block position
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global position
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;
    
    // Accumulator
    float acc = 0.0f;
    
    // Loop over K dimension in tiles
    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {
        // Collaborative load of A and B tiles to shared memory
        // TODO: Implement with vectorized loads
        
        __syncthreads();
        
        // Compute partial dot product
        // TODO: Implement with register blocking
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = __float2half(acc);
    }
}
'''


@dataclass
class BatchedGemmOp(OperationSpec):
    """
    Batched GEMM: C[b] = A[b] @ B[b] for batch dimension b.
    
    Common in attention (QK^T, attn @ V) and batched linear layers.
    """
    
    name: str = "batched_gemm"
    description: str = "Batched matrix multiplication C[B,M,N] = A[B,M,K] @ B[B,K,N]"
    
    @property
    def shape_params(self) -> Dict[str, ShapeConfig]:
        return {
            "batch": ShapeConfig(
                name="batch",
                min_value=1,
                max_value=512,
                typical_values=[1, 8, 16, 32, 64, 128, 256],
            ),
            "M": ShapeConfig(
                name="M",
                min_value=64,
                max_value=8192,
                typical_values=[512, 1024, 2048, 4096],
            ),
            "K": ShapeConfig(
                name="K",
                min_value=64,
                max_value=8192,
                typical_values=[64, 128, 256, 512, 1024],  # Often head_dim in attention
            ),
            "N": ShapeConfig(
                name="N",
                min_value=64,
                max_value=8192,
                typical_values=[512, 1024, 2048, 4096],
            ),
        }
    
    @property
    def input_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(name="A", shape=("batch", "M", "K")),
            TensorSpec(name="B", shape=("batch", "K", "N")),
        ]
    
    @property
    def output_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(name="C", shape=("batch", "M", "N"), is_output=True),
        ]
    
    def reference_impl(self, A: torch.Tensor, B: torch.Tensor) -> Dict[str, torch.Tensor]:
        """PyTorch reference implementation."""
        C = torch.bmm(A, B)
        return {"C": C}
    
    def baseline_impl(self, A: torch.Tensor, B: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Baseline using rocBLAS batched GEMM."""
        C = torch.bmm(A, B)
        return {"C": C}

