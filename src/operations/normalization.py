"""
Normalization operation specifications.

Covers:
- LayerNorm
- RMSNorm (used in LLaMA, Mistral)
- GroupNorm
"""

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from .base import DataType, OperationSpec, ShapeConfig, TensorSpec


@dataclass
class LayerNormOp(OperationSpec):
    """
    Layer Normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta
    
    Common in transformers, normalizes across the last dimension.
    """
    
    name: str = "layernorm"
    description: str = "Layer normalization across hidden dimension"
    eps: float = 1e-5
    
    @property
    def shape_params(self) -> Dict[str, ShapeConfig]:
        return {
            "batch_seq": ShapeConfig(
                name="batch_seq",
                min_value=1,
                max_value=131072,  # batch * seq_len
                typical_values=[1024, 4096, 8192, 16384, 32768, 65536],
            ),
            "hidden": ShapeConfig(
                name="hidden",
                min_value=256,
                max_value=16384,
                typical_values=[1024, 2048, 4096, 5120, 8192, 14336],
                must_be_multiple_of=256,
            ),
        }
    
    @property
    def input_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(name="X", shape=("batch_seq", "hidden")),
            TensorSpec(name="gamma", shape=("hidden",)),  # Scale
            TensorSpec(name="beta", shape=("hidden",)),   # Bias
        ]
    
    @property
    def output_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(name="Y", shape=("batch_seq", "hidden"), is_output=True),
        ]
    
    def reference_impl(
        self, 
        X: torch.Tensor, 
        gamma: torch.Tensor, 
        beta: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """PyTorch reference implementation."""
        Y = F.layer_norm(X, (X.shape[-1],), gamma, beta, self.eps)
        return {"Y": Y}
    
    def baseline_impl(
        self, 
        X: torch.Tensor, 
        gamma: torch.Tensor, 
        beta: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Baseline using PyTorch (which uses MIOpen on AMD)."""
        Y = F.layer_norm(X, (X.shape[-1],), gamma, beta, self.eps)
        return {"Y": Y}
    
    def get_hip_template(self) -> str:
        """Get HIP kernel template for LayerNorm."""
        return f'''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define EPS {self.eps}f

// LayerNorm kernel for MI300
// Key optimizations:
// - Warp-level reduction for mean/variance
// - Vectorized loads/stores
// - Fused scale and bias application

__global__ void layernorm_kernel(
    const half* __restrict__ X,      // [batch_seq, hidden]
    const half* __restrict__ gamma,  // [hidden]
    const half* __restrict__ beta,   // [hidden]
    half* __restrict__ Y,            // [batch_seq, hidden]
    int batch_seq, int hidden
) {{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Each block handles one row (one token)
    const half* x_row = X + row * hidden;
    half* y_row = Y + row * hidden;
    
    // Compute mean using warp reduction
    float sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {{
        sum += __half2float(x_row[i]);
    }}
    // Warp reduce sum
    // TODO: Implement warp reduction
    
    float mean = sum / hidden;
    
    // Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {{
        float diff = __half2float(x_row[i]) - mean;
        var_sum += diff * diff;
    }}
    // Warp reduce var_sum
    // TODO: Implement warp reduction
    
    float inv_std = rsqrtf(var_sum / hidden + EPS);
    
    // Normalize and apply scale/bias
    for (int i = tid; i < hidden; i += blockDim.x) {{
        float normalized = (__half2float(x_row[i]) - mean) * inv_std;
        float scaled = normalized * __half2float(gamma[i]) + __half2float(beta[i]);
        y_row[i] = __float2half(scaled);
    }}
}}
'''


@dataclass
class RMSNormOp(OperationSpec):
    """
    RMS Normalization: y = x / sqrt(mean(x^2) + eps) * gamma
    
    Used in LLaMA, Mistral, and other modern LLMs.
    Simpler than LayerNorm (no mean subtraction, no bias).
    """
    
    name: str = "rmsnorm"
    description: str = "Root Mean Square normalization (used in LLaMA/Mistral)"
    eps: float = 1e-6
    
    @property
    def shape_params(self) -> Dict[str, ShapeConfig]:
        return {
            "batch_seq": ShapeConfig(
                name="batch_seq",
                min_value=1,
                max_value=131072,
                typical_values=[1024, 4096, 8192, 16384, 32768, 65536],
            ),
            "hidden": ShapeConfig(
                name="hidden",
                min_value=256,
                max_value=16384,
                typical_values=[4096, 5120, 8192, 14336],  # LLaMA hidden dims
                must_be_multiple_of=256,
            ),
        }
    
    @property
    def input_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(name="X", shape=("batch_seq", "hidden")),
            TensorSpec(name="gamma", shape=("hidden",)),  # Scale only, no bias
        ]
    
    @property
    def output_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(name="Y", shape=("batch_seq", "hidden"), is_output=True),
        ]
    
    def reference_impl(
        self, 
        X: torch.Tensor, 
        gamma: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """PyTorch reference implementation."""
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(X ** 2, dim=-1, keepdim=True) + self.eps)
        Y = (X / rms) * gamma
        return {"Y": Y}
    
    def baseline_impl(
        self, 
        X: torch.Tensor, 
        gamma: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Baseline implementation."""
        rms = torch.sqrt(torch.mean(X ** 2, dim=-1, keepdim=True) + self.eps)
        Y = (X / rms) * gamma
        return {"Y": Y}
    
    def get_hip_template(self) -> str:
        """Get HIP kernel template for RMSNorm."""
        return f'''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define EPS {self.eps}f

// RMSNorm kernel for MI300
// Simpler than LayerNorm - no mean computation
// Key optimizations:
// - Single pass: compute sum of squares and normalize together
// - Warp-level reduction
// - Vectorized memory access (float4/half8)

__global__ void rmsnorm_kernel(
    const half* __restrict__ X,      // [batch_seq, hidden]
    const half* __restrict__ gamma,  // [hidden]
    half* __restrict__ Y,            // [batch_seq, hidden]
    int batch_seq, int hidden
) {{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    const half* x_row = X + row * hidden;
    half* y_row = Y + row * hidden;
    
    // Compute sum of squares
    float ss = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {{
        float val = __half2float(x_row[i]);
        ss += val * val;
    }}
    
    // Warp reduction for sum of squares
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {{
        ss += __shfl_down(ss, offset);
    }}
    
    // Broadcast to all threads in block
    __shared__ float s_ss;
    if (tid == 0) {{
        s_ss = ss;
    }}
    __syncthreads();
    ss = s_ss;
    
    // Compute RMS inverse
    float inv_rms = rsqrtf(ss / hidden + EPS);
    
    // Apply normalization and scale
    for (int i = tid; i < hidden; i += blockDim.x) {{
        float normalized = __half2float(x_row[i]) * inv_rms;
        float scaled = normalized * __half2float(gamma[i]);
        y_row[i] = __float2half(scaled);
    }}
}}
'''

