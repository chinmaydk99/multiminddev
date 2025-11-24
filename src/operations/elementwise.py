"""
Elementwise operation specifications.

Covers:
- RoPE (Rotary Position Embedding)
- SiLU (Sigmoid Linear Unit) / Swish
- GELU
- Fused operations
"""

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from .base import DataType, OperationSpec, ShapeConfig, TensorSpec


@dataclass
class RoPEOp(OperationSpec):
    """
    Rotary Position Embedding (RoPE).
    
    Applies rotary embeddings to Q and K tensors.
    Used in LLaMA, Mistral, and most modern LLMs.
    """
    
    name: str = "rope"
    description: str = "Rotary Position Embedding for Q/K in attention"
    base: float = 10000.0
    
    @property
    def shape_params(self) -> Dict[str, ShapeConfig]:
        return {
            "batch": ShapeConfig(
                name="batch",
                min_value=1,
                max_value=64,
                typical_values=[1, 2, 4, 8, 16, 32],
            ),
            "seq_len": ShapeConfig(
                name="seq_len",
                min_value=1,
                max_value=32768,
                typical_values=[1, 512, 1024, 2048, 4096, 8192],
            ),
            "num_heads": ShapeConfig(
                name="num_heads",
                min_value=1,
                max_value=128,
                typical_values=[32, 64, 96, 128],
            ),
            "head_dim": ShapeConfig(
                name="head_dim",
                min_value=32,
                max_value=256,
                typical_values=[64, 128],
                must_be_power_of_2=True,
            ),
        }
    
    @property
    def input_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(name="X", shape=("batch", "seq_len", "num_heads", "head_dim")),
            TensorSpec(name="cos", shape=("seq_len", "head_dim")),  # Precomputed cos
            TensorSpec(name="sin", shape=("seq_len", "head_dim")),  # Precomputed sin
        ]
    
    @property
    def output_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(name="Y", shape=("batch", "seq_len", "num_heads", "head_dim"), is_output=True),
        ]
    
    def reference_impl(
        self, 
        X: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """PyTorch reference implementation."""
        # X: [batch, seq_len, num_heads, head_dim]
        # cos, sin: [seq_len, head_dim]
        
        # Split into even and odd indices
        x1 = X[..., ::2]   # Even indices
        x2 = X[..., 1::2]  # Odd indices
        
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(2)
        
        cos1 = cos[..., ::2]
        cos2 = cos[..., 1::2]
        sin1 = sin[..., ::2]
        sin2 = sin[..., 1::2]
        
        # Apply rotation
        y1 = x1 * cos1 - x2 * sin1
        y2 = x1 * sin2 + x2 * cos2
        
        # Interleave back
        Y = torch.stack([y1, y2], dim=-1).flatten(-2)
        
        return {"Y": Y}
    
    def baseline_impl(
        self, 
        X: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Baseline implementation (same as reference for RoPE)."""
        return self.reference_impl(X, cos, sin)
    
    def get_hip_template(self) -> str:
        """Get HIP kernel template for RoPE."""
        return '''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// RoPE kernel for MI300
// Elementwise operation - good candidate for memory-bound optimization
// Key optimizations:
// - Coalesced memory access
// - Vectorized loads (process 2 elements at once for rotation)
// - Fuse with attention Q/K projection if possible

__global__ void rope_kernel(
    const half* __restrict__ X,    // [batch, seq_len, num_heads, head_dim]
    const half* __restrict__ cos,  // [seq_len, head_dim]
    const half* __restrict__ sin,  // [seq_len, head_dim]
    half* __restrict__ Y,          // [batch, seq_len, num_heads, head_dim]
    int batch, int seq_len, int num_heads, int head_dim
) {
    // Each thread handles a pair of elements (for rotation)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_head_dim = head_dim / 2;
    int total_pairs = batch * seq_len * num_heads * half_head_dim;
    
    if (idx >= total_pairs) return;
    
    // Decode indices
    int pair_in_head = idx % half_head_dim;
    int remaining = idx / half_head_dim;
    int head = remaining % num_heads;
    remaining = remaining / num_heads;
    int seq = remaining % seq_len;
    int b = remaining / seq_len;
    
    // Load pair of values
    int base_idx = ((b * seq_len + seq) * num_heads + head) * head_dim;
    float x1 = __half2float(X[base_idx + pair_in_head * 2]);
    float x2 = __half2float(X[base_idx + pair_in_head * 2 + 1]);
    
    // Load cos/sin
    int trig_idx = seq * head_dim + pair_in_head * 2;
    float c = __half2float(cos[trig_idx]);
    float s = __half2float(sin[trig_idx]);
    
    // Apply rotation
    float y1 = x1 * c - x2 * s;
    float y2 = x1 * s + x2 * c;
    
    // Store
    Y[base_idx + pair_in_head * 2] = __float2half(y1);
    Y[base_idx + pair_in_head * 2 + 1] = __float2half(y2);
}
'''


@dataclass
class SiLUOp(OperationSpec):
    """
    SiLU (Sigmoid Linear Unit) / Swish: y = x * sigmoid(x)
    
    Used in LLaMA, Mistral FFN layers (with gating).
    """
    
    name: str = "silu"
    description: str = "SiLU/Swish activation: y = x * sigmoid(x)"
    
    @property
    def shape_params(self) -> Dict[str, ShapeConfig]:
        return {
            "size": ShapeConfig(
                name="size",
                min_value=1024,
                max_value=100000000,  # Up to 100M elements
                typical_values=[1048576, 4194304, 16777216, 67108864],  # 1M, 4M, 16M, 64M
            ),
        }
    
    @property
    def input_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(name="X", shape=("size",)),
        ]
    
    @property
    def output_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(name="Y", shape=("size",), is_output=True),
        ]
    
    def reference_impl(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """PyTorch reference implementation."""
        Y = F.silu(X)
        return {"Y": Y}
    
    def baseline_impl(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Baseline implementation."""
        Y = F.silu(X)
        return {"Y": Y}


@dataclass
class GeluOp(OperationSpec):
    """
    GELU (Gaussian Error Linear Unit).
    
    Used in BERT, GPT-2, and other transformers.
    """
    
    name: str = "gelu"
    description: str = "GELU activation function"
    approximate: bool = True  # Use tanh approximation
    
    @property
    def shape_params(self) -> Dict[str, ShapeConfig]:
        return {
            "size": ShapeConfig(
                name="size",
                min_value=1024,
                max_value=100000000,
                typical_values=[1048576, 4194304, 16777216, 67108864],
            ),
        }
    
    @property
    def input_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(name="X", shape=("size",)),
        ]
    
    @property
    def output_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(name="Y", shape=("size",), is_output=True),
        ]
    
    def reference_impl(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """PyTorch reference implementation."""
        approximate = "tanh" if self.approximate else "none"
        Y = F.gelu(X, approximate=approximate)
        return {"Y": Y}
    
    def baseline_impl(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Baseline implementation."""
        return self.reference_impl(X)

