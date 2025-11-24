"""
Attention operation specifications.

Covers:
- Standard multi-head attention
- Flash Attention variant
- Grouped Query Attention (GQA)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .base import DataType, OperationSpec, ShapeConfig, TensorSpec


@dataclass
class AttentionOp(OperationSpec):
    """Standard scaled dot-product attention."""
    
    name: str = "attention"
    description: str = "Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) @ V"
    causal: bool = True
    
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
                min_value=128,
                max_value=32768,
                typical_values=[512, 1024, 2048, 4096, 8192, 16384],
                must_be_multiple_of=128,
            ),
            "num_heads": ShapeConfig(
                name="num_heads",
                min_value=1,
                max_value=128,
                typical_values=[8, 16, 32, 64, 96, 128],
            ),
            "head_dim": ShapeConfig(
                name="head_dim",
                min_value=32,
                max_value=256,
                typical_values=[64, 128, 256],
                must_be_power_of_2=True,
            ),
        }
    
    @property
    def input_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(name="Q", shape=("batch", "seq_len", "num_heads", "head_dim")),
            TensorSpec(name="K", shape=("batch", "seq_len", "num_heads", "head_dim")),
            TensorSpec(name="V", shape=("batch", "seq_len", "num_heads", "head_dim")),
        ]
    
    @property
    def output_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(name="O", shape=("batch", "seq_len", "num_heads", "head_dim"), is_output=True),
        ]
    
    def reference_impl(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Dict[str, torch.Tensor]:
        """PyTorch reference implementation."""
        # Q, K, V: [batch, seq_len, num_heads, head_dim]
        # Transpose to [batch, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        head_dim = Q.shape[-1]
        scale = 1.0 / (head_dim ** 0.5)
        
        # Attention scores: [batch, num_heads, seq_len, seq_len]
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # Causal mask
        if self.causal:
            seq_len = Q.shape[2]
            mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        # Output: [batch, num_heads, seq_len, head_dim]
        O = torch.matmul(attn, V)
        
        # Transpose back to [batch, seq_len, num_heads, head_dim]
        O = O.transpose(1, 2)
        
        return {"O": O}
    
    def baseline_impl(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Baseline using PyTorch's scaled_dot_product_attention (uses Flash Attention when available)."""
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        O = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=self.causal,
            scale=1.0 / (Q.shape[-1] ** 0.5)
        )
        
        O = O.transpose(1, 2)
        return {"O": O}
    
    def get_hip_template(self) -> str:
        """Get HIP kernel template for attention."""
        return '''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Attention kernel for MI300
// Optimizations to consider:
// - Use LDS (Local Data Share) for Q, K tile caching
// - Vectorized loads (float4, half8)
// - Warp-level matrix operations
// - Online softmax for numerical stability

__global__ void attention_kernel(
    const half* __restrict__ Q,    // [batch, seq_len, num_heads, head_dim]
    const half* __restrict__ K,    // [batch, seq_len, num_heads, head_dim]
    const half* __restrict__ V,    // [batch, seq_len, num_heads, head_dim]
    half* __restrict__ O,          // [batch, seq_len, num_heads, head_dim]
    int batch, int seq_len, int num_heads, int head_dim,
    float scale
) {
    // TODO: Implement optimized attention
    // Key considerations for MI300:
    // - 256 KB LDS per CU
    // - 64 KB L1 cache
    // - HBM3 bandwidth: 5.3 TB/s
}
'''


@dataclass
class FlashAttentionOp(AttentionOp):
    """
    Flash Attention variant with tiling and recomputation.
    
    Uses online softmax and tiled computation to reduce memory bandwidth.
    """
    
    name: str = "flash_attention"
    description: str = "Memory-efficient attention using tiling and online softmax"
    
    # Flash attention specific params
    block_size_q: int = 128
    block_size_kv: int = 128
    
    def get_hip_template(self) -> str:
        """Get HIP kernel template for flash attention."""
        return f'''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Flash Attention for MI300
// Block sizes: Q={self.block_size_q}, KV={self.block_size_kv}

#define BLOCK_SIZE_Q {self.block_size_q}
#define BLOCK_SIZE_KV {self.block_size_kv}

__global__ void flash_attention_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float* __restrict__ L,  // Log-sum-exp for numerical stability
    float* __restrict__ M,  // Max values for online softmax
    int batch, int seq_len, int num_heads, int head_dim,
    float scale
) {{
    // Shared memory for tiles
    extern __shared__ half smem[];
    
    // TODO: Implement Flash Attention
    // Algorithm:
    // 1. Load Q tile to shared memory
    // 2. For each KV block:
    //    a. Load K, V tiles
    //    b. Compute QK^T tile
    //    c. Update online softmax (m, l)
    //    d. Accumulate output
    // 3. Write final output
}}
'''


@dataclass  
class GroupedQueryAttentionOp(AttentionOp):
    """
    Grouped Query Attention (GQA) - used in LLaMA 2/3, Mistral.
    
    num_kv_heads < num_heads, with heads grouped to share KV.
    """
    
    name: str = "gqa"
    description: str = "Grouped Query Attention with shared KV heads"
    
    @property
    def shape_params(self) -> Dict[str, ShapeConfig]:
        params = super().shape_params
        params["num_kv_heads"] = ShapeConfig(
            name="num_kv_heads",
            min_value=1,
            max_value=32,
            typical_values=[1, 2, 4, 8],  # Often 8 for GQA
        )
        return params
    
    @property
    def input_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec(name="Q", shape=("batch", "seq_len", "num_heads", "head_dim")),
            TensorSpec(name="K", shape=("batch", "seq_len", "num_kv_heads", "head_dim")),
            TensorSpec(name="V", shape=("batch", "seq_len", "num_kv_heads", "head_dim")),
        ]

