"""
Base classes for operation specifications.

Operations define what kernels we want to generate and optimize.
Each operation has:
- Shape parameters (what dimensions can vary)
- Tensor specifications (inputs/outputs)
- Reference implementation (ground truth)
- Baseline implementation (what we're trying to beat)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


class DataType(Enum):
    """Supported data types for kernel generation."""
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    FP8 = "float8"
    INT8 = "int8"

    def to_torch(self) -> torch.dtype:
        mapping = {
            DataType.FP32: torch.float32,
            DataType.FP16: torch.float16,
            DataType.BF16: torch.bfloat16,
            DataType.INT8: torch.int8,
        }
        return mapping.get(self, torch.float32)

    def to_hip(self) -> str:
        mapping = {
            DataType.FP32: "float",
            DataType.FP16: "half",
            DataType.BF16: "__hip_bfloat16",
            DataType.INT8: "int8_t",
        }
        return mapping.get(self, "float")


@dataclass
class TensorSpec:
    """Specification for an input or output tensor."""
    name: str
    shape: Tuple[str, ...]  # Symbolic shape like ("batch", "seq_len", "hidden")
    dtype: DataType = DataType.FP16
    is_output: bool = False
    memory_layout: str = "row_major"  # or "col_major"
    
    def concrete_shape(self, shape_values: Dict[str, int]) -> Tuple[int, ...]:
        """Convert symbolic shape to concrete dimensions."""
        return tuple(shape_values[dim] for dim in self.shape)


@dataclass
class ShapeConfig:
    """Configuration for shape parameters."""
    name: str
    min_value: int
    max_value: int
    typical_values: List[int] = field(default_factory=list)
    must_be_power_of_2: bool = False
    must_be_multiple_of: Optional[int] = None
    
    def sample_value(self, difficulty: str = "medium") -> int:
        """Sample a value based on difficulty level."""
        if difficulty == "easy":
            return self.typical_values[0] if self.typical_values else self.min_value
        elif difficulty == "hard":
            return self.typical_values[-1] if self.typical_values else self.max_value
        else:
            import random
            return random.choice(self.typical_values) if self.typical_values else (self.min_value + self.max_value) // 2


@dataclass
class OperationSpec(ABC):
    """
    Abstract base class for operation specifications.
    
    Subclasses define specific operations like Attention, GEMM, LayerNorm, etc.
    """
    name: str
    description: str
    
    @property
    @abstractmethod
    def shape_params(self) -> Dict[str, ShapeConfig]:
        """Define the shape parameters for this operation."""
        pass
    
    @property
    @abstractmethod
    def input_tensors(self) -> List[TensorSpec]:
        """Define input tensor specifications."""
        pass
    
    @property
    @abstractmethod
    def output_tensors(self) -> List[TensorSpec]:
        """Define output tensor specifications."""
        pass
    
    @abstractmethod
    def reference_impl(self, **inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        PyTorch reference implementation.
        This is the ground truth for correctness checking.
        """
        pass
    
    @abstractmethod
    def baseline_impl(self, **inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Baseline implementation using vendor libraries (rocBLAS/MIOpen).
        This is what we're trying to beat.
        """
        pass
    
    def generate_test_inputs(
        self, 
        shape_values: Dict[str, int],
        dtype: DataType = DataType.FP16,
        device: str = "cuda"
    ) -> Dict[str, torch.Tensor]:
        """Generate random test inputs for the operation."""
        inputs = {}
        torch_dtype = dtype.to_torch()
        
        for tensor_spec in self.input_tensors:
            shape = tensor_spec.concrete_shape(shape_values)
            if torch_dtype in (torch.float16, torch.bfloat16, torch.float32):
                tensor = torch.randn(shape, dtype=torch_dtype, device=device)
            else:
                tensor = torch.randint(-128, 127, shape, dtype=torch_dtype, device=device)
            inputs[tensor_spec.name] = tensor
        
        return inputs
    
    def get_kernel_signature(self, shape_values: Dict[str, int], dtype: DataType) -> str:
        """Generate a HIP kernel function signature."""
        params = []
        
        for tensor_spec in self.input_tensors + self.output_tensors:
            hip_type = dtype.to_hip()
            shape = tensor_spec.concrete_shape(shape_values)
            params.append(f"{hip_type}* {tensor_spec.name}")
        
        # Add dimension parameters
        for dim_name, value in shape_values.items():
            params.append(f"int {dim_name}")
        
        return f"__global__ void {self.name}_kernel({', '.join(params)})"
    
    def get_prompt_context(self, shape_values: Dict[str, int], dtype: DataType) -> str:
        """Generate context for the LLM prompt."""
        context = f"""
Operation: {self.name}
Description: {self.description}

Shape Parameters:
"""
        for name, value in shape_values.items():
            context += f"  - {name}: {value}\n"
        
        context += f"\nData Type: {dtype.value}\n"
        context += f"\nInput Tensors:\n"
        
        for tensor in self.input_tensors:
            shape = tensor.concrete_shape(shape_values)
            context += f"  - {tensor.name}: shape={shape}, dtype={dtype.value}\n"
        
        context += f"\nOutput Tensors:\n"
        for tensor in self.output_tensors:
            shape = tensor.concrete_shape(shape_values)
            context += f"  - {tensor.name}: shape={shape}, dtype={dtype.value}\n"
        
        context += f"\nKernel Signature:\n{self.get_kernel_signature(shape_values, dtype)}\n"
        
        return context


@dataclass
class BenchmarkResult:
    """Results from benchmarking a kernel."""
    operation: str
    shape_values: Dict[str, int]
    dtype: str
    
    # Correctness
    is_correct: bool
    max_error: float
    mean_error: float
    
    # Performance
    kernel_time_ms: float
    baseline_time_ms: float
    speedup: float
    
    # Efficiency metrics
    memory_bandwidth_gbps: float
    achieved_occupancy: float
    register_usage: int
    shared_memory_bytes: int
    
    # Profiling details
    profiling_data: Dict[str, Any] = field(default_factory=dict)

