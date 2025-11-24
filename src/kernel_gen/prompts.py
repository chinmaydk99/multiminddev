"""
Prompt templates for kernel generation.

Structured prompts that provide:
- Operation specification
- Shape information
- HIP/ROCm constraints
- Optimization hints
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.operations.base import DataType, OperationSpec


SYSTEM_PROMPT = """You are an expert HIP/ROCm kernel developer specializing in high-performance GPU code for AMD MI300X GPUs.

Your task is to generate optimized HIP C++ kernels that:
1. Are correct (produce numerically accurate results)
2. Are fast (outperform standard library implementations)
3. Exploit MI300X architecture features (LDS, matrix cores, HBM3 bandwidth)

Key MI300X specifications:
- Architecture: CDNA3 (gfx942)
- Compute Units: 228 CUs
- HBM3 Memory: 192GB at 5.3 TB/s bandwidth
- LDS per CU: 64 KB
- Registers per CU: 512 VGPRs, 108 SGPRs per wavefront
- Wavefront size: 64 threads
- Max threads per block: 1024

When generating kernels:
- Use half precision (fp16/bf16) when possible for 2x throughput
- Maximize memory coalescing for HBM3 bandwidth
- Use LDS (shared memory) to reduce global memory traffic
- Consider using MFMA (Matrix Fused Multiply-Add) instructions for GEMM-like ops
- Avoid thread divergence within wavefronts
- Use vectorized loads (float4, half8) when alignment permits

Always output ONLY the kernel code in a ```cpp code block. No explanations before or after."""


@dataclass
class KernelPrompt:
    """A structured prompt for kernel generation."""
    
    operation_name: str
    operation_description: str
    shape_values: Dict[str, int]
    dtype: DataType
    
    # Input/output specifications
    input_specs: List[Dict[str, Any]] = field(default_factory=list)
    output_specs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Kernel signature
    kernel_signature: str = ""
    
    # Optional hints
    optimization_hints: List[str] = field(default_factory=list)
    previous_attempt: Optional[str] = None
    previous_error: Optional[str] = None
    previous_speedup: Optional[float] = None
    
    def to_user_prompt(self) -> str:
        """Generate the user prompt string."""
        
        prompt = f"""Generate an optimized HIP kernel for: {self.operation_name}

## Operation
{self.operation_description}

## Shape Parameters
"""
        for name, value in self.shape_values.items():
            prompt += f"- {name}: {value}\n"
        
        prompt += f"\n## Data Type\n{self.dtype.value}\n"
        
        prompt += "\n## Inputs\n"
        for spec in self.input_specs:
            prompt += f"- {spec['name']}: shape={spec['shape']}, dtype={self.dtype.value}\n"
        
        prompt += "\n## Outputs\n"
        for spec in self.output_specs:
            prompt += f"- {spec['name']}: shape={spec['shape']}, dtype={self.dtype.value}\n"
        
        prompt += f"\n## Required Kernel Signature\n```cpp\n{self.kernel_signature}\n```\n"
        
        if self.optimization_hints:
            prompt += "\n## Optimization Hints\n"
            for hint in self.optimization_hints:
                prompt += f"- {hint}\n"
        
        if self.previous_attempt:
            prompt += f"\n## Previous Attempt (needs improvement)\n```cpp\n{self.previous_attempt}\n```\n"
            
            if self.previous_error:
                prompt += f"\n## Error from previous attempt\n{self.previous_error}\n"
            
            if self.previous_speedup is not None:
                prompt += f"\n## Previous speedup: {self.previous_speedup:.2f}x (target: >2.0x)\n"
                prompt += "Please optimize further to achieve higher speedup.\n"
        
        prompt += "\nGenerate the complete HIP kernel code:"
        
        return prompt


class PromptBuilder:
    """Builds prompts from operation specifications."""
    
    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.system_prompt = system_prompt
    
    def build_prompt(
        self,
        operation: OperationSpec,
        shape_values: Dict[str, int],
        dtype: DataType = DataType.FP16,
        optimization_hints: Optional[List[str]] = None,
        previous_attempt: Optional[str] = None,
        previous_error: Optional[str] = None,
        previous_speedup: Optional[float] = None,
    ) -> KernelPrompt:
        """
        Build a prompt from an operation specification.
        
        Args:
            operation: The operation to generate a kernel for
            shape_values: Concrete shape values
            dtype: Data type to use
            optimization_hints: Optional hints for the model
            previous_attempt: Previous kernel code (for refinement)
            previous_error: Error from previous attempt
            previous_speedup: Speedup achieved by previous attempt
            
        Returns:
            KernelPrompt ready for generation
        """
        
        # Build input/output specs
        input_specs = []
        for tensor in operation.input_tensors:
            shape = tensor.concrete_shape(shape_values)
            input_specs.append({
                "name": tensor.name,
                "shape": shape,
            })
        
        output_specs = []
        for tensor in operation.output_tensors:
            shape = tensor.concrete_shape(shape_values)
            output_specs.append({
                "name": tensor.name,
                "shape": shape,
            })
        
        # Build kernel signature
        kernel_signature = operation.get_kernel_signature(shape_values, dtype)
        
        # Default optimization hints based on operation
        hints = optimization_hints or []
        hints = list(hints)  # Copy to avoid mutation
        
        # Add operation-specific hints
        if "attention" in operation.name.lower():
            hints.extend([
                "Use tiled computation to fit Q, K blocks in LDS",
                "Implement online softmax for numerical stability",
                "Consider flash attention algorithm for memory efficiency",
            ])
        elif "gemm" in operation.name.lower():
            hints.extend([
                "Use 128x128 or 256x256 tiles for good LDS utilization",
                "Consider MFMA instructions for matrix operations",
                "Double buffer tiles to hide memory latency",
            ])
        elif "norm" in operation.name.lower():
            hints.extend([
                "Use warp-level reduction for computing mean/variance",
                "Fuse normalization with scale/bias application",
                "Process multiple elements per thread for better ILP",
            ])
        
        return KernelPrompt(
            operation_name=operation.name,
            operation_description=operation.description,
            shape_values=shape_values,
            dtype=dtype,
            input_specs=input_specs,
            output_specs=output_specs,
            kernel_signature=kernel_signature,
            optimization_hints=hints,
            previous_attempt=previous_attempt,
            previous_error=previous_error,
            previous_speedup=previous_speedup,
        )
    
    def build_refinement_prompt(
        self,
        operation: OperationSpec,
        shape_values: Dict[str, int],
        dtype: DataType,
        previous_code: str,
        execution_result: Any,  # ExecutionResult
    ) -> KernelPrompt:
        """
        Build a prompt for refining a previous attempt.
        
        Uses feedback from execution to guide improvements.
        """
        
        # Determine what went wrong and build appropriate hints
        hints = []
        error = None
        speedup = None
        
        if not execution_result.compiled:
            error = execution_result.compilation_error
            hints.append("Fix the compilation error")
        elif not execution_result.numerically_correct:
            error = f"Output incorrect. Max error: {execution_result.max_absolute_error:.6f}"
            hints.append("Fix the numerical correctness issue")
            hints.append("Check boundary conditions and index calculations")
        else:
            # Compiled and correct, but need more speed
            speedup = execution_result.speedup
            hints.append(f"Current speedup is {speedup:.2f}x, target is >2.0x")
            
            # Add hints based on profiling data
            if execution_result.achieved_occupancy < 0.5:
                hints.append("Occupancy is low - try reducing register usage or increasing block size")
            
            if execution_result.memory_bandwidth_gbps < 1000:
                hints.append("Memory bandwidth utilization is low - improve memory access patterns")
        
        return self.build_prompt(
            operation=operation,
            shape_values=shape_values,
            dtype=dtype,
            optimization_hints=hints,
            previous_attempt=previous_code,
            previous_error=error,
            previous_speedup=speedup,
        )

