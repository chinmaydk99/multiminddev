from typing import Optional, Dict, Any
from .base_agent import BaseAgent, AgentResponse
from ..utils.llm_interface import LLMInterface
from ..utils.config import AgentConfig
import time
import structlog


class CUDAGeneratorAgent(BaseAgent):
    """Specialized agent for initial CUDA kernel generation from PyTorch operations."""
    
    def __init__(
        self,
        config: AgentConfig,
        llm_interface: LLMInterface,
        agent_id: Optional[str] = None,
    ):
        super().__init__(config, llm_interface, agent_id)
        self.logger = structlog.get_logger(
            agent_type=self.agent_type,
            agent_id=self.agent_id,
        )
    
    @property
    def agent_type(self) -> str:
        return "cuda_generator"
    
    @property  
    def system_prompt(self) -> str:
        return """You are a CUDA kernel generation specialist. Given a PyTorch operation, generate an equivalent CUDA kernel.

Focus on:
- Correct memory access patterns
- Thread block and grid sizing
- Basic memory coalescing
- Error handling and bounds checking
- Proper CUDA C++ syntax with __global__ kernels

Generate complete, compilable CUDA C++ code with proper headers and kernel launch parameters.

Example format:
```cuda
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void kernel_name(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // kernel logic here
        output[idx] = input[idx];
    }
}

// Host wrapper function
void launch_kernel(float* d_input, float* d_output, int size) {
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    kernel_name<<<grid, block>>>(d_input, d_output, size);
    cudaDeviceSynchronize();
}
```

Always include:
1. Proper CUDA headers
2. __global__ kernel function
3. Thread indexing with bounds checking
4. Host wrapper function with launch parameters
"""

    async def process_request(
        self, 
        pytorch_operation: str, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResponse:
        """
        Process PyTorch operation and generate equivalent CUDA kernel.
        
        Args:
            pytorch_operation: Description of PyTorch operation to convert
            context: Additional context including test inputs and target performance
            **kwargs: Additional arguments
            
        Returns:
            AgentResponse with generated CUDA kernel code
        """
        start_time = time.time()
        
        try:
            # Build context-aware prompt
            messages = self._build_messages(pytorch_operation, context)
            
            # Generate CUDA kernel
            kernel_code = await self._call_llm(messages)
            
            # Extract kernel metadata
            metadata = self._extract_kernel_metadata(kernel_code, context)
            
            execution_time = time.time() - start_time
            
            self.logger.info(
                "CUDA kernel generated successfully",
                execution_time=execution_time,
                operation=pytorch_operation,
                kernel_size=len(kernel_code)
            )
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                content=kernel_code,
                metadata=metadata,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Failed to generate CUDA kernel: {str(e)}"
            
            self.logger.error(
                "CUDA kernel generation failed",
                error=error_msg,
                execution_time=execution_time,
                operation=pytorch_operation
            )
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                content="",
                metadata={},
                execution_time=execution_time,
                error=error_msg
            )
    
    def _extract_kernel_metadata(
        self, 
        kernel_code: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract metadata from generated kernel code."""
        metadata = {
            "kernel_type": "initial_generation",
            "has_bounds_check": "idx < " in kernel_code.lower(),
            "uses_shared_memory": "__shared__" in kernel_code.lower(),
            "kernel_name": self._extract_kernel_name(kernel_code),
            "estimated_complexity": "basic"
        }
        
        if context:
            metadata.update({
                "target_performance": context.get("target_performance", 1.0),
                "input_shapes": context.get("test_inputs", []),
            })
            
        return metadata
    
    def _extract_kernel_name(self, kernel_code: str) -> Optional[str]:
        """Extract kernel function name from code."""
        lines = kernel_code.split('\n')
        for line in lines:
            if '__global__' in line and '(' in line:
                # Extract function name between __global__ and (
                parts = line.split('__global__')[1].strip()
                if 'void' in parts:
                    parts = parts.split('void')[1].strip()
                    func_name = parts.split('(')[0].strip()
                    return func_name
        return None