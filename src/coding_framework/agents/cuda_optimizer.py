from typing import Optional, Dict, Any, List
from .base_agent import BaseAgent, AgentResponse
from ..utils.llm_interface import LLMInterface
from ..utils.config import AgentConfig
import time
import structlog


class CUDAOptimizerAgent(BaseAgent):
    """Specialized agent for optimizing existing CUDA kernels for performance."""
    
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
        return "cuda_optimizer"
    
    @property
    def system_prompt(self) -> str:
        return """You are a CUDA performance optimization expert. Analyze existing kernels and apply optimizations.

Optimization techniques to consider:
- Shared memory utilization (__shared__ arrays for data reuse)
- Memory coalescing improvements (aligned, sequential access patterns)
- Thread block size optimization (multiples of warp size)
- Register usage reduction (avoid spilling to local memory)
- Vectorized memory access (float4, float2 for throughput)
- Loop unrolling and instruction-level parallelism
- Avoiding bank conflicts in shared memory
- Occupancy optimization (balance threads vs resources)

Analysis process:
1. Identify performance bottlenecks in current kernel
2. Select 1-2 most impactful optimizations
3. Apply optimizations with clear before/after explanations
4. Maintain functional correctness

Provide optimized kernel with:
- Explanation of optimizations applied
- Expected performance improvement rationale
- Any trade-offs or assumptions made
"""

    async def process_request(
        self, 
        kernel_code: str, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResponse:
        """
        Analyze and optimize existing CUDA kernel for performance.
        
        Args:
            kernel_code: Current CUDA kernel code to optimize
            context: Additional context including performance feedback and history
            **kwargs: Additional arguments
            
        Returns:
            AgentResponse with optimized CUDA kernel code
        """
        start_time = time.time()
        
        try:
            # Build optimization-focused prompt
            optimization_prompt = self._build_optimization_prompt(kernel_code, context)
            messages = self._build_messages(optimization_prompt, context)
            
            # Generate optimized kernel
            optimized_code = await self._call_llm(messages)
            
            # Extract optimization metadata
            metadata = self._extract_optimization_metadata(optimized_code, context)
            
            execution_time = time.time() - start_time
            
            self.logger.info(
                "CUDA kernel optimized successfully",
                execution_time=execution_time,
                optimizations_applied=metadata.get("optimizations_applied", []),
                original_size=len(kernel_code),
                optimized_size=len(optimized_code)
            )
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                content=optimized_code,
                metadata=metadata,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Failed to optimize CUDA kernel: {str(e)}"
            
            self.logger.error(
                "CUDA kernel optimization failed",
                error=error_msg,
                execution_time=execution_time,
                kernel_size=len(kernel_code)
            )
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                content=kernel_code,  # Return original code on failure
                metadata={},
                execution_time=execution_time,
                error=error_msg
            )
    
    def _build_optimization_prompt(
        self, 
        kernel_code: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for kernel optimization."""
        prompt = f"Analyze and optimize this CUDA kernel:\n\n```cuda\n{kernel_code}\n```\n\n"
        
        if context:
            performance_feedback = context.get("performance_feedback", "")
            if performance_feedback:
                prompt += f"Performance feedback: {performance_feedback}\n\n"
            
            target_performance = context.get("target_performance", 0)
            if target_performance > 0:
                prompt += f"Target speedup: {target_performance}x\n\n"
            
            optimization_history = context.get("optimization_history", [])
            if optimization_history:
                prompt += f"Previous optimizations tried: {', '.join(optimization_history)}\n\n"
        
        prompt += "Provide optimized kernel with explanation of improvements made."
        return prompt
    
    def _extract_optimization_metadata(
        self, 
        optimized_code: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract metadata from optimization results."""
        # Detect optimization patterns in code
        optimizations_applied = []
        
        if "__shared__" in optimized_code:
            optimizations_applied.append("shared_memory")
        
        if "float4" in optimized_code or "float2" in optimized_code:
            optimizations_applied.append("vectorized_memory")
        
        if "#pragma unroll" in optimized_code:
            optimizations_applied.append("loop_unrolling")
        
        # Count thread blocks and grid configurations
        block_configs = self._extract_block_configurations(optimized_code)
        
        metadata = {
            "kernel_type": "optimization",
            "optimizations_applied": optimizations_applied,
            "uses_shared_memory": "__shared__" in optimized_code,
            "uses_vectorization": any(vec in optimized_code for vec in ["float4", "float2", "int4"]),
            "block_configurations": block_configs,
            "estimated_improvement": self._estimate_performance_improvement(optimizations_applied)
        }
        
        if context:
            metadata["optimization_iteration"] = len(context.get("optimization_history", [])) + 1
            
        return metadata
    
    def _extract_block_configurations(self, code: str) -> List[str]:
        """Extract block size configurations from kernel code."""
        configs = []
        lines = code.split('\n')
        
        for line in lines:
            if 'dim3 block(' in line:
                config = line.strip()
                configs.append(config)
        
        return configs
    
    def _estimate_performance_improvement(self, optimizations: List[str]) -> str:
        """Estimate performance improvement based on optimizations applied."""
        improvement_map = {
            "shared_memory": "1.5-3x",
            "vectorized_memory": "1.2-2x", 
            "loop_unrolling": "1.1-1.5x",
            "memory_coalescing": "1.3-4x",
            "occupancy_optimization": "1.2-2x"
        }
        
        if not optimizations:
            return "minimal"
        
        # Return highest potential improvement
        for opt in optimizations:
            if opt in improvement_map:
                return improvement_map[opt]
        
        return "1.1-1.3x"