from typing import Optional, Dict, Any
from .base_agent import BaseAgent, AgentResponse
from ..utils.llm_interface import LLMInterface
from ..utils.config import AgentConfig
import time
import structlog


class CUDATesterAgent(BaseAgent):
    """Specialized agent for compiling, testing, and profiling CUDA kernels."""
    
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
        return "cuda_tester"
    
    @property
    def system_prompt(self) -> str:
        return """You are a CUDA testing and profiling specialist. Analyze CUDA kernels for compilation, correctness, and performance.

Your responsibilities:
1. Assess kernel compilation feasibility
2. Identify potential runtime errors
3. Evaluate performance characteristics
4. Suggest testing strategies
5. Provide profiling insights

Analysis areas:
- Compilation issues (syntax, headers, linkage)
- Memory access patterns and bounds checking
- Thread divergence and synchronization
- Resource usage (registers, shared memory)
- Performance bottlenecks
- Numerical stability and precision

Provide structured feedback including:
- Compilation assessment (will it compile?)
- Functional correctness analysis
- Performance evaluation
- Testing recommendations
- Profiling suggestions
"""

    async def process_request(
        self, 
        kernel_code: str, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResponse:
        """
        Analyze CUDA kernel for compilation, testing, and profiling.
        
        Args:
            kernel_code: CUDA kernel code to test and analyze
            context: Additional context including test inputs and PyTorch operation
            **kwargs: Additional arguments
            
        Returns:
            AgentResponse with testing and profiling analysis
        """
        start_time = time.time()
        
        try:
            # Build testing-focused prompt
            testing_prompt = self._build_testing_prompt(kernel_code, context)
            messages = self._build_messages(testing_prompt, context)
            
            # Generate testing analysis
            analysis_result = await self._call_llm(messages)
            
            # Extract testing metadata
            metadata = self._extract_testing_metadata(analysis_result, kernel_code, context)
            
            execution_time = time.time() - start_time
            
            self.logger.info(
                "CUDA kernel testing analysis completed",
                execution_time=execution_time,
                compilation_feasible=metadata.get("compilation_feasible", False),
                performance_score=metadata.get("performance_score", 0.0),
                issues_found=len(metadata.get("issues_identified", []))
            )
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                content=analysis_result,
                metadata=metadata,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Failed to analyze CUDA kernel: {str(e)}"
            
            self.logger.error(
                "CUDA kernel testing analysis failed",
                error=error_msg,
                execution_time=execution_time,
                kernel_size=len(kernel_code)
            )
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                content="",
                metadata={},
                execution_time=execution_time,
                error=error_msg
            )
    
    def _build_testing_prompt(
        self, 
        kernel_code: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for kernel testing analysis."""
        prompt = f"Analyze this CUDA kernel for compilation, correctness, and performance:\n\n```cuda\n{kernel_code}\n```\n\n"
        
        if context:
            pytorch_operation = context.get("pytorch_operation", "")
            if pytorch_operation:
                prompt += f"Original PyTorch operation: {pytorch_operation}\n\n"
            
            test_inputs = context.get("test_inputs", [])
            if test_inputs:
                prompt += f"Test inputs: {test_inputs}\n\n"
        
        prompt += """Provide analysis covering:
1. Compilation feasibility and potential issues
2. Functional correctness assessment
3. Performance characteristics and bottlenecks
4. Testing strategy recommendations
5. Overall quality score (0-10)"""
        
        return prompt
    
    def _extract_testing_metadata(
        self, 
        analysis_result: str, 
        kernel_code: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract metadata from testing analysis."""
        # Analyze kernel code characteristics
        has_bounds_check = "idx < " in kernel_code.lower() or "threadIdx.x < " in kernel_code.lower()
        has_shared_memory = "__shared__" in kernel_code
        has_synchronization = "__syncthreads()" in kernel_code
        
        # Extract quality indicators from analysis
        compilation_feasible = self._assess_compilation_feasibility(analysis_result)
        performance_score = self._extract_performance_score(analysis_result)
        issues_identified = self._extract_issues(analysis_result)
        
        metadata = {
            "kernel_type": "testing_analysis",
            "compilation_feasible": compilation_feasible,
            "has_bounds_check": has_bounds_check,
            "has_shared_memory": has_shared_memory,
            "has_synchronization": has_synchronization,
            "performance_score": performance_score,
            "issues_identified": issues_identified,
            "quality_indicators": {
                "memory_safety": has_bounds_check,
                "resource_utilization": has_shared_memory,
                "thread_coordination": has_synchronization
            }
        }
        
        # Add context-specific metadata
        if context:
            metadata.update({
                "test_inputs_provided": bool(context.get("test_inputs")),
                "baseline_operation": context.get("pytorch_operation", ""),
                "target_performance": context.get("target_performance", 1.0)
            })
        
        # Calculate overall speedup estimate
        if compilation_feasible and performance_score > 5:
            metadata["estimated_speedup"] = self._estimate_speedup(kernel_code, performance_score)
        
        return metadata
    
    def _assess_compilation_feasibility(self, analysis: str) -> bool:
        """Assess if kernel is likely to compile successfully."""
        negative_indicators = [
            "compilation error", "syntax error", "missing header", 
            "undefined", "will not compile", "compilation issues"
        ]
        
        positive_indicators = [
            "will compile", "compilation successful", "syntactically correct",
            "compiles successfully", "no compilation issues"
        ]
        
        analysis_lower = analysis.lower()
        
        # Check for explicit positive indicators
        for indicator in positive_indicators:
            if indicator in analysis_lower:
                return True
        
        # Check for negative indicators
        for indicator in negative_indicators:
            if indicator in analysis_lower:
                return False
        
        # Default to feasible if no clear indicators
        return True
    
    def _extract_performance_score(self, analysis: str) -> float:
        """Extract performance score from analysis text."""
        # Look for explicit scores
        import re
        
        # Pattern for "score: X/10" or "X out of 10"
        score_patterns = [
            r"score[:\s]+(\d+(?:\.\d+)?)[/\s]*10",
            r"(\d+(?:\.\d+)?)[/\s]+10",
            r"quality[:\s]+(\d+(?:\.\d+)?)"
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, analysis.lower())
            if match:
                score = float(match.group(1))
                return min(score, 10.0)  # Cap at 10
        
        # Fallback: analyze qualitative indicators
        quality_indicators = {
            "excellent": 9.0, "very good": 8.0, "good": 7.0,
            "decent": 6.0, "fair": 5.0, "poor": 3.0, "bad": 2.0
        }
        
        analysis_lower = analysis.lower()
        for indicator, score in quality_indicators.items():
            if indicator in analysis_lower:
                return score
        
        return 5.0  # Default middle score
    
    def _extract_issues(self, analysis: str) -> list:
        """Extract identified issues from analysis."""
        issues = []
        analysis_lower = analysis.lower()
        
        # Common CUDA issues
        issue_patterns = {
            "memory coalescing": ["coalescing", "uncoalesced access"],
            "bounds checking": ["bounds check", "out of bounds", "buffer overflow"],
            "shared memory": ["bank conflict", "shared memory"],
            "thread divergence": ["divergence", "branching"],
            "synchronization": ["race condition", "synchronization"],
            "occupancy": ["occupancy", "resource usage"]
        }
        
        for issue_type, patterns in issue_patterns.items():
            for pattern in patterns:
                if pattern in analysis_lower:
                    issues.append(issue_type)
                    break
        
        return list(set(issues))  # Remove duplicates
    
    def _estimate_speedup(self, kernel_code: str, performance_score: float) -> float:
        """Estimate potential speedup based on kernel analysis."""
        base_speedup = 1.0
        
        # Adjust based on performance score
        score_multiplier = performance_score / 5.0  # Normalize around 5
        base_speedup *= score_multiplier
        
        # Bonus for optimization features
        if "__shared__" in kernel_code:
            base_speedup *= 1.5
        
        if "float4" in kernel_code or "float2" in kernel_code:
            base_speedup *= 1.2
        
        if "#pragma unroll" in kernel_code:
            base_speedup *= 1.1
        
        # Cap reasonable speedup range
        return min(max(base_speedup, 0.8), 10.0)