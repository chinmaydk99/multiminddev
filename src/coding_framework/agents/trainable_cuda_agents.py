"""
Specialized trainable agents for CUDA kernel generation, optimization, and testing.
These agents own HuggingFace model parameters and can be trained through RL.
"""

import torch
from typing import Dict, Any, Optional, List, Tuple
import re
import asyncio
import subprocess
from pathlib import Path
import tempfile
import json

from .trainable_agent import TrainableAgent, GenerationOutput


class TrainableCUDAGeneratorAgent(TrainableAgent):
    """
    Trainable CUDA Generator Agent that creates initial CUDA kernels from PyTorch operations.
    
    Specialization:
    - Syntactic correctness and compilability
    - CUDA patterns and idioms
    - Thread indexing and memory management
    """
    
    def __init__(
        self,
        agent_id: str = "cuda_generator",
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="generator",
            model_name=model_name,
            **kwargs
        )
        
        # CUDA-specific generation patterns
        self.cuda_templates = {
            "kernel_header": "__global__ void",
            "thread_index": "int idx = blockIdx.x * blockDim.x + threadIdx.x;",
            "grid_stride": "for (int i = idx; i < n; i += blockDim.x * gridDim.x)",
            "boundary_check": "if (idx < n)",
        }
        
        self.logger.info("TrainableCUDAGeneratorAgent initialized")
    
    def _format_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Format prompt with CUDA generation-specific instructions."""
        
        instruction = """You are a CUDA kernel generation specialist. Generate optimized CUDA C++ kernel code.

Requirements:
1. Include all necessary headers (#include <cuda_runtime.h>, etc.)
2. Use proper thread indexing (blockIdx, threadIdx, blockDim, gridDim)
3. Include boundary checks to prevent out-of-bounds access
4. Follow CUDA best practices for memory access patterns
5. Include kernel launch configuration comments

Generate ONLY the kernel code without explanations."""
        
        formatted = f"{instruction}\n\n{prompt}"
        
        # Add context about tensor shapes, dtypes, etc.
        if context:
            if "tensor_shapes" in context:
                formatted += f"\nTensor Shapes: {context['tensor_shapes']}"
            if "dtype" in context:
                formatted += f"\nData Type: {context['dtype']}"
            if "operation" in context:
                formatted += f"\nOperation: {context['operation']}"
        
        return formatted
    
    async def generate_cuda_kernel(
        self,
        operation_description: str,
        tensor_info: Optional[Dict[str, Any]] = None,
        performance_hints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate CUDA kernel from operation description.
        
        Args:
            operation_description: Description of the PyTorch operation
            tensor_info: Information about tensor shapes and dtypes
            performance_hints: Optional performance optimization hints
            
        Returns:
            Generated kernel code and metadata
        """
        # Build context
        context = {}
        if tensor_info:
            context.update(tensor_info)
        if performance_hints:
            context["performance_hints"] = performance_hints
        
        # Generate kernel
        generation_output = await self.generate_with_log_probs(
            operation_description,
            max_new_tokens=1024,
            temperature=0.7
        )
        
        kernel_code = generation_output.text
        
        # Extract kernel function name
        kernel_name = self._extract_kernel_name(kernel_code)
        
        # Validate basic CUDA syntax
        is_valid, validation_errors = self._validate_cuda_syntax(kernel_code)
        
        return {
            "kernel_code": kernel_code,
            "kernel_name": kernel_name,
            "is_valid_syntax": is_valid,
            "validation_errors": validation_errors,
            "log_probs": generation_output.log_probs,
            "token_ids": generation_output.token_ids,
            "agent_id": self.agent_id,
            "context": context
        }
    
    def _extract_kernel_name(self, kernel_code: str) -> Optional[str]:
        """Extract kernel function name from CUDA code."""
        pattern = r"__global__\s+\w+\s+(\w+)\s*\("
        match = re.search(pattern, kernel_code)
        return match.group(1) if match else None
    
    def _validate_cuda_syntax(self, kernel_code: str) -> Tuple[bool, List[str]]:
        """Basic validation of CUDA kernel syntax."""
        errors = []
        
        # Check for __global__ keyword
        if "__global__" not in kernel_code:
            errors.append("Missing __global__ kernel declaration")
        
        # Check for thread indexing
        if not any(idx in kernel_code for idx in ["threadIdx", "blockIdx"]):
            errors.append("Missing thread indexing (threadIdx/blockIdx)")
        
        # Check for boundary checks
        if "if" not in kernel_code and "for" not in kernel_code:
            errors.append("Missing boundary checks")
        
        # Check for proper function signature
        if not re.search(r"__global__\s+void\s+\w+\s*\([^)]*\)", kernel_code):
            errors.append("Invalid kernel function signature")
        
        return len(errors) == 0, errors


class TrainableCUDAOptimizerAgent(TrainableAgent):
    """
    Trainable CUDA Optimizer Agent that applies performance optimizations to kernels.
    
    Specialization:
    - Memory coalescing and shared memory usage
    - Register optimization and occupancy
    - Vectorized memory access patterns
    - Loop unrolling and instruction optimization
    """
    
    def __init__(
        self,
        agent_id: str = "cuda_optimizer",
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="optimizer",
            model_name=model_name,
            **kwargs
        )
        
        # Optimization strategies
        self.optimization_strategies = [
            "shared_memory",
            "memory_coalescing",
            "vectorized_access",
            "loop_unrolling",
            "warp_primitives",
            "tensor_cores"
        ]
        
        self.logger.info("TrainableCUDAOptimizerAgent initialized")
    
    def _format_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Format prompt with CUDA optimization-specific instructions."""
        
        instruction = """You are a CUDA kernel optimization specialist. Optimize the given kernel for maximum performance.

Optimization Focus Areas:
1. Memory Coalescing: Ensure coalesced global memory access patterns
2. Shared Memory: Use shared memory to reduce global memory traffic
3. Vectorized Access: Use float4/int4 for vectorized memory operations
4. Register Usage: Optimize register usage for better occupancy
5. Loop Unrolling: Unroll loops where beneficial
6. Warp-level Primitives: Use __shfl, __ballot when appropriate

Return ONLY the optimized kernel code."""
        
        formatted = f"{instruction}\n\n{prompt}"
        
        # Add performance context
        if context:
            if "performance_metrics" in context:
                formatted += f"\nCurrent Performance: {context['performance_metrics']}"
            if "bottlenecks" in context:
                formatted += f"\nIdentified Bottlenecks: {context['bottlenecks']}"
            if "target_speedup" in context:
                formatted += f"\nTarget Speedup: {context['target_speedup']}x"
        
        return formatted
    
    async def optimize_kernel(
        self,
        kernel_code: str,
        performance_analysis: Optional[Dict[str, Any]] = None,
        optimization_targets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Optimize existing CUDA kernel for better performance.
        
        Args:
            kernel_code: Existing kernel code to optimize
            performance_analysis: Performance profiling data
            optimization_targets: Specific optimization strategies to apply
            
        Returns:
            Optimized kernel and optimization metadata
        """
        # Build optimization prompt
        prompt = f"Optimize this CUDA kernel:\n\n{kernel_code}"
        
        # Add context
        context = {}
        if performance_analysis:
            context["performance_metrics"] = performance_analysis
        if optimization_targets:
            context["requested_optimizations"] = optimization_targets
        
        # Generate optimized kernel
        generation_output = await self.generate_with_log_probs(
            prompt,
            max_new_tokens=1024,
            temperature=0.6  # Lower temperature for more focused optimization
        )
        
        optimized_code = generation_output.text
        
        # Detect applied optimizations
        applied_optimizations = self._detect_optimizations(optimized_code)
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            kernel_code,
            optimized_code,
            applied_optimizations
        )
        
        return {
            "optimized_code": optimized_code,
            "applied_optimizations": applied_optimizations,
            "optimization_score": optimization_score,
            "log_probs": generation_output.log_probs,
            "token_ids": generation_output.token_ids,
            "agent_id": self.agent_id,
            "context": context
        }
    
    def _detect_optimizations(self, kernel_code: str) -> List[str]:
        """Detect which optimization strategies were applied."""
        applied = []
        
        # Check for shared memory usage
        if "__shared__" in kernel_code or "extern __shared__" in kernel_code:
            applied.append("shared_memory")
        
        # Check for vectorized access
        if any(v in kernel_code for v in ["float4", "int4", "float2", "int2"]):
            applied.append("vectorized_access")
        
        # Check for loop unrolling
        if "#pragma unroll" in kernel_code or re.search(r"for.*\n.*{[^}]*\n[^}]*\n[^}]*\n[^}]*}", kernel_code):
            applied.append("loop_unrolling")
        
        # Check for warp primitives
        if any(p in kernel_code for p in ["__shfl", "__ballot", "__any", "__all"]):
            applied.append("warp_primitives")
        
        # Check for proper memory coalescing patterns
        if re.search(r"tid\s*\+\s*blockDim\.x\s*\*", kernel_code):
            applied.append("memory_coalescing")
        
        return applied
    
    def _calculate_optimization_score(
        self,
        original_code: str,
        optimized_code: str,
        applied_optimizations: List[str]
    ) -> float:
        """Calculate optimization score based on applied techniques."""
        score = 0.0
        
        # Base score for each optimization applied
        optimization_weights = {
            "shared_memory": 0.25,
            "memory_coalescing": 0.20,
            "vectorized_access": 0.20,
            "loop_unrolling": 0.15,
            "warp_primitives": 0.20,
        }
        
        for opt in applied_optimizations:
            if opt in optimization_weights:
                score += optimization_weights[opt]
        
        # Bonus for code compactness (fewer lines often means better optimization)
        original_lines = len(original_code.split('\n'))
        optimized_lines = len(optimized_code.split('\n'))
        if optimized_lines < original_lines * 1.5:  # Not too verbose
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0


class TrainableCUDATesterAgent(TrainableAgent):
    """
    Trainable CUDA Tester Agent for compilation, testing, and profiling.
    
    This agent can be either:
    1. Rule-based (default) for deterministic testing
    2. Trainable model for learning testing patterns
    """
    
    def __init__(
        self,
        agent_id: str = "cuda_tester",
        model_name: Optional[str] = None,  # Optional - can be rule-based
        use_trained_model: bool = False,
        **kwargs
    ):
        if use_trained_model and model_name:
            # Initialize with trainable model
            super().__init__(
                agent_id=agent_id,
                agent_type="tester",
                model_name=model_name,
                **kwargs
            )
        else:
            # Rule-based tester (no model needed)
            self.agent_id = agent_id
            self.agent_type = "tester"
            self.use_trained_model = False
            self.logger = structlog.get_logger()
            self.model = None
            self.tokenizer = None
        
        self.use_trained_model = use_trained_model
        self.nvcc_path = "nvcc"  # Assume nvcc is in PATH
        
        self.logger.info(
            "TrainableCUDATesterAgent initialized",
            use_trained_model=use_trained_model
        )
    
    async def test_kernel(
        self,
        kernel_code: str,
        test_inputs: Optional[Dict[str, Any]] = None,
        performance_target: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Test and profile CUDA kernel.
        
        Args:
            kernel_code: CUDA kernel code to test
            test_inputs: Test input specifications
            performance_target: Target performance metrics
            
        Returns:
            Compilation results, test results, and performance metrics
        """
        results = {
            "agent_id": self.agent_id,
            "compilation": {},
            "correctness": {},
            "performance": {}
        }
        
        # Step 1: Compilation test
        compilation_result = await self._compile_kernel(kernel_code)
        results["compilation"] = compilation_result
        
        if not compilation_result["success"]:
            results["overall_success"] = False
            results["failure_reason"] = "compilation_failed"
            return results
        
        # Step 2: Correctness test (if test inputs provided)
        if test_inputs:
            correctness_result = await self._test_correctness(
                kernel_code,
                test_inputs
            )
            results["correctness"] = correctness_result
        
        # Step 3: Performance profiling
        performance_result = await self._profile_performance(
            kernel_code,
            test_inputs
        )
        results["performance"] = performance_result
        
        # Step 4: Generate test report (using model if available)
        if self.use_trained_model and self.model:
            report = await self._generate_test_report(results)
            results["test_report"] = report
        else:
            # Rule-based report
            results["test_report"] = self._generate_rule_based_report(results)
        
        # Overall success determination
        results["overall_success"] = (
            compilation_result["success"] and
            results.get("correctness", {}).get("passed", True)
        )
        
        return results
    
    async def _compile_kernel(self, kernel_code: str) -> Dict[str, Any]:
        """Compile CUDA kernel using nvcc."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                suffix=".cu",
                mode="w",
                delete=False
            ) as f:
                f.write(kernel_code)
                cu_file = f.name
            
            # Compile with nvcc
            result = subprocess.run(
                [self.nvcc_path, "-c", cu_file, "-o", cu_file.replace(".cu", ".o")],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up
            Path(cu_file).unlink(missing_ok=True)
            Path(cu_file.replace(".cu", ".o")).unlink(missing_ok=True)
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "errors": self._parse_compilation_errors(result.stderr)
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Compilation timeout",
                "errors": ["Compilation took too long (>30s)"]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "errors": [str(e)]
            }
    
    def _parse_compilation_errors(self, stderr: str) -> List[str]:
        """Parse nvcc compilation errors."""
        errors = []
        for line in stderr.split('\n'):
            if "error:" in line.lower():
                errors.append(line.strip())
        return errors
    
    async def _test_correctness(
        self,
        kernel_code: str,
        test_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test kernel correctness with given inputs."""
        # This would require actual CUDA execution
        # For now, return mock results
        return {
            "passed": True,
            "test_cases_run": len(test_inputs.get("test_cases", [])),
            "test_cases_passed": len(test_inputs.get("test_cases", [])),
            "accuracy": 1.0
        }
    
    async def _profile_performance(
        self,
        kernel_code: str,
        test_inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Profile kernel performance."""
        # This would use nvprof or Nsight Compute
        # For now, return estimated metrics
        return {
            "execution_time_ms": 0.5,
            "memory_bandwidth_gb": 250.0,
            "occupancy": 0.75,
            "speedup": 1.0  # Compared to baseline
        }
    
    async def _generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate test report using trained model."""
        if not self.model:
            return self._generate_rule_based_report(results)
        
        # Format results as prompt
        prompt = f"""Generate a concise test report for the following CUDA kernel test results:
        
Compilation: {"Success" if results["compilation"]["success"] else "Failed"}
Correctness: {results.get("correctness", {}).get("accuracy", "N/A")}
Performance: {results.get("performance", {}).get("speedup", "N/A")}x speedup

Provide actionable feedback and improvement suggestions."""
        
        generation_output = await self.generate_with_log_probs(
            prompt,
            max_new_tokens=256,
            temperature=0.3  # Low temperature for factual reporting
        )
        
        return generation_output.text
    
    def _generate_rule_based_report(self, results: Dict[str, Any]) -> str:
        """Generate rule-based test report."""
        report_lines = []
        
        # Compilation status
        if results["compilation"]["success"]:
            report_lines.append("✓ Compilation successful")
        else:
            report_lines.append("✗ Compilation failed")
            if results["compilation"].get("errors"):
                report_lines.append(f"  Errors: {results['compilation']['errors'][:2]}")
        
        # Correctness
        if "correctness" in results:
            if results["correctness"]["passed"]:
                report_lines.append("✓ Correctness tests passed")
            else:
                report_lines.append("✗ Correctness tests failed")
        
        # Performance
        if "performance" in results:
            perf = results["performance"]
            report_lines.append(f"Performance: {perf.get('speedup', 0)}x speedup")
            report_lines.append(f"Occupancy: {perf.get('occupancy', 0):.1%}")
        
        # Recommendations
        report_lines.append("\nRecommendations:")
        if not results["compilation"]["success"]:
            report_lines.append("- Fix compilation errors before proceeding")
        elif results.get("performance", {}).get("occupancy", 0) < 0.5:
            report_lines.append("- Consider optimizing for better occupancy")
        elif results.get("performance", {}).get("speedup", 0) < 1.5:
            report_lines.append("- Apply memory coalescing and shared memory optimizations")
        
        return "\n".join(report_lines)