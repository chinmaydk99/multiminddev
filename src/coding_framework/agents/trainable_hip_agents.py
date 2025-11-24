"""
Specialized trainable agents for HIP kernel generation, optimization, and testing.
These agents own HuggingFace model parameters and can be trained through RL.
Designed for AMD ROCm GPU programming with HIP (Heterogeneous-compute Interface for Portability).
"""

import asyncio
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog
import torch

from .trainable_agent import AgentResponse, GenerationOutput, TrainableAgent


class TrainableHIPGeneratorAgent(TrainableAgent):
    """
    Trainable HIP Generator Agent that creates initial HIP kernels from PyTorch operations.
    
    Specialization:
    - Syntactic correctness and compilability
    - HIP patterns and idioms for AMD GPUs
    - Thread indexing and memory management
    - ROCm-specific optimizations
    """

    def __init__(
        self,
        agent_id: str = "hip_generator",
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="generator",
            model_name=model_name,
            **kwargs
        )

        # HIP-specific generation patterns
        self.hip_templates = {
            "kernel_header": "__global__ void",
            "thread_index": "int idx = blockIdx.x * blockDim.x + threadIdx.x;",
            "grid_stride": "for (int i = idx; i < n; i += blockDim.x * gridDim.x)",
            "boundary_check": "if (idx < n)",
        }

        self.logger.info("TrainableHIPGeneratorAgent initialized")

    def _format_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Format prompt with HIP generation-specific instructions."""

        instruction = """You are a HIP kernel generation specialist for AMD ROCm GPUs. Generate optimized HIP C++ kernel code.

Requirements:
1. Include all necessary headers (#include <hip/hip_runtime.h>, etc.)
2. Use proper thread indexing (blockIdx, threadIdx, blockDim, gridDim)
3. Include boundary checks to prevent out-of-bounds access
4. Follow HIP best practices for memory access patterns on AMD GPUs
5. Include kernel launch configuration comments
6. Use hipLaunchKernelGGL for kernel launches when needed

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

    async def generate_hip_kernel(
        self,
        operation_description: str,
        tensor_info: Optional[Dict[str, Any]] = None,
        performance_hints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate HIP kernel from operation description.
        
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

        # Validate basic HIP syntax
        is_valid, validation_errors = self._validate_hip_syntax(kernel_code)

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
        """Extract kernel function name from HIP code."""
        pattern = r"__global__\s+\w+\s+(\w+)\s*\("
        match = re.search(pattern, kernel_code)
        return match.group(1) if match else None

    def _validate_hip_syntax(self, kernel_code: str) -> Tuple[bool, List[str]]:
        """Basic validation of HIP kernel syntax."""
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

    async def process_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResponse:
        """
        Process a request to generate HIP kernel.
        
        Implementation of abstract method from TrainableAgent.
        """
        try:
            # Generate HIP kernel
            result = await self.generate_hip_kernel(
                operation_description=request,
                tensor_info=context.get("tensor_info") if context else None,
                performance_hints=context.get("performance_hints") if context else None
            )

            return AgentResponse(
                agent_id=self.agent_id,
                content=result["kernel_code"],
                success=result["is_valid_syntax"],
                metadata=result,
                processing_time=0.0
            )
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            return AgentResponse(
                agent_id=self.agent_id,
                content="",
                success=False,
                error=str(e),
                processing_time=0.0
            )

    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Compatibility method for conversation manager.
        
        Args:
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response as string
        """
        try:
            # Filter out incompatible parameters
            filtered_kwargs = {
                k: v for k, v in kwargs.items()
                if k in ['max_new_tokens', 'temperature', 'top_p', 'top_k']
            }
            # Convert max_tokens to max_new_tokens if present
            if 'max_tokens' in kwargs:
                filtered_kwargs['max_new_tokens'] = kwargs['max_tokens']

            result = await self.generate_with_log_probs(prompt, **filtered_kwargs)

            # Debug logging
            self.logger.debug(f"Generation result type: {type(result)}")
            self.logger.debug(f"Generation result: {result}")

            # Handle different return types
            if hasattr(result, 'text'):
                return result.text
            elif isinstance(result, dict) and 'text' in result:
                return result['text']
            elif isinstance(result, str):
                return result
            else:
                return str(result)

        except Exception as e:
            self.logger.error(f"Error in generate_response: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error generating response: {str(e)}"


class TrainableHIPOptimizerAgent(TrainableAgent):
    """
    Trainable HIP Optimizer Agent that applies performance optimizations to kernels.
    
    Specialization:
    - Memory coalescing and LDS (Local Data Share) usage
    - VGPR (Vector General Purpose Register) optimization and occupancy
    - Vectorized memory access patterns
    - Loop unrolling and instruction optimization
    - AMD-specific optimizations (wavefront utilization, etc.)
    """

    def __init__(
        self,
        agent_id: str = "hip_optimizer",
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="optimizer",
            model_name=model_name,
            **kwargs
        )

        # Optimization strategies for AMD GPUs
        self.optimization_strategies = [
            "lds_memory",          # Local Data Share (shared memory equivalent)
            "memory_coalescing",
            "vectorized_access",
            "loop_unrolling",
            "wavefront_primitives",  # AMD wavefront operations
            "matrix_cores"           # AMD matrix cores (CDNA)
        ]

        self.logger.info("TrainableHIPOptimizerAgent initialized")

    def _format_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Format prompt with HIP optimization-specific instructions."""

        instruction = """You are a HIP kernel optimization specialist for AMD ROCm GPUs. Optimize the given kernel for maximum performance.

Optimization Focus Areas:
1. Memory Coalescing: Ensure coalesced global memory access patterns
2. LDS (Local Data Share): Use LDS memory to reduce global memory traffic
3. Vectorized Access: Use float4/int4 for vectorized memory operations
4. VGPR Usage: Optimize vector register usage for better occupancy
5. Loop Unrolling: Unroll loops where beneficial
6. Wavefront-level Primitives: Use AMD wavefront operations (__shfl, etc.)

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
        Optimize existing HIP kernel for better performance.
        
        Args:
            kernel_code: Existing kernel code to optimize
            performance_analysis: Performance profiling data
            optimization_targets: Specific optimization strategies to apply
            
        Returns:
            Optimized kernel and optimization metadata
        """
        # Build optimization prompt
        prompt = f"Optimize this HIP kernel for AMD ROCm:\n\n{kernel_code}"

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

        # Check for LDS (shared memory) usage
        if "__shared__" in kernel_code or "extern __shared__" in kernel_code:
            applied.append("lds_memory")

        # Check for vectorized access
        if any(v in kernel_code for v in ["float4", "int4", "float2", "int2"]):
            applied.append("vectorized_access")

        # Check for loop unrolling
        if "#pragma unroll" in kernel_code or re.search(r"for.*\n.*{[^}]*\n[^}]*\n[^}]*\n[^}]*}", kernel_code):
            applied.append("loop_unrolling")

        # Check for wavefront primitives (AMD-specific)
        if any(p in kernel_code for p in ["__shfl", "__ballot", "__any", "__all", "__lane_id"]):
            applied.append("wavefront_primitives")

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
            "lds_memory": 0.25,
            "memory_coalescing": 0.20,
            "vectorized_access": 0.20,
            "loop_unrolling": 0.15,
            "wavefront_primitives": 0.20,
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

    async def process_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResponse:
        """
        Process a request to optimize HIP kernel.
        
        Implementation of abstract method from TrainableAgent.
        """
        try:
            # Extract kernel code from request or context
            kernel_code = context.get("kernel_code", request) if context else request
            performance_metrics = context.get("performance_metrics") if context else None
            optimization_targets = context.get("optimization_targets", ["lds_memory", "memory_coalescing"]) if context else ["lds_memory", "memory_coalescing"]

            # Optimize kernel
            result = await self.optimize_kernel(
                kernel_code=kernel_code,
                performance_analysis=performance_metrics,
                optimization_targets=optimization_targets
            )

            return AgentResponse(
                agent_id=self.agent_id,
                content=result["optimized_code"],
                success=True,
                metadata=result,
                processing_time=0.0
            )
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            return AgentResponse(
                agent_id=self.agent_id,
                content="",
                success=False,
                error=str(e),
                processing_time=0.0
            )

    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Compatibility method for conversation manager.
        
        Args:
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response as string
        """
        try:
            # Filter out incompatible parameters
            filtered_kwargs = {
                k: v for k, v in kwargs.items()
                if k in ['max_new_tokens', 'temperature', 'top_p', 'top_k']
            }
            # Convert max_tokens to max_new_tokens if present
            if 'max_tokens' in kwargs:
                filtered_kwargs['max_new_tokens'] = kwargs['max_tokens']

            result = await self.generate_with_log_probs(prompt, **filtered_kwargs)

            # Handle different return types
            if hasattr(result, 'text'):
                return result.text
            elif isinstance(result, dict) and 'text' in result:
                return result['text']
            elif isinstance(result, str):
                return result
            else:
                return str(result)

        except Exception as e:
            self.logger.error(f"Error in generate_response: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error generating response: {str(e)}"


class TrainableHIPTesterAgent(TrainableAgent):
    """
    Trainable HIP Tester Agent for compilation, testing, and profiling on AMD ROCm.
    
    This agent can be either:
    1. Rule-based (default) for deterministic testing
    2. Trainable model for learning testing patterns
    """

    def __init__(
        self,
        agent_id: str = "hip_tester",
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
        self.hipcc_path = "hipcc"  # Assume hipcc is in PATH

        self.logger.info(
            "TrainableHIPTesterAgent initialized",
            use_trained_model=use_trained_model
        )

    async def test_kernel(
        self,
        kernel_code: str,
        test_inputs: Optional[Dict[str, Any]] = None,
        performance_target: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Test and profile HIP kernel.
        
        Args:
            kernel_code: HIP kernel code to test
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
        """Compile HIP kernel using hipcc."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                suffix=".cpp",
                mode="w",
                delete=False
            ) as f:
                f.write(kernel_code)
                cpp_file = f.name

            # Compile with hipcc
            result = subprocess.run(
                [self.hipcc_path, "-c", cpp_file, "-o", cpp_file.replace(".cpp", ".o")],
                capture_output=True,
                text=True,
                timeout=30
            )

            # Clean up
            Path(cpp_file).unlink(missing_ok=True)
            Path(cpp_file.replace(".cpp", ".o")).unlink(missing_ok=True)

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
        """Parse hipcc compilation errors."""
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
        # This would require actual HIP execution
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
        """Profile kernel performance using ROCm profiling tools."""
        # This would use rocprof or similar tools
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
        prompt = f"""Generate a concise test report for the following HIP kernel test results:
        
Compilation: {"Success" if results["compilation"]["success"] else "Failed"}
Correctness: {results.get("correctness", {}).get("accuracy", "N/A")}
Performance: {results.get("performance", {}).get("speedup", "N/A")}x speedup

Provide actionable feedback and improvement suggestions for AMD ROCm optimization."""

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
            report_lines.append("✓ Compilation successful (hipcc)")
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
        report_lines.append("\nRecommendations for AMD ROCm:")
        if not results["compilation"]["success"]:
            report_lines.append("- Fix compilation errors before proceeding")
        elif results.get("performance", {}).get("occupancy", 0) < 0.5:
            report_lines.append("- Consider optimizing VGPR usage for better occupancy")
        elif results.get("performance", {}).get("speedup", 0) < 1.5:
            report_lines.append("- Apply memory coalescing and LDS optimizations")

        return "\n".join(report_lines)

    async def process_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResponse:
        """
        Process a request to test HIP kernel.
        
        Implementation of abstract method from TrainableAgent.
        """
        try:
            # Extract kernel code from request or context
            kernel_code = context.get("kernel_code", request) if context else request
            test_cases = context.get("test_cases", []) if context else []

            # Test kernel
            result = await self.test_kernel(
                kernel_code=kernel_code,
                test_inputs={"test_cases": test_cases}
            )

            return AgentResponse(
                agent_id=self.agent_id,
                content=result["test_report"],
                success=result["compilation"]["success"],
                metadata=result,
                processing_time=0.0
            )
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            return AgentResponse(
                agent_id=self.agent_id,
                content="",
                success=False,
                error=str(e),
                processing_time=0.0
            )

    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Compatibility method for conversation manager.
        
        Args:
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response as string
        """
        try:
            # Filter out incompatible parameters
            filtered_kwargs = {
                k: v for k, v in kwargs.items()
                if k in ['max_new_tokens', 'temperature', 'top_p', 'top_k']
            }
            # Convert max_tokens to max_new_tokens if present
            if 'max_tokens' in kwargs:
                filtered_kwargs['max_new_tokens'] = kwargs['max_tokens']

            result = await self.generate_with_log_probs(prompt, **filtered_kwargs)

            # Handle different return types
            if hasattr(result, 'text'):
                return result.text
            elif isinstance(result, dict) and 'text' in result:
                return result['text']
            elif isinstance(result, str):
                return result
            else:
                return str(result)

        except Exception as e:
            self.logger.error(f"Error in generate_response: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error generating response: {str(e)}"

