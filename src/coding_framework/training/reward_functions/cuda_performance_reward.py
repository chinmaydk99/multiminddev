import re
import time
import asyncio
from typing import Optional, Dict, Any, List
import structlog

from .base_reward import BaseRewardFunction
from ...cuda.compiler import CUDACompiler
from ...cuda.benchmarker import CUDABenchmarker

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class CUDAPerformanceReward(BaseRewardFunction):
    """Multi-turn reward function for CUDA kernel optimization conversations."""
    
    def __init__(
        self, 
        target_speedup: float = 2.0,
        correctness_weight: float = 0.4,
        performance_weight: float = 0.4,
        improvement_weight: float = 0.2
    ):
        super().__init__()
        self.target_speedup = target_speedup
        self.correctness_weight = correctness_weight
        self.performance_weight = performance_weight
        self.improvement_weight = improvement_weight
        
        self.compiler = CUDACompiler()
        self.benchmarker = CUDABenchmarker()
        self.logger = structlog.get_logger("cuda_performance_reward")
        
        # Cache for performance history to calculate improvement bonuses
        self.performance_cache: Dict[str, List[float]] = {}
        
        self.logger.info(
            "CUDA performance reward initialized",
            target_speedup=target_speedup,
            correctness_weight=correctness_weight,
            performance_weight=performance_weight,
            improvement_weight=improvement_weight
        )
    
    async def calculate_reward(
        self,
        problem: str,  # PyTorch operation description
        generated_code: str,  # Current conversation state (full conversation)
        test_cases: List[Dict[str, Any]],  # Test inputs and expected outputs
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate reward based on kernel performance improvement across conversation."""
        
        start_time = time.time()
        
        try:
            # Extract conversation turns from generated_code
            conversation_turns = self._parse_conversation_turns(generated_code)
            
            if not conversation_turns:
                return -1.0  # No valid conversation
            
            # Get latest kernel from conversation
            latest_kernel = self._extract_latest_kernel(conversation_turns)
            
            if not latest_kernel:
                return -0.8  # No valid kernel generated
            
            # Calculate base reward components
            compilation_reward = await self._calculate_compilation_reward(latest_kernel)
            performance_reward = await self._calculate_performance_reward(
                latest_kernel, problem, test_cases
            )
            improvement_reward = self._calculate_improvement_reward(
                conversation_turns, performance_reward, context
            )
            
            # Weighted combination
            total_reward = (
                self.correctness_weight * compilation_reward +
                self.performance_weight * performance_reward + 
                self.improvement_weight * improvement_reward
            )
            
            # Normalize to [-1, 1] range
            final_reward = max(-1.0, min(1.0, total_reward))
            
            calculation_time = time.time() - start_time
            
            self.logger.info(
                "CUDA reward calculated",
                final_reward=final_reward,
                compilation_reward=compilation_reward,
                performance_reward=performance_reward,
                improvement_reward=improvement_reward,
                calculation_time=calculation_time
            )
            
            return final_reward
            
        except Exception as e:
            calculation_time = time.time() - start_time
            error_msg = f"CUDA reward calculation failed: {str(e)}"
            
            self.logger.error(
                "CUDA reward calculation error",
                error=error_msg,
                calculation_time=calculation_time
            )
            return -1.0

    
    async def _calculate_raw_reward(
        self,
        problem: str,
        generated_code: str,
        test_cases: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate raw reward - delegates to main calculate_reward method."""
        return await self.calculate_reward(problem, generated_code, test_cases, context)
    
    async def _calculate_compilation_reward(self, kernel_code: str) -> float:
        """Calculate reward based on kernel compilation success."""
        
        try:
            # Attempt to compile the kernel
            compilation_result = await self.compiler.compile_kernel(
                kernel_code, 
                f"reward_kernel_{int(time.time())}"
            )
            
            if compilation_result.success:
                # Base reward for successful compilation
                reward = 1.0
                
                # Bonus for good coding practices
                if self._has_bounds_checking(kernel_code):
                    reward += 0.2
                
                if self._has_error_handling(kernel_code):
                    reward += 0.1
                
                if self._has_proper_headers(kernel_code):
                    reward += 0.1
                
                return min(reward, 1.0)
            else:
                # Partial credit for syntax that's close to compiling
                syntax_score = self._assess_syntax_quality(kernel_code)
                return -0.5 + 0.3 * syntax_score
                
        except Exception as e:
            self.logger.warning("Compilation reward calculation failed", error=str(e))
            return -1.0
    
    async def _calculate_performance_reward(
        self,
        kernel_code: str,
        problem_description: str,
        test_cases: List[Dict[str, Any]]
    ) -> float:
        """Calculate reward based on kernel performance."""
        
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available for performance benchmarking")
            return 0.0
        
        try:
            # Compile kernel first
            compilation_result = await self.compiler.compile_kernel(
                kernel_code,
                f"perf_kernel_{int(time.time())}"
            )
            
            if not compilation_result.success:
                return -0.8  # Heavy penalty for non-compiling code
            
            # Create test inputs from test_cases
            test_inputs = self._prepare_test_inputs(test_cases)
            if not test_inputs:
                return 0.0  # No test data available
            
            # Get PyTorch baseline operation
            baseline_operation = self._get_pytorch_baseline(problem_description)
            
            # Benchmark kernel performance
            benchmark_result = await self.benchmarker.benchmark_kernel(
                binary_path=compilation_result.binary_path,
                test_inputs=test_inputs,
                baseline_operation=baseline_operation,
                kernel_name=compilation_result.kernel_name
            )
            
            if not benchmark_result.success:
                return -0.3  # Moderate penalty for runtime failures
            
            if not benchmark_result.functional_correct:
                return -0.6  # Heavy penalty for incorrect results
            
            # Calculate performance reward based on speedup
            speedup_ratio = benchmark_result.speedup_ratio or 1.0
            
            # Sigmoid-like scaling: reward approaches 1 as speedup approaches target
            performance_reward = (speedup_ratio / self.target_speedup) - 1.0
            
            # Cap the reward to prevent extreme values
            performance_reward = max(-1.0, min(performance_reward, 1.0))
            
            return performance_reward
            
        except Exception as e:
            self.logger.warning("Performance reward calculation failed", error=str(e))
            return -0.5
    
    def _calculate_improvement_reward(
        self,
        conversation_turns: List[str],
        current_performance: float,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate bonus reward for improvement across conversation turns."""
        
        if len(conversation_turns) < 2:
            return 0.0  # No improvement possible with single turn
        
        # Track performance improvement across turns
        conversation_id = context.get("conversation_id", "default") if context else "default"
        
        if conversation_id not in self.performance_cache:
            self.performance_cache[conversation_id] = []
        
        performance_history = self.performance_cache[conversation_id]
        performance_history.append(current_performance)
        
        # Keep only recent history
        if len(performance_history) > 10:
            performance_history = performance_history[-10:]
            self.performance_cache[conversation_id] = performance_history
        
        # Calculate improvement from previous turns
        if len(performance_history) >= 2:
            previous_performance = performance_history[-2]
            improvement = current_performance - previous_performance
            
            # Reward significant improvements
            if improvement > 0.1:
                return min(improvement * 2.0, 1.0)  # Up to 1.0 bonus
            elif improvement > 0.0:
                return improvement  # Small positive reward for minor improvements
            else:
                return max(improvement * 0.5, -0.2)  # Small penalty for regression
        
        return 0.0
    
    def _parse_conversation_turns(self, generated_code: str) -> List[str]:
        """Extract conversation turns from generated code string."""
        
        # Look for agent role markers or turn separators
        turn_patterns = [
            r"Agent: (cuda_generator|cuda_optimizer|cuda_tester)",
            r"Turn \d+:",
            r"===.*?===",
            r"---.*?---"
        ]
        
        turns = []
        current_turn = []
        
        for line in generated_code.split('\n'):
            # Check if this line starts a new turn
            is_new_turn = any(re.search(pattern, line, re.IGNORECASE) for pattern in turn_patterns)
            
            if is_new_turn and current_turn:
                # Save previous turn
                turns.append('\n'.join(current_turn))
                current_turn = []
            
            current_turn.append(line)
        
        # Add final turn
        if current_turn:
            turns.append('\n'.join(current_turn))
        
        # If no clear turns found, treat entire code as one turn
        if not turns:
            turns = [generated_code]
        
        return turns
    
    def _extract_latest_kernel(self, conversation_turns: List[str]) -> Optional[str]:
        """Extract the most recent CUDA kernel from conversation."""
        
        for turn in reversed(conversation_turns):
            kernel = self._extract_cuda_code_block(turn)
            if kernel:
                return kernel
        
        return None
    
    def _extract_cuda_code_block(self, text: str) -> Optional[str]:
        """Extract CUDA code block from text."""
        
        # Look for code blocks with CUDA markers
        cuda_patterns = [
            r"```(?:cuda|c\+\+|cpp)?\s*(.*?)```",
            r"__global__.*?(?=\n\n|\n(?=\w)|\Z)",
            r"#include.*?(?=\n\n|\Z)"
        ]
        
        for pattern in cuda_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if self._is_cuda_code(match):
                    return match.strip()
        
        # Fallback: if text contains CUDA keywords, assume it's CUDA code
        if self._is_cuda_code(text):
            return text.strip()
        
        return None
    
    def _is_cuda_code(self, code: str) -> bool:
        """Check if code block contains CUDA-specific syntax."""
        cuda_indicators = [
            "__global__", "__device__", "__host__",
            "blockIdx", "threadIdx", "blockDim", "gridDim",
            "cudaMalloc", "cudaMemcpy", "cudaFree",
            "__shared__", "__syncthreads",
            "<<<", ">>>"
        ]
        
        code_lower = code.lower()
        return any(indicator.lower() in code_lower for indicator in cuda_indicators)
    
    def _has_bounds_checking(self, kernel_code: str) -> bool:
        """Check if kernel has proper bounds checking."""
        bounds_patterns = [
            r"if\s*\(\s*\w+\s*<\s*\w+\s*\)",
            r"idx\s*<\s*\w+",
            r"threadIdx\.x\s*<\s*\w+"
        ]
        return any(re.search(pattern, kernel_code) for pattern in bounds_patterns)
    
    def _has_error_handling(self, kernel_code: str) -> bool:
        """Check if code has error handling."""
        error_patterns = [
            r"CUDA_CHECK",
            r"cudaGetLastError",
            r"cudaDeviceSynchronize"
        ]
        return any(re.search(pattern, kernel_code) for pattern in error_patterns)
    
    def _has_proper_headers(self, kernel_code: str) -> bool:
        """Check if code has proper CUDA headers."""
        header_patterns = [
            r"#include\s*<cuda_runtime\.h>",
            r"#include\s*<device_launch_parameters\.h>"
        ]
        return any(re.search(pattern, kernel_code) for pattern in header_patterns)
    
    def _assess_syntax_quality(self, kernel_code: str) -> float:
        """Assess syntax quality for non-compiling code."""
        quality_score = 0.0
        
        # Check for basic CUDA syntax elements
        if "__global__" in kernel_code:
            quality_score += 0.3
        
        if any(keyword in kernel_code for keyword in ["threadIdx", "blockIdx"]):
            quality_score += 0.2
        
        if "<<<" in kernel_code and ">>>" in kernel_code:
            quality_score += 0.2
        
        if "#include" in kernel_code:
            quality_score += 0.1
        
        if "{" in kernel_code and "}" in kernel_code:
            quality_score += 0.2
        
        return min(quality_score, 1.0)
    
    def _prepare_test_inputs(self, test_cases: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """Convert test cases to PyTorch tensors."""
        if not TORCH_AVAILABLE or not test_cases:
            return []
        
        inputs = []
        for test_case in test_cases:
            # Handle different input formats
            if "tensor" in test_case:
                inputs.append(test_case["tensor"])
            elif "shape" in test_case and "dtype" in test_case:
                tensor = torch.randn(test_case["shape"], dtype=getattr(torch, test_case["dtype"]))
                inputs.append(tensor)
            elif "data" in test_case:
                tensor = torch.tensor(test_case["data"])
                inputs.append(tensor)
        
        return inputs
    
    def _get_pytorch_baseline(self, problem_description: str) -> Optional[callable]:
        """Convert problem description to PyTorch baseline operation."""
        
        if not TORCH_AVAILABLE:
            return None
        
        # Map common operations to PyTorch equivalents
        operation_map = {
            "matrix multiplication": lambda a, b: torch.mm(a, b),
            "matmul": lambda a, b: torch.mm(a, b),
            "element_wise_add": lambda a, b: torch.add(a, b),
            "element_wise_addition": lambda a, b: torch.add(a, b),
            "softmax": lambda x: torch.softmax(x, dim=-1),
            "relu": lambda x: torch.relu(x),
            "convolution": lambda x, w: torch.conv2d(x, w),
            "conv2d": lambda x, w: torch.conv2d(x, w),
            "vector_add": lambda a, b: torch.add(a, b),
            "elementwise": lambda a, b: torch.add(a, b),  # Default elementwise operation
            "scaling": lambda x: x * 2.0,  # For element-wise scaling
            "multiply": lambda x: x * 2.0   # For element-wise multiplication
        }
        
        problem_lower = problem_description.lower()
        
        for op_name, torch_op in operation_map.items():
            if op_name in problem_lower:
                return torch_op
        
        # Default: element-wise scaling
        return lambda x: x * 2.0