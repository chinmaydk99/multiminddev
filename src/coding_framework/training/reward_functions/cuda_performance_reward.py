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
    """Sophisticated multi-turn reward function for CUDA kernel optimization conversations."""
    
    def __init__(
        self, 
        target_speedup: float = 2.0,
        correctness_weight: float = 0.3,
        performance_weight: float = 0.4,
        memory_efficiency_weight: float = 0.2,
        improvement_weight: float = 0.1,
        theoretical_memory_bandwidth: float = 900.0  # GB/s for modern GPUs
    ):
        super().__init__()
        self.target_speedup = target_speedup
        self.correctness_weight = correctness_weight
        self.performance_weight = performance_weight
        self.memory_efficiency_weight = memory_efficiency_weight
        self.improvement_weight = improvement_weight
        self.theoretical_memory_bandwidth = theoretical_memory_bandwidth
        
        self.compiler = CUDACompiler()
        self.benchmarker = CUDABenchmarker()
        self.logger = structlog.get_logger("cuda_performance_reward")
        
        # Cache for performance history to calculate improvement bonuses
        self.performance_cache: Dict[str, List[Dict[str, float]]] = {}
        
        # CUDA optimization technique patterns
        self.cuda_optimization_patterns = {
            "warp_shuffle": [r"__shfl_(?:up|down|xor)_sync", r"__shfl_sync"],
            "coalesced_access": [r"threadIdx\.x\s*\*\s*sizeof", r"blockIdx\.x\s*\*\s*blockDim\.x"],
            "shared_memory": [r"__shared__", r"extern\s+__shared__"],
            "bank_conflict_avoidance": [r"\[\s*threadIdx\.x\s*\+\s*\d+\s*\*\s*blockDim\.x\s*\]"],
            "loop_unrolling": [r"#pragma\s+unroll", r"for.*unroll"],
            "occupancy_tuning": [r"__launch_bounds__\s*\(\s*\d+"],
            "memory_prefetch": [r"__ldg\s*\(", r"__builtin_assume_aligned"],
            "warp_level_primitives": [r"__ballot_sync", r"__any_sync", r"__all_sync"]
        }
        
        self.logger.info(
            "Advanced CUDA performance reward initialized",
            target_speedup=target_speedup,
            correctness_weight=correctness_weight,
            performance_weight=performance_weight,
            memory_efficiency_weight=memory_efficiency_weight,
            improvement_weight=improvement_weight,
            theoretical_memory_bandwidth=theoretical_memory_bandwidth
        )
    
    async def _calculate_raw_reward(
        self,
        problem: str,
        generated_code: str,
        test_cases: list[dict[str, Any]],
        context: Optional[dict[str, Any]] = None
    ) -> float:
        """Calculate comprehensive reward based on CUDA kernel performance and optimization quality."""
        
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
            
            # Calculate enhanced reward components
            compilation_reward = await self._calculate_compilation_reward(latest_kernel)
            
            # Gate: Must compile successfully to proceed
            if compilation_reward <= 0:
                self.logger.debug("Kernel failed to compile", kernel_preview=latest_kernel[:200])
                return max(compilation_reward, -1.0)
            
            performance_reward = await self._calculate_performance_reward(
                latest_kernel, problem, test_cases
            )
            
            # Gate: Must pass functional tests to get performance rewards
            if performance_reward <= -0.5:  # Indicates functional failure
                return max(performance_reward, -0.8)
            
            memory_efficiency_reward = await self._calculate_memory_efficiency_reward(
                latest_kernel, problem, test_cases
            )
            
            optimization_technique_reward = self._calculate_optimization_technique_reward(latest_kernel)
            
            improvement_reward = self._calculate_improvement_reward(
                conversation_turns, performance_reward, context
            )
            
            # Multi-turn trajectory reward
            trajectory_reward = self._calculate_trajectory_reward(context)
            
            # Weighted combination with gated structure
            total_reward = (
                self.correctness_weight * compilation_reward +
                self.performance_weight * performance_reward + 
                self.memory_efficiency_weight * memory_efficiency_reward +
                self.improvement_weight * improvement_reward +
                0.05 * optimization_technique_reward +
                0.05 * trajectory_reward
            )
            
            # Normalize to [-1, 1] range
            final_reward = max(-1.0, min(1.0, total_reward))
            
            calculation_time = time.time() - start_time
            
            self.logger.info(
                "Advanced CUDA reward calculated",
                final_reward=final_reward,
                compilation_reward=compilation_reward,
                performance_reward=performance_reward,
                memory_efficiency_reward=memory_efficiency_reward,
                optimization_technique_reward=optimization_technique_reward,
                improvement_reward=improvement_reward,
                trajectory_reward=trajectory_reward,
                calculation_time=calculation_time
            )
            
            # Update performance cache for trajectory tracking
            self._update_performance_cache(context, {
                'total_reward': final_reward,
                'performance_reward': performance_reward,
                'memory_efficiency_reward': memory_efficiency_reward,
                'turn_number': context.get('turn_number', 0) if context else 0
            })
            
            return final_reward
            
        except Exception as e:
            calculation_time = time.time() - start_time
            error_msg = f"Advanced CUDA reward calculation failed: {str(e)}"
            
            self.logger.error(
                "Advanced CUDA reward calculation error",
                error=error_msg,
                calculation_time=calculation_time
            )
            return -1.0

    async def _calculate_memory_efficiency_reward(
        self,
        kernel_code: str,
        problem_description: str,
        test_cases: List[Dict[str, Any]]
    ) -> float:
        """Calculate reward based on memory bandwidth utilization and efficiency."""
        
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available for memory efficiency analysis")
            return 0.0
        
        try:
            # Compile kernel first
            compilation_result = await self.compiler.compile_kernel(
                kernel_code,
                f"mem_eff_kernel_{int(time.time())}"
            )
            
            if not compilation_result.success:
                return -0.3  # Penalty for non-compiling code
            
            # Create test inputs from test_cases
            test_inputs = self._prepare_test_inputs(test_cases)
            if not test_inputs:
                return 0.0  # No test data available
            
            # Get PyTorch baseline operation
            baseline_operation = self._get_pytorch_baseline(problem_description)
            
            # Benchmark kernel performance with detailed metrics
            benchmark_result = await self.benchmarker.benchmark_kernel(
                binary_path=compilation_result.binary_path,
                test_inputs=test_inputs,
                baseline_operation=baseline_operation,
                kernel_name=compilation_result.kernel_name
            )
            
            if not benchmark_result.success or not benchmark_result.functional_correct:
                return -0.5  # Penalty for failed or incorrect execution
            
            # Calculate memory bandwidth reward
            memory_bandwidth_reward = self._calculate_memory_bandwidth_reward(benchmark_result)
            
            # Calculate occupancy reward
            occupancy_reward = self._calculate_occupancy_reward(benchmark_result)
            
            # Calculate memory access pattern reward
            memory_pattern_reward = self._calculate_memory_pattern_reward(kernel_code, benchmark_result)
            
            # Resource usage penalties
            register_penalty = self._calculate_register_pressure_penalty(compilation_result)
            shared_memory_penalty = self._calculate_shared_memory_penalty(compilation_result)
            
            # Composite memory efficiency reward
            memory_efficiency = (
                0.4 * memory_bandwidth_reward +
                0.3 * occupancy_reward +
                0.3 * memory_pattern_reward -
                0.1 * register_penalty -
                0.1 * shared_memory_penalty
            )
            
            return max(0.0, min(1.0, memory_efficiency))
            
        except Exception as e:
            self.logger.warning("Memory efficiency reward calculation failed", error=str(e))
            return 0.0
    
    def _calculate_memory_bandwidth_reward(self, benchmark_result) -> float:
        """Reward high memory bandwidth utilization."""
        
        memory_throughput = getattr(benchmark_result, 'memory_throughput_gb_s', 0)
        
        if memory_throughput <= 0:
            return 0.0
        
        # Calculate bandwidth utilization ratio
        utilization_ratio = memory_throughput / self.theoretical_memory_bandwidth
        
        # Reward function: sigmoid-like curve with optimal range
        if utilization_ratio >= 0.8:
            return 1.0  # Excellent bandwidth utilization
        elif utilization_ratio >= 0.6:
            return 0.8 + 0.2 * ((utilization_ratio - 0.6) / 0.2)  # Good utilization
        elif utilization_ratio >= 0.4:
            return 0.5 + 0.3 * ((utilization_ratio - 0.4) / 0.2)  # Moderate utilization
        elif utilization_ratio >= 0.2:
            return 0.2 + 0.3 * ((utilization_ratio - 0.2) / 0.2)  # Low utilization
        else:
            return utilization_ratio  # Very low utilization
    
    def _calculate_occupancy_reward(self, benchmark_result) -> float:
        """Reward optimal occupancy (not necessarily 100%)."""
        
        occupancy = getattr(benchmark_result, 'occupancy_achieved', 0.5)
        
        # Optimal occupancy is usually 50-75%, not 100%
        if 0.5 <= occupancy <= 0.75:
            return 1.0  # Optimal occupancy range
        elif 0.4 <= occupancy < 0.5:
            return 0.8 + 0.2 * ((occupancy - 0.4) / 0.1)  # Slightly low but acceptable
        elif 0.75 < occupancy <= 0.85:
            return 0.9 - 0.1 * ((occupancy - 0.75) / 0.1)  # High but may indicate underutilization
        elif occupancy > 0.85:
            return 0.7  # Very high occupancy may limit flexibility
        else:
            return occupancy / 0.4  # Linear scaling for low occupancy
    
    def _calculate_memory_pattern_reward(self, kernel_code: str, benchmark_result) -> float:
        """Analyze memory access patterns and coalescing efficiency."""
        
        coalescing_reward = 0.0
        
        # Check for coalesced access patterns
        if self._has_coalesced_access_pattern(kernel_code):
            coalescing_reward += 0.5
        
        # Check for shared memory usage
        if self._uses_shared_memory_effectively(kernel_code):
            coalescing_reward += 0.3
        
        # Check for bank conflict avoidance
        if self._avoids_bank_conflicts(kernel_code):
            coalescing_reward += 0.2
        
        # Warp efficiency bonus from benchmark results
        warp_efficiency = getattr(benchmark_result, 'warp_efficiency', 0.8)
        warp_reward = warp_efficiency * 0.3
        
        return min(1.0, coalescing_reward + warp_reward)
    
    def _calculate_register_pressure_penalty(self, compilation_result) -> float:
        """Calculate penalty for high register pressure."""
        
        register_count = getattr(compilation_result, 'register_pressure', 0)
        
        if register_count <= 0:
            return 0.0
        
        # Modern GPUs have 255 registers per thread maximum
        if register_count > 200:
            return 1.0  # Severe penalty for very high register usage
        elif register_count > 128:
            return 0.5 + 0.5 * ((register_count - 128) / 72)  # Moderate to high penalty
        elif register_count > 64:
            return 0.2 * ((register_count - 64) / 64)  # Light penalty for moderate usage
        else:
            return 0.0  # No penalty for low register usage
    
    def _calculate_shared_memory_penalty(self, compilation_result) -> float:
        """Calculate penalty for excessive shared memory usage."""
        
        shared_memory_bytes = getattr(compilation_result, 'shared_memory_usage', 0)
        
        if shared_memory_bytes <= 0:
            return 0.0
        
        # Modern GPUs have 48KB shared memory per SM
        max_shared_memory = 48 * 1024  # 48KB
        
        if shared_memory_bytes > max_shared_memory:
            return 1.0  # Severe penalty for exceeding limits
        elif shared_memory_bytes > max_shared_memory * 0.8:
            ratio = (shared_memory_bytes - max_shared_memory * 0.8) / (max_shared_memory * 0.2)
            return 0.3 + 0.7 * ratio  # Progressive penalty for high usage
        elif shared_memory_bytes > max_shared_memory * 0.6:
            ratio = (shared_memory_bytes - max_shared_memory * 0.6) / (max_shared_memory * 0.2)
            return 0.1 * ratio  # Light penalty for moderate usage
        else:
            return 0.0  # No penalty for reasonable usage
    
    def _calculate_optimization_technique_reward(self, kernel_code: str) -> float:
        """Reward usage of advanced CUDA optimization techniques."""
        
        technique_rewards = 0.0
        
        for technique_name, patterns in self.cuda_optimization_patterns.items():
            if any(re.search(pattern, kernel_code, re.MULTILINE | re.IGNORECASE) for pattern in patterns):
                technique_rewards += 0.1
                self.logger.debug(f"Detected CUDA optimization: {technique_name}")
        
        # Bonus for using multiple techniques
        technique_count = len([
            technique for technique, patterns in self.cuda_optimization_patterns.items()
            if any(re.search(pattern, kernel_code, re.MULTILINE | re.IGNORECASE) for pattern in patterns)
        ])
        
        if technique_count >= 3:
            technique_rewards += 0.2  # Bonus for comprehensive optimization
        elif technique_count >= 2:
            technique_rewards += 0.1  # Bonus for good optimization
        
        return min(1.0, technique_rewards)
    
    def _calculate_trajectory_reward(self, context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate reward based on multi-turn improvement trajectory."""
        
        if not context:
            return 0.0
        
        conversation_id = context.get("conversation_id", "default")
        turn_number = context.get("turn_number", 0)
        
        if conversation_id not in self.performance_cache or turn_number < 1:
            return 0.0
        
        performance_history = self.performance_cache[conversation_id]
        
        if len(performance_history) < 2:
            return 0.0
        
        # Calculate improvement trajectory
        recent_performance = [entry['total_reward'] for entry in performance_history[-3:]]
        
        if len(recent_performance) >= 2:
            # Reward consistent improvement
            improvements = [recent_performance[i] - recent_performance[i-1] 
                          for i in range(1, len(recent_performance))]
            
            avg_improvement = sum(improvements) / len(improvements)
            
            # Turn discount factor - later turns get less reward
            turn_discount = 0.9 ** turn_number
            
            trajectory_reward = avg_improvement * turn_discount
            return max(-0.3, min(0.5, trajectory_reward))
        
        return 0.0
    
    def _update_performance_cache(self, context: Optional[Dict[str, Any]], metrics: Dict[str, float]) -> None:
        """Update performance cache for trajectory tracking."""
        
        if not context:
            return
        
        conversation_id = context.get("conversation_id", "default")
        
        if conversation_id not in self.performance_cache:
            self.performance_cache[conversation_id] = []
        
        self.performance_cache[conversation_id].append(metrics)
        
        # Keep only recent history to manage memory
        if len(self.performance_cache[conversation_id]) > 10:
            self.performance_cache[conversation_id] = self.performance_cache[conversation_id][-5:]
    
    def _has_coalesced_access_pattern(self, kernel_code: str) -> bool:
        """Check for memory coalescing patterns."""
        coalescing_patterns = [
            r"threadIdx\.x\s*\*\s*sizeof",
            r"blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x",
            r"tid\s*=\s*blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x"
        ]
        return any(re.search(pattern, kernel_code, re.MULTILINE) for pattern in coalescing_patterns)
    
    def _uses_shared_memory_effectively(self, kernel_code: str) -> bool:
        """Check for effective shared memory usage patterns."""
        shared_memory_patterns = [
            r"__shared__.*\[.*blockDim\.x.*\]",
            r"extern\s+__shared__.*\[\s*\]",
            r"__syncthreads\s*\(\s*\).*__shared__"
        ]
        return any(re.search(pattern, kernel_code, re.MULTILINE | re.DOTALL) for pattern in shared_memory_patterns)
    
    def _avoids_bank_conflicts(self, kernel_code: str) -> bool:
        """Check for bank conflict avoidance patterns."""
        bank_conflict_patterns = [
            r"\[\s*threadIdx\.x\s*\+\s*\d+\s*\*\s*blockDim\.x\s*\]",  # Stride access
            r"\[\s*threadIdx\.x\s*\*\s*\d+\s*\]",  # Non-unit stride
            r"#define\s+AVOID_BANK_CONFLICTS"  # Explicit bank conflict avoidance
        ]
        return any(re.search(pattern, kernel_code, re.MULTILINE) for pattern in bank_conflict_patterns)