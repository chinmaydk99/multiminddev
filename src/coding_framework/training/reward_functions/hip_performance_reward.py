from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import structlog


@dataclass
class RewardComponents:
    """Individual components of the reward function."""
    compilation_success: float = 0.0
    functional_correctness: float = 0.0
    performance_score: float = 0.0
    efficiency_score: float = 0.0
    code_quality_score: float = 0.0
    improvement_bonus: float = 0.0

    def total_score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted total score."""
        return (
            weights.get("compilation", 0.3) * self.compilation_success +
            weights.get("correctness", 0.3) * self.functional_correctness +
            weights.get("performance", 0.25) * self.performance_score +
            weights.get("efficiency", 0.1) * self.efficiency_score +
            weights.get("code_quality", 0.05) * self.code_quality_score
        )


class HIPPerformanceReward:
    """Comprehensive HIP performance reward function for AMD ROCm."""

    def __init__(
        self,
        target_speedup: float = 2.0,
        max_speedup_reward: float = 4.0,
        memory_bandwidth_weight: float = 0.3,
        compute_efficiency_weight: float = 0.4,
        occupancy_weight: float = 0.3,
        reward_weights: Optional[Dict[str, float]] = None
    ):
        self.target_speedup = target_speedup
        self.max_speedup_reward = max_speedup_reward
        self.memory_bandwidth_weight = memory_bandwidth_weight
        self.compute_efficiency_weight = compute_efficiency_weight
        self.occupancy_weight = occupancy_weight

        # Default reward component weights
        self.reward_weights = reward_weights or {
            "compilation": 0.30,    # Must compile successfully
            "correctness": 0.30,    # Must produce correct results
            "performance": 0.25,    # Performance improvements
            "efficiency": 0.10,     # Resource utilization efficiency
            "code_quality": 0.05    # Code quality metrics
        }

        self.logger = structlog.get_logger()

    async def calculate_reward(
        self,
        problem: Dict[str, Any],
        generated_code: str,
        compilation_result,
        benchmark_result,
        context: Dict[str, Any]
    ) -> Tuple[float, RewardComponents]:
        """
        Calculate comprehensive HIP performance reward.
        
        Returns:
            Tuple of (total_reward, reward_components)
        """

        components = RewardComponents()

        # 1. Compilation Success (30%)
        components.compilation_success = self._calculate_compilation_reward(
            compilation_result, generated_code
        )

        # 2. Functional Correctness (30%)
        components.functional_correctness = self._calculate_correctness_reward(
            benchmark_result
        )

        # 3. Performance Score (25%)
        components.performance_score = self._calculate_performance_reward(
            benchmark_result, problem.get("baseline_performance")
        )

        # 4. Efficiency Score (10%)
        components.efficiency_score = self._calculate_efficiency_reward(
            compilation_result, benchmark_result
        )

        # 5. Code Quality Score (5%)
        components.code_quality_score = self._calculate_code_quality_reward(
            generated_code, compilation_result
        )

        # 6. Improvement Bonus (applied to final score)
        components.improvement_bonus = self._calculate_improvement_bonus(context)

        # Calculate final weighted score
        base_score = components.total_score(self.reward_weights)
        final_score = base_score * (1.0 + components.improvement_bonus)

        # Clamp to reasonable range
        final_score = max(0.0, min(2.0, final_score))

        self.logger.debug(
            "HIP reward calculated",
            total_reward=final_score,
            compilation=components.compilation_success,
            correctness=components.functional_correctness,
            performance=components.performance_score,
            efficiency=components.efficiency_score,
            code_quality=components.code_quality_score,
            improvement_bonus=components.improvement_bonus
        )

        return final_score, components

    def _calculate_compilation_reward(
        self,
        compilation_result,
        generated_code: str
    ) -> float:
        """Calculate reward for compilation success and quality."""

        if not compilation_result or not compilation_result.success:
            return 0.0

        base_reward = 1.0  # Full points for successful compilation

        # Bonus for clean compilation (no warnings)
        if compilation_result.compilation_warnings:
            warning_penalty = min(len(compilation_result.compilation_warnings) * 0.1, 0.3)
            base_reward -= warning_penalty

        # Bonus for good VGPR (Vector General Purpose Register) usage
        if compilation_result.register_pressure > 0:
            # Optimal VGPR usage varies by AMD GPU architecture
            # Typically 32-64 VGPRs is good for occupancy
            if 16 <= compilation_result.register_pressure <= 64:
                base_reward += 0.1
            elif compilation_result.register_pressure > 128:
                base_reward -= 0.2  # High VGPR pressure reduces occupancy

        # Bonus for reasonable LDS (Local Data Share) usage
        if compilation_result.shared_memory_usage > 0:
            # Using LDS is generally good for optimization
            base_reward += 0.1

        return max(0.0, min(1.0, base_reward))

    def _calculate_correctness_reward(self, benchmark_result) -> float:
        """Calculate reward for functional correctness."""

        if not benchmark_result or not benchmark_result.success:
            return 0.0

        if not benchmark_result.functional_correct:
            return 0.0

        # Base reward for correctness
        correctness_reward = 1.0

        # Bonus for high numerical accuracy
        if hasattr(benchmark_result, 'numerical_accuracy'):
            accuracy = benchmark_result.numerical_accuracy
            if accuracy >= 0.99:
                correctness_reward = 1.0
            elif accuracy >= 0.95:
                correctness_reward = 0.9
            elif accuracy >= 0.90:
                correctness_reward = 0.7
            else:
                correctness_reward = 0.5  # Low accuracy

        return correctness_reward

    def _calculate_performance_reward(
        self,
        benchmark_result,
        baseline_performance: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate reward for performance improvements."""

        if not benchmark_result or not benchmark_result.success or not benchmark_result.functional_correct:
            return 0.0

        performance_reward = 0.0

        # Primary metric: speedup vs PyTorch
        if hasattr(benchmark_result, 'speedup_vs_torch'):
            speedup = benchmark_result.speedup_vs_torch

            if speedup >= self.target_speedup:
                # Achieved target speedup
                performance_reward = 1.0

                # Bonus for exceeding target
                if speedup > self.target_speedup:
                    excess_speedup = min(speedup - self.target_speedup, self.max_speedup_reward - self.target_speedup)
                    bonus = excess_speedup / (self.max_speedup_reward - self.target_speedup)
                    performance_reward += 0.5 * bonus  # Up to 50% bonus
            else:
                # Partial credit for partial speedup
                performance_reward = speedup / self.target_speedup

        # Secondary metrics: memory bandwidth and compute efficiency
        secondary_score = 0.0

        if hasattr(benchmark_result, 'memory_bandwidth_gb_s'):
            # Good memory bandwidth for AMD GPUs (HBM2/HBM3) is 500-1600 GB/s
            bandwidth = benchmark_result.memory_bandwidth_gb_s
            bandwidth_score = min(bandwidth / 1000.0, 1.0)  # Normalize to 1000 GB/s (MI250 level)
            secondary_score += self.memory_bandwidth_weight * bandwidth_score

        if hasattr(benchmark_result, 'compute_throughput_gflops'):
            # Good compute throughput varies by operation and GPU
            throughput = benchmark_result.compute_throughput_gflops
            throughput_score = min(throughput / 1000.0, 1.0)  # Normalize to 1 TFLOPS
            secondary_score += self.compute_efficiency_weight * throughput_score

        if hasattr(benchmark_result, 'occupancy_percent'):
            # Good occupancy is typically 50-100% (AMD wavefront occupancy)
            occupancy = benchmark_result.occupancy_percent / 100.0
            occupancy_score = min(occupancy / 0.75, 1.0)  # Normalize to 75% occupancy
            secondary_score += self.occupancy_weight * occupancy_score

        # Combine primary and secondary metrics
        total_performance_reward = 0.7 * performance_reward + 0.3 * secondary_score

        return max(0.0, min(1.5, total_performance_reward))  # Allow up to 150% for exceptional performance

    def _calculate_efficiency_reward(self, compilation_result, benchmark_result) -> float:
        """Calculate reward for resource utilization efficiency."""

        efficiency_score = 0.0

        # Memory efficiency
        if benchmark_result and hasattr(benchmark_result, 'peak_memory_usage_mb'):
            # Lower memory usage is generally better (efficient kernels)
            memory_mb = benchmark_result.peak_memory_usage_mb
            if memory_mb > 0:
                # Normalize based on problem size - this is simplified
                memory_efficiency = 1.0 / (1.0 + memory_mb / 1000.0)  # Penalty for using >1GB
                efficiency_score += 0.4 * memory_efficiency

        # VGPR efficiency (AMD-specific)
        if compilation_result and compilation_result.register_pressure > 0:
            vgprs = compilation_result.register_pressure
            if vgprs <= 32:
                register_efficiency = 1.0  # Excellent for occupancy
            elif vgprs <= 64:
                register_efficiency = 0.8
            elif vgprs <= 96:
                register_efficiency = 0.6
            else:
                register_efficiency = 0.2  # Very high VGPR usage

            efficiency_score += 0.3 * register_efficiency

        # Compilation time efficiency
        if compilation_result and compilation_result.compilation_time > 0:
            compile_time = compilation_result.compilation_time
            if compile_time <= 5.0:
                compile_efficiency = 1.0
            elif compile_time <= 15.0:
                compile_efficiency = 0.8
            elif compile_time <= 30.0:
                compile_efficiency = 0.6
            else:
                compile_efficiency = 0.2  # Very slow compilation

            efficiency_score += 0.3 * compile_efficiency

        return max(0.0, min(1.0, efficiency_score))

    def _calculate_code_quality_reward(self, generated_code: str, compilation_result) -> float:
        """Calculate reward for code quality and best practices in HIP."""

        quality_score = 0.5  # Start with neutral score

        # Check for HIP best practices
        code_lower = generated_code.lower()

        # Positive indicators
        if '__syncthreads()' in generated_code:
            quality_score += 0.1  # Using synchronization

        if '__shared__' in generated_code:
            quality_score += 0.1  # Using LDS (Local Data Share)

        if 'coalesced' in code_lower or 'coalesce' in code_lower:
            quality_score += 0.1  # Mentions memory coalescing

        if 'occupancy' in code_lower:
            quality_score += 0.05  # Considers occupancy

        # Check for HIP error handling
        if 'hipGetLastError' in generated_code or 'HIP_CHECK' in generated_code:
            quality_score += 0.1  # Good error handling

        # Check for HIP runtime includes
        if 'hip/hip_runtime.h' in generated_code:
            quality_score += 0.05  # Proper HIP includes

        # Negative indicators
        if 'goto' in code_lower:
            quality_score -= 0.1  # Avoid goto

        if generated_code.count('malloc') > 0 and generated_code.count('free') == 0:
            quality_score -= 0.2  # Memory leaks

        # AMD-specific quality checks
        if 'hipMalloc' in generated_code and 'hipFree' not in generated_code:
            quality_score -= 0.15  # HIP memory leak

        # Check compilation warnings for quality issues
        if compilation_result and compilation_result.compilation_warnings:
            for warning in compilation_result.compilation_warnings:
                if 'unused' in warning.lower():
                    quality_score -= 0.05  # Unused variables
                if 'deprecated' in warning.lower():
                    quality_score -= 0.1  # Using deprecated features

        return max(0.0, min(1.0, quality_score))

    def _calculate_improvement_bonus(self, context: Dict[str, Any]) -> float:
        """Calculate bonus for improvements over previous turns."""

        improvement_bonus = 0.0

        # Get previous performance if available
        previous_performance = context.get("previous_performance", {})
        current_performance = context.get("current_performance", {})

        if previous_performance and current_performance:
            # Speedup improvement
            prev_speedup = previous_performance.get("speedup", 0.0)
            curr_speedup = current_performance.get("speedup", 0.0)

            if curr_speedup > prev_speedup:
                speedup_improvement = (curr_speedup - prev_speedup) / max(prev_speedup, 0.1)
                improvement_bonus += min(speedup_improvement * 0.1, 0.2)  # Up to 20% bonus

        # Turn efficiency bonus (fewer turns to solve is better)
        turn_number = context.get("turn_number", 0)
        max_turns = context.get("max_turns", 5)

        if turn_number > 0:
            turn_efficiency = (max_turns - turn_number) / max_turns
            improvement_bonus += 0.1 * turn_efficiency  # Up to 10% bonus for efficiency

        return improvement_bonus

    def get_reward_explanation(self, components: RewardComponents) -> str:
        """Generate human-readable explanation of reward calculation."""

        explanation = f"""
HIP/ROCm Reward Breakdown:
- Compilation Success: {components.compilation_success:.3f}
- Functional Correctness: {components.functional_correctness:.3f}
- Performance Score: {components.performance_score:.3f}
- Efficiency Score: {components.efficiency_score:.3f}
- Code Quality Score: {components.code_quality_score:.3f}
- Improvement Bonus: {components.improvement_bonus:.3f}

Total Score: {components.total_score(self.reward_weights):.3f}
"""
        return explanation.strip()

