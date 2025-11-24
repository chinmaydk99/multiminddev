"""
Execution-based reward function.

Calculates reward from real kernel execution:
1. Compilation success (gate)
2. Numerical correctness (gate)
3. Performance speedup (main signal)
4. Efficiency bonus (secondary signal)
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import structlog

from src.execution.sandbox import ExecutionResult

logger = structlog.get_logger()


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    
    # Reward weights
    compilation_weight: float = 0.0  # Binary gate, not weighted
    correctness_weight: float = 0.0  # Binary gate, not weighted
    performance_weight: float = 0.8
    efficiency_weight: float = 0.2
    
    # Performance scaling
    target_speedup: float = 2.0
    max_speedup_bonus: float = 5.0  # Cap for diminishing returns
    speedup_scale: str = "log"  # "linear" or "log"
    
    # Penalties
    compilation_failure_penalty: float = -1.0
    correctness_failure_penalty: float = -0.5
    timeout_penalty: float = -1.5
    
    # Efficiency thresholds (for MI300X)
    good_occupancy: float = 0.7
    good_bandwidth_utilization: float = 0.6
    
    # Correctness tolerance
    numerical_tolerance: float = 1e-3


class ExecutionReward:
    """
    Calculate reward from kernel execution results.
    
    Reward structure:
    - Compilation failed: negative penalty
    - Correctness failed: negative penalty (but less than compilation)
    - Correct but slow: small positive reward
    - Correct and fast: large positive reward
    - Correct, fast, and efficient: bonus on top
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.logger = logger.bind(component="ExecutionReward")
    
    def calculate(
        self,
        execution_result: ExecutionResult,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward from execution result.
        
        Args:
            execution_result: Result from kernel execution
            context: Optional context (e.g., previous best performance)
            
        Returns:
            Tuple of (total_reward, component_breakdown)
        """
        components = {}
        
        # Stage 1: Compilation gate
        if not execution_result.compiled:
            components["compilation"] = self.config.compilation_failure_penalty
            total = self.config.compilation_failure_penalty
            
            self.logger.debug(
                "Reward: compilation failed",
                reward=total
            )
            return total, components
        
        components["compilation"] = 0.0  # Passed gate
        
        # Stage 2: Correctness gate
        if not execution_result.executed:
            components["execution"] = self.config.correctness_failure_penalty
            total = self.config.correctness_failure_penalty
            
            self.logger.debug(
                "Reward: execution failed",
                reward=total
            )
            return total, components
        
        if not execution_result.numerically_correct:
            # Partial credit based on how close we are
            error_penalty = self._correctness_penalty(execution_result)
            components["correctness"] = error_penalty
            total = error_penalty
            
            self.logger.debug(
                "Reward: incorrect output",
                max_error=execution_result.max_absolute_error,
                reward=total
            )
            return total, components
        
        components["correctness"] = 0.1  # Small bonus for being correct
        
        # Stage 3: Performance reward (main signal)
        performance_reward = self._performance_reward(execution_result)
        components["performance"] = performance_reward
        
        # Stage 4: Efficiency bonus
        efficiency_bonus = self._efficiency_bonus(execution_result)
        components["efficiency"] = efficiency_bonus
        
        # Stage 5: Improvement bonus (if context provided)
        improvement_bonus = 0.0
        if context and "previous_speedup" in context:
            improvement_bonus = self._improvement_bonus(
                execution_result.speedup,
                context["previous_speedup"]
            )
            components["improvement"] = improvement_bonus
        
        # Calculate total
        total = (
            components["correctness"] +
            self.config.performance_weight * performance_reward +
            self.config.efficiency_weight * efficiency_bonus +
            improvement_bonus
        )
        
        # Clamp to reasonable range
        total = max(-2.0, min(2.0, total))
        
        self.logger.debug(
            "Reward calculated",
            speedup=f"{execution_result.speedup:.2f}x",
            total=f"{total:.3f}",
            components=components
        )
        
        return total, components
    
    def _correctness_penalty(self, result: ExecutionResult) -> float:
        """Calculate penalty for incorrect output."""
        
        # If very close, give partial credit
        if result.max_absolute_error < self.config.numerical_tolerance * 10:
            # Close but not quite - small penalty
            return -0.2
        elif result.max_absolute_error < self.config.numerical_tolerance * 100:
            # Somewhat close - medium penalty
            return -0.4
        else:
            # Way off - full penalty
            return self.config.correctness_failure_penalty
    
    def _performance_reward(self, result: ExecutionResult) -> float:
        """Calculate reward based on speedup."""
        
        speedup = result.speedup
        target = self.config.target_speedup
        max_bonus = self.config.max_speedup_bonus
        
        if speedup <= 0:
            return -0.5  # Something went wrong
        
        if self.config.speedup_scale == "log":
            # Logarithmic scaling: diminishing returns for high speedups
            # speedup=1.0 -> 0.0, speedup=2.0 -> 0.69, speedup=4.0 -> 1.39
            if speedup < 1.0:
                # Slower than baseline - negative reward
                return math.log(speedup)  # Negative value
            else:
                # Faster than baseline - positive reward
                raw_reward = math.log(speedup)
                # Scale relative to target
                scaled = raw_reward / math.log(target)
                return min(scaled, math.log(max_bonus) / math.log(target))
        else:
            # Linear scaling
            if speedup < 1.0:
                return speedup - 1.0  # Negative
            else:
                normalized = (speedup - 1.0) / (target - 1.0)
                return min(normalized, max_bonus / target)
    
    def _efficiency_bonus(self, result: ExecutionResult) -> float:
        """Calculate bonus for efficient resource utilization."""
        
        bonus = 0.0
        
        # Occupancy bonus
        if result.achieved_occupancy >= self.config.good_occupancy:
            bonus += 0.2
        elif result.achieved_occupancy >= self.config.good_occupancy * 0.7:
            bonus += 0.1
        
        # Memory bandwidth bonus
        # MI300X has ~5.3 TB/s theoretical bandwidth
        # Good utilization would be 60%+ for memory-bound kernels
        if result.memory_bandwidth_gbps > 0:
            # Rough estimate of utilization
            theoretical_bw = 5300  # GB/s for MI300X
            utilization = result.memory_bandwidth_gbps / theoretical_bw
            if utilization >= self.config.good_bandwidth_utilization:
                bonus += 0.2
            elif utilization >= self.config.good_bandwidth_utilization * 0.5:
                bonus += 0.1
        
        # Penalty for register spilling (if we have that info)
        # Spilling is bad for performance
        if result.profiling_data.get("spill_vgpr", 0) > 0:
            bonus -= 0.1
        
        return bonus
    
    def _improvement_bonus(
        self, 
        current_speedup: float, 
        previous_speedup: float
    ) -> float:
        """Bonus for improving over previous best."""
        
        if previous_speedup <= 0:
            return 0.0
        
        improvement_ratio = current_speedup / previous_speedup
        
        if improvement_ratio > 1.1:  # 10% improvement
            return 0.2
        elif improvement_ratio > 1.0:  # Any improvement
            return 0.1
        else:
            return 0.0


def create_reward_function(
    target_speedup: float = 2.0,
    performance_weight: float = 0.8,
    efficiency_weight: float = 0.2,
) -> ExecutionReward:
    """Factory function to create reward function with custom config."""
    
    config = RewardConfig(
        target_speedup=target_speedup,
        performance_weight=performance_weight,
        efficiency_weight=efficiency_weight,
    )
    
    return ExecutionReward(config)

