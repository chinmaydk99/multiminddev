"""
Curriculum Learning Manager for progressive CUDA kernel training difficulty.

Manages tier progression, advancement criteria, and performance tracking
for systematic skill development in CUDA optimization.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import structlog

from pydantic import BaseModel, Field


class CurriculumTier(Enum):
    """Curriculum difficulty tiers for CUDA kernel training."""
    BASIC = 0
    INTERMEDIATE = 1
    ADVANCED = 2
    EXPERT = 3


@dataclass
class TierCriteria:
    """Advancement criteria for curriculum tiers."""
    compile_rate: float
    pass_rate: float
    target_performance: float
    max_complexity: int
    operations: List[str] = field(default_factory=list)
    min_episodes: int = 50
    consistency_window: int = 20


@dataclass 
class PerformanceMetrics:
    """Performance tracking for curriculum advancement."""
    compilation_success: bool
    tests_passed: bool
    final_speedup: float
    turns_required: int
    memory_efficiency: float
    optimization_techniques_used: int
    timestamp: float = field(default_factory=time.time)


class WorkflowResult(BaseModel):
    """Result from a CUDA optimization workflow."""
    success: bool
    compilation_success: bool = False
    tests_passed: bool = False
    final_speedup: float = 0.0
    turns_required: int = 1
    memory_efficiency: float = 0.0
    optimization_techniques_used: int = 0
    total_reward: float = 0.0


class CUDACurriculumManager:
    """
    Manages progressive curriculum for CUDA kernel training.
    
    Implements tier-based learning with automatic advancement based on
    performance criteria and consistency requirements.
    """
    
    def __init__(self, initial_tier: CurriculumTier = CurriculumTier.BASIC):
        """
        Initialize curriculum manager with tier definitions.
        
        Args:
            initial_tier: Starting curriculum tier
        """
        self.current_tier = initial_tier
        self.logger = structlog.get_logger("cuda_curriculum_manager")
        
        # Define curriculum tiers with advancement criteria
        self.curriculum_tiers = {
            CurriculumTier.BASIC: TierCriteria(
                operations=["vector_add", "scalar_multiply", "element_wise"],
                compile_rate=0.9,
                pass_rate=0.75,
                target_performance=2.0,  # 2x speedup minimum
                max_complexity=100,  # Lines of kernel code
                min_episodes=50,
                consistency_window=20
            ),
            CurriculumTier.INTERMEDIATE: TierCriteria(
                operations=["reduction", "transpose", "matrix_vector", "convolution_1d"],
                compile_rate=0.85,
                pass_rate=0.70,
                target_performance=3.0,  # 3x speedup minimum
                max_complexity=200,
                min_episodes=75,
                consistency_window=25
            ),
            CurriculumTier.ADVANCED: TierCriteria(
                operations=["matrix_multiply", "softmax", "layer_norm", "conv2d"],
                compile_rate=0.80,
                pass_rate=0.65,
                target_performance=5.0,  # 5x speedup minimum
                max_complexity=300,
                min_episodes=100,
                consistency_window=30
            ),
            CurriculumTier.EXPERT: TierCriteria(
                operations=["fused_attention", "custom_layers", "optimized_gemm", "flash_attention"],
                compile_rate=0.75,
                pass_rate=0.60,
                target_performance=10.0,  # 10x speedup minimum
                max_complexity=500,
                min_episodes=150,
                consistency_window=40
            )
        }
        
        # Performance tracking
        self.tier_performance_history: Dict[CurriculumTier, List[PerformanceMetrics]] = defaultdict(list)
        self.advancement_attempts: Dict[CurriculumTier, int] = defaultdict(int)
        self.tier_mastery_timestamps: Dict[CurriculumTier, float] = {}
        
        # Statistics tracking
        self.total_episodes = 0
        self.successful_episodes = 0
        self.tier_episode_counts: Dict[CurriculumTier, int] = defaultdict(int)
        
        self.logger.info(
            "CUDA curriculum manager initialized",
            initial_tier=initial_tier.name,
            total_tiers=len(self.curriculum_tiers)
        )
    
    def get_current_tier_info(self) -> Dict[str, Any]:
        """Get information about current curriculum tier."""
        criteria = self.curriculum_tiers[self.current_tier]
        recent_performance = self._get_recent_performance_stats()
        
        return {
            "tier": self.current_tier.name,
            "tier_value": self.current_tier.value,
            "operations": criteria.operations,
            "advancement_criteria": {
                "compile_rate": criteria.compile_rate,
                "pass_rate": criteria.pass_rate,
                "target_performance": criteria.target_performance,
                "min_episodes": criteria.min_episodes
            },
            "current_performance": recent_performance,
            "episodes_completed": self.tier_episode_counts[self.current_tier],
            "advancement_ready": self._check_advancement_readiness(),
            "max_complexity": criteria.max_complexity
        }
    
    def get_training_operations(self, batch_size: int = 1) -> List[str]:
        """
        Get operations for current tier training.
        
        Args:
            batch_size: Number of operations to return
            
        Returns:
            List of operation types for current tier
        """
        criteria = self.curriculum_tiers[self.current_tier]
        operations = criteria.operations
        
        # Return cyclic selection if batch_size > available operations
        if batch_size <= len(operations):
            return operations[:batch_size]
        else:
            # Cycle through operations to fill batch
            result = []
            for i in range(batch_size):
                result.append(operations[i % len(operations)])
            return result
    
    def record_episode_result(self, result: WorkflowResult) -> None:
        """
        Record the result of a training episode.
        
        Args:
            result: Workflow result to record
        """
        # Convert to internal performance metrics
        metrics = PerformanceMetrics(
            compilation_success=result.compilation_success,
            tests_passed=result.tests_passed,
            final_speedup=result.final_speedup,
            turns_required=result.turns_required,
            memory_efficiency=result.memory_efficiency,
            optimization_techniques_used=result.optimization_techniques_used
        )
        
        # Record in tier-specific history
        self.tier_performance_history[self.current_tier].append(metrics)
        
        # Update episode counts
        self.total_episodes += 1
        self.tier_episode_counts[self.current_tier] += 1
        
        if result.success:
            self.successful_episodes += 1
        
        # Prune history to manage memory
        self._prune_performance_history()
        
        self.logger.debug(
            "Episode result recorded",
            tier=self.current_tier.name,
            success=result.success,
            speedup=result.final_speedup,
            tier_episodes=self.tier_episode_counts[self.current_tier]
        )
    
    async def should_advance_tier(self) -> bool:
        """
        Check if agent should advance to next curriculum tier.
        
        Returns:
            True if advancement criteria are met
        """
        if self.current_tier == CurriculumTier.EXPERT:
            return False  # Already at highest tier
        
        if not self._check_advancement_readiness():
            return False
        
        criteria = self.curriculum_tiers[self.current_tier]
        recent_results = self._get_recent_performance_window(criteria.consistency_window)
        
        if len(recent_results) < criteria.min_episodes:
            self.logger.debug(
                "Insufficient episodes for advancement",
                current_count=len(recent_results),
                required=criteria.min_episodes
            )
            return False
        
        # Calculate performance metrics
        performance_stats = self._calculate_advancement_metrics(recent_results)
        
        # Check all advancement criteria
        meets_compile_threshold = performance_stats["compile_rate"] >= criteria.compile_rate
        meets_pass_threshold = performance_stats["pass_rate"] >= criteria.pass_rate
        meets_performance_target = performance_stats["avg_speedup"] >= criteria.target_performance
        meets_consistency = performance_stats["consistency_score"] >= 0.8
        
        advancement_ready = (
            meets_compile_threshold and 
            meets_pass_threshold and 
            meets_performance_target and
            meets_consistency
        )
        
        if advancement_ready:
            self.logger.info(
                "Advancement criteria met",
                current_tier=self.current_tier.name,
                compile_rate=performance_stats["compile_rate"],
                pass_rate=performance_stats["pass_rate"],
                avg_speedup=performance_stats["avg_speedup"],
                consistency_score=performance_stats["consistency_score"]
            )
            return True
        else:
            self.logger.debug(
                "Advancement criteria not met",
                current_tier=self.current_tier.name,
                meets_compile=meets_compile_threshold,
                meets_pass=meets_pass_threshold,
                meets_performance=meets_performance_target,
                meets_consistency=meets_consistency,
                performance_stats=performance_stats
            )
            return False
    
    def advance_tier(self) -> bool:
        """
        Advance to the next curriculum tier.
        
        Returns:
            True if advancement successful, False if already at max tier
        """
        if self.current_tier == CurriculumTier.EXPERT:
            self.logger.warning("Already at expert tier, cannot advance further")
            return False
        
        previous_tier = self.current_tier
        self.current_tier = CurriculumTier(self.current_tier.value + 1)
        self.tier_mastery_timestamps[previous_tier] = time.time()
        self.advancement_attempts[self.current_tier] = 0
        
        self.logger.info(
            "Advanced to next curriculum tier",
            previous_tier=previous_tier.name,
            new_tier=self.current_tier.name,
            episodes_in_previous=self.tier_episode_counts[previous_tier]
        )
        
        return True
    
    def should_regress_tier(self, regression_threshold: float = 0.5) -> bool:
        """
        Check if performance has degraded enough to warrant tier regression.
        
        Args:
            regression_threshold: Performance threshold for regression
            
        Returns:
            True if tier regression is recommended
        """
        if self.current_tier == CurriculumTier.BASIC:
            return False  # Can't regress from basic tier
        
        criteria = self.curriculum_tiers[self.current_tier]
        recent_results = self._get_recent_performance_window(criteria.consistency_window)
        
        if len(recent_results) < 20:  # Need sufficient data
            return False
        
        performance_stats = self._calculate_advancement_metrics(recent_results)
        
        # Check if performance has significantly degraded
        poor_compile_rate = performance_stats["compile_rate"] < criteria.compile_rate * regression_threshold
        poor_pass_rate = performance_stats["pass_rate"] < criteria.pass_rate * regression_threshold
        poor_performance = performance_stats["avg_speedup"] < criteria.target_performance * regression_threshold
        
        regression_needed = poor_compile_rate or poor_pass_rate or poor_performance
        
        if regression_needed:
            self.logger.warning(
                "Performance degradation detected",
                tier=self.current_tier.name,
                compile_rate=performance_stats["compile_rate"],
                pass_rate=performance_stats["pass_rate"],
                avg_speedup=performance_stats["avg_speedup"]
            )
        
        return regression_needed
    
    def regress_tier(self) -> bool:
        """
        Regress to the previous curriculum tier.
        
        Returns:
            True if regression successful, False if already at basic tier
        """
        if self.current_tier == CurriculumTier.BASIC:
            self.logger.warning("Already at basic tier, cannot regress further")
            return False
        
        previous_tier = self.current_tier
        self.current_tier = CurriculumTier(self.current_tier.value - 1)
        
        self.logger.info(
            "Regressed to previous curriculum tier",
            previous_tier=previous_tier.name,
            new_tier=self.current_tier.name
        )
        
        return True
    
    def get_curriculum_progress(self) -> Dict[str, Any]:
        """Get comprehensive curriculum progress statistics."""
        
        progress_by_tier = {}
        for tier in CurriculumTier:
            episodes = self.tier_episode_counts[tier]
            if episodes > 0:
                recent_results = self.tier_performance_history[tier][-50:]  # Last 50 episodes
                if recent_results:
                    stats = self._calculate_advancement_metrics(recent_results)
                    progress_by_tier[tier.name] = {
                        "episodes": episodes,
                        "compile_rate": stats["compile_rate"],
                        "pass_rate": stats["pass_rate"],
                        "avg_speedup": stats["avg_speedup"],
                        "mastered": tier in self.tier_mastery_timestamps
                    }
        
        return {
            "current_tier": self.current_tier.name,
            "total_episodes": self.total_episodes,
            "success_rate": self.successful_episodes / max(self.total_episodes, 1),
            "progress_by_tier": progress_by_tier,
            "mastery_timestamps": {
                tier.name: timestamp for tier, timestamp in self.tier_mastery_timestamps.items()
            }
        }
    
    def reset_curriculum(self, tier: CurriculumTier = CurriculumTier.BASIC) -> None:
        """Reset curriculum to specified tier."""
        self.current_tier = tier
        self.tier_performance_history.clear()
        self.advancement_attempts.clear()
        self.tier_mastery_timestamps.clear()
        self.total_episodes = 0
        self.successful_episodes = 0
        self.tier_episode_counts.clear()
        
        self.logger.info("Curriculum reset", new_tier=tier.name)
    
    def _check_advancement_readiness(self) -> bool:
        """Check basic readiness for tier advancement."""
        criteria = self.curriculum_tiers[self.current_tier]
        episode_count = self.tier_episode_counts[self.current_tier]
        
        return episode_count >= criteria.min_episodes
    
    def _get_recent_performance_stats(self, window_size: int = 50) -> Dict[str, float]:
        """Get recent performance statistics for current tier."""
        recent_results = self._get_recent_performance_window(window_size)
        
        if not recent_results:
            return {
                "compile_rate": 0.0,
                "pass_rate": 0.0,
                "avg_speedup": 0.0,
                "avg_turns": 0.0,
                "episodes": 0
            }
        
        return self._calculate_advancement_metrics(recent_results)
    
    def _get_recent_performance_window(self, window_size: int) -> List[PerformanceMetrics]:
        """Get recent performance metrics within specified window."""
        all_results = self.tier_performance_history[self.current_tier]
        return all_results[-window_size:] if all_results else []
    
    def _calculate_advancement_metrics(self, results: List[PerformanceMetrics]) -> Dict[str, float]:
        """Calculate metrics for advancement decision."""
        if not results:
            return {
                "compile_rate": 0.0,
                "pass_rate": 0.0,
                "avg_speedup": 0.0,
                "avg_turns": 0.0,
                "consistency_score": 0.0,
                "episodes": 0
            }
        
        compile_successes = sum(1 for r in results if r.compilation_success)
        test_passes = sum(1 for r in results if r.tests_passed)
        speedups = [r.final_speedup for r in results if r.final_speedup > 0]
        turns = [r.turns_required for r in results]
        
        # Calculate consistency score (lower variance = higher consistency)
        if len(speedups) > 1:
            speedup_variance = np.var(speedups)
            consistency_score = 1.0 / (1.0 + speedup_variance)  # Inverse variance
        else:
            consistency_score = 0.5  # Neutral for insufficient data
        
        return {
            "compile_rate": compile_successes / len(results),
            "pass_rate": test_passes / len(results),
            "avg_speedup": np.mean(speedups) if speedups else 0.0,
            "avg_turns": np.mean(turns) if turns else 0.0,
            "consistency_score": consistency_score,
            "episodes": len(results)
        }
    
    def _prune_performance_history(self, max_history_per_tier: int = 1000) -> None:
        """Prune performance history to manage memory usage."""
        for tier in self.tier_performance_history:
            if len(self.tier_performance_history[tier]) > max_history_per_tier:
                # Keep most recent results
                self.tier_performance_history[tier] = (
                    self.tier_performance_history[tier][-max_history_per_tier//2:]
                )
    
    def __repr__(self) -> str:
        """String representation of curriculum manager."""
        return (
            f"CUDACurriculumManager("
            f"current_tier={self.current_tier.name}, "
            f"total_episodes={self.total_episodes}, "
            f"success_rate={self.successful_episodes/max(self.total_episodes, 1):.3f})"
        )