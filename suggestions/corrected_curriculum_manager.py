"""
Curriculum Learning Manager for SakanaAI CUDA dataset.
Properly handles level_1, level_2, level_3 progression with performance tracking.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from collections import deque
import structlog
import time


@dataclass 
class SakanaLevelConfig:
    """Configuration for a SakanaAI dataset level."""
    level_name: str          # level_1, level_2, level_3
    level_id: int           # 1, 2, 3
    difficulty_tier: str    # easy, medium, hard
    
    # Performance targets for advancement
    min_compilation_rate: float = 0.7
    min_success_rate: float = 0.5
    target_speedup: float = 1.5
    
    # Training constraints
    min_episodes: int = 100
    max_episodes: int = 1000
    
    # Example operations at this level
    typical_operations: List[str] = field(default_factory=list)
    
    def meets_advancement_criteria(self, metrics: Dict[str, float]) -> bool:
        """Check if performance metrics meet advancement criteria."""
        return (
            metrics.get("compilation_rate", 0.0) >= self.min_compilation_rate and
            metrics.get("success_rate", 0.0) >= self.min_success_rate and
            metrics.get("avg_speedup", 0.0) >= self.target_speedup
        )


@dataclass
class LevelPerformanceHistory:
    """Track performance history for a specific SakanaAI level."""
    level_name: str
    
    # Performance tracking
    compilation_successes: deque = field(default_factory=lambda: deque(maxlen=100))
    speedups: deque = field(default_factory=lambda: deque(maxlen=100))
    correctness_checks: deque = field(default_factory=lambda: deque(maxlen=100))
    final_rewards: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Metadata
    episode_count: int = 0
    level_start_time: float = field(default_factory=time.time)
    total_training_time: float = 0.0
    
    def add_episode_result(
        self,
        compilation_success: bool,
        speedup: float,
        is_correct: bool,
        final_reward: float
    ):
        """Add a training episode result."""
        self.compilation_successes.append(1.0 if compilation_success else 0.0)
        self.speedups.append(max(0.0, speedup))  # Ensure non-negative
        self.correctness_checks.append(1.0 if is_correct else 0.0)
        self.final_rewards.append(final_reward)
        self.episode_count += 1
    
    def get_recent_metrics(self, window: int = 50) -> Dict[str, float]:
        """Get recent performance metrics for advancement decisions."""
        if len(self.compilation_successes) < window:
            window = len(self.compilation_successes)
        
        if window == 0:
            return {
                "compilation_rate": 0.0,
                "success_rate": 0.0,
                "avg_speedup": 0.0,
                "avg_reward": 0.0,
                "episodes": 0
            }
        
        recent_compilations = list(self.compilation_successes)[-window:]
        recent_speedups = list(self.speedups)[-window:]
        recent_correctness = list(self.correctness_checks)[-window:]
        recent_rewards = list(self.final_rewards)[-window:]
        
        # Only consider speedups from successful compilations
        successful_speedups = [
            speedup for i, speedup in enumerate(recent_speedups)
            if recent_compilations[i] > 0.5 and recent_correctness[i] > 0.5
        ]
        
        return {
            "compilation_rate": np.mean(recent_compilations),
            "success_rate": np.mean(recent_correctness),
            "avg_speedup": np.mean(successful_speedups) if successful_speedups else 0.0,
            "avg_reward": np.mean(recent_rewards),
            "episodes": window,
            "successful_episodes": len(successful_speedups)
        }


class SakanaCurriculumManager:
    """Curriculum manager specifically designed for SakanaAI dataset structure."""
    
    def __init__(
        self,
        initial_level: str = "level_1",
        advancement_window: int = 50,
        patience_episodes: int = 200,
        enable_level_reset: bool = True
    ):
        self.advancement_window = advancement_window
        self.patience_episodes = patience_episodes
        self.enable_level_reset = enable_level_reset
        self.logger = structlog.get_logger("sakana_curriculum")
        
        # Define SakanaAI level configurations
        self.level_configs = {
            "level_1": SakanaLevelConfig(
                level_name="level_1",
                level_id=1,
                difficulty_tier="easy",
                min_compilation_rate=0.80,
                min_success_rate=0.60,
                target_speedup=1.3,
                min_episodes=100,
                max_episodes=500,
                typical_operations=[
                    "Square_matrix_multiplication",
                    "vector_add", 
                    "element_wise_operations",
                    "basic_reductions"
                ]
            ),
            "level_2": SakanaLevelConfig(
                level_name="level_2", 
                level_id=2,
                difficulty_tier="medium",
                min_compilation_rate=0.75,
                min_success_rate=0.50,
                target_speedup=1.8,
                min_episodes=200,
                max_episodes=800,
                typical_operations=[
                    "matrix_vector_multiply",
                    "transpose_operations",
                    "reduction_sum",
                    "softmax",
                    "layer_norm"
                ]
            ),
            "level_3": SakanaLevelConfig(
                level_name="level_3",
                level_id=3,
                difficulty_tier="hard", 
                min_compilation_rate=0.70,
                min_success_rate=0.40,
                target_speedup=2.5,
                min_episodes=300,
                max_episodes=1000,
                typical_operations=[
                    "complex_matrix_multiply",
                    "convolution_2d",
                    "fused_operations",
                    "attention_mechanisms",
                    "optimized_gemm"
                ]
            )
        }
        
        # Initialize state
        self.current_level = initial_level
        self.level_order = ["level_1", "level_2", "level_3"]
        
        # Performance tracking
        self.performance_history = {
            level: LevelPerformanceHistory(level) 
            for level in self.level_configs.keys()
        }
        
        # Training state
        self.episodes_without_progress = 0
        self.total_episodes_trained = 0
        
        self.logger.info(
            "SakanaAI curriculum manager initialized",
            initial_level=initial_level,
            available_levels=list(self.level_configs.keys()),
            advancement_window=advancement_window
        )
    
    def get_current_level(self) -> str:
        """Get current curriculum level."""
        return self.current_level
    
    def get_current_config(self) -> SakanaLevelConfig:
        """Get configuration for current level."""
        return self.level_configs[self.current_level]
    
    def get_level_config(self, level: str) -> SakanaLevelConfig:
        """Get configuration for specific level."""
        return self.level_configs.get(level)
    
    def record_episode_result(
        self,
        conversation_result,
        level: Optional[str] = None
    ):
        """Record result of a training episode."""
        level = level or self.current_level
        
        if level not in self.performance_history:
            self.logger.warning(f"Unknown level: {level}")
            return
        
        history = self.performance_history[level]
        
        # Extract metrics from conversation result
        compilation_success = any(
            turn.compilation_success for turn in conversation_result.turns
            if hasattr(turn, 'compilation_success')
        )
        
        # Extract speedup from final performance
        speedup = conversation_result.current_performance.get("speedup", 0.0)
        
        # Check if conversation was successful
        is_correct = conversation_result.conversation_success
        
        final_reward = conversation_result.final_reward
        
        # Record the result
        history.add_episode_result(
            compilation_success=compilation_success,
            speedup=speedup,
            is_correct=is_correct,
            final_reward=final_reward
        )
        
        self.total_episodes_trained += 1
        
        # Check for progress
        recent_metrics = history.get_recent_metrics(self.advancement_window)
        current_config = self.level_configs[level]
        
        if current_config.meets_advancement_criteria(recent_metrics):
            self.episodes_without_progress = 0
        else:
            self.episodes_without_progress += 1
        
        self.logger.debug(
            "Episode result recorded",
            level=level,
            episode_count=history.episode_count,
            compilation_success=compilation_success,
            speedup=speedup,
            correct=is_correct,
            reward=final_reward,
            episodes_without_progress=self.episodes_without_progress
        )
    
    def should_advance_level(self, level: Optional[str] = None) -> bool:
        """Check if should advance to next level."""
        level = level or self.current_level
        
        # Can't advance from highest level
        current_index = self.level_order.index(level)
        if current_index >= len(self.level_order) - 1:
            return False
        
        history = self.performance_history[level]
        config = self.level_configs[level]
        
        # Check minimum episodes requirement
        if history.episode_count < config.min_episodes:
            return False
        
        # Check recent performance
        recent_metrics = history.get_recent_metrics(self.advancement_window)
        
        advancement_ready = config.meets_advancement_criteria(recent_metrics)
        
        self.logger.info(
            "Advancement check",
            level=level,
            episode_count=history.episode_count,
            min_required=config.min_episodes,
            recent_metrics=recent_metrics,
            advancement_criteria={
                "min_compilation_rate": config.min_compilation_rate,
                "min_success_rate": config.min_success_rate,
                "target_speedup": config.target_speedup
            },
            advancement_ready=advancement_ready
        )
        
        return advancement_ready
    
    def advance_level(self) -> Optional[str]:
        """Advance to next curriculum level."""
        current_index = self.level_order.index(self.current_level)
        
        if current_index < len(self.level_order) - 1:
            old_level = self.current_level
            self.current_level = self.level_order[current_index + 1]
            
            # Reset progress tracking
            self.episodes_without_progress = 0
            
            # Update timing
            self.performance_history[old_level].total_training_time = (
                time.time() - self.performance_history[old_level].level_start_time
            )
            self.performance_history[self.current_level].level_start_time = time.time()
            
            self.logger.info(
                "Advanced curriculum level",
                old_level=old_level,
                new_level=self.current_level,
                old_level_episodes=self.performance_history[old_level].episode_count,
                total_training_time_hours=self.performance_history[old_level].total_training_time / 3600
            )
            
            return self.current_level
        
        return None
    
    def should_reset_level(self, level: Optional[str] = None) -> bool:
        """Check if should reset to easier level due to poor performance."""
        if not self.enable_level_reset:
            return False
        
        level = level or self.current_level
        
        # Don't reset from easiest level
        if level == self.level_order[0]:
            return False
        
        history = self.performance_history[level]
        
        # Only consider reset after substantial training
        if history.episode_count < self.patience_episodes:
            return False
        
        # Check if performance is very poor
        recent_metrics = history.get_recent_metrics(self.advancement_window)
        
        very_poor_performance = (
            recent_metrics["compilation_rate"] < 0.3 or
            recent_metrics["success_rate"] < 0.1 or
            self.episodes_without_progress > self.patience_episodes
        )
        
        if very_poor_performance:
            self.logger.warning(
                "Poor performance detected",
                level=level,
                recent_metrics=recent_metrics,
                episodes_without_progress=self.episodes_without_progress,
                patience_episodes=self.patience_episodes
            )
        
        return very_poor_performance
    
    def reset_level(self) -> Optional[str]:
        """Reset to previous (easier) curriculum level."""
        current_index = self.level_order.index(self.current_level)
        
        if current_index > 0:
            old_level = self.current_level
            self.current_level = self.level_order[current_index - 1]
            
            # Reset progress tracking
            self.episodes_without_progress = 0
            
            self.logger.warning(
                "Reset curriculum level due to poor performance",
                old_level=old_level,
                new_level=self.current_level,
                failed_episodes=self.performance_history[old_level].episode_count
            )
            
            return self.current_level
        
        return None
    
    def get_level_summary(self, level: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive performance summary for a level."""
        level = level or self.current_level
        
        if level not in self.performance_history:
            return {"error": f"Level {level} not found"}
        
        history = self.performance_history[level]
        config = self.level_configs[level]
        
        if history.episode_count == 0:
            return {
                "level": level,
                "episode_count": 0,
                "status": "not_started"
            }
        
        recent_metrics = history.get_recent_metrics(self.advancement_window)
        overall_metrics = history.get_recent_metrics(history.episode_count)
        
        return {
            "level": level,
            "level_id": config.level_id,
            "difficulty_tier": config.difficulty_tier,
            "episode_count": history.episode_count,
            "training_time_hours": (time.time() - history.level_start_time) / 3600,
            
            # Overall performance
            "overall_metrics": overall_metrics,
            
            # Recent performance
            "recent_metrics": recent_metrics,
            
            # Advancement status
            "advancement_criteria": {
                "min_compilation_rate": config.min_compilation_rate,
                "min_success_rate": config.min_success_rate,
                "target_speedup": config.target_speedup,
                "min_episodes": config.min_episodes
            },
            "meets_advancement": config.meets_advancement_criteria(recent_metrics),
            "ready_for_advancement": self.should_advance_level(level),
            
            # Progress tracking
            "episodes_without_progress": self.episodes_without_progress,
            "typical_operations": config.typical_operations
        }
    
    def get_training_progress_report(self) -> Dict[str, Any]:
        """Get comprehensive training progress report."""
        
        report = {
            "current_level": self.current_level,
            "total_episodes": self.total_episodes_trained,
            "episodes_without_progress": self.episodes_without_progress,
            "level_summaries": {}
        }
        
        for level in self.level_order:
            report["level_summaries"][level] = self.get_level_summary(level)
        
        # Calculate overall progress
        completed_levels = 0
        for level in self.level_order:
            if level == self.current_level:
                break
            completed_levels += 1
        
        current_level_progress = 0.0
        current_config = self.level_configs[self.current_level]
        current_history = self.performance_history[self.current_level]
        
        if current_history.episode_count > 0:
            current_level_progress = min(
                current_history.episode_count / current_config.min_episodes,
                1.0
            )
        
        overall_progress = (completed_levels + current_level_progress) / len(self.level_order)
        report["overall_progress"] = overall_progress
        
        return report
