"""
Curriculum Learning Manager for progressive CUDA kernel training.
Manages tier progression, advancement criteria, and performance tracking.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from collections import deque
import structlog
import time


@dataclass 
class CurriculumTier:
    """Definition of a curriculum learning tier."""
    name: str
    difficulty: str
    target_performance: Dict[str, float]
    advancement_criteria: Dict[str, float]
    max_episodes: int = 1000
    min_episodes: int = 100
    operations: List[str] = field(default_factory=list)


@dataclass
class PerformanceHistory:
    """Track performance history for curriculum advancement."""
    compilation_successes: deque = field(default_factory=lambda: deque(maxlen=100))
    speedups: deque = field(default_factory=lambda: deque(maxlen=100))
    final_rewards: deque = field(default_factory=lambda: deque(maxlen=100))
    episode_count: int = 0
    tier_start_time: float = field(default_factory=time.time)
    
    def add_result(self, compilation_success: bool, speedup: float, reward: float):
        """Add a training result to history."""
        self.compilation_successes.append(compilation_success)
        self.speedups.append(speedup)
        self.final_rewards.append(reward)
        self.episode_count += 1
    
    def get_recent_stats(self, window: int = 50) -> Dict[str, float]:
        """Get recent performance statistics."""
        recent_compilations = list(self.compilation_successes)[-window:]
        recent_speedups = list(self.speedups)[-window:]
        recent_rewards = list(self.final_rewards)[-window:]
        
        if not recent_compilations:
            return {
                "compilation_rate": 0.0,
                "avg_speedup": 0.0,
                "avg_reward": 0.0,
                "success_rate": 0.0
            }
        
        compilation_rate = sum(recent_compilations) / len(recent_compilations)
        avg_speedup = np.mean(recent_speedups) if recent_speedups else 0.0
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        success_rate = sum(1 for s in recent_speedups if s > 1.0) / len(recent_speedups)
        
        return {
            "compilation_rate": compilation_rate,
            "avg_speedup": avg_speedup,
            "avg_reward": avg_reward,
            "success_rate": success_rate
        }


class CurriculumManager:
    """Manages curriculum learning progression for CUDA training."""
    
    def __init__(
        self,
        initial_tier: str = "easy",
        advancement_window: int = 50,
        min_success_rate: float = 0.75,
        min_avg_speedup: float = 1.5
    ):
        self.advancement_window = advancement_window
        self.min_success_rate = min_success_rate
        self.min_avg_speedup = min_avg_speedup
        self.logger = structlog.get_logger("curriculum_manager")
        
        # Define curriculum tiers with updated operations from spec
        self.tiers = {
            "easy": CurriculumTier(
                name="easy",
                difficulty="easy",
                target_performance={"speedup": 1.5, "min_speedup": 1.2},
                advancement_criteria={
                    "compilation_rate": 0.8,
                    "success_rate": 0.6,
                    "avg_speedup": 1.2,
                    "min_episodes": 100
                },
                max_episodes=500,
                min_episodes=100,
                operations=["vector_add", "scalar_multiply", "element_wise"]
            ),
            "medium": CurriculumTier(
                name="medium", 
                difficulty="medium",
                target_performance={"speedup": 2.0, "min_speedup": 1.5},
                advancement_criteria={
                    "compilation_rate": 0.75,
                    "success_rate": 0.5,
                    "avg_speedup": 1.8,
                    "min_episodes": 200
                },
                max_episodes=800,
                min_episodes=200,
                operations=["reduction", "transpose", "matrix_vector", "convolution_1d"]
            ),
            "hard": CurriculumTier(
                name="hard",
                difficulty="hard", 
                target_performance={"speedup": 3.0, "min_speedup": 2.0},
                advancement_criteria={
                    "compilation_rate": 0.7,
                    "success_rate": 0.4,
                    "avg_speedup": 2.5,
                    "min_episodes": 300
                },
                max_episodes=1000,
                min_episodes=300,
                operations=["matrix_multiply", "softmax", "layer_norm", "conv2d", 
                           "fused_attention", "custom_layers", "optimized_gemm"]
            )
        }
        
        # Initialize state
        self.current_tier = initial_tier
        self.performance_history = {
            tier: PerformanceHistory() for tier in self.tiers.keys()
        }
        self.total_episodes = 0
        self.tier_transitions = []
        
        self.logger.info(
            "Curriculum manager initialized",
            initial_tier=initial_tier,
            available_tiers=list(self.tiers.keys())
        )
    
    def get_current_tier(self) -> str:
        """Get current curriculum tier."""
        return self.current_tier
    
    def get_tier_info(self) -> Dict[str, Any]:
        """Get detailed information about current tier."""
        tier_config = self.tiers[self.current_tier]
        history = self.performance_history[self.current_tier]
        
        return {
            "tier": self.current_tier,
            "difficulty": tier_config.difficulty,
            "operations": tier_config.operations,
            "target_performance": tier_config.target_performance,
            "advancement_criteria": tier_config.advancement_criteria,
            "episode_count": history.episode_count,
            "recent_stats": history.get_recent_stats(self.advancement_window),
            "total_episodes": self.total_episodes
        }
    
    def get_tier_config(self, tier: Optional[str] = None) -> CurriculumTier:
        """Get configuration for specified tier (or current tier)."""
        tier = tier or self.current_tier
        return self.tiers[tier]
    
    def record_episode_result(
        self,
        compilation_success: bool,
        speedup: float,
        final_reward: float,
        tier: Optional[str] = None
    ):
        """Record result of a training episode."""
        tier = tier or self.current_tier
        history = self.performance_history[tier]
        
        # Record the result
        history.add_result(compilation_success, speedup, final_reward)
        self.total_episodes += 1
        
        self.logger.debug(
            "Episode result recorded",
            tier=tier,
            compilation_success=compilation_success,
            speedup=speedup,
            reward=final_reward,
            episode_count=history.episode_count
        )
        
        # Check for advancement if enough episodes
        if history.episode_count >= self.tiers[tier].min_episodes:
            if self._should_advance(tier):
                self._advance_tier()
    
    def _should_advance(self, tier: str) -> bool:
        """Check if advancement criteria are met."""
        if tier == "hard":
            return False  # Already at highest tier
        
        tier_config = self.tiers[tier]
        history = self.performance_history[tier]
        stats = history.get_recent_stats(self.advancement_window)
        
        # Check all advancement criteria
        criteria = tier_config.advancement_criteria
        
        meets_compilation = stats["compilation_rate"] >= criteria["compilation_rate"]
        meets_success = stats["success_rate"] >= criteria["success_rate"]
        meets_speedup = stats["avg_speedup"] >= criteria["avg_speedup"]
        meets_episodes = history.episode_count >= criteria["min_episodes"]
        
        should_advance = meets_compilation and meets_success and meets_speedup and meets_episodes
        
        if should_advance:
            self.logger.info(
                "Advancement criteria met",
                tier=tier,
                stats=stats,
                criteria=criteria
            )
        
        return should_advance
    
    def _advance_tier(self):
        """Advance to next tier."""
        tier_order = ["easy", "medium", "hard"]
        current_index = tier_order.index(self.current_tier)
        
        if current_index < len(tier_order) - 1:
            old_tier = self.current_tier
            new_tier = tier_order[current_index + 1]
            
            self.current_tier = new_tier
            self.tier_transitions.append({
                "from": old_tier,
                "to": new_tier,
                "episode": self.total_episodes,
                "timestamp": time.time()
            })
            
            self.logger.info(
                "Advanced to next tier",
                old_tier=old_tier,
                new_tier=new_tier,
                total_episodes=self.total_episodes
            )
    
    def should_stop_training(self) -> bool:
        """Check if training should stop based on curriculum completion."""
        if self.current_tier == "hard":
            history = self.performance_history["hard"]
            stats = history.get_recent_stats(self.advancement_window)
            
            # Stop if achieved target performance on hard tier
            if (stats["avg_speedup"] >= 3.0 and 
                stats["compilation_rate"] >= 0.7 and
                history.episode_count >= 300):
                
                self.logger.info(
                    "Training complete - achieved target on hard tier",
                    stats=stats
                )
                return True
        
        # Stop if max episodes reached
        if self.total_episodes >= 3000:
            self.logger.info("Training complete - max episodes reached")
            return True
        
        return False
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get comprehensive progress summary."""
        summary = {
            "current_tier": self.current_tier,
            "total_episodes": self.total_episodes,
            "tier_transitions": self.tier_transitions,
            "tier_performance": {}
        }
        
        for tier_name, history in self.performance_history.items():
            if history.episode_count > 0:
                summary["tier_performance"][tier_name] = {
                    "episodes": history.episode_count,
                    "stats": history.get_recent_stats(self.advancement_window),
                    "time_spent": time.time() - history.tier_start_time
                }
        
        return summary
    
    def reset(self):
        """Reset curriculum manager to initial state."""
        self.current_tier = "easy"
        self.performance_history = {
            tier: PerformanceHistory() for tier in self.tiers.keys()
        }
        self.total_episodes = 0
        self.tier_transitions = []
        
        self.logger.info("Curriculum manager reset")