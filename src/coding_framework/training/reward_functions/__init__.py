"""
Reward functions for reinforcement learning training.

This module provides reward functions for evaluating HIP code generation performance
on AMD ROCm GPUs in the VERL training pipeline.
"""

from .base_reward import BaseRewardFunction
from .hip_performance_reward import HIPPerformanceReward, RewardComponents

__all__ = [
    "BaseRewardFunction",
    "HIPPerformanceReward",
    "RewardComponents",
]
