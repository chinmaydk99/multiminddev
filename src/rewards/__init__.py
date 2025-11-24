"""
Reward functions for kernel generation RL training.

Rewards are based on real execution feedback:
- Compilation success
- Numerical correctness
- Performance (speedup vs baseline)
- Efficiency metrics
"""

from .execution_reward import ExecutionReward, RewardConfig

__all__ = [
    "ExecutionReward",
    "RewardConfig",
]

