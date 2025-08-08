"""
Training module for VERL reinforcement learning integration.

This module provides the core training infrastructure for the multi-agent coding framework,
including VERL PPO training, reward functions, and data pipeline components.
"""

from .base_trainer import BaseTrainer, TrainingResults
from .data_loader import TrainingDataLoader, TrainingProblem
from .reward_functions import (
    BaseRewardFunction,
    CompositeReward,
    CorrectnessReward,
    EfficiencyReward,
    StyleReward,
)
from .verl_trainer import VERLConfig, VERLPPOTrainer

__all__ = [
    "BaseTrainer",
    "TrainingResults",
    "VERLPPOTrainer",
    "VERLConfig",
    "TrainingDataLoader",
    "TrainingProblem",
    "BaseRewardFunction",
    "CorrectnessReward",
    "StyleReward",
    "EfficiencyReward",
    "CompositeReward",
]
