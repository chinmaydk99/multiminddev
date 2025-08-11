"""
Training module for VERL reinforcement learning integration.

This module provides the core training infrastructure for the multi-agent coding framework,
including VERL PPO training, reward functions, and data pipeline components.
"""

from .base_trainer import BaseTrainer, TrainingResults
from .data_loader import TrainingDataLoader, TrainingProblem
from .multi_turn_conversation import MultiTurnConversationManager, CUDAConversationState, ConversationTurn
from .reward_functions import BaseRewardFunction, CUDAPerformanceReward, RewardComponents
from .verl_integration import MultiAgentVERLTrainer
from .verl_trainer import VERLConfig, VERLPPOTrainer

__all__ = [
    "BaseTrainer",
    "TrainingResults",
    "VERLPPOTrainer",
    "VERLConfig",
    "TrainingDataLoader",
    "TrainingProblem",
    "BaseRewardFunction",
    "CUDAPerformanceReward",
    "RewardComponents",
    "MultiTurnConversationManager",
    "CUDAConversationState",
    "ConversationTurn",
    "MultiAgentVERLTrainer",
]
