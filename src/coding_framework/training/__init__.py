"""
Training module for multi-turn CUDA RL training.

This module provides the core training infrastructure for the multi-agent CUDA framework,
including VERL GRPO training, multi-turn conversations, and reward functions.
"""

from .multi_turn_conversation import MultiTurnConversationManager, CUDAConversationState, ConversationTurn
from .reward_functions import BaseRewardFunction, CUDAPerformanceReward, RewardComponents
from .verl_integration import MultiAgentVERLTrainer

__all__ = [
    "MultiTurnConversationManager",
    "CUDAConversationState", 
    "ConversationTurn",
    "BaseRewardFunction",
    "CUDAPerformanceReward",
    "RewardComponents",
    "MultiAgentVERLTrainer",
]
