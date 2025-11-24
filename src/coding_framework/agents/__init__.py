"""
Agent implementations for the Multi-Turn RL Training Framework.

This module contains trainable agents with owned model parameters that
can be improved through reinforcement learning via VERL.
Specialized for HIP kernel generation on AMD ROCm GPUs.
"""

from .trainable_agent import AgentResponse, GenerationOutput, TrainableAgent
from .trainable_hip_agents import (
    TrainableHIPGeneratorAgent,
    TrainableHIPOptimizerAgent,
    TrainableHIPTesterAgent,
)

__all__ = [
    "TrainableAgent",
    "AgentResponse",
    "GenerationOutput",
    "TrainableHIPGeneratorAgent",
    "TrainableHIPOptimizerAgent",
    "TrainableHIPTesterAgent",
]
