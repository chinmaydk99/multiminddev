"""
Agent implementations for the Multi-Turn RL Training Framework.

This module contains trainable agents with owned model parameters that
can be improved through reinforcement learning via VERL.
"""

from .trainable_agent import TrainableAgent, AgentResponse, GenerationOutput
from .trainable_cuda_agents import (
    TrainableCUDAGeneratorAgent,
    TrainableCUDAOptimizerAgent,
    TrainableCUDATesterAgent,
)

__all__ = [
    "TrainableAgent",
    "AgentResponse",
    "GenerationOutput",
    "TrainableCUDAGeneratorAgent",
    "TrainableCUDAOptimizerAgent",
    "TrainableCUDATesterAgent",
]
