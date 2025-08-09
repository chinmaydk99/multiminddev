"""
Multi-Turn RL CUDA Code Generation Framework

A multi-turn reinforcement learning framework using VERL for training
specialized agents that generate and optimize CUDA kernels.
"""

__version__ = "0.1.0"
__author__ = "MultiMindDev"
__email__ = "multiminddev@example.com"

from .agents import (
    TrainableAgent,
    TrainableCUDAGeneratorAgent,
    TrainableCUDAOptimizerAgent,
    TrainableCUDATesterAgent,
)
from .utils import load_config, setup_logging

__all__ = [
    "TrainableAgent",
    "TrainableCUDAGeneratorAgent",
    "TrainableCUDAOptimizerAgent", 
    "TrainableCUDATesterAgent",
    "setup_logging",
    "load_config",
]
