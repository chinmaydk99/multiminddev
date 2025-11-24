"""
Multi-Turn RL HIP Code Generation Framework for AMD ROCm

A multi-turn reinforcement learning framework using VERL for training
specialized agents that generate and optimize HIP kernels on AMD GPUs.
"""

__version__ = "0.1.0"
__author__ = "MultiMindDev"
__email__ = "multiminddev@example.com"

from .agents import (
    TrainableAgent,
    TrainableHIPGeneratorAgent,
    TrainableHIPOptimizerAgent,
    TrainableHIPTesterAgent,
)
from .utils import load_config, setup_logging

__all__ = [
    "TrainableAgent",
    "TrainableHIPGeneratorAgent",
    "TrainableHIPOptimizerAgent",
    "TrainableHIPTesterAgent",
    "setup_logging",
    "load_config",
]
