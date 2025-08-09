"""
VERL + LangGraph Multi-Agent Coding Framework

A sophisticated multi-agent system that combines LangGraph orchestration with VERL
reinforcement learning for intelligent code generation, review, and execution.
"""

__version__ = "0.1.0"
__author__ = "MultiMindDev"
__email__ = "multiminddev@example.com"

from .agents import (
    BaseAgent,
    CUDAGeneratorAgent,
    CUDAOptimizerAgent,
    CUDATesterAgent,
)
from .orchestration import CodingSupervisor
from .utils import load_config, setup_logging

__all__ = [
    "BaseAgent",
    "CUDAGeneratorAgent",
    "CUDAOptimizerAgent", 
    "CUDATesterAgent",
    "CodingSupervisor",
    "setup_logging",
    "load_config",
]
