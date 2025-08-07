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
    CodeGeneratorAgent, 
    CodeReviewerAgent,
    CodeExecutorAgent,
)
from .orchestration import CodingSupervisor
from .utils import setup_logging, load_config

__all__ = [
    "BaseAgent",
    "CodeGeneratorAgent",
    "CodeReviewerAgent", 
    "CodeExecutorAgent",
    "CodingSupervisor",
    "setup_logging",
    "load_config",
]