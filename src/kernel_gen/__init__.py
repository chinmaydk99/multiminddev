"""
Kernel generation using LLMs.

Supports:
- Prompting with any OpenAI-compatible API
- Local models via HuggingFace
- Fine-tuned models (future)
"""

from .generator import KernelGenerator, GenerationResult
from .prompts import PromptBuilder, KernelPrompt

__all__ = [
    "KernelGenerator",
    "GenerationResult",
    "PromptBuilder",
    "KernelPrompt",
]

