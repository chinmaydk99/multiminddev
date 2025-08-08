# Project Overview: VERL + LangGraph Multi-Agent Coding Framework

## Purpose
A sophisticated multi-agent system that combines LangGraph orchestration with VERL reinforcement learning for intelligent code generation, review, and execution with continuous learning capabilities.

## Architecture
- **Three Specialized Agents**: CodeGenerator, CodeReviewer, CodeExecutor
- **LangGraph Orchestration**: State machine workflows with conditional routing
- **VERL Integration**: Reinforcement learning for continuous agent improvement
- **Production Features**: Docker sandboxing, monitoring, multi-language support

## Tech Stack
- **Core**: Python 3.8+, Pydantic v2, AsyncIO
- **AI/ML**: LangGraph, LangChain, OpenAI/Anthropic APIs, VERL (PyTorch ecosystem)
- **Execution**: Docker, Ray (distributed computing)
- **Development**: UV package management, Ruff, MyPy, Pytest
- **Monitoring**: Structlog, W&B

## Current Status
- Phase 1: Complete orchestration framework ✅
- Phase 2: VERL training pipeline integration (in progress)

## Key Entry Points
- CLI: `python -m coding_framework.cli`
- Main API: `CodingSupervisor` class
- Configuration: YAML-based hierarchical config system