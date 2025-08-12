# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **CUDA Multi-Agent RL Training Framework** that combines VERL (reinforcement learning) with multi-agent systems to train CUDA code generation and optimization agents. The system uses a two-phase approach:

1. **SFT Phase**: Supervised fine-tuning with QLoRA using the SakanaAI CUDA dataset
2. **RL Phase**: Multi-turn reinforcement learning with VERL/GRPO for agent collaboration

## Core Architecture

### Agent System
The framework implements three specialized agents:
- **Generator Agent**: Creates initial CUDA kernel implementations
- **Optimizer Agent**: Improves and optimizes CUDA code for performance  
- **Tester Agent**: Validates correctness and benchmarks performance

### Key Components
```
src/coding_framework/
├── agents/                    # Trainable CUDA agents
├── cuda/                      # CUDA compilation and benchmarking
├── data/                      # Data pipeline and curriculum learning
├── training/                  # VERL integration and multi-turn conversations
├── evaluation/                # Benchmark evaluation systems
├── deployment/                # A/B testing and model serving
└── monitoring/                # Training and deployment monitoring
```

## Development Environment Setup

### Critical Dependency Management
**⚠️ NEVER use pip directly!** This project has complex ML dependencies that require UV for proper resolution.

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up environment with VERL dependencies
uv sync --extra verl

# Install local VERL (if using local development version)
cd verl && uv pip install -e . --no-deps && cd ..

# Verify critical imports
uv run python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
uv run python -c "import transformers, datasets, peft, accelerate; print('✅ Core ML OK')"
```

### Development Commands

```bash
# Code quality
uv run ruff format .
uv run ruff check --fix .
uv run mypy src/

# Testing
uv run pytest
uv run pytest --cov=src/coding_framework --cov-report=html
uv run pytest -m "unit"
uv run pytest -m "integration"

# Main training commands
uv run python run_sft_training.py --num-examples 500 --epochs 3
uv run python run_multiturn_rl_training.py --num-episodes 200 --num-gpus 8
uv run python launch_complete_cuda_training.py --quick-test
```

## Training System Architecture

### Two-Phase Training Pipeline

#### Phase 1: SFT with QLoRA (`run_sft_training.py`)
- **Input**: SakanaAI dataset (30k+ CUDA kernels)
- **Method**: QLoRA fine-tuning on Qwen2.5-Coder-1.5B
- **Output**: Generator and optimizer agent checkpoints
- **Requirements**: Single GPU (12GB+ VRAM)

#### Phase 2: Multi-Turn RL (`run_multiturn_rl_training.py`)
- **Input**: SFT checkpoints from Phase 1
- **Method**: VERL/GRPO multi-agent reinforcement learning
- **Output**: RL-trained collaborative agents
- **Requirements**: Multi-GPU cluster (8x A100 recommended)

### Data Pipeline Features

#### SakanaAI Dataset Integration
```python
# Curriculum learning with 3 difficulty tiers
Level 1 (Easy): 12,157 examples - vector operations, element-wise ops
Level 2 (Medium): 12,938 examples - reductions, matrix-vector ops  
Level 3 (Hard): 5,520 examples - matrix multiplication, convolutions
```

#### Curriculum Manager
- Automatic progression from easy → medium → hard
- Performance-based advancement thresholds
- Adaptive difficulty based on success rates

## Key System Components

### CUDA Compilation & Benchmarking
```python
# Safe compilation with Docker sandboxing
from coding_framework.cuda.compiler import CUDACompiler
from coding_framework.cuda.benchmarker import CUDABenchmarker

compiler = CUDACompiler()
benchmarker = CUDABenchmarker()
```

### Multi-Turn Conversations
```python
# Three-phase conversation flow: Generation → Optimization → Evaluation
from coding_framework.training.multi_turn_conversation import MultiTurnConversationManager

conversation_manager = MultiTurnConversationManager(
    generator_agent, optimizer_agent, tester_agent,
    compiler, benchmarker
)
```

### Reward Function
```python
# Multi-component reward calculation
from coding_framework.training.reward_functions.cuda_performance_reward import CUDAPerformanceReward

# Components: Compilation (30%) + Correctness (30%) + Performance (25%) + Efficiency (10%) + Quality (5%)
reward_function = CUDAPerformanceReward()
```

## Hardware Requirements

### Minimum (SFT Phase)
- **GPU**: 1x RTX 3080/4080 (12GB+ VRAM)
- **RAM**: 32GB
- **Storage**: 100GB

### Recommended (RL Phase)
- **GPU**: 8x A100 (40GB VRAM) or 8x RTX 4090
- **RAM**: 128GB+
- **Storage**: 500GB+
- **Docker**: Required for safe CUDA compilation

## Configuration Management

### Training Configuration
```python
# Located in training scripts - highly configurable
@dataclass
class SFTConfig:
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    num_examples: int = 500
    epochs: int = 3
    qlora_r: int = 16
    qlora_alpha: int = 32

@dataclass  
class MultiTurnRLConfig:
    num_episodes: int = 200
    verl_algorithm: str = "grpo"  # grpo, dapo, ppo
    num_gpus: int = 8
```

## Code Style & Conventions

### Python Standards
- **Line length**: 100 characters (Ruff configuration)
- **Functions**: <50 lines with single responsibility
- **Classes**: <100 lines representing single concepts
- **Files**: <500 lines - refactor if exceeded
- **Type hints**: Required for all function signatures
- **Docstrings**: Google-style for public functions

### ML/AI Specific Standards
- **Library-first approach**: Use established ML libraries (transformers, PEFT, VERL)
- **No custom training loops**: Use SFTTrainer, PPOTrainer, etc.
- **Reproducibility**: Always set seeds, track experiments
- **Configuration over hardcoding**: All hyperparameters in config files

## Testing Strategy

### Test Organization
```
tests/
├── unit/                      # Individual component tests
├── integration/               # Multi-component tests  
└── system/                    # End-to-end training tests
```

### Test Categories
```bash
pytest -m "unit"              # Fast unit tests
pytest -m "integration"       # Integration tests
pytest -m "slow"              # Full training tests
pytest -m "cuda"              # CUDA hardware tests
```

## Safety & Security

### CUDA Safety Measures
- **Docker sandboxing**: All CUDA compilation in isolated containers
- **Security validation**: Pattern-based detection of dangerous operations
- **Resource limits**: Timeout enforcement and memory constraints
- **Code analysis**: Detection of common CUDA pitfalls and security issues

### Training Safety
- **Curriculum learning**: Gradual difficulty progression
- **Early termination**: Stop on performance thresholds
- **Checkpoint management**: Regular saving for recovery
- **Memory monitoring**: GPU memory tracking and cleanup

## Monitoring & Observability

### Experiment Tracking
```bash
# WandB integration for both training phases
export WANDB_API_KEY=your_api_key
# Projects: "CUDA-SFT-Training" and "CUDA-MultiTurn-RL"
```

### Key Metrics
- **SFT Phase**: Training loss, learning rate, model convergence
- **RL Phase**: Episode rewards, success rates, curriculum advancement
- **System**: GPU utilization, memory usage, compilation times

## Common Workflows

### Quick Training Test
```bash
# Test system components
uv run python test_complete_system.py

# Quick SFT + RL test (minimal settings)
uv run python launch_complete_cuda_training.py --quick-test
```

### Full Training Pipeline
```bash
# Phase 1: SFT training
uv run python run_sft_training.py --num-examples 1000 --epochs 5

# Phase 2: RL training  
uv run python run_multiturn_rl_training.py \
  --generator-checkpoint ./sft_checkpoints/generator \
  --optimizer-checkpoint ./sft_checkpoints/optimizer \
  --num-episodes 200 --num-gpus 8
```

### Debugging & Troubleshooting

#### Dependency Issues (Most Common)
```bash
# Check for proper UV installation
uv sync --extra verl
cd verl && uv pip install -e . --no-deps && cd ..

# Verify critical imports
uv run python -c "import numpy; print('NumPy:', numpy.__version__)"  # Should be < 2.0
```

#### GPU/CUDA Issues
```bash
# Verify CUDA setup
nvidia-smi
uv run python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

#### Training Issues
```bash
# Reduce memory usage
uv run python run_sft_training.py --batch-size 2
uv run python run_multiturn_rl_training.py --rl-batch-size 128

# Disable Docker if needed (less safe)
uv run python run_multiturn_rl_training.py --no-docker
```

## File Structure Context

### Training Entry Points
- `run_sft_training.py` - SFT phase with QLoRA  
- `run_multiturn_rl_training.py` - RL phase with VERL
- `launch_complete_cuda_training.py` - Complete pipeline orchestrator

### Core Implementation
- `src/coding_framework/data/` - Data pipeline and curriculum
- `src/coding_framework/training/` - VERL integration and conversations
- `src/coding_framework/cuda/` - Compilation and benchmarking
- `src/coding_framework/agents/` - Trainable CUDA agents

### Configuration & Docs
- `TRAINING_GUIDE.md` - Detailed training instructions
- `cuda_rl_system_spec.md` - System specification
- `pyproject.toml` - Dependency management with UV

## Important Notes

### Dependency Management
- **NEVER edit pyproject.toml directly** - always use `uv add`/`uv remove`
- **Complex ML dependencies** require UV for proper resolution
- **NumPy < 2.0 constraint** due to VERL compatibility requirements
- **PyTorch 2.1-2.7 range** for VERL/VLLM compatibility

### Training Considerations
- **Docker required** for safe CUDA compilation in RL phase
- **Multi-GPU setup** needed for effective RL training
- **Wandb integration** for experiment tracking and monitoring
- **Curriculum learning** automatically progresses difficulty

### Performance Expectations
- **SFT**: ~2-4 hours for full training on single GPU
- **RL**: ~12-24 hours on 8x A100 cluster
- **Final models**: Should achieve >2x CUDA speedups with correct implementations

This framework represents a complete production-ready system for training collaborative CUDA optimization agents using state-of-the-art ML techniques.