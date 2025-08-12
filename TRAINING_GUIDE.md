# CUDA Multi-Turn RL Training Guide

This guide shows how to use the separated training approach for CUDA agents with QLoRA SFT and VERL-based multi-turn RL.

## Overview

The training is split into two phases:

1. **SFT Phase (QLoRA)**: Fine-tune generator and optimizer agents using supervised learning
2. **RL Phase (VERL + GRPO)**: Multi-turn reinforcement learning with curriculum progression

## Architecture

### SFT Phase
- **File**: `run_sft_training.py`
- **Requirements**: Single GPU, transformers + PEFT
- **Data**: SakanaAI dataset with 30K+ CUDA examples
- **Output**: Generator and optimizer checkpoints

### RL Phase  
- **File**: `run_multiturn_rl_training.py`
- **Requirements**: Multi-GPU, VERL, Docker (for safe compilation)
- **Input**: SFT checkpoints
- **Output**: RL-trained multi-turn agents

## Data Pipeline Features

### SakanaAI Dataset Integration
- **Total Examples**: 30,615 CUDA kernels
- **Level 1 (Easy)**: 12,157 examples - vector operations, element-wise ops
- **Level 2 (Medium)**: 12,938 examples - reductions, matrix-vector ops  
- **Level 3 (Hard)**: 5,520 examples - matrix multiplication, convolutions

### Curriculum Learning
- **Automatic progression** from easy → medium → hard
- **Performance thresholds** for advancement
- **Adaptive difficulty** based on success rates

## Quick Start

### 1. **CRITICAL**: Proper Dependency Management Setup

**DO NOT use pip directly!** This project uses UV for robust dependency management to avoid version conflicts.

#### Install UV (if not already installed)
```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Setup Clean Environment
```bash
# Navigate to project root
cd /path/to/MultiMindDev

# Create virtual environment and sync all dependencies
uv sync --extra verl

# For local VERL development (install VERL from ./verl directory)
cd verl && uv pip install -e . --no-deps
cd ..
```

#### Verify Installation
```bash
# Test critical imports
uv run python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
uv run python -c "import transformers, datasets, peft, accelerate, wandb; print('✅ Core ML libs OK')"
uv run python -c "import vllm; print('✅ VLLM OK')"

# Test VERL import
uv run python -c "import verl; print('✅ VERL OK')"
```

#### Why This Matters
- **Prevents import errors** (numpy 2.0 conflicts, scipy issues, torch version mismatches)
- **Ensures VERL compatibility** (specific tensordict, numpy constraints)
- **Includes VLLM for RL rollouts** (essential for multi-turn RL training)
- **Handles CUDA dependencies** properly (flash-attn, bitsandbytes)
- **Reproducible across environments**

**⚠️ If you skip this step, you WILL encounter import failures during training!**

### 2. Run SFT Training

```bash
# Set WandB API key for experiment tracking
export WANDB_API_KEY=your_wandb_api_key_here

# Basic SFT training (single GPU)
uv run python run_sft_training.py --num-examples 500 --epochs 3

# Customized SFT training
uv run python run_sft_training.py \
  --num-examples 1000 \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --output-dir ./my_sft_checkpoints
```

**Expected Output:**
- `./sft_checkpoints/generator/` - Generator agent checkpoint
- `./sft_checkpoints/optimizer/` - Optimizer agent checkpoint  
- `./sft_checkpoints/config.json` - Training configuration

### 3. Run Multi-Turn RL Training

```bash
# Install Docker first (required for safe CUDA compilation)
# Then run RL training
uv run python run_multiturn_rl_training.py \
  --generator-checkpoint ./sft_checkpoints/generator \
  --optimizer-checkpoint ./sft_checkpoints/optimizer \
  --num-episodes 200 \
  --num-gpus 8
```

**Expected Output:**
- `./rl_checkpoints/best_model_episode_X/` - Best performing model
- `./rl_checkpoints/final_model/` - Final trained model

## Configuration Options

### SFT Training Configuration

```python
@dataclass
class SFTConfig:
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    num_examples: int = 500
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_length: int = 2048
    
    # QLoRA settings
    qlora_r: int = 16
    qlora_alpha: int = 32
    qlora_dropout: float = 0.1
```

### RL Training Configuration

```python
@dataclass
class MultiTurnRLConfig:
    num_episodes: int = 200
    max_turns_per_episode: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-5
    
    # VERL settings
    verl_algorithm: str = "grpo"  # grpo, dapo, or ppo
    verl_rollout_batch_size: int = 128
    
    # Multi-GPU
    num_gpus: int = 8
    use_docker_sandbox: bool = True
```

## Monitoring Training

### WandB Integration
Both scripts automatically log to Weights & Biases:
- **SFT Project**: "CUDA-SFT-Training"
- **RL Project**: "CUDA-MultiTurn-RL"

### Key Metrics to Monitor

**SFT Phase:**
- Training loss
- Learning rate schedule
- Model convergence

**RL Phase:**
- Episode rewards
- Success rates per difficulty
- Curriculum advancement
- Multi-turn conversation quality

## Troubleshooting

### Dependency Issues (Most Common!)

1. **Import Errors (transformers, numpy, scipy)**
   ```bash
   # ❌ WRONG: Don't use pip directly
   pip install transformers
   
   # ✅ CORRECT: Use UV for clean dependency resolution
   uv sync --extra verl
   cd verl && uv pip install -e . --no-deps && cd ..
   ```

2. **Flash Attention Build Failures**
   ```bash
   # If flash-attn fails to build during uv sync --extra verl:
   
   # Option 1: Install torch first, then sync
   uv pip install torch --no-build-isolation
   uv sync --extra verl --no-build-isolation
   
   # Option 2: Sync without verl extra first, then install VERL manually
   uv sync  # Basic dependencies first
   cd verl && uv pip install -e . --no-deps && cd ..
   ```

3. **NumPy 2.0 Conflicts**
   ```bash
   # Check numpy version - should be < 2.0
   uv run python -c "import numpy; print(numpy.__version__)"
   # Should show: 1.26.x (not 2.x.x)
   ```

4. **PyTorch CUDA Version Mismatch**
   ```bash
   # Verify PyTorch CUDA compatibility
   uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
   ```

5. **TrainingExample Attribute Errors**
   ```bash
   # If you see "AttributeError: 'TrainingExample' object has no attribute 'prompt'"
   # This is fixed in the updated run_sft_training.py - the formatting function was
   # referencing a non-existent 'prompt' attribute
   ```

6. **Device Mapping Errors for Quantized Models**
   ```bash
   # If you see "You can't train a model that has been loaded in 8-bit or 4-bit precision on a different device"
   # This is fixed by using device_map={"": torch.cuda.current_device()} instead of device_map="auto"
   ```

7. **Missing pydantic-settings**
   ```bash
   # This should NOT happen with UV - but if it does:
   uv add pydantic-settings
   ```

### Training Issues

5. **GPU Memory Issues**
   ```bash
   # Reduce batch size
   uv run python run_sft_training.py --batch-size 2
   ```

6. **Dataset Loading Issues**
   ```bash
   # The system automatically handles trust_remote_code issues
   # and falls back to proper curriculum mapping
   ```

7. **Docker Issues (RL Phase)**
   ```bash
   # Install Docker with GPU support
   sudo apt install docker.io
   sudo systemctl start docker
   sudo usermod -aG docker $USER
   
   # Disable Docker if needed (less safe)
   uv run python run_multiturn_rl_training.py --no-docker
   ```

## Hardware Requirements

### SFT Phase (Minimum)
- **GPU**: 1x RTX 3080/4080 (12GB+ VRAM)
- **RAM**: 32GB
- **Storage**: 100GB

### RL Phase (Recommended)
- **GPU**: 8x A100 (40GB VRAM) or 8x RTX 4090
- **RAM**: 128GB+
- **Storage**: 500GB+
- **Network**: High-bandwidth for distributed training

## File Structure

```
├── run_sft_training.py              # SFT training script
├── run_multiturn_rl_training.py     # RL training script  
├── launch_complete_cuda_training.py # Original monolithic script
├── src/coding_framework/
│   ├── data/
│   │   ├── sakana_loader.py         # Updated with curriculum mapping
│   │   ├── data_pipeline.py         # Main data interface
│   │   └── curriculum_manager.py    # Curriculum progression logic
│   ├── training/
│   │   ├── verl_integration.py      # VERL + GRPO integration
│   │   └── multi_turn_conversation.py # Multi-agent conversations
│   └── cuda/
│       ├── compiler.py              # Safe CUDA compilation
│       └── benchmarker.py           # Performance evaluation
├── sft_checkpoints/                 # SFT outputs
├── rl_checkpoints/                  # RL outputs
└── cache/datasets/                  # Dataset cache
```

## Next Steps

1. **Run SFT training** to get base agent checkpoints
2. **Verify checkpoints** are saved correctly
3. **Set up Docker** for safe CUDA compilation
4. **Run RL training** with multi-turn conversations
5. **Monitor training** via WandB dashboards
6. **Evaluate final models** on held-out test problems

## Performance Expectations

### SFT Training
- **Time**: ~1-2 minutes for small batches (50 examples), ~2-4 hours for full training
- **Memory**: ~8-12GB VRAM
- **Success**: Agents should learn CUDA syntax and basic optimization patterns
- **Example Results**: 
  ```
  Training completed successfully:
  - Trainable params: 18,464,768 (1.18% of total model)
  - Train loss: 0.1546
  - Runtime: ~47 seconds total (both generator and optimizer)
  - Checkpoints saved to: ./sft_checkpoints/generator and ./sft_checkpoints/optimizer
  ```

### RL Training
- **Time**: ~12-24 hours on 8x A100 cluster
- **Memory**: ~25-30GB VRAM per GPU
- **Success**: Agents should learn multi-turn collaboration and achieve >2x CUDA speedups

## Support

For issues or questions:
1. Check logs in `wandb` dashboard
2. Review error messages in console output
3. Verify hardware requirements are met
4. Ensure all dependencies are properly installed