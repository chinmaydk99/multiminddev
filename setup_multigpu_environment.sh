#!/bin/bash
# Multi-GPU Environment Setup Script for Phase 3 VERL Integration

set -e  # Exit on any error

echo "🚀 Setting up Multi-GPU Environment for Phase 3 VERL Integration..."

# Check GPU availability
echo "📊 Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA drivers not found. Please install NVIDIA drivers first."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "✅ Found $GPU_COUNT GPUs available"

if [ $GPU_COUNT -lt 4 ]; then
    echo "⚠️  Warning: Found only $GPU_COUNT GPUs, but 4 were expected"
fi

# Create and activate virtual environment
echo "🐍 Creating Python virtual environment..."
if [ ! -d "venv_multigpu" ]; then
    python3 -m venv venv_multigpu
fi

source venv_multigpu/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip and install wheel
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Ray for distributed computing
echo "⚡ Installing Ray for distributed training..."
pip install -U "ray[default,train,tune]" ray[rllib]

# Install VERL dependencies
echo "🧠 Installing VERL dependencies..."
pip install transformers accelerate datasets tokenizers
pip install deepspeed fairscale
pip install wandb tensorboard

# Install evaluation dependencies
echo "📊 Installing evaluation framework dependencies..."
pip install human-eval  # For HumanEval benchmark
pip install datasets  # For MBPP and other benchmarks
pip install matplotlib seaborn pandas numpy  # For BigCodeBench

# Install monitoring dependencies
echo "📈 Installing monitoring dependencies..."
pip install psutil GPUtil prometheus_client
pip install structlog rich

# Install testing dependencies
echo "🧪 Installing testing dependencies..."
pip install pytest pytest-asyncio pytest-cov
pip install httpx aiofiles

# Install additional utilities
echo "🛠️ Installing additional utilities..."
pip install pydantic pydantic-settings
pip install uvloop  # For better async performance
pip install jinja2  # For templating

# Verify installations
echo "✅ Verifying critical installations..."

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')

import ray
print(f'Ray version: {ray.__version__}')

import transformers
print(f'Transformers version: {transformers.__version__}')
"

echo "🎉 Multi-GPU environment setup complete!"
echo "📋 Next steps:"
echo "   1. Source the environment: source venv_multigpu/bin/activate"
echo "   2. Run the configuration script: python configure_multigpu_tests.py"
echo "   3. Execute tests: python run_phase3_integration_test.py"