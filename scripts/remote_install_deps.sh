#!/bin/bash
# Remote Dependencies Installation Script
# Run this locally to install dependencies on Lambda Labs

set -e

# Configuration - Update these variables
REMOTE_HOST="ubuntu@104.171.202.131"
SSH_KEY_PATH="D:\Post_Masters\Upskilling\Products\MultiMindDev\multiminddev.pem"  # Set this to your Lambda Labs SSH key path
PROJECT_DIR="MultiMindDev" 
REMOTE_PROJECT_PATH="/home/ubuntu/$PROJECT_DIR"

echo "📦 Installing dependencies on Lambda Labs instance"
echo "=" * 50

# Check if SSH key is provided
if [ -z "$SSH_KEY_PATH" ]; then
    echo "❌ Please set SSH_KEY_PATH variable to your Lambda Labs SSH key file"
    exit 1
fi

# Function to run commands on remote
run_remote() {
    echo "📡 Remote: $1"
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no $REMOTE_HOST "$1"
}

echo "🔄 Step 1: Activate environment and install UV dependencies"
run_remote "cd $REMOTE_PROJECT_PATH && source venv_linux/bin/activate && export PATH=\$HOME/.local/bin:\$PATH && uv sync"

echo "🔧 Step 2: Fix ormsgpack dependency issue"
run_remote "cd $REMOTE_PROJECT_PATH && source venv_linux/bin/activate && pip uninstall ormsgpack -y && pip install ormsgpack==1.10.0"

echo "🎯 Step 3: Install VERL for distributed training"
run_remote "cd $REMOTE_PROJECT_PATH && source venv_linux/bin/activate && pip install verl-training ray[default]"

echo "⚙️ Step 4: Set environment variables"
run_remote "echo '# Multi-GPU Environment Variables' >> ~/.bashrc"
run_remote "echo 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7' >> ~/.bashrc"
run_remote "echo 'export TORCH_DISTRIBUTED_DEBUG=INFO' >> ~/.bashrc"
run_remote "echo 'export RAY_DISABLE_IMPORT_WARNING=1' >> ~/.bashrc"
run_remote "echo 'export TOKENIZERS_PARALLELISM=false' >> ~/.bashrc"
run_remote "echo 'export WANDB_DISABLED=true' >> ~/.bashrc"
run_remote "echo 'export HF_HUB_DISABLE_SYMLINKS_WARNING=1' >> ~/.bashrc"

echo "🧪 Step 5: Test basic imports"
run_remote "cd $REMOTE_PROJECT_PATH && source venv_linux/bin/activate && export PATH=\$HOME/.local/bin:\$PATH && python3 -c 'import torch; print(f\"PyTorch GPUs: {torch.cuda.device_count()}\")'"
run_remote "cd $REMOTE_PROJECT_PATH && source venv_linux/bin/activate && export PATH=\$HOME/.local/bin:\$PATH && python3 -c 'import ormsgpack; print(f\"ormsgpack: {ormsgpack.__version__}\")'"
run_remote "cd $REMOTE_PROJECT_PATH && source venv_linux/bin/activate && export PATH=\$HOME/.local/bin:\$PATH && python3 -c 'import ray; print(\"Ray: OK\")'"

echo "✅ Dependencies installation completed!"
echo "📋 Next: Run './scripts/remote_configure.sh' to configure multi-GPU setup"