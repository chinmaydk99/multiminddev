#!/bin/bash
# Remote Configuration Script
# Run this locally to configure multi-GPU setup on Lambda Labs

set -e

# Configuration - Update these variables
REMOTE_HOST="ubuntu@104.171.202.131"
SSH_KEY_PATH="D:\Post_Masters\Upskilling\Products\MultiMindDev\multiminddev.pem"  # Set this to your Lambda Labs SSH key path
PROJECT_DIR="MultiMindDev"
REMOTE_PROJECT_PATH="/home/ubuntu/$PROJECT_DIR"

echo "⚙️ Configuring multi-GPU setup on Lambda Labs"
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

echo "🔍 Step 1: Check GPU configuration"
run_remote "cd $REMOTE_PROJECT_PATH && source venv_linux/bin/activate && export PATH=\$HOME/.local/bin:\$PATH && nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader"

echo "⚙️ Step 2: Run multi-GPU configuration"
run_remote "cd $REMOTE_PROJECT_PATH && source venv_linux/bin/activate && export PATH=\$HOME/.local/bin:\$PATH && python configure_multigpu_tests.py"

echo "✅ Multi-GPU configuration completed!"
echo "📋 Next: Run './scripts/remote_test.sh' to run validation tests"