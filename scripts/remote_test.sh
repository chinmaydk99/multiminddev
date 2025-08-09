#!/bin/bash
# Remote Testing Script
# Run this locally to execute tests on Lambda Labs

set -e

# Configuration - Update these variables
REMOTE_HOST="ubuntu@104.171.202.62"
SSH_KEY_PATH="D:\Post_Masters\Upskilling\Products\MultiMindDev\multiminddev.pem"  # Set this to your Lambda Labs SSH key path
PROJECT_DIR="MultiMindDev"
REMOTE_PROJECT_PATH="/home/ubuntu/$PROJECT_DIR"

echo "🧪 Running tests on Lambda Labs instance"
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

echo "🔧 Step 1: Fix ormsgpack dependency"
run_remote "cd $REMOTE_PROJECT_PATH && source venv_linux/bin/activate && export PATH=\$HOME/.local/bin:\$PATH && pip uninstall ormsgpack -y && pip install ormsgpack==1.10.0"

echo "🧪 Step 2: Run CUDA components test"
run_remote "cd $REMOTE_PROJECT_PATH && source venv_linux/bin/activate && export PATH=\$HOME/.local/bin:\$PATH && python test_cuda_components.py"

echo "🎯 Step 3: Run CUDA training example (quick test)"
run_remote "cd $REMOTE_PROJECT_PATH && source venv_linux/bin/activate && export PATH=\$HOME/.local/bin:\$PATH && cd examples/cuda_training && python train_cuda_agents.py --quick-test --validate-cuda"

echo "🚀 Step 4: Run full CUDA training pipeline (if components test passes)"
run_remote "cd $REMOTE_PROJECT_PATH && source venv_linux/bin/activate && export PATH=\$HOME/.local/bin:\$PATH && cd examples/cuda_training && python train_cuda_agents.py --episodes 5"

echo "📊 Step 4: Generate test report"
run_remote "cd $REMOTE_PROJECT_PATH && source venv_linux/bin/activate && export PATH=\$HOME/.local/bin:\$PATH && ls -la test_results/ 2>/dev/null || echo 'No test results directory yet'"

echo "✅ All tests completed!"
echo "📋 Check the output above for results"