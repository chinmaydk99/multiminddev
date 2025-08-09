#!/bin/bash
# Remote Setup Script for Lambda Labs
# Run this locally to setup the Lambda Labs instance

set -e  # Exit on any error

# Configuration - Update these variables
REMOTE_HOST="ubuntu@104.171.202.62"
SSH_KEY_PATH="D:\Post_Masters\Upskilling\Products\MultiMindDev\multiminddev.pem"  # Set this to your Lambda Labs SSH key path
PROJECT_DIR="MultiMindDev"
REMOTE_PROJECT_PATH="/home/ubuntu/$PROJECT_DIR"

echo "🚀 Setting up Lambda Labs instance: $REMOTE_HOST"
echo "=" * 60

# Check if SSH key is provided
if [ -z "$SSH_KEY_PATH" ]; then
    echo "❌ Please set SSH_KEY_PATH variable to your Lambda Labs SSH key file"
    echo "   Example: SSH_KEY_PATH=\"~/.ssh/lambda_labs_key.pem\""
    exit 1
fi

# Verify SSH key exists
if [ ! -f "$SSH_KEY_PATH" ]; then
    echo "❌ SSH key not found: $SSH_KEY_PATH"
    echo "   Please download your SSH key from Lambda Cloud console"
    exit 1
fi

# Set proper permissions on SSH key
chmod 600 "$SSH_KEY_PATH"

echo "✅ Using SSH key: $SSH_KEY_PATH"

# Function to run commands on remote
run_remote() {
    echo "📡 Running on remote: $1"
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no $REMOTE_HOST "$1"
}

# Function to copy files to remote
copy_to_remote() {
    echo "📤 Copying $1 to remote"
    scp -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$1" $REMOTE_HOST:"$2"
}

echo "🔍 Step 1: Check connection and basic setup"
run_remote "echo '✅ Connected to Lambda Labs instance' && whoami && pwd"

echo "🔍 Step 2: Check GPU availability"
run_remote "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | nl"

echo "📁 Step 3: Setup project directory"
run_remote "mkdir -p $REMOTE_PROJECT_PATH && cd $REMOTE_PROJECT_PATH && pwd"

echo "🐍 Step 4: Setup Python environment"
run_remote "cd $REMOTE_PROJECT_PATH && python3 -m venv venv_linux && echo '✅ Virtual environment created'"

echo "📦 Step 5: Install UV package manager"
run_remote "curl -LsSf https://astral.sh/uv/install.sh | sh && echo '✅ UV installed'"

echo "🔄 Step 6: Copy project files"
# Copy essential files
copy_to_remote "configure_multigpu_tests.py" "$REMOTE_PROJECT_PATH/"
copy_to_remote "pyproject.toml" "$REMOTE_PROJECT_PATH/"
copy_to_remote "uv.lock" "$REMOTE_PROJECT_PATH/"

# Copy source directory
echo "📤 Copying source directory..."
scp -r -o StrictHostKeyChecking=no "src/" $REMOTE_HOST:"$REMOTE_PROJECT_PATH/"

echo "✅ Remote setup completed!"
echo "📋 Next: Run './scripts/remote_install_deps.sh' to install dependencies"