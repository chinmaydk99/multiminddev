#!/bin/bash
# Master Script to Run All Remote Operations
# This script orchestrates the entire Lambda Labs setup and testing process

set -e

# Configuration - EDIT THESE VARIABLES
REMOTE_HOST="ubuntu@104.171.202.131"
SSH_KEY_PATH="D:\Post_Masters\Upskilling\Products\MultiMindDev\multiminddev.pem"  # SET THIS TO YOUR LAMBDA LABS SSH KEY PATH
PROJECT_DIR="MultiMindDev"

echo "🚀 Lambda Labs Multi-GPU Setup & Testing Pipeline"
echo "=" * 60

# Check if SSH key is provided
if [ -z "$SSH_KEY_PATH" ]; then
    echo "❌ Please edit this script and set SSH_KEY_PATH variable"
    echo "   Download your SSH key from Lambda Cloud console"
    echo "   Example: SSH_KEY_PATH=\"~/.ssh/lambda_labs_key.pem\""
    exit 1
fi

# Update all scripts with the SSH key path
echo "🔧 Updating scripts with SSH key path..."
sed -i "s|SSH_KEY_PATH=\"\"|SSH_KEY_PATH=\"$SSH_KEY_PATH\"|g" scripts/remote_setup.sh
sed -i "s|SSH_KEY_PATH=\"\"|SSH_KEY_PATH=\"$SSH_KEY_PATH\"|g" scripts/remote_install_deps.sh
sed -i "s|SSH_KEY_PATH=\"\"|SSH_KEY_PATH=\"$SSH_KEY_PATH\"|g" scripts/remote_configure.sh
sed -i "s|SSH_KEY_PATH=\"\"|SSH_KEY_PATH=\"$SSH_KEY_PATH\"|g" scripts/remote_test.sh

echo "✅ Scripts updated with SSH key path"

echo ""
echo "🎯 Running full pipeline..."
echo ""

echo "📋 Phase 1: Remote Setup"
./scripts/remote_setup.sh

echo ""
echo "📋 Phase 2: Install Dependencies"
./scripts/remote_install_deps.sh

echo ""
echo "📋 Phase 3: Configure Multi-GPU"
./scripts/remote_configure.sh

echo ""
echo "📋 Phase 4: Run Tests"
./scripts/remote_test.sh

echo ""
echo "🎉 Lambda Labs setup and testing completed!"
echo "=" * 60
echo "✅ Your multi-GPU VERL setup is ready for production training"