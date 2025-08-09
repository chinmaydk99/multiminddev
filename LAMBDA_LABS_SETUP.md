# Lambda Labs Remote Setup Guide

This guide helps you set up and test your multi-GPU VERL training environment on Lambda Labs from your local machine.

## 🔑 Prerequisites

1. **Lambda Labs SSH Key**: Download your SSH private key from the Lambda Cloud console
2. **Instance Running**: Your Lambda Labs instance should be running (ubuntu@104.171.203.41)
3. **Local Environment**: Bash shell (Linux/macOS) or WSL/Git Bash (Windows)

## 🚀 Quick Start

### Option 1: Run Everything at Once
```bash
# 1. Edit the SSH key path in the master script
nano scripts/run_all_remote.sh
# Set: SSH_KEY_PATH="/path/to/your/lambda_key.pem"

# 2. Make scripts executable
chmod +x scripts/*.sh

# 3. Run the complete setup
./scripts/run_all_remote.sh
```

### Option 2: Step-by-Step Execution

#### Step 1: Setup Remote Environment
```bash
# Edit SSH key path
nano scripts/remote_setup.sh
# Set: SSH_KEY_PATH="/path/to/your/lambda_key.pem"

# Run setup
chmod +x scripts/remote_setup.sh
./scripts/remote_setup.sh
```

#### Step 2: Install Dependencies
```bash
# Edit SSH key path
nano scripts/remote_install_deps.sh
# Set: SSH_KEY_PATH="/path/to/your/lambda_key.pem"

# Install dependencies
chmod +x scripts/remote_install_deps.sh
./scripts/remote_install_deps.sh
```

#### Step 3: Configure Multi-GPU
```bash
# Edit SSH key path
nano scripts/remote_configure.sh
# Set: SSH_KEY_PATH="/path/to/your/lambda_key.pem"

# Configure multi-GPU
chmod +x scripts/remote_configure.sh
./scripts/remote_configure.sh
```

#### Step 4: Run Tests
```bash
# Edit SSH key path
nano scripts/remote_test.sh
# Set: SSH_KEY_PATH="/path/to/your/lambda_key.pem"

# Run tests
chmod +x scripts/remote_test.sh
./scripts/remote_test.sh
```

## 🛠️ What Each Script Does

### `remote_setup.sh`
- Connects to Lambda Labs instance
- Verifies GPU availability
- Creates project directory
- Sets up Python virtual environment
- Copies project files to remote

### `remote_install_deps.sh`
- Installs UV package manager
- Installs project dependencies
- Fixes ormsgpack compatibility issue
- Installs VERL for distributed training
- Sets up environment variables

### `remote_configure.sh`
- Configures multi-GPU setup
- Runs the configuration script
- Verifies GPU detection

### `remote_test.sh`
- Fixes any remaining dependency issues
- Runs quick validation tests
- Runs comprehensive integration tests
- Generates test reports

## 🔧 Troubleshooting

### SSH Connection Issues
```bash
# Test manual connection
ssh -i "/path/to/your/key.pem" ubuntu@104.171.203.41

# Fix key permissions
chmod 600 /path/to/your/key.pem
```

### Dependency Issues
```bash
# If ormsgpack fails, run this manually:
ssh -i "/path/to/your/key.pem" ubuntu@104.171.203.41 "cd MultiMindDev && source venv_linux/bin/activate && pip uninstall ormsgpack -y && pip install ormsgpack==1.10.0"
```

### GPU Detection Issues
```bash
# Check GPUs manually:
ssh -i "/path/to/your/key.pem" ubuntu@104.171.203.41 "nvidia-smi"
```

## 📊 Expected Output

### Successful Setup
```
🚀 Multi-GPU Test Configuration for Phase 3 VERL Integration
✅ Found 8 CUDA devices
✅ Ray cluster initialized with 8 GPUs
✅ Configuration files created in configs/
✅ GPU memory test completed successfully
```

### Successful Tests
```
🚀 Running Quick Multi-GPU Setup Verification...
📊 Available GPUs: 8
📊 Ray initialized: True
✅ VERL coordinator test passed
✅ Evaluation framework test passed
📊 Quick Test Results: 2/2 passed
```

## 📝 Important Notes

1. **SSH Key Security**: Keep your SSH key private and secure
2. **Instance Costs**: Remember to stop your Lambda Labs instance when not in use
3. **File Synchronization**: Scripts will overwrite remote files with local versions
4. **Environment Variables**: Scripts set up GPU environment variables automatically

## 🎯 Next Steps After Setup

Once setup is complete, you can:
1. Run training jobs using VERL
2. Monitor GPU usage via Ray dashboard
3. Scale up/down GPU usage as needed
4. Deploy models for inference

## 🆘 Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your SSH key and Lambda Labs instance status
3. Run scripts individually to isolate problems
4. Check Lambda Labs console for instance logs