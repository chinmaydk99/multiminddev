# 🚀 Multi-GPU Setup and Testing Guide for Phase 3 VERL Integration

This guide provides step-by-step instructions for setting up and testing Phase 3 VERL integration on a multi-GPU system (4 GPUs).

## 📋 Prerequisites

### System Requirements
- **4 GPUs**: NVIDIA GPUs with CUDA support (RTX 3080/4080, A100, H100, etc.)
- **CUDA**: Version 11.8 or 12.1
- **Python**: 3.9-3.11
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ free space

### Initial Verification
```bash
# Check GPU availability
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check Python version
python --version
```

## 🛠️ Step-by-Step Setup

### Step 1: Environment Setup
```bash
# Clone/navigate to the project
cd /path/to/MultiMindDev

# Make setup script executable (Linux/Mac)
chmod +x setup_multigpu_environment.sh

# Run environment setup
./setup_multigpu_environment.sh

# On Windows, use:
# python -m pip install -r requirements.txt
# (then follow manual installation steps below)
```

### Step 2: Manual Installation (if needed)
```bash
# Activate virtual environment
source venv_multigpu/bin/activate  # Linux/Mac
# or
venv_multigpu\Scripts\activate     # Windows

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install "ray[default,train,tune]" ray[rllib]
pip install transformers accelerate datasets tokenizers
pip install wandb tensorboard psutil GPUtil
pip install pytest pytest-asyncio pytest-cov
pip install pydantic structlog rich
```

### Step 3: Configure Multi-GPU Environment
```bash
# Run configuration script
python configure_multigpu_tests.py
```

This script will:
- ✅ Detect and verify your 4 GPUs
- ⚡ Initialize Ray cluster with GPU resources
- 📝 Create optimized configuration files
- 📁 Setup test directories and environment variables
- 🧪 Run GPU memory tests

### Step 4: Quick Setup Verification
```bash
# Run quick verification test
python quick_setup_test.py
```

Expected output:
```
🚀 Running Quick Multi-GPU Setup Verification...
📊 Available GPUs: 4
📊 Ray initialized: True

==================================================
Testing: VERL Components
==================================================
✅ VERL coordinator test passed

==================================================
Testing: Evaluation Framework  
==================================================
✅ Evaluation framework test passed (164 problems loaded)

📊 Quick Test Results: 2/2 passed
🎉 Setup verification complete! Ready to run full tests.
```

## 🧪 Running Tests

### Option 1: Full Multi-GPU Integration Tests
```bash
# Run comprehensive multi-GPU tests
python run_multigpu_integration_tests.py
```

This runs:
- 🔥 **Multi-GPU VERL Integration**: Distributed training across 4 GPUs
- 📊 **Parallel Evaluation Framework**: Concurrent benchmark evaluation
- 🚀 **Distributed Deployment**: GPU-aware model serving
- 📈 **Multi-GPU Monitoring**: Resource tracking and observability
- 🔄 **End-to-End Integration**: Complete workflow validation

### Option 2: Individual Component Tests
```bash
# Test specific components
python -m pytest src/coding_framework/tests/test_verl_integration_comprehensive.py -v
python -m pytest src/coding_framework/tests/test_deployment_integration.py -v
```

### Option 3: Original Single-GPU Tests
```bash
# Run original integration tests (single GPU)
python run_phase3_integration_test.py
```

## 📊 Understanding Test Results

### Successful Multi-GPU Test Output
```
🚀 PHASE 3 VERL INTEGRATION - MULTI-GPU TEST SUITE (4 GPUs Available)
================================================================================
🔥 CUDA Version: 11.8
   GPU 0: NVIDIA RTX 4080 (16.0 GB)
   GPU 1: NVIDIA RTX 4080 (16.0 GB) 
   GPU 2: NVIDIA RTX 4080 (16.0 GB)
   GPU 3: NVIDIA RTX 4080 (16.0 GB)
⚡ Ray Cluster: 4.0 GPUs, 16 CPUs

======================================================================
Running: Multi-GPU VERL Integration
======================================================================
[4xGPU] VERL Coordinator: Initializing with GPU distribution...
[4xGPU] Ray Cluster: Available GPUs: 4.0
✅ Multi-GPU VERL Integration: PASSED (15.32s)
    GPU: 87.2% peak utilization, 24.8 GB total memory used

... (additional tests) ...

📈 SUMMARY: 5/5 tests passed (100.0%)
⏱️  Total Duration: 45.67 seconds
🔥 GPUs Available: 4
🎉 ALL MULTI-GPU TESTS PASSED! Phase 3 is ready for production.

✨ Multi-GPU Phase 3 Implementation Status: COMPLETE (4x GPU) ✨
```

## 🔧 Configuration Files

The setup creates optimized configurations in `configs/`:

### `verl_multigpu_config.json`
```json
{
  "algorithm": "ppo",
  "num_gpus": 4,
  "strategy": "fsdp2", 
  "ray_config": {
    "num_workers": 4,
    "num_cpus_per_worker": 4,
    "num_gpus_per_worker": 1
  },
  "training_config": {
    "batch_size": 128,
    "gradient_accumulation_steps": 2,
    "learning_rate": 1e-4
  }
}
```

### `evaluation_multigpu_config.json`
```json
{
  "parallel_execution": true,
  "max_concurrent_evaluations": 3,
  "benchmarks": {
    "humaneval": {"max_samples": 10, "timeout": 30},
    "mbpp": {"max_samples": 10, "timeout": 30},
    "bigcodebench": {"max_samples": 5, "timeout": 60}
  }
}
```

## 📈 Monitoring and Observability

### Ray Dashboard
- **URL**: http://localhost:8265
- **Features**: Real-time GPU utilization, memory usage, task distribution

### Test Results
- **Location**: `./test_results/`
- **Reports**: JSON reports with detailed GPU utilization metrics
- **Logs**: Comprehensive logging with GPU-specific information

## ⚡ Performance Optimization Tips

### GPU Memory Management
```bash
# Monitor GPU memory usage
nvidia-smi -l 1

# Clear GPU cache if needed
python -c "import torch; torch.cuda.empty_cache()"
```

### Batch Size Tuning
- **4 GPUs**: Use batch_size = 128-256 (32-64 per GPU)
- **2 GPUs**: Use batch_size = 64-128 (32-64 per GPU)
- **1 GPU**: Use batch_size = 32-64

### Ray Cluster Optimization
```python
# Optimal Ray configuration for 4 GPUs
ray_config = {
    "num_workers": 4,           # One worker per GPU
    "num_cpus_per_worker": 4,   # 4 CPUs per worker
    "num_gpus_per_worker": 1,   # 1 GPU per worker
    "resources_per_worker": {
        "GPU": 1
    }
}
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory
```bash
# Reduce batch size in configs/verl_multigpu_config.json
"batch_size": 64  # Instead of 128

# Or enable gradient checkpointing
"gradient_checkpointing": true
```

#### 2. Ray Cluster Issues
```bash
# Reset Ray cluster
python -c "import ray; ray.shutdown()"
python configure_multigpu_tests.py
```

#### 3. GPU Not Detected
```bash
# Check CUDA installation
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Slow Performance
```bash
# Enable optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Use mixed precision training
"mixed_precision": true
```

### Environment Variables
```bash
# Set these for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_DISTRIBUTED_DEBUG=INFO
export RAY_DISABLE_IMPORT_WARNING=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
```

## 📊 Expected Performance Metrics

### Training Performance (4 GPUs)
- **Throughput**: 400-800 samples/second
- **GPU Utilization**: 80-95%
- **Memory Usage**: 12-14 GB per GPU
- **Gradient Sync**: 50-100ms

### Evaluation Performance
- **HumanEval**: 10 problems in <30 seconds
- **MBPP**: 10 problems in <30 seconds  
- **BigCodeBench**: 5 problems in <60 seconds
- **Parallel Execution**: 3 concurrent evaluations

### Deployment Performance
- **Model Loading**: 5-15 seconds per GPU
- **Inference**: 50-200ms per request
- **Throughput**: 100-500 requests/second
- **Health Checks**: <5 seconds

## 🎯 Next Steps After Successful Testing

1. **Production Deployment**:
   ```bash
   # Use the deployment orchestrator for production
   python -c "
   from src.coding_framework.deployment.deployment_orchestrator import DeploymentOrchestrator
   # Deploy trained models with multi-GPU support
   "
   ```

2. **Training Real Models**:
   ```bash
   # Start VERL distributed training
   python -c "
   from src.coding_framework.verl_integration.verl_coordinator import VERLCoordinator
   # Initialize real training with your dataset
   "
   ```

3. **Continuous Monitoring**:
   ```bash
   # Setup production monitoring
   python -c "
   from src.coding_framework.monitoring.verl_training_monitor import VERLTrainingMonitor
   # Monitor production training
   "
   ```

## 📝 Support and Resources

- **Ray Documentation**: https://docs.ray.io/
- **VERL Repository**: https://github.com/volcengine/verl
- **PyTorch Distributed**: https://pytorch.org/tutorials/beginner/dist_overview.html
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit

---

**🎉 Congratulations! You now have a fully functional multi-GPU Phase 3 VERL integration system ready for production use.**