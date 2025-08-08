#!/usr/bin/env python3
"""
Multi-GPU Configuration and Test Setup

This script configures the system for multi-GPU VERL training and testing.
Run this after setting up the environment but before running tests.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import torch
import ray


def check_gpu_setup():
    """Check GPU configuration and availability."""
    print("🔍 Checking GPU Setup...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Please check your PyTorch installation.")
        return False
        
    gpu_count = torch.cuda.device_count()
    print(f"✅ Found {gpu_count} CUDA devices")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
    if gpu_count < 2:
        print("⚠️  Warning: Multi-GPU features require at least 2 GPUs")
        
    return True


def setup_ray_cluster():
    """Initialize Ray cluster for distributed training."""
    print("\n⚡ Setting up Ray cluster...")
    
    try:
        # Shutdown any existing Ray instance
        if ray.is_initialized():
            ray.shutdown()
            
        # Initialize Ray with GPU resources
        gpu_count = torch.cuda.device_count()
        
        ray_config = {
            "num_cpus": os.cpu_count(),
            "num_gpus": gpu_count,
            "object_store_memory": 2000000000,  # 2GB
            "dashboard_host": "0.0.0.0",
            "dashboard_port": 8265,
            "_temp_dir": str(Path.cwd() / "ray_temp")
        }
        
        ray.init(**ray_config)
        
        print(f"✅ Ray cluster initialized with {gpu_count} GPUs")
        print(f"📊 Ray Dashboard: http://localhost:8265")
        
        # Verify cluster resources
        resources = ray.cluster_resources()
        print(f"   Available CPUs: {resources.get('CPU', 0)}")
        print(f"   Available GPUs: {resources.get('GPU', 0)}")
        print(f"   Available Memory: {resources.get('memory', 0) / (1024**3):.1f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize Ray cluster: {e}")
        return False


def create_test_configs():
    """Create configuration files for multi-GPU testing."""
    print("\n📝 Creating test configurations...")
    
    gpu_count = torch.cuda.device_count()
    
    # VERL Multi-GPU Configuration
    verl_config = {
        "algorithm": "ppo",
        "num_gpus": min(gpu_count, 4),  # Use up to 4 GPUs
        "strategy": "fsdp2",
        "ray_config": {
            "num_workers": min(gpu_count, 4),
            "num_cpus_per_worker": max(1, os.cpu_count() // gpu_count),
            "num_gpus_per_worker": 1,
            "resources_per_worker": {
                "GPU": 1
            }
        },
        "training_config": {
            "batch_size": 32 * min(gpu_count, 4),  # Scale batch size with GPUs
            "gradient_accumulation_steps": 2,
            "learning_rate": 1e-4,
            "max_steps": 100,
            "eval_steps": 25,
            "save_steps": 50
        },
        "model_config": {
            "model_name": "microsoft/DialoGPT-small",  # Lightweight model for testing
            "max_length": 512,
            "padding_side": "left"
        }
    }
    
    # Save VERL config
    config_path = Path("configs")
    config_path.mkdir(exist_ok=True)
    
    with open(config_path / "verl_multigpu_config.json", "w") as f:
        json.dump(verl_config, f, indent=2)
        
    # Evaluation Configuration for Multi-GPU
    eval_config = {
        "parallel_execution": True,
        "max_concurrent_evaluations": min(gpu_count, 3),
        "benchmarks": {
            "humaneval": {
                "max_samples": 10,  # Reduced for testing
                "timeout": 30
            },
            "mbpp": {
                "max_samples": 10,
                "use_three_shot": True,
                "timeout": 30
            },
            "bigcodebench": {
                "max_samples": 5,  # Most resource intensive
                "mode": "instruct",
                "timeout": 60
            }
        }
    }
    
    with open(config_path / "evaluation_multigpu_config.json", "w") as f:
        json.dump(eval_config, f, indent=2)
        
    # Deployment Configuration
    deployment_config = {
        "target_environment": "development",
        "deployment_strategy": "blue_green",
        "replicas": min(gpu_count, 2),
        "resource_requirements": {
            "gpu": 1,
            "cpu": 2,
            "memory": "8Gi"
        },
        "health_check_timeout": 60,
        "max_deployment_time": 600
    }
    
    with open(config_path / "deployment_multigpu_config.json", "w") as f:
        json.dump(deployment_config, f, indent=2)
        
    # Monitoring Configuration
    monitoring_config = {
        "metrics_collection_interval": 15,
        "enable_gpu_monitoring": True,
        "enable_distributed_monitoring": True,
        "export_metrics_to_file": True,
        "metrics_export_path": "./test_results/monitoring",
        "alert_rules": {
            "high_gpu_memory": {
                "threshold": 90.0,
                "condition": "greater_than",
                "severity": "warning"
            },
            "low_gpu_utilization": {
                "threshold": 20.0,
                "condition": "less_than", 
                "severity": "info"
            }
        }
    }
    
    with open(config_path / "monitoring_multigpu_config.json", "w") as f:
        json.dump(monitoring_config, f, indent=2)
        
    print(f"✅ Configuration files created in {config_path}/")
    
    return True


def setup_test_environment():
    """Set up directories and environment variables for testing."""
    print("\n📁 Setting up test environment...")
    
    # Create necessary directories
    directories = [
        "test_results",
        "test_results/monitoring", 
        "test_results/evaluation",
        "test_results/deployment",
        "test_models",
        "logs",
        "ray_temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    # Set environment variables for multi-GPU testing
    env_vars = {
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(torch.cuda.device_count())),
        "TORCH_DISTRIBUTED_DEBUG": "INFO",
        "RAY_DISABLE_IMPORT_WARNING": "1",
        "TOKENIZERS_PARALLELISM": "false",  # Avoid tokenizer warnings
        "WANDB_DISABLED": "true",  # Disable wandb for testing
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        
    print("✅ Test environment configured")
    
    return True


def run_gpu_memory_test():
    """Run a quick GPU memory test to verify setup."""
    print("\n🧪 Running GPU memory test...")
    
    try:
        gpu_count = torch.cuda.device_count()
        
        for i in range(gpu_count):
            torch.cuda.set_device(i)
            
            # Allocate a small tensor on each GPU
            x = torch.randn(1000, 1000, device=f'cuda:{i}')
            y = torch.randn(1000, 1000, device=f'cuda:{i}')
            z = torch.matmul(x, y)
            
            memory_used = torch.cuda.memory_allocated(i) / (1024**2)  # MB
            print(f"   GPU {i}: Memory test passed ({memory_used:.1f} MB used)")
            
            # Clean up
            del x, y, z
            torch.cuda.empty_cache()
            
        print("✅ GPU memory test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ GPU memory test failed: {e}")
        return False


def create_quick_test_script():
    """Create a quick test script to verify the setup."""
    
    quick_test_script = '''#!/usr/bin/env python3
"""
Quick Multi-GPU Setup Verification Test

Run this to quickly verify that your multi-GPU setup is working.
"""

import torch
import ray
import asyncio
import sys
from pathlib import Path

async def quick_verl_test():
    """Quick test of VERL components."""
    print("🔄 Testing VERL components...")
    
    try:
        # Import and test VERL coordinator
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from coding_framework.verl_integration.verl_coordinator import VERLCoordinator
        from coding_framework.verl_integration.verl_config import VERLDistributedConfig
        
        config = VERLDistributedConfig(
            algorithm="ppo",
            num_gpus=torch.cuda.device_count(),
            strategy="fsdp2"
        )
        
        coordinator = VERLCoordinator(config)
        result = await coordinator.initialize()
        
        if result.get("success"):
            print("✅ VERL coordinator test passed")
            return True
        else:
            print(f"❌ VERL coordinator test failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ VERL test failed: {e}")
        return False

async def quick_evaluation_test():
    """Quick test of evaluation framework."""
    print("🔄 Testing evaluation framework...")
    
    try:
        from coding_framework.evaluation.evaluators.humaneval_evaluator import HumanEvalEvaluator
        from coding_framework.agents.code_generator_agent import CodeGeneratorAgent
        from coding_framework.agents.agent_config import CodeGeneratorConfig
        
        # Create test agent
        agent = CodeGeneratorAgent(CodeGeneratorConfig())
        
        # Test evaluator
        evaluator = HumanEvalEvaluator()
        problems = evaluator.load_dataset()
        
        if len(problems) > 0:
            print(f"✅ Evaluation framework test passed ({len(problems)} problems loaded)")
            return True
        else:
            print("❌ Evaluation framework test failed: No problems loaded")
            return False
            
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False

async def main():
    """Run quick verification tests."""
    print("🚀 Running Quick Multi-GPU Setup Verification...")
    print(f"📊 Available GPUs: {torch.cuda.device_count()}")
    print(f"📊 Ray initialized: {ray.is_initialized()}")
    
    tests = [
        ("VERL Components", quick_verl_test),
        ("Evaluation Framework", quick_evaluation_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\\n{'='*50}")
        print(f"Testing: {test_name}")
        print('='*50)
        
        success = await test_func()
        if success:
            passed += 1
            
    print(f"\\n📊 Quick Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 Setup verification complete! Ready to run full tests.")
        print("\\n📋 Next steps:")
        print("   python run_phase3_integration_test.py")
    else:
        print("⚠️ Some tests failed. Please check the setup.")

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open("quick_setup_test.py", "w") as f:
        f.write(quick_test_script)
        
    print("✅ Quick test script created: quick_setup_test.py")


def main():
    """Main configuration function."""
    print("🚀 Multi-GPU Test Configuration for Phase 3 VERL Integration")
    print("=" * 70)
    
    success_steps = []
    
    # Step 1: Check GPU setup
    if check_gpu_setup():
        success_steps.append("GPU Setup")
        
    # Step 2: Setup Ray cluster
    if setup_ray_cluster():
        success_steps.append("Ray Cluster")
        
    # Step 3: Create test configurations
    if create_test_configs():
        success_steps.append("Test Configurations")
        
    # Step 4: Setup test environment
    if setup_test_environment():
        success_steps.append("Test Environment")
        
    # Step 5: Run GPU memory test
    if run_gpu_memory_test():
        success_steps.append("GPU Memory Test")
        
    # Step 6: Create quick test script
    create_quick_test_script()
    success_steps.append("Quick Test Script")
    
    print("\n" + "=" * 70)
    print("📊 Configuration Summary:")
    print(f"✅ Completed steps: {len(success_steps)}/6")
    
    for step in success_steps:
        print(f"   ✅ {step}")
        
    if len(success_steps) == 6:
        print("\n🎉 Multi-GPU configuration complete!")
        print("\n📋 Ready to run tests:")
        print("   1. Quick verification: python quick_setup_test.py")
        print("   2. Full integration tests: python run_phase3_integration_test.py")
        print("   3. Individual component tests: python -m pytest src/coding_framework/tests/")
        print("\n📊 Monitor Ray cluster: http://localhost:8265")
    else:
        print("\n⚠️ Configuration incomplete. Please fix the issues above.")
        
    print("=" * 70)


if __name__ == "__main__":
    main()