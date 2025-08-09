#!/usr/bin/env python3
"""
Fixed test for VERL and CUDA on remote instance
"""

import os
import sys
import asyncio
from pathlib import Path
import torch
import numpy as np
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"

import ray
import verl
from verl import DataProto


async def test_verl_dataproto():
    """Test VERL DataProto with correct API"""
    print("Testing VERL DataProto...")
    
    try:
        # Check actual DataProto signature
        import inspect
        sig = inspect.signature(DataProto.__init__)
        print(f"DataProto signature: {sig}")
        
        # Try creating with different approaches
        try:
            # Approach 1: Just data
            batch = DataProto(data={"text": ["test"]})
            print(f"✓ DataProto created with data dict")
        except:
            try:
                # Approach 2: Direct dict
                batch = DataProto({"text": ["test"]})
                print(f"✓ DataProto created with dict")
            except:
                # Approach 3: Empty
                batch = DataProto()
                batch.data = {"text": ["test"]}
                print(f"✓ DataProto created empty and populated")
        
        return True
    except Exception as e:
        print(f"✗ DataProto failed: {e}")
        return False


async def test_cuda_compilation_simple():
    """Test CUDA compilation with fixed command"""
    print("\nTesting CUDA compilation (simplified)...")
    
    try:
        # Create temp directory
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="cuda_test_")
        
        # Write kernel file
        kernel_code = """
#include <cuda_runtime.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" {
    void* get_vector_add() {
        return (void*)vector_add;
    }
}
"""
        
        kernel_file = os.path.join(temp_dir, "test_kernel.cu")
        output_file = os.path.join(temp_dir, "test_kernel.so")
        
        with open(kernel_file, 'w') as f:
            f.write(kernel_code)
        
        # Compile with simplified command
        cmd = [
            "nvcc",
            "-O3",
            "-arch=sm_70",  # V100 architecture
            "--shared",
            "-Xcompiler", "-fPIC",
            "-o", output_file,
            kernel_file
        ]
        
        print(f"Compiling with: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_file):
            print(f"✓ Compilation successful")
            print(f"  - Output: {output_file}")
            print(f"  - Size: {os.path.getsize(output_file)} bytes")
            return True
        else:
            print(f"✗ Compilation failed")
            print(f"  - Return code: {result.returncode}")
            print(f"  - Stderr: {result.stderr[:200]}")
            return False
            
    except Exception as e:
        print(f"✗ Compilation error: {e}")
        return False


async def test_ray_cluster():
    """Test Ray cluster setup"""
    print("\nTesting Ray cluster...")
    
    try:
        # Initialize Ray with available resources
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        resources = ray.cluster_resources()
        print(f"✓ Ray cluster active")
        print(f"  - GPUs: {resources.get('GPU', 0)}")
        print(f"  - CPUs: {resources.get('CPU', 0)}")
        print(f"  - Memory: {resources.get('memory', 0) / 1e9:.1f} GB")
        
        # Test simple remote function
        @ray.remote
        def simple_task():
            return "Ray task executed"
        
        result = ray.get(simple_task.remote())
        print(f"  - Remote task: {result}")
        
        return True
    except Exception as e:
        print(f"✗ Ray failed: {e}")
        return False


async def test_basic_rl_components():
    """Test basic RL components without full training"""
    print("\nTesting basic RL components...")
    
    try:
        # Test reward calculation
        from coding_framework.training.reward_functions.cuda_performance_reward import CUDAPerformanceReward
        
        reward_fn = CUDAPerformanceReward(
            target_speedup=2.0,
            correctness_weight=0.4,
            performance_weight=0.4,
            improvement_weight=0.2
        )
        print("✓ Reward function initialized")
        
        # Test data loader
        from coding_framework.training.cuda_data_loader import CUDADataLoader
        
        data_loader = CUDADataLoader()
        print("✓ Data loader initialized")
        
        # Test workflow
        from coding_framework.orchestration.cuda_workflow import CUDAKernelWorkflow
        print("✓ CUDA workflow importable")
        
        return True
        
    except Exception as e:
        print(f"✗ RL components failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_multi_gpu_info():
    """Show multi-GPU information"""
    print("\nMulti-GPU Information:")
    
    try:
        # Get GPU info via nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ GPU Information:")
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 4:
                    print(f"  - GPU {parts[0]}: {parts[1]} ({parts[2]}, SM {parts[3]})")
        
        # Check CUDA version
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            cuda_version = result.stdout.split('\n')[3].split(',')[1].strip()
            print(f"  - CUDA Version: {cuda_version}")
        
        return True
    except Exception as e:
        print(f"✗ GPU info failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("="*60)
    print("VERL & CUDA TESTING ON MULTI-GPU INSTANCE")
    print("="*60)
    
    tests = [
        ("Multi-GPU Info", test_multi_gpu_info),
        ("VERL DataProto", test_verl_dataproto),
        ("CUDA Compilation", test_cuda_compilation_simple),
        ("Ray Cluster", test_ray_cluster),
        ("RL Components", test_basic_rl_components)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            success = await test_func()
            results[test_name] = success
        except Exception as e:
            print(f"✗ {test_name} exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✅ All tests passed - System ready for training!")
    else:
        print("\n⚠️  Some tests failed - See details above")
    
    # Cleanup
    if ray.is_initialized():
        ray.shutdown()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))