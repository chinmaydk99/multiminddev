#!/usr/bin/env python3
"""
Basic test to verify VERL and run a simple RL training loop
Adapted to work with actual VERL structure
"""

import os
import sys
import asyncio
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"

import ray
import verl
from verl import DataProto

# Our components
from coding_framework.agents.cuda_generator import CUDAGeneratorAgent
from coding_framework.agents.cuda_optimizer import CUDAOptimizerAgent
from coding_framework.agents.cuda_tester import CUDATesterAgent
from coding_framework.cuda.compiler import CUDACompiler
from coding_framework.cuda.benchmarker import CUDABenchmarker
from coding_framework.utils.config import load_config, AgentConfig
from coding_framework.utils.llm_interface import LLMInterface


async def test_verl_basic():
    """Test basic VERL functionality"""
    print("Testing VERL basic functionality...")
    
    # Test DataProto
    try:
        batch = DataProto(
            prompts=["Generate CUDA kernel for vector addition"],
            responses=["__global__ void add() {}"],
            rewards=torch.tensor([0.5])
        )
        print(f"✓ DataProto created: {batch}")
    except Exception as e:
        print(f"✗ DataProto failed: {e}")
        return False
    
    # Test Ray
    try:
        if not ray.is_initialized():
            ray.init(num_gpus=1, num_cpus=4, ignore_reinit_error=True)
        resources = ray.cluster_resources()
        print(f"✓ Ray initialized - GPUs: {resources.get('GPU', 0)}, CPUs: {resources.get('CPU', 0)}")
    except Exception as e:
        print(f"✗ Ray failed: {e}")
        return False
    
    return True


async def test_cuda_compilation():
    """Test actual CUDA compilation"""
    print("\nTesting CUDA compilation...")
    
    compiler = CUDACompiler()
    
    # Detect CUDA architecture
    cuda_arch = compiler.detect_cuda_arch()
    print(f"Detected CUDA architecture: {cuda_arch}")
    
    # Simple kernel
    kernel_code = """
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""
    
    result = await compiler.compile_kernel(
        kernel_code=kernel_code,
        kernel_name="vector_add_test"
    )
    
    if result.success:
        print(f"✓ Compilation successful")
        print(f"  - Binary: {result.binary_path}")
        print(f"  - Time: {result.compilation_time:.3f}s")
        if result.register_pressure:
            print(f"  - Registers: {result.register_pressure}")
        return True
    else:
        print(f"✗ Compilation failed: {result.stderr[:200]}")
        return False


async def test_simple_rl_loop():
    """Test a simple RL training loop with our agents"""
    print("\nTesting simple RL loop...")
    
    try:
        # Initialize agents with minimal config
        config = load_config()
        config.llm.provider = "huggingface"
        config.llm.model = "bigcode/starcoder2-3b"
        
        llm_interface = LLMInterface(config.llm)
        await llm_interface.initialize()
        
        agent_config = AgentConfig(
            max_retries=1,
            timeout=30,
            enable_logging=True
        )
        
        # Create agents
        cuda_generator = CUDAGeneratorAgent(agent_config, llm_interface)
        
        print("✓ Agent initialized")
        
        # Simple training loop
        problems = [
            {
                "description": "Vector addition",
                "prompt": "Generate CUDA kernel for element-wise vector addition C = A + B",
                "torch_reference": "torch.add(A, B)"
            }
        ]
        
        rewards = []
        for i in range(3):  # 3 episodes
            print(f"\nEpisode {i+1}:")
            
            problem = problems[0]
            
            # Generate code
            response = await cuda_generator.process_request(
                pytorch_operation=problem["torch_reference"]
            )
            
            # Simple reward (random for now since we're testing)
            reward = np.random.random()
            rewards.append(reward)
            
            print(f"  Generated code length: {len(response.get('code', ''))} chars")
            print(f"  Reward: {reward:.3f}")
        
        print(f"\n✓ RL loop completed - Mean reward: {np.mean(rewards):.3f}")
        return True
        
    except Exception as e:
        print(f"✗ RL loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("="*60)
    print("VERL BASIC TESTING")
    print("="*60)
    
    tests = [
        ("VERL Basic", test_verl_basic),
        ("CUDA Compilation", test_cuda_compilation),
        ("Simple RL Loop", test_simple_rl_loop)
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
    print("SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    # Cleanup
    if ray.is_initialized():
        ray.shutdown()
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))