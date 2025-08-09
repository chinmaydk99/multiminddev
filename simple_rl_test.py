#!/usr/bin/env python3
"""
Simple RL Training Test for CUDA Code Generation
Test without emojis for Windows compatibility
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from coding_framework.utils.config import load_config, AgentConfig


async def test_basic_initialization():
    """Test basic component initialization."""
    print("Testing basic component initialization...")
    
    try:
        # Load configuration
        config = load_config()
        print(f"Configuration loaded successfully")
        print(f"LLM provider: {config.llm.provider}")
        print(f"LLM model: {config.llm.model}")
        
        return True
        
    except Exception as e:
        print(f"Failed to load configuration: {str(e)}")
        return False


async def test_llm_interface():
    """Test LLM interface initialization."""
    print("Testing LLM interface...")
    
    try:
        config = load_config()
        config.llm.provider = "huggingface" 
        config.llm.model = "bigcode/starcoder2-3b"
        
        from coding_framework.utils.llm_interface import LLMInterface
        llm_interface = LLMInterface(config.llm)
        await llm_interface.initialize()
        
        # Test a simple generation
        response = await llm_interface.generate("Hello, this is a test.")
        print(f"LLM response received: {len(response)} characters")
        
        return True
        
    except Exception as e:
        print(f"LLM interface test failed: {str(e)}")
        return False


async def test_cuda_agents():
    """Test CUDA agents initialization."""
    print("Testing CUDA agents...")
    
    try:
        from coding_framework.agents.cuda_generator import CUDAGeneratorAgent
        from coding_framework.agents.cuda_optimizer import CUDAOptimizerAgent  
        from coding_framework.agents.cuda_tester import CUDATesterAgent
        
        # Load configuration
        config = load_config()
        config.llm.provider = "huggingface" 
        config.llm.model = "bigcode/starcoder2-3b"
        
        from coding_framework.utils.llm_interface import LLMInterface
        llm_interface = LLMInterface(config.llm)
        await llm_interface.initialize()
        
        # Create agent config
        agent_config = AgentConfig(
            max_retries=2,
            timeout=60,
            enable_logging=True
        )
        
        # Initialize agents
        cuda_generator = CUDAGeneratorAgent(agent_config, llm_interface)
        cuda_optimizer = CUDAOptimizerAgent(agent_config, llm_interface)
        cuda_tester = CUDATesterAgent(agent_config, llm_interface)
        
        print("CUDA agents initialized successfully")
        
        # Health check
        gen_health = await cuda_generator.health_check()
        opt_health = await cuda_optimizer.health_check()
        test_health = await cuda_tester.health_check()
        
        print(f"Generator health: {gen_health['status']}")
        print(f"Optimizer health: {opt_health['status']}")
        print(f"Tester health: {test_health['status']}")
        
        return True
        
    except Exception as e:
        print(f"CUDA agents test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_reward_function():
    """Test reward function."""
    print("Testing reward function...")
    
    try:
        from coding_framework.training.reward_functions.cuda_performance_reward import CUDAPerformanceReward
        
        reward_function = CUDAPerformanceReward(
            target_speedup=2.0,
            correctness_weight=0.4,
            performance_weight=0.4,
            improvement_weight=0.2
        )
        
        # Test with sample data
        sample_kernel = """
__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}
"""
        
        test_cases = [{"shape": [1024], "dtype": "float32"}]
        
        reward = await reward_function.calculate_reward(
            problem="Vector addition: C = A + B",
            generated_code=sample_kernel,
            test_cases=test_cases,
            context={"turn": 0, "conversation_id": "test"}
        )
        
        print(f"Reward calculated: {reward}")
        return True
        
    except Exception as e:
        print(f"Reward function test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test runner."""
    print("CUDA RL Training Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Initialization", test_basic_initialization),
        ("LLM Interface", test_llm_interface),
        ("CUDA Agents", test_cuda_agents),
        ("Reward Function", test_reward_function)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nRunning test: {test_name}")
        print("-" * 30)
        
        start_time = time.time()
        try:
            success = await test_func()
            test_time = time.time() - start_time
            results[test_name] = {
                "success": success,
                "time": test_time
            }
            status = "PASSED" if success else "FAILED"
            print(f"Test {test_name}: {status} ({test_time:.2f}s)")
            
        except Exception as e:
            test_time = time.time() - start_time
            results[test_name] = {
                "success": False,
                "time": test_time,
                "error": str(e)
            }
            print(f"Test {test_name}: FAILED ({test_time:.2f}s) - {str(e)}")
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(1 for r in results.values() if r["success"])
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for test_name, result in results.items():
        status = "PASS" if result["success"] else "FAIL"
        print(f"  {test_name}: {status}")
        if not result["success"] and "error" in result:
            print(f"    Error: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())