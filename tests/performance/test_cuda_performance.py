# Consolidated CUDA performance tests
# This replaces: performance_validation_test.py, test_cuda_components.py

import pytest
import asyncio
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from coding_framework.cuda.compiler import CUDACompiler
from coding_framework.cuda.benchmarker import CUDABenchmarker
from coding_framework.training.reward_functions.cuda_performance_reward import CUDAPerformanceReward

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="CUDA not available")
@pytest.mark.asyncio
async def test_cuda_compilation_performance():
    """Test CUDA kernel compilation performance."""
    compiler = CUDACompiler()
    
    test_kernel = """
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}"""
    
    start_time = time.time()
    result = await compiler.compile_kernel(test_kernel, "perf_test_kernel")
    compilation_time = time.time() - start_time
    
    assert result.success, f"Compilation failed: {result.stderr}"
    assert compilation_time < 5.0, f"Compilation too slow: {compilation_time}s"


@pytest.mark.asyncio
async def test_reward_function_performance():
    """Test CUDA reward function performance."""
    reward_function = CUDAPerformanceReward(target_speedup=2.0)
    
    test_conversation = """Agent: cuda_generator
#include <cuda_runtime.h>
__global__ void test_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 2.0f;
}"""
    
    start_time = time.time()
    reward = await reward_function.calculate_reward(
        problem="Element-wise scaling",
        generated_code=test_conversation,
        test_cases=[{"shape": [1024], "dtype": "float32"}]
    )
    calculation_time = time.time() - start_time
    
    assert -1.0 <= reward <= 1.0, f"Invalid reward: {reward}"
    assert calculation_time < 10.0, f"Reward calculation too slow: {calculation_time}s"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="CUDA not available")
def test_cuda_environment():
    """Test CUDA environment setup."""
    import torch
    assert torch.cuda.is_available(), "CUDA should be available"
    assert torch.cuda.device_count() > 0, "Should have at least one CUDA device"


if __name__ == "__main__":
    # Allow running directly for quick testing
    if TORCH_AVAILABLE:
        asyncio.run(test_cuda_compilation_performance())
        asyncio.run(test_reward_function_performance())
        test_cuda_environment()
        print("✅ All CUDA performance tests passed!")
    else:
        print("⚠️ CUDA not available, skipping performance tests")