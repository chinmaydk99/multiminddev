# Consolidated CUDA workflow integration tests
# This replaces: core_integration_test.py, simple_verl_test.py, integration_test_cuda_workflow.py

import pytest
import asyncio
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from coding_framework.agents.cuda_generator import CUDAGeneratorAgent
from coding_framework.agents.cuda_optimizer import CUDAOptimizerAgent
from coding_framework.agents.cuda_tester import CUDATesterAgent
from coding_framework.orchestration.cuda_workflow import CUDAKernelWorkflow


class MockConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def dict(self):
        return {k: v for k, v in self.__dict__.items()}


class MockLLMInterface:
    def __init__(self):
        self.call_count = 0
        
    async def call(self, messages, **kwargs):
        self.call_count += 1
        context = str(messages).lower()
        
        if "optimize" in context:
            return """#include <cuda_runtime.h>
__global__ void optimized_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 2.0f;
}"""
        elif "test" in context:
            return "Compilation: SUCCESS, Performance: 2.1x speedup"
        else:
            return """#include <cuda_runtime.h>
__global__ void basic_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 2.0f;
}"""


@pytest.mark.asyncio
async def test_cuda_agents_integration():
    """Test multi-agent CUDA workflow integration."""
    config = MockConfig(temperature=0.7, max_tokens=2048)
    llm = MockLLMInterface()
    
    generator = CUDAGeneratorAgent(config, llm)
    optimizer = CUDAOptimizerAgent(config, llm)
    tester = CUDATesterAgent(config, llm)
    
    # Test multi-turn conversation
    gen_response = await generator.process_request("Create vector add kernel")
    assert gen_response.success
    
    test_response = await tester.process_request(gen_response.content)
    assert test_response.success
    
    opt_response = await optimizer.process_request(gen_response.content)
    assert opt_response.success


@pytest.mark.asyncio
async def test_cuda_workflow_orchestration():
    """Test CUDA workflow orchestration."""
    config = MockConfig(temperature=0.7, max_tokens=2048)
    llm = MockLLMInterface()
    
    generator = CUDAGeneratorAgent(config, llm)
    optimizer = CUDAOptimizerAgent(config, llm)
    tester = CUDATesterAgent(config, llm)
    
    workflow = CUDAKernelWorkflow(generator, optimizer, tester)
    
    # Test workflow status
    status = await workflow.get_workflow_status()
    assert status is not None
    assert isinstance(status, dict)


if __name__ == "__main__":
    # Allow running directly for quick testing
    asyncio.run(test_cuda_agents_integration())
    asyncio.run(test_cuda_workflow_orchestration())
    print("✅ All CUDA integration tests passed!")