# Consolidated system validation tests
# This replaces: validate_cuda_implementation.py, validate_simple.py, test_fixes.py, test_fixes_simple.py

import pytest
import asyncio
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from coding_framework.agents.cuda_generator import CUDAGeneratorAgent
from coding_framework.agents.cuda_optimizer import CUDAOptimizerAgent
from coding_framework.agents.cuda_tester import CUDATesterAgent

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False


class MockConfig:
    temperature = 0.7
    max_tokens = 2048
    def dict(self):
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}


class MockLLMInterface:
    async def call(self, messages, **kwargs):
        return "__global__ void test_kernel() {}"


@pytest.mark.asyncio
async def test_agent_initialization():
    """Test that all CUDA agents can be initialized properly."""
    config = MockConfig()
    llm = MockLLMInterface()
    
    # Test agent creation
    generator = CUDAGeneratorAgent(config, llm)
    optimizer = CUDAOptimizerAgent(config, llm)
    tester = CUDATesterAgent(config, llm)
    
    # Test health checks
    gen_health = await generator.health_check()
    opt_health = await optimizer.health_check()
    test_health = await tester.health_check()
    
    assert gen_health["status"] == "healthy"
    assert opt_health["status"] == "healthy"
    assert test_health["status"] == "healthy"


@pytest.mark.asyncio
async def test_basic_workflow():
    """Test basic multi-agent workflow."""
    config = MockConfig()
    llm = MockLLMInterface()
    
    generator = CUDAGeneratorAgent(config, llm)
    
    # Test basic request processing
    response = await generator.process_request("Create simple kernel")
    
    assert response.success
    assert response.content is not None
    assert len(response.content) > 0


def test_import_validation():
    """Test that all critical imports work."""
    # Test core agent imports
    from coding_framework.agents import (
        BaseAgent, CUDAGeneratorAgent, CUDAOptimizerAgent, CUDATesterAgent
    )
    
    # Test CUDA component imports
    from coding_framework.cuda import CUDACompiler, CUDABenchmarker
    
    # Test orchestration imports
    from coding_framework.orchestration import CUDAKernelWorkflow
    
    # Test training imports
    from coding_framework.training.reward_functions import CUDAPerformanceReward
    
    assert True  # If we get here, all imports worked


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="CUDA not available")
def test_cuda_availability():
    """Test CUDA environment when available."""
    import torch
    
    assert torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    assert device_count > 0
    
    # Test basic tensor operations
    x = torch.randn(100, device='cuda')
    y = torch.randn(100, device='cuda')
    z = x + y
    
    assert z.is_cuda
    assert z.shape == (100,)


if __name__ == "__main__":
    # Allow running directly for quick testing
    test_import_validation()
    
    asyncio.run(test_agent_initialization())
    asyncio.run(test_basic_workflow())
    
    if TORCH_AVAILABLE:
        test_cuda_availability()
        
    print("✅ All system validation tests passed!")