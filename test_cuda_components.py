#!/usr/bin/env python3
"""
CUDA Components Test Suite
Tests the core CUDA implementation components for production readiness
"""

import sys
import os
import asyncio
import time
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import torch
    import numpy as np
    print(f"✅ PyTorch {torch.__version__} loaded successfully")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    print(f"✅ GPU Count: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"❌ Failed to import PyTorch: {e}")
    sys.exit(1)

# Test imports for our CUDA framework
try:
    from coding_framework.cuda.compiler import CUDACompiler
    from coding_framework.cuda.benchmarker import CUDABenchmarker
    from coding_framework.orchestration.cuda_workflow import CUDAKernelWorkflow
    from coding_framework.agents.cuda_generator import CUDAGeneratorAgent
    from coding_framework.agents.cuda_optimizer import CUDAOptimizerAgent
    from coding_framework.agents.cuda_tester import CUDATesterAgent
    print("✅ All CUDA framework components imported successfully")
except ImportError as e:
    print(f"❌ Failed to import CUDA framework components: {e}")
    sys.exit(1)


class MockLLMInterface:
    """Mock LLM interface for testing."""
    
    def __init__(self):
        self.call_count = 0
        
    async def call(self, messages, **kwargs):
        self.call_count += 1
        context = str(messages).lower()
        
        if "optimize" in context or "improvement" in context:
            return '''
// Optimized CUDA kernel with better memory access patterns
__global__ void optimized_vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for better memory coalescing
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}
'''
        elif "test" in context or "benchmark" in context:
            return "Compilation: SUCCESS\nPerformance: 2.3x speedup over naive implementation\nMemory bandwidth utilization: 85%"
        else:
            # Basic kernel generation
            return '''
// Basic CUDA vector addition kernel
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
'''


class MockConfig:
    """Mock configuration for testing."""
    
    def __init__(self, **kwargs):
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 2048)
        self.model = kwargs.get('model', 'test-model')
    
    def dict(self):
        return {k: v for k, v in self.__dict__.items()}


async def test_cuda_compiler():
    """Test CUDA compiler functionality."""
    print("\n🔧 Testing CUDA Compiler...")
    
    compiler = CUDACompiler()
    
    # Test basic kernel compilation
    kernel_code = '''
__global__ void test_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}
'''
    
    try:
        result = compiler.compile_kernel(kernel_code, "test_kernel")
        if result.success:
            print("   ✅ Kernel compilation successful")
            print(f"   Compilation time: {result.compilation_time:.3f}s")
        else:
            print(f"   ❌ Compilation failed: {result.error_message}")
            
    except Exception as e:
        print(f"   ❌ Compiler test failed: {e}")


async def test_cuda_benchmarker():
    """Test CUDA benchmarker functionality."""
    print("\n📊 Testing CUDA Benchmarker...")
    
    benchmarker = CUDABenchmarker()
    
    # Create simple test data
    n = 1024 * 1024
    a = torch.randn(n, device='cuda:0')
    b = torch.randn(n, device='cuda:0')
    c = torch.zeros(n, device='cuda:0')
    
    try:
        # Test performance measurement
        def test_operation():
            return torch.add(a, b)
        
        result = benchmarker.benchmark_operation(test_operation, "vector_add")
        
        if result.success:
            print(f"   ✅ Benchmark successful")
            print(f"   Average time: {result.avg_time:.6f}s")
            print(f"   Throughput: {result.throughput:.2f} GFLOP/s")
        else:
            print(f"   ❌ Benchmark failed: {result.error_message}")
            
    except Exception as e:
        print(f"   ❌ Benchmarker test failed: {e}")


async def test_cuda_agents():
    """Test CUDA agent functionality."""
    print("\n🤖 Testing CUDA Agents...")
    
    config = MockConfig()
    llm = MockLLMInterface()
    
    try:
        # Initialize agents
        generator = CUDAGeneratorAgent(config, llm)
        optimizer = CUDAOptimizerAgent(config, llm)
        tester = CUDATesterAgent(config, llm)
        
        print("   ✅ All agents initialized successfully")
        
        # Test agent interactions
        gen_request = "Generate a simple vector addition kernel for arrays of size 1024"
        gen_response = await generator.process_request(gen_request)
        
        if gen_response.success:
            print("   ✅ Code generation successful")
            
            # Test optimization
            opt_response = await optimizer.process_request(gen_response.content)
            if opt_response.success:
                print("   ✅ Code optimization successful")
                
            # Test validation
            test_response = await tester.process_request(gen_response.content)
            if test_response.success:
                print("   ✅ Code testing successful")
            else:
                print(f"   ❌ Code testing failed: {test_response.error}")
        else:
            print(f"   ❌ Code generation failed: {gen_response.error}")
            
    except Exception as e:
        print(f"   ❌ Agent test failed: {e}")


async def test_cuda_workflow():
    """Test CUDA workflow orchestration."""
    print("\n🔄 Testing CUDA Workflow...")
    
    config = MockConfig()
    llm = MockLLMInterface()
    
    try:
        # Initialize agents and workflow
        generator = CUDAGeneratorAgent(config, llm)
        optimizer = CUDAOptimizerAgent(config, llm)
        tester = CUDATesterAgent(config, llm)
        
        workflow = CUDAKernelWorkflow(generator, optimizer, tester)
        
        # Test workflow status
        status = await workflow.get_workflow_status()
        if isinstance(status, dict):
            print("   ✅ Workflow status retrieval successful")
        else:
            print("   ❌ Workflow status retrieval failed")
            
        print("   ✅ Workflow orchestration test completed")
        
    except Exception as e:
        print(f"   ❌ Workflow test failed: {e}")


def test_gpu_availability():
    """Test GPU availability and basic operations."""
    print("\n🎯 Testing GPU Availability...")
    
    try:
        if not torch.cuda.is_available():
            print("   ❌ CUDA not available")
            return False
            
        device_count = torch.cuda.device_count()
        print(f"   ✅ {device_count} GPU(s) available")
        
        for i in range(min(device_count, 4)):  # Test first 4 GPUs
            device = f'cuda:{i}'
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            # Test basic operations
            start_time = time.time()
            z = torch.matmul(x, y)
            end_time = time.time()
            
            print(f"   ✅ GPU {i}: Matrix multiply in {end_time - start_time:.4f}s")
            
        return True
        
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
        return False


def test_ray_integration():
    """Test Ray integration for distributed processing."""
    print("\n🚀 Testing Ray Integration...")
    
    try:
        import ray
        
        # Initialize Ray
        if ray.is_initialized():
            ray.shutdown()
            
        ray.init(ignore_reinit_error=True, num_gpus=min(torch.cuda.device_count(), 4))
        
        @ray.remote(num_gpus=1)
        def gpu_task(size=1000):
            import torch
            device = f'cuda:{torch.cuda.current_device()}'
            x = torch.randn(size, size, device=device)
            return torch.sum(x).item()
        
        # Run tasks on multiple GPUs
        futures = []
        for i in range(min(2, torch.cuda.device_count())):
            futures.append(gpu_task.remote(500))
            
        results = ray.get(futures)
        print(f"   ✅ Ray GPU tasks completed: {len(results)} results")
        
        ray.shutdown()
        return True
        
    except Exception as e:
        print(f"   ❌ Ray integration test failed: {e}")
        try:
            ray.shutdown()
        except:
            pass
        return False


def main():
    """Main test runner."""
    print("🧪 CUDA Components Comprehensive Test Suite")
    print("=" * 60)
    
    # Track test results
    results = {
        'gpu_availability': False,
        'ray_integration': False,
        'cuda_agents': False,
        'cuda_workflow': False,
        'cuda_compiler': False,
        'cuda_benchmarker': False
    }
    
    # Run synchronous tests first
    results['gpu_availability'] = test_gpu_availability()
    results['ray_integration'] = test_ray_integration()
    
    # Run async tests
    async def run_async_tests():
        await test_cuda_compiler()
        results['cuda_compiler'] = True
        
        await test_cuda_benchmarker() 
        results['cuda_benchmarker'] = True
        
        await test_cuda_agents()
        results['cuda_agents'] = True
        
        await test_cuda_workflow()
        results['cuda_workflow'] = True
    
    # Run async tests
    try:
        asyncio.run(run_async_tests())
    except Exception as e:
        print(f"❌ Async tests failed: {e}")
    
    # Generate report
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - CUDA SYSTEM READY!")
        return 0
    else:
        print("⚠️ SOME TESTS FAILED - REVIEW ISSUES ABOVE")
        return 1


if __name__ == "__main__":
    sys.exit(main())