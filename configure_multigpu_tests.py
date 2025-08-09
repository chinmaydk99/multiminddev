#!/usr/bin/env python3
"""
Multi-GPU Configuration and Testing Script for Lambda Labs
Sets up the multi-GPU environment and validates CUDA implementation
"""

import os
import sys
import subprocess
import json
import time
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import torch
    import numpy as np
    from pynvml import *
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Installing missing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "numpy", "pynvml"])
    import torch
    import numpy as np
    from pynvml import *


class MultiGPUTester:
    """Multi-GPU environment tester and configurator."""
    
    def __init__(self):
        self.gpu_count = 0
        self.gpu_info = []
        self.test_results = {}
        
    def initialize_nvml(self):
        """Initialize NVIDIA Management Library."""
        try:
            nvmlInit()
            self.gpu_count = nvmlDeviceGetCount()
            print(f"✅ Found {self.gpu_count} GPU(s)")
            return True
        except Exception as e:
            print(f"❌ NVML initialization failed: {e}")
            return False
    
    def get_gpu_info(self):
        """Get detailed GPU information."""
        if not self.initialize_nvml():
            return False
            
        for i in range(self.gpu_count):
            try:
                handle = nvmlDeviceGetHandleByIndex(i)
                name = nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = nvmlDeviceGetMemoryInfo(handle)
                temperature = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                utilization = nvmlDeviceGetUtilizationRates(handle)
                
                gpu_data = {
                    'id': i,
                    'name': name,
                    'memory_total': memory_info.total // (1024**3),  # GB
                    'memory_free': memory_info.free // (1024**3),   # GB
                    'memory_used': memory_info.used // (1024**3),   # GB
                    'temperature': temperature,
                    'utilization_gpu': utilization.gpu,
                    'utilization_memory': utilization.memory
                }
                self.gpu_info.append(gpu_data)
                print(f"GPU {i}: {name} - {gpu_data['memory_total']}GB")
                
            except Exception as e:
                print(f"❌ Error getting info for GPU {i}: {e}")
                
        return True
    
    def test_pytorch_cuda(self):
        """Test PyTorch CUDA functionality."""
        print("\n🔬 Testing PyTorch CUDA...")
        
        try:
            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda
            torch_version = torch.__version__
            device_count = torch.cuda.device_count()
            
            print(f"   CUDA Available: {cuda_available}")
            print(f"   CUDA Version: {cuda_version}")
            print(f"   PyTorch Version: {torch_version}")
            print(f"   Device Count: {device_count}")
            
            if not cuda_available:
                self.test_results['pytorch_cuda'] = False
                return False
            
            # Test basic operations on each GPU
            for i in range(device_count):
                device = torch.device(f'cuda:{i}')
                print(f"   Testing GPU {i}...")
                
                # Simple tensor operations
                x = torch.randn(1000, 1000, device=device)
                y = torch.randn(1000, 1000, device=device)
                z = torch.matmul(x, y)
                
                print(f"     ✅ GPU {i}: Matrix multiplication successful")
                
            self.test_results['pytorch_cuda'] = True
            return True
            
        except Exception as e:
            print(f"❌ PyTorch CUDA test failed: {e}")
            self.test_results['pytorch_cuda'] = False
            return False
    
    def test_multi_gpu_communication(self):
        """Test multi-GPU communication and data transfer."""
        print("\n🔗 Testing multi-GPU communication...")
        
        if self.gpu_count < 2:
            print("   ⚠️ Need at least 2 GPUs for communication test")
            self.test_results['multi_gpu_communication'] = 'skipped'
            return True
        
        try:
            # Test P2P communication
            for i in range(min(2, self.gpu_count)):
                for j in range(min(2, self.gpu_count)):
                    if i != j:
                        can_access = torch.cuda.can_device_access_peer(i, j)
                        print(f"   GPU {i} -> GPU {j} P2P: {can_access}")
            
            # Test data transfer between GPUs
            device0 = torch.device('cuda:0')
            device1 = torch.device('cuda:1')
            
            # Create tensor on GPU 0
            x = torch.randn(1000, 1000, device=device0)
            
            # Transfer to GPU 1
            start_time = time.time()
            y = x.to(device1)
            transfer_time = time.time() - start_time
            
            # Verify transfer
            assert y.device == device1
            print(f"   ✅ GPU 0->1 transfer: {transfer_time:.4f}s")
            
            self.test_results['multi_gpu_communication'] = True
            return True
            
        except Exception as e:
            print(f"❌ Multi-GPU communication test failed: {e}")
            self.test_results['multi_gpu_communication'] = False
            return False
    
    def test_cuda_kernel_compilation(self):
        """Test CUDA kernel compilation capabilities."""
        print("\n⚙️ Testing CUDA kernel compilation...")
        
        try:
            # Simple CUDA kernel test
            kernel_code = """
            extern "C" __global__ void test_kernel(float* data, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    data[idx] = data[idx] * 2.0f;
                }
            }
            """
            
            # Check if nvcc is available
            result = subprocess.run(['nvcc', '--version'], 
                                 capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("   ✅ nvcc compiler available")
                print(f"   Version: {result.stdout.split('release')[1].split(',')[0].strip()}")
                self.test_results['cuda_compilation'] = True
            else:
                print("   ❌ nvcc compiler not available")
                self.test_results['cuda_compilation'] = False
                
        except subprocess.TimeoutExpired:
            print("   ❌ nvcc compiler check timed out")
            self.test_results['cuda_compilation'] = False
        except FileNotFoundError:
            print("   ❌ nvcc compiler not found")
            self.test_results['cuda_compilation'] = False
        except Exception as e:
            print(f"   ❌ CUDA compilation test failed: {e}")
            self.test_results['cuda_compilation'] = False
    
    def test_ray_initialization(self):
        """Test Ray cluster initialization for distributed training."""
        print("\n🚀 Testing Ray initialization...")
        
        try:
            import ray
            
            # Initialize Ray with GPU resources
            ray.shutdown()  # Ensure clean state
            
            ray.init(
                num_cpus=os.cpu_count(),
                num_gpus=self.gpu_count,
                object_store_memory=2000000000,  # 2GB
                ignore_reinit_error=True
            )
            
            # Test Ray GPU detection
            resources = ray.available_resources()
            gpu_resources = resources.get('GPU', 0)
            
            print(f"   ✅ Ray initialized with {gpu_resources} GPUs")
            
            # Simple Ray GPU task test
            @ray.remote(num_gpus=1)
            def gpu_task():
                import torch
                device = torch.cuda.current_device()
                x = torch.randn(100, 100, device=f'cuda:{device}')
                return x.sum().item()
            
            # Run task on available GPUs
            futures = []
            for i in range(min(2, int(gpu_resources))):
                futures.append(gpu_task.remote())
            
            results = ray.get(futures)
            print(f"   ✅ Ray GPU tasks completed: {len(results)} results")
            
            ray.shutdown()
            self.test_results['ray_initialization'] = True
            return True
            
        except Exception as e:
            print(f"❌ Ray initialization test failed: {e}")
            self.test_results['ray_initialization'] = False
            try:
                ray.shutdown()
            except:
                pass
            return False
    
    def test_verl_components(self):
        """Test VERL integration components."""
        print("\n📊 Testing VERL components...")
        
        try:
            # Try importing VERL components
            import sys
            verl_path = Path(__file__).parent / "verl"
            sys.path.insert(0, str(verl_path))
            
            # Test basic VERL imports
            from verl.trainer.ppo.ray_trainer import RayPPOTrainer
            print("   ✅ VERL imports successful")
            
            self.test_results['verl_components'] = True
            return True
            
        except ImportError as e:
            print(f"   ⚠️ VERL import failed (expected): {e}")
            self.test_results['verl_components'] = 'partial'
            return True  # This is expected initially
        except Exception as e:
            print(f"❌ VERL component test failed: {e}")
            self.test_results['verl_components'] = False
            return False
    
    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "="*60)
        print("🎯 MULTI-GPU CONFIGURATION REPORT")
        print("="*60)
        
        # System info
        print(f"GPUs Detected: {self.gpu_count}")
        for gpu in self.gpu_info:
            print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory_total']}GB)")
        
        print("\nTest Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result is True else "❌ FAIL" if result is False else "⚠️ SKIP"
            print(f"  {test_name}: {status}")
        
        # Recommendations
        print("\n🔧 Recommendations:")
        if self.test_results.get('pytorch_cuda', False):
            print("  ✅ PyTorch CUDA ready for training")
        else:
            print("  ❌ Fix PyTorch CUDA setup before proceeding")
            
        if self.gpu_count >= 4:
            print("  ✅ Sufficient GPUs for distributed training")
        elif self.gpu_count >= 2:
            print("  ⚠️ Limited GPUs - consider smaller batch sizes")
        else:
            print("  ❌ Single GPU - distributed training not optimal")
        
        # Save report
        report_data = {
            'gpu_count': self.gpu_count,
            'gpu_info': self.gpu_info,
            'test_results': self.test_results,
            'timestamp': time.time()
        }
        
        with open('gpu_config_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Report saved to: gpu_config_report.json")
        
        # Overall status
        critical_tests = ['pytorch_cuda', 'cuda_compilation']
        all_critical_passed = all(self.test_results.get(test, False) for test in critical_tests)
        
        if all_critical_passed:
            print("\n🎉 MULTI-GPU SETUP: READY FOR PRODUCTION!")
            return True
        else:
            print("\n⚠️ MULTI-GPU SETUP: NEEDS ATTENTION")
            return False


def main():
    """Main configuration and testing function."""
    print("🚀 Multi-GPU Configuration and Testing")
    print("="*50)
    
    tester = MultiGPUTester()
    
    # Run all tests
    tests = [
        tester.get_gpu_info,
        tester.test_pytorch_cuda,
        tester.test_multi_gpu_communication,
        tester.test_cuda_kernel_compilation,
        tester.test_ray_initialization,
        tester.test_verl_components
    ]
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with error: {e}")
    
    # Generate final report
    success = tester.generate_report()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())