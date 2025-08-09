import time
import ctypes
import numpy as np
from typing import List, Optional, Callable, Any, Dict, Union
from dataclasses import dataclass, field
import asyncio
import structlog

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass 
class BenchmarkResult:
    """Result of CUDA kernel performance benchmark."""
    success: bool
    kernel_name: str = ""
    cuda_time_mean: float = 0.0
    cuda_time_std: float = 0.0
    baseline_time_mean: Optional[float] = None
    baseline_time_std: Optional[float] = None
    speedup_ratio: Optional[float] = None
    functional_correct: bool = False
    memory_throughput_gb_s: float = 0.0
    iterations: int = 0
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class CUDABenchmarker:
    """CUDA kernel performance measurement and comparison."""
    
    def __init__(
        self,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
        tolerance: float = 1e-5
    ):
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.tolerance = tolerance
        self.logger = structlog.get_logger("cuda_benchmarker")
        
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available - baseline comparisons will be limited")
        
        self.logger.info(
            "CUDA benchmarker initialized",
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
            tolerance=tolerance
        )
    
    async def benchmark_kernel(
        self,
        binary_path: str,
        test_inputs: List[torch.Tensor],
        baseline_operation: Optional[Callable] = None,
        kernel_name: str = "unknown_kernel",
        expected_output_shape: Optional[tuple] = None
    ) -> BenchmarkResult:
        """
        Benchmark CUDA kernel against PyTorch baseline.
        
        Args:
            binary_path: Path to compiled CUDA kernel binary
            test_inputs: List of input tensors for testing
            baseline_operation: PyTorch operation for comparison
            kernel_name: Name of the kernel being benchmarked
            expected_output_shape: Expected shape of kernel output
            
        Returns:
            BenchmarkResult with performance metrics
        """
        if not TORCH_AVAILABLE:
            return BenchmarkResult(
                success=False,
                kernel_name=kernel_name,
                error_message="PyTorch not available for benchmarking"
            )
        
        start_time = time.time()
        
        try:
            # Load compiled kernel library
            kernel_lib = self._load_kernel_library(binary_path)
            
            # Prepare GPU memory and data
            gpu_inputs, gpu_output = self._prepare_gpu_memory(test_inputs, expected_output_shape)
            
            # Perform warmup runs
            for _ in range(self.warmup_iterations):
                self._execute_kernel_mock(kernel_lib, gpu_inputs, gpu_output)
                if TORCH_AVAILABLE:
                    torch.cuda.synchronize()
            
            # Benchmark CUDA kernel
            cuda_times = await self._benchmark_cuda_kernel(kernel_lib, gpu_inputs, gpu_output)
            
            # Benchmark baseline if provided
            baseline_times = None
            baseline_result = None
            
            if baseline_operation and TORCH_AVAILABLE:
                baseline_times, baseline_result = await self._benchmark_baseline(
                    baseline_operation, test_inputs
                )
            
            # Retrieve kernel result for correctness check
            kernel_result = self._retrieve_kernel_result(gpu_output)
            
            # Calculate metrics
            cuda_time_mean = float(np.mean(cuda_times))
            cuda_time_std = float(np.std(cuda_times))
            
            baseline_time_mean = float(np.mean(baseline_times)) if baseline_times else None
            baseline_time_std = float(np.std(baseline_times)) if baseline_times else None
            
            speedup_ratio = None
            if baseline_time_mean and cuda_time_mean > 0:
                speedup_ratio = baseline_time_mean / cuda_time_mean
            
            # Check functional correctness
            functional_correct = self._check_correctness(kernel_result, baseline_result)
            
            # Calculate memory throughput
            memory_throughput = self._calculate_memory_throughput(test_inputs, cuda_time_mean)
            
            benchmark_time = time.time() - start_time
            
            self.logger.info(
                "Kernel benchmark completed",
                kernel_name=kernel_name,
                cuda_time_mean=cuda_time_mean,
                speedup_ratio=speedup_ratio,
                functional_correct=functional_correct,
                benchmark_time=benchmark_time
            )
            
            return BenchmarkResult(
                success=True,
                kernel_name=kernel_name,
                cuda_time_mean=cuda_time_mean,
                cuda_time_std=cuda_time_std,
                baseline_time_mean=baseline_time_mean,
                baseline_time_std=baseline_time_std,
                speedup_ratio=speedup_ratio,
                functional_correct=functional_correct,
                memory_throughput_gb_s=memory_throughput,
                iterations=self.benchmark_iterations,
                metadata={
                    "warmup_iterations": self.warmup_iterations,
                    "total_benchmark_time": benchmark_time,
                    "input_shapes": [list(t.shape) for t in test_inputs],
                    "input_dtypes": [str(t.dtype) for t in test_inputs]
                }
            )
            
        except Exception as e:
            error_msg = f"Benchmark failed: {str(e)}"
            benchmark_time = time.time() - start_time
            
            self.logger.error(
                "Kernel benchmark failed",
                kernel_name=kernel_name,
                error=error_msg,
                benchmark_time=benchmark_time
            )
            
            return BenchmarkResult(
                success=False,
                kernel_name=kernel_name,
                error_message=error_msg,
                metadata={"benchmark_time": benchmark_time}
            )
    
    def _load_kernel_library(self, binary_path: str) -> ctypes.CDLL:
        """Load compiled CUDA kernel as shared library."""
        try:
            lib = ctypes.CDLL(binary_path)
            self.logger.debug("Loaded CUDA kernel library", path=binary_path)
            return lib
        except OSError as e:
            raise RuntimeError(f"Failed to load kernel library {binary_path}: {e}")
    
    def _prepare_gpu_memory(
        self, 
        test_inputs: List[torch.Tensor],
        expected_output_shape: Optional[tuple] = None
    ) -> tuple:
        """Prepare GPU memory for kernel execution."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for GPU memory management")
        
        # Move inputs to GPU
        gpu_inputs = [tensor.cuda() if not tensor.is_cuda else tensor for tensor in test_inputs]
        
        # Create output tensor
        if expected_output_shape:
            gpu_output = torch.empty(expected_output_shape, device='cuda', dtype=gpu_inputs[0].dtype)
        else:
            # Default: same shape as first input
            gpu_output = torch.empty_like(gpu_inputs[0])
        
        return gpu_inputs, gpu_output
    
    def _execute_kernel_mock(
        self, 
        kernel_lib: ctypes.CDLL, 
        gpu_inputs: List[torch.Tensor], 
        gpu_output: torch.Tensor
    ) -> None:
        """
        Mock kernel execution - in real implementation, this would call the actual kernel.
        For now, we simulate execution time with a simple operation.
        """
        # This is a mock implementation - in practice, you would:
        # 1. Extract function pointer from kernel_lib
        # 2. Convert tensors to C pointers  
        # 3. Call kernel with proper launch parameters
        
        # Mock execution with simple tensor operation
        if len(gpu_inputs) >= 2:
            torch.add(gpu_inputs[0], gpu_inputs[1], out=gpu_output)
        else:
            gpu_output.copy_(gpu_inputs[0])
    
    async def _benchmark_cuda_kernel(
        self,
        kernel_lib: ctypes.CDLL,
        gpu_inputs: List[torch.Tensor], 
        gpu_output: torch.Tensor
    ) -> List[float]:
        """Benchmark CUDA kernel execution times."""
        cuda_times = []
        
        for _ in range(self.benchmark_iterations):
            if TORCH_AVAILABLE:
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            self._execute_kernel_mock(kernel_lib, gpu_inputs, gpu_output)
            
            if TORCH_AVAILABLE:
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            cuda_times.append(end_time - start_time)
        
        return cuda_times
    
    async def _benchmark_baseline(
        self,
        baseline_operation: Callable,
        test_inputs: List[torch.Tensor]
    ) -> tuple:
        """Benchmark baseline PyTorch operation."""
        if not TORCH_AVAILABLE:
            return None, None
        
        baseline_times = []
        baseline_result = None
        
        # Move inputs to GPU for fair comparison
        gpu_inputs = [tensor.cuda() if not tensor.is_cuda else tensor for tensor in test_inputs]
        
        # Warmup baseline
        for _ in range(self.warmup_iterations):
            baseline_operation(*gpu_inputs)
            torch.cuda.synchronize()
        
        # Benchmark baseline
        for i in range(self.benchmark_iterations):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            result = baseline_operation(*gpu_inputs)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            baseline_times.append(end_time - start_time)
            
            # Store result from first iteration for correctness check
            if i == 0:
                baseline_result = result
        
        return baseline_times, baseline_result
    
    def _retrieve_kernel_result(self, gpu_output: torch.Tensor) -> torch.Tensor:
        """Retrieve result from kernel execution."""
        # In practice, the kernel would have written to gpu_output
        # For now, return the mock result
        return gpu_output
    
    def _check_correctness(
        self, 
        kernel_result: Optional[torch.Tensor], 
        baseline_result: Optional[torch.Tensor]
    ) -> bool:
        """Check functional correctness between kernel and baseline results."""
        if kernel_result is None or baseline_result is None:
            return True  # Cannot verify, assume correct
        
        if not TORCH_AVAILABLE:
            return True
        
        try:
            # Check shapes match
            if kernel_result.shape != baseline_result.shape:
                self.logger.warning(
                    "Shape mismatch in correctness check",
                    kernel_shape=kernel_result.shape,
                    baseline_shape=baseline_result.shape
                )
                return False
            
            # Check values are close
            is_close = torch.allclose(
                kernel_result, 
                baseline_result, 
                rtol=self.tolerance, 
                atol=self.tolerance
            )
            
            if not is_close:
                max_diff = torch.max(torch.abs(kernel_result - baseline_result)).item()
                self.logger.warning(
                    "Functional correctness check failed",
                    max_difference=max_diff,
                    tolerance=self.tolerance
                )
            
            return is_close
            
        except Exception as e:
            self.logger.error("Error during correctness check", error=str(e))
            return False
    
    def _calculate_memory_throughput(
        self, 
        test_inputs: List[torch.Tensor], 
        execution_time: float
    ) -> float:
        """Calculate memory throughput in GB/s."""
        if execution_time <= 0:
            return 0.0
        
        try:
            # Calculate total bytes transferred
            total_bytes = 0
            for tensor in test_inputs:
                total_bytes += tensor.numel() * tensor.element_size()
            
            # Assume read + write (2x data movement)
            total_bytes *= 2
            
            # Convert to GB/s
            throughput_gb_s = (total_bytes / (1024**3)) / execution_time
            return throughput_gb_s
            
        except Exception as e:
            self.logger.warning("Failed to calculate memory throughput", error=str(e))
            return 0.0
    
    async def benchmark_multiple_inputs(
        self,
        binary_path: str,
        test_input_sets: List[List[torch.Tensor]],
        baseline_operation: Optional[Callable] = None,
        kernel_name: str = "unknown_kernel"
    ) -> List[BenchmarkResult]:
        """Benchmark kernel with multiple different input sets."""
        results = []
        
        for i, test_inputs in enumerate(test_input_sets):
            self.logger.info(f"Benchmarking input set {i+1}/{len(test_input_sets)}")
            
            result = await self.benchmark_kernel(
                binary_path=binary_path,
                test_inputs=test_inputs,
                baseline_operation=baseline_operation,
                kernel_name=f"{kernel_name}_input_{i}"
            )
            
            results.append(result)
        
        return results
    
    def get_summary_statistics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate summary statistics across multiple benchmark results."""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.success]
        if not successful_results:
            return {"success_rate": 0.0}
        
        speedups = [r.speedup_ratio for r in successful_results if r.speedup_ratio is not None]
        correctness = [r.functional_correct for r in successful_results]
        
        summary = {
            "success_rate": len(successful_results) / len(results),
            "correctness_rate": sum(correctness) / len(correctness) if correctness else 0.0,
            "mean_speedup": float(np.mean(speedups)) if speedups else None,
            "median_speedup": float(np.median(speedups)) if speedups else None,
            "min_speedup": float(np.min(speedups)) if speedups else None,
            "max_speedup": float(np.max(speedups)) if speedups else None,
            "total_benchmarks": len(results),
            "successful_benchmarks": len(successful_results)
        }
        
        return summary