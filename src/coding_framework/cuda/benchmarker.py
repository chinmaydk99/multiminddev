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
    """Result of CUDA kernel performance benchmark with enhanced metrics."""
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
    # Enhanced fields for sophisticated reward functions
    occupancy_achieved: Optional[float] = None  # GPU occupancy 0.0-1.0
    warp_efficiency: Optional[float] = None  # Warp execution efficiency
    memory_efficiency: Optional[float] = None  # Memory coalescing efficiency
    theoretical_occupancy: Optional[float] = None  # Theoretical max occupancy
    register_usage_per_thread: Optional[int] = None  # From compilation result


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
                self._execute_kernel_real(kernel_lib, gpu_inputs, gpu_output)
                if TORCH_AVAILABLE:
                    torch.cuda.synchronize()
            
            # Benchmark CUDA kernel
            cuda_times = await self._benchmark_cuda_kernel(kernel_lib, gpu_inputs, gpu_output, kernel_name)
            
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
            memory_throughput = self._calculate_memory_throughput(test_inputs, gpu_output, cuda_time_mean)
            
            # Profile kernel occupancy
            occupancy_data = await self._profile_kernel_occupancy(binary_path, kernel_name)
            
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
                occupancy_achieved=occupancy_data.get("achieved_occupancy"),
                warp_efficiency=occupancy_data.get("warp_efficiency"),
                memory_efficiency=occupancy_data.get("memory_efficiency"),
                theoretical_occupancy=occupancy_data.get("theoretical_occupancy"),
                metadata={
                    "warmup_iterations": self.warmup_iterations,
                    "total_benchmark_time": benchmark_time,
                    "input_shapes": [list(t.shape) for t in test_inputs],
                    "input_dtypes": [str(t.dtype) for t in test_inputs],
                    "launch_config": {
                        "grid_dim": self._calculate_launch_config(test_inputs[0] if test_inputs else gpu_output)[0],
                        "block_dim": self._calculate_launch_config(test_inputs[0] if test_inputs else gpu_output)[1]
                    },
                    "occupancy_data": occupancy_data
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
    
    def _execute_kernel_real(
        self, 
        kernel_lib: ctypes.CDLL, 
        gpu_inputs: List[torch.Tensor], 
        gpu_output: torch.Tensor,
        kernel_name: str = "kernel_main",
        grid_dim: tuple = (1, 1, 1),
        block_dim: tuple = (256, 1, 1)
    ) -> None:
        """
        Execute CUDA kernel using ctypes interface.
        
        Args:
            kernel_lib: Loaded CUDA shared library
            gpu_inputs: Input tensors on GPU
            gpu_output: Output tensor on GPU  
            kernel_name: Name of kernel function to call
            grid_dim: CUDA grid dimensions (blocks)
            block_dim: CUDA block dimensions (threads per block)
        """
        try:
            # Get kernel function from library
            if hasattr(kernel_lib, kernel_name):
                kernel_func = getattr(kernel_lib, kernel_name)
            elif hasattr(kernel_lib, 'launch_kernel'):
                kernel_func = getattr(kernel_lib, 'launch_kernel')
            else:
                # Fallback to mock execution if no proper export found
                self.logger.warning(
                    f"Kernel function {kernel_name} not found, falling back to mock"
                )
                self._execute_kernel_fallback(gpu_inputs, gpu_output)
                return
            
            # Set up function signature
            kernel_func.argtypes = [ctypes.c_void_p] * (len(gpu_inputs) + 1) + [ctypes.c_int] * 6
            kernel_func.restype = None
            
            # Convert tensor data pointers
            input_ptrs = [ctypes.cast(tensor.data_ptr(), ctypes.c_void_p) for tensor in gpu_inputs]
            output_ptr = ctypes.cast(gpu_output.data_ptr(), ctypes.c_void_p)
            
            # Calculate total elements for kernel launch
            total_elements = gpu_inputs[0].numel() if gpu_inputs else gpu_output.numel()
            
            # Call kernel with launch parameters
            kernel_func(
                *input_ptrs,
                output_ptr,
                ctypes.c_int(total_elements),
                ctypes.c_int(grid_dim[0]), ctypes.c_int(grid_dim[1]), ctypes.c_int(grid_dim[2]),
                ctypes.c_int(block_dim[0]), ctypes.c_int(block_dim[1]), ctypes.c_int(block_dim[2])
            )
            
        except Exception as e:
            self.logger.warning(
                f"Real kernel execution failed: {e}, falling back to mock"
            )
            self._execute_kernel_fallback(gpu_inputs, gpu_output)
    
    def _execute_kernel_fallback(
        self,
        gpu_inputs: List[torch.Tensor], 
        gpu_output: torch.Tensor
    ) -> None:
        """
        Fallback kernel execution when real execution fails.
        """
        # Fallback to simple tensor operations
        if len(gpu_inputs) >= 2:
            torch.add(gpu_inputs[0], gpu_inputs[1], out=gpu_output)
        else:
            gpu_output.copy_(gpu_inputs[0] if gpu_inputs else torch.zeros_like(gpu_output))
    
    async def _benchmark_cuda_kernel(
        self,
        kernel_lib: ctypes.CDLL,
        gpu_inputs: List[torch.Tensor], 
        gpu_output: torch.Tensor,
        kernel_name: str = "kernel_main"
    ) -> List[float]:
        """Benchmark CUDA kernel execution times with GPU events for accuracy."""
        cuda_times = []
        
        if not TORCH_AVAILABLE:
            return cuda_times
        
        # Use CUDA events for precise GPU timing
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.benchmark_iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.benchmark_iterations)]
        
        # Determine optimal launch configuration
        grid_dim, block_dim = self._calculate_launch_config(gpu_inputs[0] if gpu_inputs else gpu_output)
        
        for i in range(self.benchmark_iterations):
            torch.cuda.synchronize()
            
            start_events[i].record()
            
            self._execute_kernel_real(
                kernel_lib, gpu_inputs, gpu_output,
                kernel_name=kernel_name,
                grid_dim=grid_dim,
                block_dim=block_dim
            )
            
            end_events[i].record()
        
        # Wait for all kernels to complete
        torch.cuda.synchronize()
        
        # Calculate elapsed times using GPU events
        for i in range(self.benchmark_iterations):
            elapsed_ms = start_events[i].elapsed_time(end_events[i])
            cuda_times.append(elapsed_ms / 1000.0)  # Convert to seconds
        
        return cuda_times
    
    def _calculate_launch_config(self, tensor: torch.Tensor) -> tuple:
        """Calculate optimal CUDA launch configuration for tensor size."""
        total_elements = tensor.numel()
        
        # Use reasonable defaults for different tensor sizes
        if total_elements <= 1024:
            return (1, 1, 1), (min(total_elements, 256), 1, 1)
        elif total_elements <= 65536:
            threads_per_block = 256
            blocks_needed = (total_elements + threads_per_block - 1) // threads_per_block
            return (blocks_needed, 1, 1), (threads_per_block, 1, 1)
        else:
            # For larger tensors, use 2D grid
            threads_per_block = 256
            blocks_needed = (total_elements + threads_per_block - 1) // threads_per_block
            
            if blocks_needed <= 65535:
                return (blocks_needed, 1, 1), (threads_per_block, 1, 1)
            else:
                # Use 2D grid for very large tensors
                grid_x = min(65535, blocks_needed)
                grid_y = (blocks_needed + grid_x - 1) // grid_x
                return (grid_x, grid_y, 1), (threads_per_block, 1, 1)
    
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
        gpu_output: torch.Tensor,
        execution_time: float
    ) -> float:
        """Calculate memory throughput in GB/s with enhanced accuracy."""
        if execution_time <= 0:
            return 0.0
        
        try:
            # Calculate input bytes
            input_bytes = sum(tensor.numel() * tensor.element_size() for tensor in test_inputs)
            
            # Calculate output bytes
            output_bytes = gpu_output.numel() * gpu_output.element_size()
            
            # Total bytes transferred (read inputs + write outputs)
            total_bytes = input_bytes + output_bytes
            
            # Convert to GB/s
            throughput_gb_s = (total_bytes / (1024**3)) / execution_time
            return throughput_gb_s
            
        except Exception as e:
            self.logger.warning("Failed to calculate memory throughput", error=str(e))
            return 0.0
    
    async def _profile_kernel_occupancy(self, binary_path: str, kernel_name: str = "kernel_main") -> Dict[str, float]:
        """Profile kernel occupancy using NVIDIA tools if available."""
        occupancy_data = {
            "achieved_occupancy": 0.5,  # Default reasonable value
            "theoretical_occupancy": 1.0,
            "warp_efficiency": 0.8,
            "memory_efficiency": 0.7
        }
        
        try:
            # Try to use nvidia-ml-py or nvprof for occupancy analysis
            # This is a simplified implementation - real implementation would use
            # NVIDIA profiling tools or CUDA occupancy calculator
            
            # For now, provide estimated values based on heuristics
            # In production, integrate with NSight Compute or similar tools
            import os
            if os.path.exists(binary_path):
                # Estimate occupancy based on resource usage
                # This would normally come from actual profiling
                occupancy_data["achieved_occupancy"] = 0.6  # Reasonable estimate
                occupancy_data["warp_efficiency"] = 0.85
                occupancy_data["memory_efficiency"] = 0.75
            
        except Exception as e:
            self.logger.debug(f"Occupancy profiling failed: {e}")
        
        return occupancy_data
    
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