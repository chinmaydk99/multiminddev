import ctypes
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import structlog
import torch


@dataclass
class BenchmarkResult:
    """Comprehensive benchmark results with multiple performance metrics."""
    success: bool
    execution_time_ms: float = 0.0
    memory_bandwidth_gb_s: float = 0.0
    compute_throughput_gflops: float = 0.0
    gpu_utilization_percent: float = 0.0

    # Correctness metrics
    functional_correct: bool = False
    numerical_accuracy: float = 0.0  # For floating point operations

    # Performance comparisons
    speedup_vs_baseline: float = 1.0
    speedup_vs_torch: float = 1.0
    speedup_ratio: Optional[float] = None  # For compatibility with existing code

    # Resource utilization
    peak_memory_usage_mb: float = 0.0
    occupancy_percent: float = 0.0

    # Error information
    error_message: str = ""
    benchmark_details: Optional[Dict[str, Any]] = None


class HIPBenchmarker:
    """Production HIP kernel benchmarker with comprehensive metrics for AMD ROCm."""

    def __init__(
        self,
        warmup_iterations: int = 3,
        benchmark_iterations: int = 10,
        memory_check_tolerance: float = 1e-5
    ):
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.memory_check_tolerance = memory_check_tolerance
        self.logger = structlog.get_logger()

        # Initialize ROCm context (PyTorch uses torch.cuda API for ROCm too)
        self._initialize_rocm_context()

    def _initialize_rocm_context(self):
        """Initialize ROCm context and check GPU availability."""
        # PyTorch with ROCm uses the same torch.cuda API
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm/HIP not available for benchmarking")

        self.device = torch.cuda.current_device()
        self.device_props = torch.cuda.get_device_properties(self.device)

        self.logger.info(
            "HIP benchmarker initialized",
            device=self.device,
            device_name=self.device_props.name,
            gcn_arch=self._get_gcn_arch(),
            total_memory_gb=self.device_props.total_memory / (1024**3)
        )

    def _get_gcn_arch(self) -> str:
        """Get the GCN/RDNA architecture string for the AMD GPU."""
        # PyTorch with ROCm exposes this through device properties
        device_name = self.device_props.name.lower()

        # Map device names to architectures
        if "mi300" in device_name:
            return "CDNA3 (gfx942)"
        elif "mi250" in device_name or "mi210" in device_name:
            return "CDNA2 (gfx90a)"
        elif "mi100" in device_name:
            return "CDNA (gfx908)"
        elif "7900" in device_name:
            return "RDNA3 (gfx1100)"
        elif "6900" in device_name or "6800" in device_name:
            return "RDNA2 (gfx1030)"
        elif "vega" in device_name:
            return "GCN5 (gfx906)"
        else:
            return f"Unknown ({device_name})"

    async def benchmark_kernel(
        self,
        binary_path: str,
        kernel_name: str,
        test_inputs: List[torch.Tensor],
        baseline_operation: Optional[callable] = None,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> BenchmarkResult:
        """
        Benchmark HIP kernel against test cases with comprehensive metrics.
        
        Args:
            binary_path: Path to compiled kernel shared library
            kernel_name: Name of the kernel function
            test_inputs: Test input tensors
            baseline_operation: Optional baseline for speedup comparison
            test_cases: Optional test case specifications for advanced benchmarking
            
        Returns:
            BenchmarkResult with comprehensive performance metrics
        """

        try:
            # Load compiled kernel
            kernel_lib = ctypes.CDLL(binary_path)
            launch_func = getattr(kernel_lib, f"launch_{kernel_name}")

            # Configure function signature
            launch_func.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),  # args array
                ctypes.c_int,                     # num_args
                ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # grid dims
                ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # block dims
                ctypes.c_size_t                   # shared memory (LDS)
            ]
            launch_func.restype = None

            # If test_cases provided, use advanced benchmarking
            if test_cases:
                return await self._benchmark_with_test_cases(launch_func, test_cases, baseline_operation)
            else:
                # Use simple benchmarking with provided inputs
                return await self._benchmark_simple(launch_func, test_inputs, baseline_operation)

        except Exception as e:
            error_msg = f"Benchmark failed: {str(e)}"
            self.logger.error("Kernel benchmark failed", error=error_msg)

            return BenchmarkResult(
                success=False,
                error_message=error_msg
            )

    async def _benchmark_with_test_cases(
        self,
        launch_func: ctypes.CFUNCTYPE,
        test_cases: List[Dict[str, Any]],
        baseline_implementation: Optional[callable] = None
    ) -> BenchmarkResult:
        """Benchmark kernel with multiple test cases."""

        # Run benchmarks on all test cases
        all_results = []

        for i, test_case in enumerate(test_cases):
            self.logger.debug(f"Running benchmark for test case {i+1}/{len(test_cases)}")

            case_result = await self._benchmark_single_case(
                launch_func, test_case, baseline_implementation
            )
            all_results.append(case_result)

        # Aggregate results
        return self._aggregate_benchmark_results(all_results)

    async def _benchmark_simple(
        self,
        launch_func: ctypes.CFUNCTYPE,
        test_inputs: List[torch.Tensor],
        baseline_operation: Optional[callable] = None
    ) -> BenchmarkResult:
        """Simple benchmarking with provided test inputs."""

        # Create test case from inputs
        test_case = {
            "input_shapes": [list(tensor.shape) for tensor in test_inputs],
            "dtype": test_inputs[0].dtype,
            "grid_dims": (32, 1, 1),
            "block_dims": (256, 1, 1),
            "shared_memory": 0  # LDS memory
        }

        # Generate expected output (simple heuristic)
        expected_output = self._generate_expected_output(test_inputs)
        gpu_output = torch.zeros_like(expected_output)

        # Prepare arguments for kernel launch
        all_tensors = test_inputs + [gpu_output]
        args_array = self._prepare_kernel_args(all_tensors)

        # Warmup runs
        for _ in range(self.warmup_iterations):
            try:
                launch_func(
                    args_array, len(all_tensors),
                    test_case["grid_dims"][0], test_case["grid_dims"][1], test_case["grid_dims"][2],
                    test_case["block_dims"][0], test_case["block_dims"][1], test_case["block_dims"][2],
                    test_case["shared_memory"]
                )
                torch.cuda.synchronize()  # Works for ROCm via PyTorch
            except Exception as e:
                return BenchmarkResult(
                    success=False,
                    error_message=f"Warmup failed: {str(e)}"
                )

        # Benchmark runs with timing
        execution_times = []
        memory_stats = []

        for iteration in range(self.benchmark_iterations):
            # Reset output
            gpu_output.zero_()

            # Record memory before execution
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated(self.device)

            # Time execution using HIP events (via PyTorch's cuda API)
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            try:
                launch_func(
                    args_array, len(all_tensors),
                    test_case["grid_dims"][0], test_case["grid_dims"][1], test_case["grid_dims"][2],
                    test_case["block_dims"][0], test_case["block_dims"][1], test_case["block_dims"][2],
                    test_case["shared_memory"]
                )
            except Exception as e:
                return BenchmarkResult(
                    success=False,
                    error_message=f"Kernel execution failed on iteration {iteration}: {str(e)}"
                )

            end_event.record()
            torch.cuda.synchronize()

            # Record metrics
            execution_time_ms = start_event.elapsed_time(end_event)
            memory_after = torch.cuda.memory_allocated(self.device)

            execution_times.append(execution_time_ms)
            memory_stats.append(memory_after - memory_before)

        # Verify correctness
        functional_correct, numerical_accuracy = self._verify_correctness(
            gpu_output.cpu(), expected_output.cpu()
        )

        # Calculate performance metrics
        avg_execution_time = statistics.mean(execution_times)

        # Memory bandwidth calculation (HBM2/HBM3 on AMD GPUs)
        total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in all_tensors)
        memory_bandwidth = (total_bytes / (1024**3)) / (avg_execution_time / 1000)  # GB/s

        # Calculate speedup vs baseline if provided
        speedup_vs_baseline = 1.0
        if baseline_operation:
            baseline_time = self._benchmark_baseline(baseline_operation, test_inputs, expected_output)
            if baseline_time > 0:
                speedup_vs_baseline = baseline_time / avg_execution_time

        # Calculate speedup vs PyTorch equivalent
        speedup_vs_torch = self._benchmark_torch_equivalent(test_inputs, expected_output, avg_execution_time)

        return BenchmarkResult(
            success=True,
            execution_time_ms=avg_execution_time,
            memory_bandwidth_gb_s=memory_bandwidth,
            functional_correct=functional_correct,
            numerical_accuracy=numerical_accuracy,
            speedup_vs_baseline=speedup_vs_baseline,
            speedup_vs_torch=speedup_vs_torch,
            speedup_ratio=speedup_vs_baseline,  # For compatibility
            peak_memory_usage_mb=max(memory_stats) / (1024**2) if memory_stats else 0.0,
            benchmark_details={
                "execution_times": execution_times,
                "memory_usage": memory_stats,
                "test_case_params": test_case,
                "total_data_bytes": total_bytes,
                "gpu_arch": self._get_gcn_arch()
            }
        )

    async def _benchmark_single_case(
        self,
        launch_func: ctypes.CFUNCTYPE,
        test_case: Dict[str, Any],
        baseline_implementation: Optional[callable] = None
    ) -> BenchmarkResult:
        """Benchmark kernel on a single test case."""

        # Extract test case parameters
        input_shapes = test_case.get("input_shapes", [[1024]])
        data_type = test_case.get("dtype", torch.float32)
        grid_dims = test_case.get("grid_dims", (32, 1, 1))
        block_dims = test_case.get("block_dims", (256, 1, 1))
        shared_memory = test_case.get("shared_memory", 0)  # LDS memory

        # Generate test data
        gpu_inputs, expected_output = self._generate_test_data(input_shapes, data_type)
        gpu_output = torch.zeros_like(expected_output)

        # Prepare arguments for kernel launch
        args_array = self._prepare_kernel_args(gpu_inputs + [gpu_output])

        # Warmup runs
        for _ in range(self.warmup_iterations):
            try:
                launch_func(
                    args_array, len(gpu_inputs) + 1,
                    grid_dims[0], grid_dims[1], grid_dims[2],
                    block_dims[0], block_dims[1], block_dims[2],
                    shared_memory
                )
                torch.cuda.synchronize()
            except Exception as e:
                return BenchmarkResult(
                    success=False,
                    error_message=f"Warmup failed: {str(e)}"
                )

        # Benchmark runs with timing
        execution_times = []
        memory_stats = []

        for iteration in range(self.benchmark_iterations):
            # Reset output
            gpu_output.zero_()

            # Record memory before execution
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated(self.device)

            # Time execution
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            try:
                launch_func(
                    args_array, len(gpu_inputs) + 1,
                    grid_dims[0], grid_dims[1], grid_dims[2],
                    block_dims[0], block_dims[1], block_dims[2],
                    shared_memory
                )
            except Exception as e:
                return BenchmarkResult(
                    success=False,
                    error_message=f"Kernel execution failed on iteration {iteration}: {str(e)}"
                )

            end_event.record()
            torch.cuda.synchronize()

            # Record metrics
            execution_time_ms = start_event.elapsed_time(end_event)
            memory_after = torch.cuda.memory_allocated(self.device)

            execution_times.append(execution_time_ms)
            memory_stats.append(memory_after - memory_before)

        # Verify correctness
        functional_correct, numerical_accuracy = self._verify_correctness(
            gpu_output.cpu(), expected_output.cpu()
        )

        if not functional_correct:
            return BenchmarkResult(
                success=False,
                functional_correct=False,
                error_message="Kernel output does not match expected result"
            )

        # Calculate performance metrics
        avg_execution_time = statistics.mean(execution_times)

        # Memory bandwidth calculation
        total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in gpu_inputs + [gpu_output])
        memory_bandwidth = (total_bytes / (1024**3)) / (avg_execution_time / 1000)  # GB/s

        # Calculate speedup vs baseline if provided
        speedup_vs_baseline = 1.0
        if baseline_implementation:
            baseline_time = self._benchmark_baseline(baseline_implementation, gpu_inputs, expected_output)
            if baseline_time > 0:
                speedup_vs_baseline = baseline_time / avg_execution_time

        # Calculate speedup vs PyTorch equivalent
        speedup_vs_torch = self._benchmark_torch_equivalent(gpu_inputs, expected_output, avg_execution_time)

        return BenchmarkResult(
            success=True,
            execution_time_ms=avg_execution_time,
            memory_bandwidth_gb_s=memory_bandwidth,
            functional_correct=functional_correct,
            numerical_accuracy=numerical_accuracy,
            speedup_vs_baseline=speedup_vs_baseline,
            speedup_vs_torch=speedup_vs_torch,
            speedup_ratio=speedup_vs_baseline,  # For compatibility
            peak_memory_usage_mb=max(memory_stats) / (1024**2) if memory_stats else 0.0,
            benchmark_details={
                "execution_times": execution_times,
                "memory_usage": memory_stats,
                "test_case_params": test_case,
                "total_data_bytes": total_bytes
            }
        )

    def _generate_test_data(
        self,
        input_shapes: List[List[int]],
        data_type: torch.dtype
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Generate test input data and expected output."""

        # Generate random input tensors
        gpu_inputs = []
        for shape in input_shapes:
            tensor = torch.randn(shape, dtype=data_type, device='cuda')
            gpu_inputs.append(tensor)

        # For simple operations, calculate expected output
        expected_output = self._generate_expected_output(gpu_inputs)

        return gpu_inputs, expected_output

    def _generate_expected_output(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Generate expected output based on input tensors."""
        if len(inputs) == 2 and inputs[0].shape == inputs[1].shape:
            # Assume element-wise addition as default
            return inputs[0] + inputs[1]
        elif len(inputs) == 1:
            # For single input, assume identity operation
            return inputs[0].clone()
        else:
            # For more complex operations, return zeros with same shape as first input
            return torch.zeros_like(inputs[0])

    def _prepare_kernel_args(self, tensors: List[torch.Tensor]) -> ctypes.Array:
        """Prepare tensor arguments for kernel launch."""
        args = []
        for tensor in tensors:
            # Get raw data pointer as void*
            ptr = ctypes.cast(tensor.data_ptr(), ctypes.c_void_p)
            args.append(ptr)

        # Create array of void pointers
        args_array_type = ctypes.c_void_p * len(args)
        return args_array_type(*args)

    def _verify_correctness(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor
    ) -> Tuple[bool, float]:
        """Verify functional correctness and calculate numerical accuracy."""

        if actual.shape != expected.shape:
            return False, 0.0

        # Check for exact equality first (for integer operations)
        if torch.equal(actual, expected):
            return True, 1.0

        # For floating point, check relative tolerance
        if actual.dtype.is_floating_point:
            abs_diff = torch.abs(actual - expected)
            rel_diff = abs_diff / (torch.abs(expected) + 1e-8)

            max_rel_error = torch.max(rel_diff).item()

            # Consider correct if relative error is within tolerance
            is_correct = max_rel_error < self.memory_check_tolerance

            # Accuracy score based on how many elements are within tolerance
            within_tolerance = rel_diff < self.memory_check_tolerance
            accuracy = torch.sum(within_tolerance).item() / actual.numel()

            return is_correct, accuracy
        else:
            # For integer types, require exact match
            return torch.equal(actual, expected), 1.0 if torch.equal(actual, expected) else 0.0

    def _benchmark_baseline(
        self,
        baseline_func: callable,
        inputs: List[torch.Tensor],
        expected_output: torch.Tensor
    ) -> float:
        """Benchmark baseline implementation for speedup comparison."""
        try:
            # Warmup
            for _ in range(self.warmup_iterations):
                baseline_func(*inputs)
                torch.cuda.synchronize()

            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()

            for _ in range(self.benchmark_iterations):
                baseline_func(*inputs)

            torch.cuda.synchronize()
            end_time = time.time()

            avg_time_ms = ((end_time - start_time) / self.benchmark_iterations) * 1000
            return avg_time_ms

        except Exception as e:
            self.logger.warning(f"Baseline benchmark failed: {e}")
            return 0.0

    def _benchmark_torch_equivalent(
        self,
        inputs: List[torch.Tensor],
        expected_output: torch.Tensor,
        kernel_time_ms: float
    ) -> float:
        """Benchmark PyTorch equivalent operation."""
        try:
            # Simple heuristic: if two inputs of same shape, assume addition
            if len(inputs) == 2 and inputs[0].shape == inputs[1].shape:

                # Warmup
                for _ in range(self.warmup_iterations):
                    _ = torch.add(inputs[0], inputs[1])
                    torch.cuda.synchronize()

                # Benchmark
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                for _ in range(self.benchmark_iterations):
                    _ = torch.add(inputs[0], inputs[1])
                end_event.record()

                torch.cuda.synchronize()
                torch_time_ms = start_event.elapsed_time(end_event) / self.benchmark_iterations

                return torch_time_ms / kernel_time_ms

        except Exception as e:
            self.logger.warning(f"PyTorch benchmark failed: {e}")

        return 1.0  # No speedup if comparison fails

    def _aggregate_benchmark_results(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        """Aggregate results from multiple test cases."""

        if not results:
            return BenchmarkResult(success=False, error_message="No benchmark results")

        # Check if any failed
        failed_results = [r for r in results if not r.success]
        if failed_results:
            return failed_results[0]  # Return first failure

        # Aggregate successful results
        successful_results = [r for r in results if r.success and r.functional_correct]

        if not successful_results:
            return BenchmarkResult(
                success=False,
                functional_correct=False,
                error_message="No functionally correct results"
            )

        return BenchmarkResult(
            success=True,
            execution_time_ms=statistics.mean([r.execution_time_ms for r in successful_results]),
            memory_bandwidth_gb_s=statistics.mean([r.memory_bandwidth_gb_s for r in successful_results]),
            functional_correct=True,
            numerical_accuracy=statistics.mean([r.numerical_accuracy for r in successful_results]),
            speedup_vs_baseline=statistics.mean([r.speedup_vs_baseline for r in successful_results]),
            speedup_vs_torch=statistics.mean([r.speedup_vs_torch for r in successful_results]),
            speedup_ratio=statistics.mean([r.speedup_vs_baseline for r in successful_results]),  # For compatibility
            peak_memory_usage_mb=max([r.peak_memory_usage_mb for r in successful_results]),
            benchmark_details={
                "num_test_cases": len(successful_results),
                "individual_results": successful_results,
                "gpu_arch": self._get_gcn_arch()
            }
        )

