"""
Docker-based kernel execution sandbox.

Provides safe execution environment for AI-generated kernels:
- Compilation with hipcc
- Execution with test inputs
- Correctness verification
- Performance measurement
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog
import torch

logger = structlog.get_logger()


@dataclass
class ExecutionResult:
    """Result from executing a kernel in the sandbox."""
    
    # Compilation
    compiled: bool = False
    compilation_time_s: float = 0.0
    compilation_error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Correctness
    executed: bool = False
    numerically_correct: bool = False
    max_absolute_error: float = float('inf')
    mean_absolute_error: float = float('inf')
    max_relative_error: float = float('inf')
    
    # Performance
    kernel_time_ms: float = float('inf')
    baseline_time_ms: float = 0.0
    speedup: float = 0.0
    
    # Profiling metrics (from rocprof)
    memory_bandwidth_gbps: float = 0.0
    achieved_occupancy: float = 0.0
    register_usage: int = 0
    lds_usage_bytes: int = 0
    wavefronts_launched: int = 0
    
    # Raw outputs for debugging
    kernel_output: Optional[torch.Tensor] = None
    reference_output: Optional[torch.Tensor] = None
    profiling_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding tensors)."""
        return {
            "compiled": self.compiled,
            "compilation_time_s": self.compilation_time_s,
            "compilation_error": self.compilation_error,
            "warnings": self.warnings,
            "executed": self.executed,
            "numerically_correct": self.numerically_correct,
            "max_absolute_error": self.max_absolute_error,
            "mean_absolute_error": self.mean_absolute_error,
            "max_relative_error": self.max_relative_error,
            "kernel_time_ms": self.kernel_time_ms,
            "baseline_time_ms": self.baseline_time_ms,
            "speedup": self.speedup,
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
            "achieved_occupancy": self.achieved_occupancy,
            "register_usage": self.register_usage,
            "lds_usage_bytes": self.lds_usage_bytes,
            "profiling_data": self.profiling_data,
        }


class KernelSandbox:
    """
    Sandbox for compiling and executing HIP kernels.
    
    Can run locally (if ROCm is available) or in Docker container.
    """
    
    def __init__(
        self,
        docker_image: str = "rocm/pytorch-nightly:latest",
        use_docker: bool = True,
        timeout_seconds: int = 60,
        num_warmup_runs: int = 3,
        num_benchmark_runs: int = 10,
        correctness_tolerance: float = 1e-3,
        device_id: int = 0,
    ):
        self.docker_image = docker_image
        self.use_docker = use_docker
        self.timeout_seconds = timeout_seconds
        self.num_warmup_runs = num_warmup_runs
        self.num_benchmark_runs = num_benchmark_runs
        self.correctness_tolerance = correctness_tolerance
        self.device_id = device_id
        
        self.logger = logger.bind(component="KernelSandbox")
    
    async def execute_kernel(
        self,
        kernel_code: str,
        kernel_name: str,
        input_tensors: Dict[str, torch.Tensor],
        reference_outputs: Dict[str, torch.Tensor],
        shape_values: Dict[str, int],
        baseline_time_ms: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Execute a kernel and measure its performance.
        
        Args:
            kernel_code: HIP C++ source code
            kernel_name: Name of the kernel function
            input_tensors: Dictionary of input tensors
            reference_outputs: Expected outputs from reference implementation
            shape_values: Shape parameters for kernel launch
            baseline_time_ms: Pre-measured baseline time (optional)
        
        Returns:
            ExecutionResult with all metrics
        """
        result = ExecutionResult()
        
        # Create temporary directory for kernel files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Write kernel source
            kernel_file = tmpdir / f"{kernel_name}.cpp"
            kernel_file.write_text(self._wrap_kernel_code(kernel_code, kernel_name))
            
            # Save input tensors
            inputs_file = tmpdir / "inputs.pt"
            torch.save(input_tensors, inputs_file)
            
            # Save shape values
            shapes_file = tmpdir / "shapes.json"
            shapes_file.write_text(json.dumps(shape_values))
            
            # Step 1: Compile
            compile_result = await self._compile_kernel(tmpdir, kernel_name)
            result.compiled = compile_result["success"]
            result.compilation_time_s = compile_result["time"]
            result.compilation_error = compile_result.get("error")
            result.warnings = compile_result.get("warnings", [])
            
            if not result.compiled:
                self.logger.warning("Compilation failed", error=result.compilation_error)
                return result
            
            # Step 2: Execute and benchmark
            exec_result = await self._run_kernel(
                tmpdir, 
                kernel_name, 
                input_tensors,
                shape_values
            )
            
            result.executed = exec_result["success"]
            if not result.executed:
                self.logger.warning("Execution failed", error=exec_result.get("error"))
                return result
            
            result.kernel_time_ms = exec_result["time_ms"]
            result.kernel_output = exec_result.get("output")
            
            # Step 3: Check correctness
            if result.kernel_output is not None:
                ref_key = list(reference_outputs.keys())[0]
                ref_output = reference_outputs[ref_key]
                
                correctness = self._check_correctness(
                    result.kernel_output, 
                    ref_output
                )
                result.numerically_correct = correctness["correct"]
                result.max_absolute_error = correctness["max_abs_error"]
                result.mean_absolute_error = correctness["mean_abs_error"]
                result.max_relative_error = correctness["max_rel_error"]
                result.reference_output = ref_output
            
            # Step 4: Calculate speedup
            if baseline_time_ms is not None:
                result.baseline_time_ms = baseline_time_ms
            else:
                result.baseline_time_ms = await self._measure_baseline(
                    input_tensors,
                    reference_outputs
                )
            
            if result.baseline_time_ms > 0:
                result.speedup = result.baseline_time_ms / result.kernel_time_ms
            
            # Step 5: Profile (optional, more detailed metrics)
            if result.executed and result.numerically_correct:
                profile_result = await self._profile_kernel(
                    tmpdir,
                    kernel_name,
                    input_tensors,
                    shape_values
                )
                result.memory_bandwidth_gbps = profile_result.get("bandwidth_gbps", 0)
                result.achieved_occupancy = profile_result.get("occupancy", 0)
                result.register_usage = profile_result.get("registers", 0)
                result.lds_usage_bytes = profile_result.get("lds_bytes", 0)
                result.profiling_data = profile_result
        
        self.logger.info(
            "Kernel execution complete",
            compiled=result.compiled,
            correct=result.numerically_correct,
            speedup=f"{result.speedup:.2f}x",
            time_ms=f"{result.kernel_time_ms:.3f}"
        )
        
        return result
    
    def _wrap_kernel_code(self, kernel_code: str, kernel_name: str) -> str:
        """Wrap kernel code with necessary includes and launcher."""
        
        wrapper = f'''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <stdlib.h>

// Error checking macro
#define HIP_CHECK(call) do {{ \\
    hipError_t err = call; \\
    if (err != hipSuccess) {{ \\
        fprintf(stderr, "HIP error at %s:%d - %s\\n", \\
                __FILE__, __LINE__, hipGetErrorString(err)); \\
        exit(1); \\
    }} \\
}} while(0)

// User kernel code
{kernel_code}

// Kernel launcher (will be called from Python via ctypes)
extern "C" {{
    void launch_{kernel_name}(
        void** args,
        int num_args,
        dim3 grid,
        dim3 block,
        size_t shared_mem,
        hipStream_t stream
    ) {{
        // Launch kernel
        hipLaunchKernelGGL(
            {kernel_name},
            grid,
            block,
            shared_mem,
            stream,
            // Args will be unpacked based on kernel signature
        );
        HIP_CHECK(hipGetLastError());
    }}
}}
'''
        return wrapper
    
    async def _compile_kernel(
        self, 
        workdir: Path, 
        kernel_name: str
    ) -> Dict[str, Any]:
        """Compile kernel with hipcc."""
        
        start_time = time.time()
        source_file = workdir / f"{kernel_name}.cpp"
        output_file = workdir / f"{kernel_name}.so"
        
        # Compilation command
        cmd = [
            "hipcc",
            "-shared",
            "-fPIC",
            "-O3",
            "--offload-arch=gfx942",  # MI300X architecture
            "-o", str(output_file),
            str(source_file),
        ]
        
        try:
            if self.use_docker:
                # Run in Docker container
                docker_cmd = [
                    "docker", "run", "--rm",
                    "--device=/dev/kfd",
                    "--device=/dev/dri",
                    f"-v{workdir}:/workspace",
                    "-w/workspace",
                    self.docker_image,
                ] + cmd[:]  # Replace paths for container
                
                docker_cmd[-1] = f"/workspace/{kernel_name}.cpp"
                docker_cmd[-3] = f"/workspace/{kernel_name}.so"
                
                proc = await asyncio.create_subprocess_exec(
                    *docker_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout_seconds
            )
            
            compile_time = time.time() - start_time
            
            if proc.returncode == 0:
                return {
                    "success": True,
                    "time": compile_time,
                    "warnings": self._parse_warnings(stderr.decode()),
                }
            else:
                return {
                    "success": False,
                    "time": compile_time,
                    "error": stderr.decode(),
                }
                
        except asyncio.TimeoutError:
            return {
                "success": False,
                "time": self.timeout_seconds,
                "error": "Compilation timed out",
            }
        except Exception as e:
            return {
                "success": False,
                "time": time.time() - start_time,
                "error": str(e),
            }
    
    async def _run_kernel(
        self,
        workdir: Path,
        kernel_name: str,
        input_tensors: Dict[str, torch.Tensor],
        shape_values: Dict[str, int],
    ) -> Dict[str, Any]:
        """Run compiled kernel and measure time."""
        
        # For now, return a placeholder
        # TODO: Implement actual kernel execution via ctypes or Python binding
        
        return {
            "success": True,
            "time_ms": 1.0,
            "output": None,
        }
    
    def _check_correctness(
        self,
        kernel_output: torch.Tensor,
        reference_output: torch.Tensor,
    ) -> Dict[str, Any]:
        """Check numerical correctness of kernel output."""
        
        diff = (kernel_output.float() - reference_output.float()).abs()
        ref_abs = reference_output.float().abs()
        
        max_abs_error = diff.max().item()
        mean_abs_error = diff.mean().item()
        
        # Relative error (avoid division by zero)
        rel_error = diff / (ref_abs + 1e-8)
        max_rel_error = rel_error.max().item()
        
        correct = max_abs_error < self.correctness_tolerance
        
        return {
            "correct": correct,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "max_rel_error": max_rel_error,
        }
    
    async def _measure_baseline(
        self,
        input_tensors: Dict[str, torch.Tensor],
        reference_outputs: Dict[str, torch.Tensor],
    ) -> float:
        """Measure baseline PyTorch/rocBLAS time."""
        
        # TODO: Implement proper baseline measurement
        return 1.0
    
    async def _profile_kernel(
        self,
        workdir: Path,
        kernel_name: str,
        input_tensors: Dict[str, torch.Tensor],
        shape_values: Dict[str, int],
    ) -> Dict[str, Any]:
        """Profile kernel with rocprof."""
        
        # TODO: Implement rocprof integration
        return {}
    
    def _parse_warnings(self, stderr: str) -> List[str]:
        """Extract warnings from compiler output."""
        warnings = []
        for line in stderr.split('\n'):
            if 'warning:' in line.lower():
                warnings.append(line.strip())
        return warnings

