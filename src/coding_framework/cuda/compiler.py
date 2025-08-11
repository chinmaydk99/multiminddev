import asyncio
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import structlog


@dataclass
class CompilationResult:
    """Enhanced compilation result with metrics for reward calculation."""
    success: bool
    binary_path: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    compilation_time: float = 0.0
    kernel_name: str = ""

    # CUDA-specific metrics
    register_pressure: int = 0  # Registers per thread
    shared_memory_usage: int = 0  # Bytes of shared memory
    compilation_warnings: List[str] = field(default_factory=list)
    ptx_code: Optional[str] = None
    resource_usage: Optional[Dict] = None
    temp_files: List[str] = field(default_factory=list)

class CUDACompiler:
    """Production CUDA compiler with Docker safety and comprehensive metrics."""

    def __init__(
        self,
        nvcc_path: str = "nvcc",
        cuda_arch: str = "auto",
        optimization_level: str = "-O3",
        temp_dir: Optional[str] = None,
        use_docker: bool = True,
        docker_image: str = "nvidia/cuda:12.0-devel-ubuntu20.04"
    ):
        self.nvcc_path = nvcc_path
        self.cuda_arch = cuda_arch
        self.optimization_level = optimization_level
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="cuda_compiler_")
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.logger = structlog.get_logger()

        # Validate environment
        self._validate_environment()

    def _validate_environment(self):
        """Validate CUDA and Docker environment."""
        if self.use_docker:
            if not self._check_docker_available():
                raise RuntimeError("Docker not available but use_docker=True")
            if not self._check_docker_gpu_support():
                self.logger.warning("Docker GPU support not detected")
        else:
            if not self._check_nvcc_available():
                raise RuntimeError(f"nvcc not found at {self.nvcc_path}")

    def _check_docker_available(self) -> bool:
        """Check if Docker is available and running."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_docker_gpu_support(self) -> bool:
        """Check if Docker supports GPU access."""
        try:
            result = subprocess.run(
                ["docker", "run", "--rm", "--gpus=all", "nvidia/cuda:12.0-base-ubuntu20.04", "nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_nvcc_available(self) -> bool:
        """Check if nvcc is available."""
        try:
            result = subprocess.run(
                [self.nvcc_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def detect_gpu_architecture(self) -> str:
        """Detect GPU architecture for compilation."""
        if self.cuda_arch != "auto":
            return self.cuda_arch

        try:
            # Use nvidia-ml-py or nvidia-smi to detect architecture
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                compute_cap = result.stdout.strip().split('\n')[0]
                major, minor = compute_cap.split('.')
                return f"sm_{major}{minor}"
            else:
                self.logger.warning("Could not detect GPU architecture, using sm_75")
                return "sm_75"

        except Exception as e:
            self.logger.warning(f"GPU architecture detection failed: {e}, using sm_75")
            return "sm_75"

    async def compile_kernel(
        self,
        kernel_code: str,
        kernel_name: str = "kernel",
        additional_flags: Optional[List[str]] = None,
        include_dirs: Optional[List[str]] = None
    ) -> CompilationResult:
        """Compile CUDA kernel with comprehensive error handling and metrics."""

        if self.use_docker:
            return await self._compile_in_docker(kernel_code, kernel_name, additional_flags)
        else:
            return await self._compile_native(kernel_code, kernel_name, additional_flags)

    async def _compile_in_docker(
        self,
        kernel_code: str,
        kernel_name: str,
        additional_flags: Optional[List[str]] = None
    ) -> CompilationResult:
        """Compile CUDA kernel in Docker container for maximum safety."""
        start_time = time.time()

        try:
            # Create container-specific temp directory
            container_temp_dir = os.path.join(self.temp_dir, f"container_{kernel_name}_{int(time.time())}")
            os.makedirs(container_temp_dir, exist_ok=True)

            # Wrap kernel code with necessary includes and wrapper
            full_kernel_code = self._wrap_kernel_code(kernel_code, kernel_name)

            # Write kernel to file
            kernel_file = os.path.join(container_temp_dir, f"{kernel_name}.cu")
            with open(kernel_file, 'w') as f:
                f.write(full_kernel_code)

            # Detect GPU architecture
            arch = self.detect_gpu_architecture()

            # Build Docker compilation command
            docker_cmd = [
                "docker", "run", "--rm",
                "--gpus=all",  # Enable GPU access
                "--memory=8g",  # Memory limit
                "--cpus=4.0",   # CPU limit
                "--network=none",  # No network access for security
                f"--volume={container_temp_dir}:/workspace:rw",
                "--workdir=/workspace",
                "--user=root",  # Needed for file permissions
                self.docker_image,
                "bash", "-c", f"""
                    set -e
                    nvcc {self.optimization_level} -arch={arch} \\
                         -shared -Xcompiler -fPIC \\
                         --ptxas-options=-v \\
                         -o {kernel_name}.so {kernel_name}.cu 2>&1
                """
            ]

            # Add additional flags if provided
            if additional_flags:
                # Insert additional flags before the output specification
                insert_index = docker_cmd.index("bash") + 2
                cmd_str = docker_cmd[insert_index]
                cmd_str = cmd_str.replace(f"-o {kernel_name}.so", f"{' '.join(additional_flags)} -o {kernel_name}.so")
                docker_cmd[insert_index] = cmd_str

            # Execute compilation with timeout
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=60.0  # 60 second timeout
                )

                stdout_text = stdout.decode('utf-8') if stdout else ""
                stderr_text = stderr.decode('utf-8') if stderr else ""

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                compilation_time = time.time() - start_time

                return CompilationResult(
                    success=False,
                    stderr="Compilation timeout after 60 seconds",
                    compilation_time=compilation_time,
                    kernel_name=kernel_name
                )

            success = process.returncode == 0
            compilation_time = time.time() - start_time

            # Extract CUDA-specific metrics from compilation output
            register_pressure = self._extract_register_pressure(stderr_text)
            shared_memory_usage = self._extract_shared_memory_usage(stderr_text)
            warnings = self._parse_compilation_warnings(stderr_text)

            # Set binary path if compilation succeeded
            binary_path = None
            if success:
                binary_path = os.path.join(container_temp_dir, f"{kernel_name}.so")
                if not os.path.exists(binary_path):
                    success = False
                    stderr_text += "\nERROR: Binary file was not created"

            self.logger.info(
                "Docker CUDA compilation completed",
                kernel_name=kernel_name,
                success=success,
                compilation_time=compilation_time,
                register_pressure=register_pressure,
                shared_memory_usage=shared_memory_usage,
                warnings_count=len(warnings)
            )

            return CompilationResult(
                success=success,
                binary_path=binary_path,
                stdout=stdout_text,
                stderr=stderr_text,
                compilation_time=compilation_time,
                kernel_name=kernel_name,
                register_pressure=register_pressure,
                shared_memory_usage=shared_memory_usage,
                compilation_warnings=warnings,
                temp_files=[kernel_file, binary_path] if binary_path else [kernel_file],
                resource_usage={
                    "compilation_method": "docker",
                    "cuda_arch": arch,
                    "container_timeout": 60,
                    "memory_limit": "8g",
                    "cpu_limit": "4.0"
                }
            )

        except Exception as e:
            compilation_time = time.time() - start_time
            error_msg = f"Docker compilation error: {str(e)}"

            self.logger.error(
                "Docker CUDA compilation failed",
                kernel_name=kernel_name,
                error=error_msg,
                compilation_time=compilation_time
            )

            return CompilationResult(
                success=False,
                stderr=error_msg,
                compilation_time=compilation_time,
                kernel_name=kernel_name
            )

    async def _compile_native(
        self,
        kernel_code: str,
        kernel_name: str,
        additional_flags: Optional[List[str]] = None
    ) -> CompilationResult:
        """Compile CUDA kernel natively on the host system."""
        start_time = time.time()

        try:
            # Create temp directory for this compilation
            compile_temp_dir = os.path.join(self.temp_dir, f"native_{kernel_name}_{int(time.time())}")
            os.makedirs(compile_temp_dir, exist_ok=True)

            # Wrap kernel code
            full_kernel_code = self._wrap_kernel_code(kernel_code, kernel_name)

            # Write kernel to file
            kernel_file = os.path.join(compile_temp_dir, f"{kernel_name}.cu")
            with open(kernel_file, 'w') as f:
                f.write(full_kernel_code)

            # Detect GPU architecture
            arch = self.detect_gpu_architecture()

            # Build compilation command
            compile_cmd = [
                self.nvcc_path,
                self.optimization_level,
                f"-arch={arch}",
                "-shared",
                "-Xcompiler", "-fPIC",
                "--ptxas-options=-v",
                "-o", os.path.join(compile_temp_dir, f"{kernel_name}.so"),
                kernel_file
            ]

            if additional_flags:
                # Insert additional flags before output specification
                compile_cmd.extend(additional_flags)

            # Execute compilation with timeout
            process = await asyncio.create_subprocess_exec(
                *compile_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=compile_temp_dir
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=60.0
                )

                stdout_text = stdout.decode('utf-8') if stdout else ""
                stderr_text = stderr.decode('utf-8') if stderr else ""

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                compilation_time = time.time() - start_time

                return CompilationResult(
                    success=False,
                    stderr="Native compilation timeout after 60 seconds",
                    compilation_time=compilation_time,
                    kernel_name=kernel_name
                )

            success = process.returncode == 0
            compilation_time = time.time() - start_time

            # Extract metrics
            register_pressure = self._extract_register_pressure(stderr_text)
            shared_memory_usage = self._extract_shared_memory_usage(stderr_text)
            warnings = self._parse_compilation_warnings(stderr_text)

            # Set binary path
            binary_path = None
            if success:
                binary_path = os.path.join(compile_temp_dir, f"{kernel_name}.so")
                if not os.path.exists(binary_path):
                    success = False
                    stderr_text += "\nERROR: Binary file was not created"

            self.logger.info(
                "Native CUDA compilation completed",
                kernel_name=kernel_name,
                success=success,
                compilation_time=compilation_time,
                register_pressure=register_pressure,
                shared_memory_usage=shared_memory_usage,
                warnings_count=len(warnings)
            )

            return CompilationResult(
                success=success,
                binary_path=binary_path,
                stdout=stdout_text,
                stderr=stderr_text,
                compilation_time=compilation_time,
                kernel_name=kernel_name,
                register_pressure=register_pressure,
                shared_memory_usage=shared_memory_usage,
                compilation_warnings=warnings,
                temp_files=[kernel_file, binary_path] if binary_path else [kernel_file],
                resource_usage={
                    "compilation_method": "native",
                    "cuda_arch": arch,
                    "timeout": 60
                }
            )

        except Exception as e:
            compilation_time = time.time() - start_time
            error_msg = f"Native compilation error: {str(e)}"

            self.logger.error(
                "Native CUDA compilation failed",
                kernel_name=kernel_name,
                error=error_msg,
                compilation_time=compilation_time
            )

            return CompilationResult(
                success=False,
                stderr=error_msg,
                compilation_time=compilation_time,
                kernel_name=kernel_name
            )

    def _wrap_kernel_code(self, kernel_code: str, kernel_name: str) -> str:
        """Wrap user kernel code with necessary includes and launch wrapper."""

        # Basic includes
        includes = """
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Error checking macro
#define CUDA_CHECK(call) do { \\
    cudaError_t err = call; \\
    if (err != cudaSuccess) { \\
        fprintf(stderr, "CUDA error at %s:%d - %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \\
        exit(1); \\
    } \\
} while(0)

"""

        # Extract kernel function name from code (simple regex)
        kernel_match = re.search(r'__global__\s+\w+\s+(\w+)\s*\(', kernel_code)
        actual_kernel_name = kernel_match.group(1) if kernel_match else "unknown_kernel"

        # Create C wrapper for easier calling from Python
        wrapper_code = f"""

// C wrapper for Python ctypes interface
extern "C" {{
    void launch_{kernel_name}(void** args, int num_args, dim3 grid, dim3 block, size_t shared_mem) {{
        // This is a simplified launcher - in practice you'd need more sophisticated parameter handling
        switch(num_args) {{
            case 3:
                {actual_kernel_name}<<<grid, block, shared_mem>>>((float*)args[0], (float*)args[1], (float*)args[2]);
                break;
            case 4:
                {actual_kernel_name}<<<grid, block, shared_mem>>>((float*)args[0], (float*)args[1], (float*)args[2], *(int*)args[3]);
                break;
            // Add more cases as needed
            default:
                fprintf(stderr, "Unsupported number of arguments: %d\\n", num_args);
        }}
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }}
}}
"""

        return includes + kernel_code + wrapper_code

    def _extract_register_pressure(self, compilation_output: str) -> int:
        """Extract register usage from nvcc compilation output."""
        # Look for ptxas info about register usage
        register_match = re.search(r'(\d+) registers', compilation_output)
        if register_match:
            return int(register_match.group(1))
        return 0

    def _extract_shared_memory_usage(self, compilation_output: str) -> int:
        """Extract shared memory usage from compilation output."""
        # Look for shared memory usage info
        smem_match = re.search(r'(\d+) bytes smem', compilation_output)
        if smem_match:
            return int(smem_match.group(1))
        return 0

    def _parse_compilation_warnings(self, compilation_output: str) -> List[str]:
        """Parse compilation warnings from nvcc output."""
        warnings = []
        lines = compilation_output.split('\n')

        for line in lines:
            line = line.strip()
            if 'warning:' in line.lower():
                warnings.append(line)
            elif 'note:' in line.lower() and 'optimization' in line.lower():
                warnings.append(line)

        return warnings

    def cleanup_temp_files(self, compilation_result: CompilationResult):
        """Clean up temporary files created during compilation."""
        if compilation_result.temp_files:
            for temp_file in compilation_result.temp_files:
                try:
                    if temp_file and os.path.exists(temp_file):
                        os.remove(temp_file)
                        self.logger.debug(f"Cleaned up temp file: {temp_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up {temp_file}: {e}")
