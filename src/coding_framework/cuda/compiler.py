import os
import tempfile
import subprocess
import time
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import structlog


@dataclass
class CompilationResult:
    """Result of CUDA kernel compilation with enhanced metrics for reward functions."""
    success: bool
    binary_path: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    compilation_time: float = 0.0
    kernel_name: str = ""
    temp_files: List[str] = None
    # Enhanced fields for reward functions
    register_pressure: Optional[int] = None  # Register usage per thread
    shared_memory_usage: Optional[int] = None  # Shared memory bytes
    compilation_warnings: List[str] = None
    ptx_code: Optional[str] = None  # PTX intermediate code
    resource_usage: Dict[str, Any] = None  # Additional resource metrics
    
    def __post_init__(self):
        if self.temp_files is None:
            self.temp_files = []
        if self.compilation_warnings is None:
            self.compilation_warnings = []
        if self.resource_usage is None:
            self.resource_usage = {}


class CUDACompiler:
    """CUDA kernel compilation and execution management."""
    
    def __init__(
        self, 
        nvcc_path: str = "nvcc",
        cuda_arch: str = "sm_75",
        optimization_level: str = "-O3",
        temp_dir: Optional[str] = None
    ):
        self.nvcc_path = nvcc_path
        self.cuda_arch = cuda_arch
        self.optimization_level = optimization_level
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="cuda_kernels_")
        self.logger = structlog.get_logger("cuda_compiler")
        
        # Ensure temp directory exists
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        
        # Track compiled kernels for cleanup
        self.compiled_kernels: List[str] = []
        
        self.logger.info(
            "CUDA compiler initialized",
            nvcc_path=nvcc_path,
            cuda_arch=cuda_arch,
            temp_dir=self.temp_dir
        )
    
    async def compile_kernel(
        self, 
        kernel_code: str, 
        kernel_name: str,
        additional_flags: Optional[List[str]] = None,
        use_docker: bool = False
    ) -> CompilationResult:
        """
        Compile CUDA kernel and return compilation status with enhanced metrics.
        
        Args:
            kernel_code: CUDA kernel source code
            kernel_name: Name identifier for the kernel
            additional_flags: Optional additional compiler flags
            use_docker: Whether to use Docker for safe compilation
            
        Returns:
            CompilationResult with compilation status and enhanced metrics
        """
        start_time = time.time()
        
        try:
            # Use Docker compilation if requested and available
            if use_docker and self._docker_available():
                return await self._compile_in_container(kernel_code, kernel_name, additional_flags)
            
            # Prepare kernel code with necessary headers
            full_kernel_code = self._wrap_kernel_code(kernel_code)
            
            # Create temporary files
            kernel_file = os.path.join(self.temp_dir, f"{kernel_name}.cu")
            binary_file = os.path.join(self.temp_dir, f"{kernel_name}.so")
            ptx_file = os.path.join(self.temp_dir, f"{kernel_name}.ptx")
            
            # Write kernel to file
            with open(kernel_file, 'w') as f:
                f.write(full_kernel_code)
            
            # Build compilation command with PTX generation for analysis
            cmd = self._build_compile_command(kernel_file, binary_file, additional_flags, ptx_file)
            
            # Execute compilation
            result = await self._run_compilation(cmd)
            
            compilation_time = time.time() - start_time
            
            # Check if compilation was successful
            success = result.returncode == 0 and os.path.exists(binary_file)
            
            # Parse compilation warnings and extract resource usage
            warnings = self._parse_nvcc_warnings(result.stderr) if result.stderr else []
            register_pressure = self._extract_register_info(result.stderr) if result.stderr else None
            shared_memory_usage = self._extract_smem_info(result.stderr) if result.stderr else None
            
            # Read PTX code if available
            ptx_code = None
            if os.path.exists(ptx_file):
                try:
                    with open(ptx_file, 'r') as f:
                        ptx_code = f.read()
                except Exception as e:
                    self.logger.debug("Failed to read PTX file", error=str(e))
            
            compilation_result = CompilationResult(
                success=success,
                binary_path=binary_file if success else None,
                stdout=result.stdout,
                stderr=result.stderr,
                compilation_time=compilation_time,
                kernel_name=kernel_name,
                temp_files=[kernel_file, binary_file, ptx_file] if success else [kernel_file],
                register_pressure=register_pressure,
                shared_memory_usage=shared_memory_usage,
                compilation_warnings=warnings,
                ptx_code=ptx_code,
                resource_usage={
                    "cuda_arch": self.cuda_arch,
                    "optimization_level": self.optimization_level,
                    "additional_flags": additional_flags or []
                }
            )
            
            if success:
                self.compiled_kernels.append(binary_file)
                self.logger.info(
                    "CUDA kernel compiled successfully",
                    kernel_name=kernel_name,
                    compilation_time=compilation_time,
                    binary_path=binary_file,
                    register_pressure=register_pressure,
                    warnings_count=len(warnings)
                )
            else:
                self.logger.error(
                    "CUDA kernel compilation failed",
                    kernel_name=kernel_name,
                    compilation_time=compilation_time,
                    stderr=result.stderr[:500]  # Truncate long errors
                )
            
            return compilation_result
            
        except Exception as e:
            compilation_time = time.time() - start_time
            error_msg = f"Compilation error: {str(e)}"
            
            self.logger.error(
                "CUDA compilation exception",
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
    
    def _wrap_kernel_code(self, kernel_code: str) -> str:
        """Wrap user kernel code with necessary headers and host code."""
        
        # Check if headers are already included
        has_cuda_headers = any(header in kernel_code for header in [
            "#include <cuda_runtime.h>",
            "#include <device_launch_parameters.h>"
        ])
        
        if has_cuda_headers:
            return kernel_code
        
        # Add standard CUDA headers and wrapper
        wrapped_code = f"""#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

{kernel_code}

// Error checking macro
#define CUDA_CHECK(call) do {{ \\
    cudaError_t error = call; \\
    if (error != cudaSuccess) {{ \\
        fprintf(stderr, "CUDA error at %s:%d - %s\\n", __FILE__, __LINE__, cudaGetErrorString(error)); \\
        exit(1); \\
    }} \\
}} while(0)

// Host utility functions will be generated here if needed
extern "C" {{
    // Export functions for Python ctypes interface
}}
"""
        return wrapped_code
    
    def _build_compile_command(
        self, 
        source_file: str, 
        output_file: str,
        additional_flags: Optional[List[str]] = None,
        ptx_file: Optional[str] = None
    ) -> List[str]:
        """Build nvcc compilation command with enhanced options."""
        
        cmd = [
            self.nvcc_path,
            self.optimization_level,
            f"-arch={self.cuda_arch}",
            "-shared",
            "-Xcompiler", "-fPIC",
            "--compiler-options", "-ffast-math",
            "--ptxas-options=-v",  # Verbose PTX assembler for register usage
            "-lineinfo",  # Include line info for debugging
            "-o", output_file,
            source_file
        ]
        
        # Generate PTX intermediate code for analysis
        if ptx_file:
            cmd.extend(["-ptx", "-o", ptx_file, source_file])
        
        # Add additional flags if provided
        if additional_flags:
            cmd.extend(additional_flags)
        
        return cmd
    
    async def _run_compilation(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Execute compilation command asynchronously."""
        
        self.logger.debug("Running compilation command", cmd=" ".join(cmd))
        
        # Use asyncio subprocess for non-blocking execution
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.temp_dir
        )
        
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout.decode('utf-8', errors='replace'),
            stderr=stderr.decode('utf-8', errors='replace')
        )
    
    def detect_cuda_arch(self) -> str:
        """Dynamically detect CUDA architecture of available GPU."""
        try:
            # Try nvidia-smi first for compute capability
            result = subprocess.run([
                "nvidia-smi", 
                "--query-gpu=compute_cap", 
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                compute_caps = result.stdout.strip().split('\n')
                if compute_caps and compute_caps[0]:
                    compute_cap = compute_caps[0].strip()
                    arch = f"sm_{compute_cap.replace('.', '')}"
                    self.logger.info("Detected CUDA architecture via nvidia-smi", arch=arch)
                    return arch
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.debug("nvidia-smi failed, trying deviceQuery", error=str(e))
        
        try:
            # Fallback: try nvcc deviceQuery if available
            result = subprocess.run([
                "nvcc", "--help"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                self.logger.info("nvcc available, using default modern architecture")
                return "sm_80"  # Modern default for RTX 30xx/40xx series
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Final fallback
        self.logger.warning("Could not detect GPU architecture, using fallback")
        return "sm_75"
    
    def cleanup_old_kernels(self, max_kernels: int = 20) -> None:
        """Clean up old compiled kernel binaries to save disk space."""
        if len(self.compiled_kernels) > max_kernels:
            # Remove oldest kernels
            kernels_to_remove = self.compiled_kernels[:len(self.compiled_kernels) - max_kernels]
            
            for kernel_path in kernels_to_remove:
                try:
                    if os.path.exists(kernel_path):
                        os.remove(kernel_path)
                    self.logger.debug("Removed old kernel binary", path=kernel_path)
                except OSError as e:
                    self.logger.warning("Failed to remove old kernel", path=kernel_path, error=str(e))
            
            # Update tracked kernels
            self.compiled_kernels = self.compiled_kernels[len(kernels_to_remove):]
            
            self.logger.info("Cleaned up old kernels", removed_count=len(kernels_to_remove))
    
    def get_compilation_info(self) -> Dict[str, Any]:
        """Get information about compiler configuration and status."""
        return {
            "nvcc_path": self.nvcc_path,
            "cuda_arch": self.cuda_arch,
            "optimization_level": self.optimization_level,
            "temp_dir": self.temp_dir,
            "compiled_kernels_count": len(self.compiled_kernels),
            "available_disk_space": self._get_disk_space()
        }
    
    def _get_disk_space(self) -> Optional[int]:
        """Get available disk space in temp directory."""
        try:
            import shutil
            _, _, free = shutil.disk_usage(self.temp_dir)
            return free
        except Exception:
            return None
    
    def _parse_nvcc_warnings(self, stderr_output: str) -> List[str]:
        """Parse nvcc compiler warnings from stderr output."""
        warnings = []
        lines = stderr_output.split('\n')
        
        for line in lines:
            line = line.strip()
            if 'warning' in line.lower():
                warnings.append(line)
            elif 'ptxas info' in line.lower():
                # PTX assembler info can contain useful resource information
                warnings.append(line)
        
        return warnings
    
    def _extract_register_info(self, stderr_output: str) -> Optional[int]:
        """Extract register usage information from nvcc output."""
        import re
        
        # Look for register usage patterns in ptxas output
        # Pattern: "ptxas info    : Used N registers"
        register_pattern = r'ptxas info.*?Used (\d+) registers'
        match = re.search(register_pattern, stderr_output, re.IGNORECASE)
        
        if match:
            return int(match.group(1))
        
        return None
    
    def _extract_smem_info(self, stderr_output: str) -> Optional[int]:
        """Extract shared memory usage information from nvcc output."""
        import re
        
        # Look for shared memory patterns
        # Pattern: "ptxas info    : Compiling entry function 'kernel_name' for 'sm_XX'"
        # followed by shared memory info
        smem_pattern = r'(\d+)\s*bytes\s*smem'
        match = re.search(smem_pattern, stderr_output, re.IGNORECASE)
        
        if match:
            return int(match.group(1))
        
        return None
    
    def _docker_available(self) -> bool:
        """Check if Docker is available for containerized compilation."""
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    async def _compile_in_container(
        self, 
        kernel_code: str, 
        kernel_name: str,
        additional_flags: Optional[List[str]] = None
    ) -> CompilationResult:
        """Compile CUDA kernel in Docker container for safety."""
        start_time = time.time()
        
        try:
            # Create temporary directory for container compilation
            container_temp_dir = os.path.join(self.temp_dir, f"container_{kernel_name}")
            os.makedirs(container_temp_dir, exist_ok=True)
            
            # Prepare kernel code
            full_kernel_code = self._wrap_kernel_code(kernel_code)
            
            # Write kernel to container temp directory
            kernel_file = os.path.join(container_temp_dir, f"{kernel_name}.cu")
            with open(kernel_file, 'w') as f:
                f.write(full_kernel_code)
            
            # Build Docker compilation command
            docker_cmd = [
                "docker", "run", "--rm",
                "--gpus=all",  # Enable GPU access
                "--memory=8g", "--cpus=4.0",  # Resource limits
                "--network=none",  # No network access for security
                f"--volume={container_temp_dir}:/workspace",
                "--workdir=/workspace",
                "nvidia/cuda:12.0-devel-ubuntu20.04",
                "nvcc", 
                self.optimization_level,
                f"-arch={self.cuda_arch}",
                "-shared", "-Xcompiler", "-fPIC",
                "--ptxas-options=-v",
                "-o", f"{kernel_name}.so",
                f"{kernel_name}.cu"
            ]
            
            # Add additional flags
            if additional_flags:
                docker_cmd.extend(additional_flags)
            
            # Execute Docker compilation with timeout
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=30  # 30 second timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise RuntimeError("Docker compilation timed out after 30 seconds")
            
            compilation_time = time.time() - start_time
            
            # Check compilation success
            binary_file = os.path.join(container_temp_dir, f"{kernel_name}.so")
            success = process.returncode == 0 and os.path.exists(binary_file)
            
            # Parse output
            stdout_text = stdout.decode('utf-8', errors='replace')
            stderr_text = stderr.decode('utf-8', errors='replace')
            
            warnings = self._parse_nvcc_warnings(stderr_text)
            register_pressure = self._extract_register_info(stderr_text)
            shared_memory_usage = self._extract_smem_info(stderr_text)
            
            # Move binary to main temp directory if successful
            final_binary_path = None
            if success:
                final_binary_path = os.path.join(self.temp_dir, f"{kernel_name}.so")
                import shutil
                shutil.move(binary_file, final_binary_path)
                self.compiled_kernels.append(final_binary_path)
            
            self.logger.info(
                "Docker CUDA compilation completed",
                kernel_name=kernel_name,
                success=success,
                compilation_time=compilation_time,
                warnings_count=len(warnings)
            )
            
            return CompilationResult(
                success=success,
                binary_path=final_binary_path,
                stdout=stdout_text,
                stderr=stderr_text,
                compilation_time=compilation_time,
                kernel_name=kernel_name,
                temp_files=[kernel_file, final_binary_path] if success else [kernel_file],
                register_pressure=register_pressure,
                shared_memory_usage=shared_memory_usage,
                compilation_warnings=warnings,
                resource_usage={
                    "compilation_method": "docker",
                    "cuda_arch": self.cuda_arch,
                    "container_timeout": 30
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

    def __del__(self):
        """Cleanup on object destruction."""
        try:
            # Optional cleanup of temp directory
            # Note: Be careful with automatic cleanup in production
            pass
        except Exception:
            pass