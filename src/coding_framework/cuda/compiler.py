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
    """Result of CUDA kernel compilation."""
    success: bool
    binary_path: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    compilation_time: float = 0.0
    kernel_name: str = ""
    temp_files: List[str] = None
    
    def __post_init__(self):
        if self.temp_files is None:
            self.temp_files = []


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
        additional_flags: Optional[List[str]] = None
    ) -> CompilationResult:
        """
        Compile CUDA kernel and return compilation status.
        
        Args:
            kernel_code: CUDA kernel source code
            kernel_name: Name identifier for the kernel
            additional_flags: Optional additional compiler flags
            
        Returns:
            CompilationResult with compilation status and artifacts
        """
        start_time = time.time()
        
        try:
            # Prepare kernel code with necessary headers
            full_kernel_code = self._wrap_kernel_code(kernel_code)
            
            # Create temporary files
            kernel_file = os.path.join(self.temp_dir, f"{kernel_name}.cu")
            binary_file = os.path.join(self.temp_dir, f"{kernel_name}.so")
            
            # Write kernel to file
            with open(kernel_file, 'w') as f:
                f.write(full_kernel_code)
            
            # Build compilation command
            cmd = self._build_compile_command(kernel_file, binary_file, additional_flags)
            
            # Execute compilation
            result = await self._run_compilation(cmd)
            
            compilation_time = time.time() - start_time
            
            # Check if compilation was successful
            success = result.returncode == 0 and os.path.exists(binary_file)
            
            compilation_result = CompilationResult(
                success=success,
                binary_path=binary_file if success else None,
                stdout=result.stdout,
                stderr=result.stderr,
                compilation_time=compilation_time,
                kernel_name=kernel_name,
                temp_files=[kernel_file, binary_file] if success else [kernel_file]
            )
            
            if success:
                self.compiled_kernels.append(binary_file)
                self.logger.info(
                    "CUDA kernel compiled successfully",
                    kernel_name=kernel_name,
                    compilation_time=compilation_time,
                    binary_path=binary_file
                )
            else:
                self.logger.error(
                    "CUDA kernel compilation failed",
                    kernel_name=kernel_name,
                    compilation_time=compilation_time,
                    stderr=result.stderr
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
        additional_flags: Optional[List[str]] = None
    ) -> List[str]:
        """Build nvcc compilation command."""
        
        cmd = [
            self.nvcc_path,
            self.optimization_level,
            f"-arch={self.cuda_arch}",
            "-shared",
            "-Xcompiler", "-fPIC",
            "--compiler-options", "-ffast-math",
            "-o", output_file,
            source_file
        ]
        
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
            result = subprocess.run([
                "nvidia-smi", 
                "--query-gpu=compute_cap", 
                "--format=csv,noheader"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                compute_cap = result.stdout.strip().split('\n')[0]
                arch = f"sm_{compute_cap.replace('.', '')}"
                self.logger.info("Detected CUDA architecture", arch=arch)
                return arch
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.warning("Failed to detect CUDA architecture", error=str(e))
        
        # Fallback to default
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
    
    def __del__(self):
        """Cleanup on object destruction."""
        try:
            # Optional cleanup of temp directory
            # Note: Be careful with automatic cleanup in production
            pass
        except Exception:
            pass