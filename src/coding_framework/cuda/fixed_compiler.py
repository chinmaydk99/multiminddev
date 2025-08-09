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
    """Fixed CUDA kernel compilation without duplicate output file issues."""
    
    def __init__(
        self, 
        nvcc_path: str = "nvcc",
        cuda_arch: str = "sm_70",  # Default to V100 architecture
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
        Compile CUDA kernel with fixed command generation.
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
            
            # Build compilation command - FIXED: no duplicate output files
            cmd = [
                self.nvcc_path,
                self.optimization_level,
                f"-arch={self.cuda_arch}",
                "--shared",
                "-Xcompiler", "-fPIC",
                "--ptxas-options=-v",  # Verbose PTX assembler for register usage
                "-o", binary_file,
                kernel_file
            ]
            
            # Add additional flags if provided
            if additional_flags:
                cmd.extend(additional_flags)
            
            # Execute compilation
            self.logger.debug("Running compilation command", cmd=" ".join(cmd))
            
            # Use asyncio subprocess for non-blocking execution
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.temp_dir
            )
            
            stdout, stderr = await process.communicate()
            stdout_text = stdout.decode('utf-8', errors='replace')
            stderr_text = stderr.decode('utf-8', errors='replace')
            
            compilation_time = time.time() - start_time
            
            # Check if compilation was successful
            success = process.returncode == 0 and os.path.exists(binary_file)
            
            # Parse compilation output for metrics
            warnings = self._parse_nvcc_warnings(stderr_text) if stderr_text else []
            register_pressure = self._extract_register_info(stderr_text) if stderr_text else None
            shared_memory_usage = self._extract_smem_info(stderr_text) if stderr_text else None
            
            compilation_result = CompilationResult(
                success=success,
                binary_path=binary_file if success else None,
                stdout=stdout_text,
                stderr=stderr_text,
                compilation_time=compilation_time,
                kernel_name=kernel_name,
                temp_files=[kernel_file, binary_file] if success else [kernel_file],
                register_pressure=register_pressure,
                shared_memory_usage=shared_memory_usage,
                compilation_warnings=warnings,
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
                    register_pressure=register_pressure
                )
            else:
                self.logger.error(
                    "CUDA kernel compilation failed",
                    kernel_name=kernel_name,
                    compilation_time=compilation_time,
                    stderr=stderr_text[:500]  # Truncate long errors
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
        """Wrap user kernel code with necessary headers."""
        
        # Check if headers are already included
        has_cuda_headers = any(header in kernel_code for header in [
            "#include <cuda_runtime.h>",
            "#include <device_launch_parameters.h>",
            "__global__"
        ])
        
        if has_cuda_headers or "__global__" in kernel_code:
            # Already has CUDA code
            if not "#include" in kernel_code:
                # Add headers if missing
                return f"""#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

{kernel_code}
"""
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
"""
        return wrapped_code
    
    def _parse_nvcc_warnings(self, stderr_output: str) -> List[str]:
        """Parse nvcc compiler warnings from stderr output."""
        warnings = []
        lines = stderr_output.split('\n')
        
        for line in lines:
            line = line.strip()
            if 'warning' in line.lower() and 'incompatible redefinition' not in line:
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
        register_pattern = r'Used (\d+) registers'
        match = re.search(register_pattern, stderr_output, re.IGNORECASE)
        
        if match:
            return int(match.group(1))
        
        return None
    
    def _extract_smem_info(self, stderr_output: str) -> Optional[int]:
        """Extract shared memory usage information from nvcc output."""
        import re
        
        # Look for shared memory patterns
        smem_pattern = r'(\d+)\s*bytes\s*smem'
        match = re.search(smem_pattern, stderr_output, re.IGNORECASE)
        
        if match:
            return int(match.group(1))
        
        # Also try pattern: "Used N bytes of shared memory"
        smem_pattern2 = r'Used\s+(\d+)\s+bytes.*shared'
        match = re.search(smem_pattern2, stderr_output, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        return None
    
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
            self.logger.debug("nvidia-smi failed, using default", error=str(e))
        
        # Default to V100 architecture for our multi-GPU instance
        return "sm_70"
    
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
            "compiled_kernels_count": len(self.compiled_kernels)
        }
    
    def __del__(self):
        """Cleanup on object destruction."""
        try:
            # Optional cleanup of temp directory
            pass
        except Exception:
            pass