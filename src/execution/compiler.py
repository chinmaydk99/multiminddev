"""
HIP kernel compiler.

Handles compilation of HIP C++ code using hipcc.
"""

import asyncio
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class CompilationResult:
    """Result from compiling a HIP kernel."""
    
    success: bool = False
    binary_path: Optional[str] = None
    compilation_time_s: float = 0.0
    
    # Errors and warnings
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Resource usage (from compiler output)
    register_usage: int = 0
    lds_usage_bytes: int = 0
    spill_sgpr: int = 0
    spill_vgpr: int = 0
    
    # Architecture info
    target_arch: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "binary_path": self.binary_path,
            "compilation_time_s": self.compilation_time_s,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "register_usage": self.register_usage,
            "lds_usage_bytes": self.lds_usage_bytes,
            "spill_sgpr": self.spill_sgpr,
            "spill_vgpr": self.spill_vgpr,
            "target_arch": self.target_arch,
        }


class HIPCompiler:
    """
    Compiler for HIP C++ kernels.
    
    Supports:
    - Local compilation (requires ROCm installed)
    - Docker-based compilation
    - Multiple target architectures
    """
    
    # MI300X architecture
    MI300X_ARCH = "gfx942"
    
    # Common architectures
    SUPPORTED_ARCHS = {
        "mi300x": "gfx942",
        "mi300a": "gfx940",
        "mi250x": "gfx90a",
        "mi210": "gfx90a",
        "mi100": "gfx908",
        "rx7900xtx": "gfx1100",
        "rx7900xt": "gfx1100",
    }
    
    def __init__(
        self,
        target_arch: str = "gfx942",  # MI300X default
        optimization_level: str = "-O3",
        use_docker: bool = False,
        docker_image: str = "rocm/pytorch-nightly:latest",
        timeout_seconds: int = 120,
        keep_temp_files: bool = False,
    ):
        self.target_arch = target_arch
        self.optimization_level = optimization_level
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.timeout_seconds = timeout_seconds
        self.keep_temp_files = keep_temp_files
        
        self.logger = logger.bind(component="HIPCompiler")
        
        # Verify hipcc is available (if not using Docker)
        if not use_docker:
            self._verify_hipcc()
    
    def _verify_hipcc(self):
        """Verify hipcc is available."""
        try:
            result = subprocess.run(
                ["hipcc", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                self.logger.warning("hipcc not found, will use Docker")
                self.use_docker = True
        except FileNotFoundError:
            self.logger.warning("hipcc not found, will use Docker")
            self.use_docker = True
    
    async def compile(
        self,
        source_code: str,
        kernel_name: str,
        output_dir: Optional[Path] = None,
        extra_flags: Optional[List[str]] = None,
    ) -> CompilationResult:
        """
        Compile HIP source code to shared library.
        
        Args:
            source_code: HIP C++ source code
            kernel_name: Name for the output binary
            output_dir: Directory for output files (uses temp if None)
            extra_flags: Additional compiler flags
            
        Returns:
            CompilationResult with success status and metrics
        """
        result = CompilationResult(target_arch=self.target_arch)
        
        # Create output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="hip_kernel_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write source file
        source_file = output_dir / f"{kernel_name}.cpp"
        source_file.write_text(source_code)
        
        # Output binary
        binary_file = output_dir / f"{kernel_name}.so"
        
        # Build compilation command
        cmd = self._build_compile_command(
            source_file,
            binary_file,
            extra_flags or []
        )
        
        self.logger.debug("Compiling kernel", cmd=" ".join(cmd))
        
        start_time = time.time()
        
        try:
            if self.use_docker:
                stdout, stderr, returncode = await self._compile_in_docker(
                    source_code,
                    kernel_name,
                    extra_flags or []
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout_seconds
                )
                stdout = stdout_bytes.decode()
                stderr = stderr_bytes.decode()
                returncode = proc.returncode
            
            result.compilation_time_s = time.time() - start_time
            
            if returncode == 0:
                result.success = True
                result.binary_path = str(binary_file)
                
                # Parse compiler output for resource usage
                resource_info = self._parse_resource_usage(stderr)
                result.register_usage = resource_info.get("vgpr", 0)
                result.lds_usage_bytes = resource_info.get("lds", 0)
                result.spill_sgpr = resource_info.get("spill_sgpr", 0)
                result.spill_vgpr = resource_info.get("spill_vgpr", 0)
                
                result.warnings = self._parse_warnings(stderr)
                
                self.logger.info(
                    "Compilation successful",
                    kernel=kernel_name,
                    time_s=f"{result.compilation_time_s:.2f}",
                    vgpr=result.register_usage,
                    lds=result.lds_usage_bytes
                )
            else:
                result.success = False
                result.error_message = stderr
                
                self.logger.warning(
                    "Compilation failed",
                    kernel=kernel_name,
                    error=stderr[:500]  # Truncate long errors
                )
        
        except asyncio.TimeoutError:
            result.compilation_time_s = self.timeout_seconds
            result.error_message = f"Compilation timed out after {self.timeout_seconds}s"
            self.logger.error("Compilation timeout", kernel=kernel_name)
        
        except Exception as e:
            result.compilation_time_s = time.time() - start_time
            result.error_message = str(e)
            self.logger.error("Compilation error", kernel=kernel_name, error=str(e))
        
        finally:
            # Cleanup temp files if requested
            if not self.keep_temp_files and output_dir.name.startswith("hip_kernel_"):
                import shutil
                shutil.rmtree(output_dir, ignore_errors=True)
        
        return result
    
    def _build_compile_command(
        self,
        source_file: Path,
        output_file: Path,
        extra_flags: List[str],
    ) -> List[str]:
        """Build hipcc command line."""
        
        cmd = [
            "hipcc",
            "-shared",
            "-fPIC",
            self.optimization_level,
            f"--offload-arch={self.target_arch}",
            "-Rpass-analysis=kernel-resource-usage",  # Get resource usage info
            "-o", str(output_file),
            str(source_file),
        ]
        
        cmd.extend(extra_flags)
        
        return cmd
    
    async def _compile_in_docker(
        self,
        source_code: str,
        kernel_name: str,
        extra_flags: List[str],
    ) -> tuple:
        """Compile kernel inside Docker container."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Write source
            source_file = tmpdir / f"{kernel_name}.cpp"
            source_file.write_text(source_code)
            
            # Docker command
            docker_cmd = [
                "docker", "run", "--rm",
                "--device=/dev/kfd",
                "--device=/dev/dri",
                f"-v{tmpdir}:/workspace",
                "-w/workspace",
                self.docker_image,
                "hipcc",
                "-shared",
                "-fPIC",
                self.optimization_level,
                f"--offload-arch={self.target_arch}",
                "-Rpass-analysis=kernel-resource-usage",
                "-o", f"/workspace/{kernel_name}.so",
                f"/workspace/{kernel_name}.cpp",
            ]
            docker_cmd.extend(extra_flags)
            
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout_seconds
            )
            
            return stdout.decode(), stderr.decode(), proc.returncode
    
    def _parse_resource_usage(self, compiler_output: str) -> Dict[str, int]:
        """Parse register and LDS usage from compiler output."""
        
        result = {}
        
        # Look for VGPR usage
        vgpr_match = re.search(r'vgpr:\s*(\d+)', compiler_output, re.IGNORECASE)
        if vgpr_match:
            result["vgpr"] = int(vgpr_match.group(1))
        
        # Look for SGPR usage
        sgpr_match = re.search(r'sgpr:\s*(\d+)', compiler_output, re.IGNORECASE)
        if sgpr_match:
            result["sgpr"] = int(sgpr_match.group(1))
        
        # Look for LDS usage
        lds_match = re.search(r'lds:\s*(\d+)', compiler_output, re.IGNORECASE)
        if lds_match:
            result["lds"] = int(lds_match.group(1))
        
        # Look for spills
        spill_sgpr_match = re.search(r'spill\s+sgpr:\s*(\d+)', compiler_output, re.IGNORECASE)
        if spill_sgpr_match:
            result["spill_sgpr"] = int(spill_sgpr_match.group(1))
        
        spill_vgpr_match = re.search(r'spill\s+vgpr:\s*(\d+)', compiler_output, re.IGNORECASE)
        if spill_vgpr_match:
            result["spill_vgpr"] = int(spill_vgpr_match.group(1))
        
        return result
    
    def _parse_warnings(self, compiler_output: str) -> List[str]:
        """Extract warnings from compiler output."""
        
        warnings = []
        for line in compiler_output.split('\n'):
            if 'warning:' in line.lower():
                warnings.append(line.strip())
        return warnings

