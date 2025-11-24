"""
ROCm profiler integration.

Uses rocprof to collect detailed performance metrics.
"""

import asyncio
import csv
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class ProfileResult:
    """Profiling result from rocprof."""
    
    success: bool = False
    error_message: Optional[str] = None
    
    # Timing
    kernel_time_ns: int = 0
    kernel_time_ms: float = 0.0
    
    # Compute metrics
    wavefronts: int = 0
    work_groups: int = 0
    waves_per_cu: float = 0.0
    occupancy: float = 0.0
    
    # Memory metrics
    memory_bandwidth_gbps: float = 0.0
    l2_cache_hit_rate: float = 0.0
    lds_bank_conflicts: int = 0
    
    # Instruction metrics
    valu_utilization: float = 0.0
    mfma_utilization: float = 0.0  # Matrix core utilization
    vmem_utilization: float = 0.0
    lds_utilization: float = 0.0
    
    # Resource usage
    vgpr_count: int = 0
    sgpr_count: int = 0
    lds_size_bytes: int = 0
    
    # Raw counters
    raw_counters: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "error_message": self.error_message,
            "kernel_time_ms": self.kernel_time_ms,
            "wavefronts": self.wavefronts,
            "occupancy": self.occupancy,
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
            "l2_cache_hit_rate": self.l2_cache_hit_rate,
            "valu_utilization": self.valu_utilization,
            "mfma_utilization": self.mfma_utilization,
            "vgpr_count": self.vgpr_count,
            "sgpr_count": self.sgpr_count,
            "lds_size_bytes": self.lds_size_bytes,
            "raw_counters": self.raw_counters,
        }


class ROCProfiler:
    """
    Interface to rocprof for detailed kernel profiling.
    
    Collects:
    - Kernel execution time
    - Hardware counter values
    - Memory bandwidth
    - Occupancy metrics
    """
    
    # Common hardware counters for MI300X
    MI300X_COUNTERS = [
        "GRBM_COUNT",
        "GRBM_GUI_ACTIVE",
        "SQ_WAVES",
        "SQ_INSTS_VALU",
        "SQ_INSTS_VMEM",
        "SQ_INSTS_LDS",
        "SQ_INSTS_MFMA",  # Matrix core instructions
        "TCC_HIT",
        "TCC_MISS",
        "TCC_EA_WRREQ",
        "TCC_EA_RDREQ",
    ]
    
    def __init__(
        self,
        rocprof_path: str = "rocprof",
        use_docker: bool = False,
        docker_image: str = "rocm/pytorch-nightly:latest",
        timeout_seconds: int = 60,
        counters: Optional[List[str]] = None,
    ):
        self.rocprof_path = rocprof_path
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.timeout_seconds = timeout_seconds
        self.counters = counters or self.MI300X_COUNTERS
        
        self.logger = logger.bind(component="ROCProfiler")
    
    async def profile_kernel(
        self,
        binary_path: str,
        kernel_name: str,
        launch_config: Dict[str, Any],
        input_tensors: Dict[str, Any],
    ) -> ProfileResult:
        """
        Profile a kernel with rocprof.
        
        Args:
            binary_path: Path to compiled kernel .so
            kernel_name: Name of kernel function
            launch_config: Grid and block dimensions
            input_tensors: Input data for kernel
            
        Returns:
            ProfileResult with all metrics
        """
        result = ProfileResult()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create input file for rocprof
            input_file = self._create_input_file(tmpdir, self.counters)
            
            # Create runner script
            runner_script = self._create_runner_script(
                tmpdir,
                binary_path,
                kernel_name,
                launch_config,
                input_tensors
            )
            
            # Output file
            output_file = tmpdir / "profile_output.csv"
            
            try:
                # Run rocprof
                cmd = [
                    self.rocprof_path,
                    "-i", str(input_file),
                    "-o", str(output_file),
                    "--stats",
                    "python", str(runner_script),
                ]
                
                if self.use_docker:
                    # Wrap in Docker
                    cmd = self._wrap_docker_command(cmd, tmpdir)
                
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout_seconds
                )
                
                if proc.returncode == 0:
                    # Parse results
                    result = self._parse_output(output_file, stdout.decode())
                    result.success = True
                else:
                    result.error_message = stderr.decode()
                    self.logger.warning("Profiling failed", error=result.error_message)
            
            except asyncio.TimeoutError:
                result.error_message = f"Profiling timed out after {self.timeout_seconds}s"
                self.logger.error("Profiling timeout")
            
            except Exception as e:
                result.error_message = str(e)
                self.logger.error("Profiling error", error=str(e))
        
        return result
    
    def _create_input_file(self, tmpdir: Path, counters: List[str]) -> Path:
        """Create rocprof input file with counter specifications."""
        
        input_file = tmpdir / "rocprof_input.txt"
        
        # Format: pmc: counter1, counter2, ...
        # rocprof can only collect 4 counters at a time
        lines = []
        for i in range(0, len(counters), 4):
            chunk = counters[i:i+4]
            lines.append(f"pmc: {', '.join(chunk)}")
        
        input_file.write_text('\n'.join(lines))
        return input_file
    
    def _create_runner_script(
        self,
        tmpdir: Path,
        binary_path: str,
        kernel_name: str,
        launch_config: Dict[str, Any],
        input_tensors: Dict[str, Any],
    ) -> Path:
        """Create Python script to run the kernel."""
        
        script = f'''
import ctypes
import torch

# Load kernel
lib = ctypes.CDLL("{binary_path}")

# Setup inputs
# TODO: Implement based on actual kernel signature

# Run kernel
grid = {launch_config.get("grid", (1, 1, 1))}
block = {launch_config.get("block", (256, 1, 1))}

# Launch
# TODO: Implement kernel launch

# Sync
torch.cuda.synchronize()
'''
        
        script_file = tmpdir / "run_kernel.py"
        script_file.write_text(script)
        return script_file
    
    def _wrap_docker_command(self, cmd: List[str], tmpdir: Path) -> List[str]:
        """Wrap command to run in Docker."""
        
        return [
            "docker", "run", "--rm",
            "--device=/dev/kfd",
            "--device=/dev/dri",
            f"-v{tmpdir}:/workspace",
            "-w/workspace",
            self.docker_image,
        ] + cmd
    
    def _parse_output(self, output_file: Path, stdout: str) -> ProfileResult:
        """Parse rocprof output CSV."""
        
        result = ProfileResult()
        
        if not output_file.exists():
            result.error_message = "Output file not found"
            return result
        
        try:
            with open(output_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if not rows:
                    result.error_message = "No profiling data"
                    return result
                
                # Get first kernel invocation
                row = rows[0]
                
                # Parse timing
                if "DurationNs" in row:
                    result.kernel_time_ns = int(row["DurationNs"])
                    result.kernel_time_ms = result.kernel_time_ns / 1e6
                
                # Parse counters
                for key, value in row.items():
                    if key not in ["Index", "KernelName", "DurationNs"]:
                        try:
                            result.raw_counters[key] = int(value)
                        except ValueError:
                            pass
                
                # Calculate derived metrics
                result = self._calculate_metrics(result)
        
        except Exception as e:
            result.error_message = f"Failed to parse output: {e}"
        
        return result
    
    def _calculate_metrics(self, result: ProfileResult) -> ProfileResult:
        """Calculate derived metrics from raw counters."""
        
        counters = result.raw_counters
        
        # Wavefronts
        result.wavefronts = counters.get("SQ_WAVES", 0)
        
        # VALU utilization
        valu_insts = counters.get("SQ_INSTS_VALU", 0)
        if result.kernel_time_ns > 0:
            # Rough estimate assuming clock speed
            result.valu_utilization = min(1.0, valu_insts / (result.kernel_time_ns * 0.001))
        
        # MFMA utilization (matrix cores)
        mfma_insts = counters.get("SQ_INSTS_MFMA", 0)
        if result.kernel_time_ns > 0:
            result.mfma_utilization = min(1.0, mfma_insts / (result.kernel_time_ns * 0.0001))
        
        # L2 cache hit rate
        l2_hits = counters.get("TCC_HIT", 0)
        l2_misses = counters.get("TCC_MISS", 0)
        total_l2 = l2_hits + l2_misses
        if total_l2 > 0:
            result.l2_cache_hit_rate = l2_hits / total_l2
        
        # Memory bandwidth (rough estimate)
        # TCC_EA_RDREQ + TCC_EA_WRREQ = memory transactions
        # Assuming 64 bytes per transaction
        rd_req = counters.get("TCC_EA_RDREQ", 0)
        wr_req = counters.get("TCC_EA_WRREQ", 0)
        total_bytes = (rd_req + wr_req) * 64
        if result.kernel_time_ns > 0:
            result.memory_bandwidth_gbps = total_bytes / result.kernel_time_ns  # GB/s
        
        return result

