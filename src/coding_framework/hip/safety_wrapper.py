"""
Safety wrapper for HIP kernel compilation and execution on AMD ROCm.
Provides sandboxed execution environment using Docker containers with resource limits.
"""

import asyncio
import hashlib
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog


@dataclass
class SafetyViolation:
    """Details about a safety violation."""
    violation_type: str
    description: str
    severity: str  # "critical", "high", "medium", "low"
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of safety validation."""
    is_safe: bool
    violations: List[SafetyViolation]
    risk_score: float  # 0.0 (safe) to 1.0 (dangerous)
    sanitized_code: Optional[str] = None


class SecurityValidator:
    """
    Validates HIP code for security risks before compilation.
    """

    # Dangerous patterns that should be blocked
    DANGEROUS_PATTERNS = [
        # System calls
        (r'system\s*\(', 'System call detected', 'critical'),
        (r'exec[lv]?[pe]?\s*\(', 'Process execution detected', 'critical'),
        (r'fork\s*\(', 'Process forking detected', 'critical'),

        # File operations
        (r'fopen\s*\(', 'File operation detected', 'high'),
        (r'open\s*\(', 'File open detected', 'high'),
        (r'unlink\s*\(', 'File deletion detected', 'critical'),
        (r'remove\s*\(', 'File removal detected', 'critical'),

        # Network operations
        (r'socket\s*\(', 'Network socket creation detected', 'critical'),
        (r'connect\s*\(', 'Network connection detected', 'critical'),
        (r'bind\s*\(', 'Network binding detected', 'critical'),

        # Memory operations that could be dangerous
        (r'mmap\s*\(', 'Memory mapping detected', 'high'),
        (r'dlopen\s*\(', 'Dynamic library loading detected', 'critical'),

        # Assembly injection (HIP/AMD specific)
        (r'asm\s*\(', 'Inline assembly detected', 'high'),
        (r'__asm__', 'Inline assembly detected', 'high'),
        (r'__builtin_amdgcn', 'AMD GCN builtin detected', 'medium'),

        # Preprocessor abuse
        (r'#\s*include\s*<[^>]*\.h>', 'System header inclusion', 'medium'),
        (r'#\s*define\s+[A-Z_]+\s+system', 'Macro aliasing system calls', 'critical'),
    ]

    # Suspicious patterns that need review
    SUSPICIOUS_PATTERNS = [
        (r'malloc\s*\(\s*\d{10,}', 'Excessive memory allocation', 'medium'),
        (r'hipMalloc[^;]{100,}', 'Complex memory allocation', 'low'),
        (r'while\s*\(\s*1\s*\)', 'Infinite loop detected', 'medium'),
        (r'for\s*\(\s*;\s*;\s*\)', 'Infinite loop detected', 'medium'),
    ]

    def __init__(self):
        self.logger = structlog.get_logger()

    def validate_code(self, code: str) -> ValidationResult:
        """
        Validate HIP code for security risks.
        
        Args:
            code: HIP kernel code to validate
            
        Returns:
            ValidationResult with violations and risk assessment
        """
        violations = []

        # Check for dangerous patterns
        for pattern, description, severity in self.DANGEROUS_PATTERNS:
            matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                violations.append(SafetyViolation(
                    violation_type="dangerous_pattern",
                    description=description,
                    severity=severity,
                    line_number=line_num,
                    code_snippet=match.group(0)
                ))

        # Check for suspicious patterns
        for pattern, description, severity in self.SUSPICIOUS_PATTERNS:
            matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                violations.append(SafetyViolation(
                    violation_type="suspicious_pattern",
                    description=description,
                    severity=severity,
                    line_number=line_num,
                    code_snippet=match.group(0)
                ))

        # Calculate risk score
        risk_score = self._calculate_risk_score(violations)

        # Determine if code is safe
        critical_violations = [v for v in violations if v.severity == "critical"]
        is_safe = len(critical_violations) == 0 and risk_score < 0.7

        # Attempt to sanitize if not safe but fixable
        sanitized_code = None
        if not is_safe and risk_score < 0.9:
            sanitized_code = self._sanitize_code(code, violations)

        return ValidationResult(
            is_safe=is_safe,
            violations=violations,
            risk_score=risk_score,
            sanitized_code=sanitized_code
        )

    def _calculate_risk_score(self, violations: List[SafetyViolation]) -> float:
        """Calculate overall risk score from violations."""

        if not violations:
            return 0.0

        severity_weights = {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.2
        }

        total_weight = sum(severity_weights[v.severity] for v in violations)
        max_possible = len(violations) * 1.0  # All critical

        return min(1.0, total_weight / max(max_possible, 1.0))

    def _sanitize_code(
        self,
        code: str,
        violations: List[SafetyViolation]
    ) -> Optional[str]:
        """
        Attempt to sanitize code by removing dangerous constructs.
        
        Args:
            code: Original code
            violations: List of violations found
            
        Returns:
            Sanitized code or None if not possible
        """
        # For critical violations, we can't sanitize
        if any(v.severity == "critical" for v in violations):
            return None

        sanitized = code

        # Remove suspicious patterns that can be safely removed
        for pattern, _, severity in self.SUSPICIOUS_PATTERNS:
            if severity in ["low", "medium"]:
                # Comment out the suspicious lines
                sanitized = re.sub(
                    f'({pattern})',
                    r'/* SANITIZED: \1 */',
                    sanitized,
                    flags=re.IGNORECASE | re.MULTILINE
                )

        return sanitized


class DockerSandbox:
    """
    Provides Docker-based sandboxed execution environment for HIP kernels on ROCm.
    """

    def __init__(
        self,
        docker_image: str = "rocm/dev-ubuntu-22.04:6.0",
        memory_limit: str = "8g",
        cpu_limit: float = 4.0,
        gpu_device_ids: Optional[List[int]] = None,
        enable_network: bool = False
    ):
        """
        Initialize Docker sandbox for ROCm.
        
        Args:
            docker_image: Docker image to use (ROCm development image)
            memory_limit: Memory limit for container
            cpu_limit: CPU limit for container
            gpu_device_ids: Specific GPU devices to expose (via render nodes)
            enable_network: Whether to enable network (default: False for safety)
        """
        self.docker_image = docker_image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.gpu_device_ids = gpu_device_ids or [0]
        self.enable_network = enable_network
        self.logger = structlog.get_logger()

        # Validate Docker availability
        self._validate_docker()

    def _validate_docker(self):
        """Validate Docker is available and properly configured."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("Docker is not running")
        except FileNotFoundError:
            raise RuntimeError("Docker is not installed")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Docker is not responding")

    async def execute_in_sandbox(
        self,
        code: str,
        timeout_seconds: int = 60,
        working_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute code in Docker sandbox with resource limits.
        
        Args:
            code: Code to execute
            timeout_seconds: Execution timeout
            working_dir: Working directory for execution
            
        Returns:
            Execution results including output and metrics
        """
        # Create temporary directory for code
        with tempfile.TemporaryDirectory(prefix="hip_sandbox_") as temp_dir:
            # Write code to file
            code_file = os.path.join(temp_dir, "kernel.cpp")
            with open(code_file, 'w') as f:
                f.write(code)

            # Build Docker command
            docker_cmd = self._build_docker_command(temp_dir, timeout_seconds)

            # Execute in sandbox
            start_time = time.time()

            try:
                process = await asyncio.create_subprocess_exec(
                    *docker_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_seconds + 5  # Grace period
                )

                execution_time = time.time() - start_time

                return {
                    "success": process.returncode == 0,
                    "stdout": stdout.decode('utf-8') if stdout else "",
                    "stderr": stderr.decode('utf-8') if stderr else "",
                    "return_code": process.returncode,
                    "execution_time": execution_time,
                    "timeout": False
                }

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "Execution timeout",
                    "return_code": -1,
                    "execution_time": timeout_seconds,
                    "timeout": True
                }

            except Exception as e:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": str(e),
                    "return_code": -1,
                    "execution_time": time.time() - start_time,
                    "timeout": False
                }

    def _build_docker_command(
        self,
        mount_dir: str,
        timeout_seconds: int
    ) -> List[str]:
        """Build Docker command with security constraints for ROCm."""

        cmd = [
            "docker", "run",
            "--rm",  # Remove container after execution
            f"--memory={self.memory_limit}",
            f"--cpus={self.cpu_limit}",
            "--read-only",  # Read-only root filesystem
            "--security-opt=no-new-privileges",  # No privilege escalation
            f"--volume={mount_dir}:/workspace:ro",  # Mount as read-only
            "--workdir=/workspace",
            "--user=nobody",  # Run as unprivileged user
            f"--stop-timeout={timeout_seconds}",
            # ROCm-specific device access
            "--device=/dev/kfd",  # Kernel Fusion Driver
            "--device=/dev/dri",  # Direct Rendering Infrastructure
            "--group-add=video",  # Video group for GPU access
        ]

        # Network isolation
        if not self.enable_network:
            cmd.append("--network=none")

        # Add container image and command
        cmd.extend([
            self.docker_image,
            "timeout", str(timeout_seconds),
            "hipcc", "-o", "/tmp/kernel", "kernel.cpp"
        ])

        return cmd


class SafetyWrapper:
    """
    Main safety wrapper combining validation and sandboxed execution for HIP/ROCm.
    """

    def __init__(
        self,
        enable_validation: bool = True,
        enable_sandbox: bool = True,
        max_risk_score: float = 0.5,
        docker_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize safety wrapper.
        
        Args:
            enable_validation: Whether to validate code before execution
            enable_sandbox: Whether to use Docker sandbox
            max_risk_score: Maximum acceptable risk score
            docker_config: Docker sandbox configuration
        """
        self.enable_validation = enable_validation
        self.enable_sandbox = enable_sandbox
        self.max_risk_score = max_risk_score

        self.validator = SecurityValidator() if enable_validation else None
        self.sandbox = DockerSandbox(**docker_config) if enable_sandbox and docker_config else None

        self.logger = structlog.get_logger()

    async def safe_compile(
        self,
        code: str,
        kernel_name: str = "kernel",
        timeout_seconds: int = 60
    ) -> Dict[str, Any]:
        """
        Safely compile HIP code with validation and sandboxing.
        
        Args:
            code: HIP kernel code
            kernel_name: Name of the kernel
            timeout_seconds: Compilation timeout
            
        Returns:
            Compilation results with safety information
        """
        result = {
            "success": False,
            "kernel_name": kernel_name,
            "safety_checks": {},
            "compilation_result": None
        }

        # Step 1: Validate code
        if self.enable_validation:
            validation_result = self.validator.validate_code(code)
            result["safety_checks"]["validation"] = {
                "is_safe": validation_result.is_safe,
                "risk_score": validation_result.risk_score,
                "violations": len(validation_result.violations)
            }

            if not validation_result.is_safe:
                if validation_result.risk_score > self.max_risk_score:
                    result["error"] = "Code failed safety validation"
                    result["violations"] = [
                        {
                            "type": v.violation_type,
                            "description": v.description,
                            "severity": v.severity
                        }
                        for v in validation_result.violations
                    ]

                    self.logger.warning(
                        "Code rejected due to safety violations",
                        kernel_name=kernel_name,
                        risk_score=validation_result.risk_score,
                        violations=len(validation_result.violations)
                    )

                    return result

                # Use sanitized code if available
                if validation_result.sanitized_code:
                    code = validation_result.sanitized_code
                    result["safety_checks"]["sanitized"] = True

        # Step 2: Compile in sandbox
        if self.enable_sandbox and self.sandbox:
            try:
                compilation_result = await self.sandbox.execute_in_sandbox(
                    code,
                    timeout_seconds=timeout_seconds
                )

                result["compilation_result"] = compilation_result
                result["success"] = compilation_result["success"]

                self.logger.info(
                    "Safe HIP compilation completed",
                    kernel_name=kernel_name,
                    success=compilation_result["success"],
                    execution_time=compilation_result["execution_time"]
                )

            except Exception as e:
                result["error"] = f"Sandbox execution failed: {str(e)}"
                self.logger.error(
                    "Sandbox execution failed",
                    kernel_name=kernel_name,
                    error=str(e)
                )
        else:
            # Fallback to direct compilation (not recommended for untrusted code)
            result["warning"] = "Sandbox disabled - executing without isolation"
            # Would integrate with existing HIPCompiler here

        return result

    def get_safety_report(self, code: str) -> Dict[str, Any]:
        """
        Generate a detailed safety report for code without executing it.
        
        Args:
            code: Code to analyze
            
        Returns:
            Detailed safety report
        """
        if not self.enable_validation:
            return {"error": "Validation not enabled"}

        validation_result = self.validator.validate_code(code)

        report = {
            "is_safe": validation_result.is_safe,
            "risk_score": validation_result.risk_score,
            "risk_level": self._get_risk_level(validation_result.risk_score),
            "total_violations": len(validation_result.violations),
            "violations_by_severity": {},
            "detailed_violations": [],
            "can_be_sanitized": validation_result.sanitized_code is not None
        }

        # Count violations by severity
        for severity in ["critical", "high", "medium", "low"]:
            count = sum(1 for v in validation_result.violations if v.severity == severity)
            if count > 0:
                report["violations_by_severity"][severity] = count

        # Add detailed violations
        for violation in validation_result.violations:
            report["detailed_violations"].append({
                "type": violation.violation_type,
                "description": violation.description,
                "severity": violation.severity,
                "line_number": violation.line_number,
                "code_snippet": violation.code_snippet
            })

        return report

    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level."""
        if risk_score < 0.2:
            return "low"
        elif risk_score < 0.5:
            return "medium"
        elif risk_score < 0.8:
            return "high"
        else:
            return "critical"

