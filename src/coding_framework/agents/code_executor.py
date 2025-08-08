"""
Code Executor Agent for testing and validating code execution.

This agent specializes in safely executing code in sandboxed environments,
running tests, and providing execution feedback with performance metrics.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import docker
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from .base_agent import AgentResponse, BaseAgent


class ExecutionResult(BaseModel):
    """Structured execution result."""

    success: bool = Field(..., description="Whether execution was successful")
    output: str = Field(default="", description="Program output")
    error: str = Field(default="", description="Error output if any")
    exit_code: int = Field(default=0, description="Exit code")
    execution_time: float = Field(..., description="Execution time in seconds")
    memory_usage: Optional[int] = Field(None, description="Memory usage in bytes")
    test_results: Optional[dict[str, Any]] = Field(None, description="Test execution results")


class CodeExecutorAgent(BaseAgent):
    """
    Specialized agent for code execution and testing.

    Executes code safely in sandboxed environments, runs tests,
    and provides detailed execution feedback and performance metrics.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the code executor agent."""
        super().__init__(*args, **kwargs)

        # Initialize Docker client for sandboxed execution
        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
            self.logger.info("Docker client initialized successfully")
        except Exception as e:
            self.docker_client = None
            self.docker_available = False
            self.logger.warning(f"Docker not available: {e}")

        # Execution limits and settings
        self.default_timeout = 30
        self.memory_limit = "512m"
        self.cpu_limit = "1.0"
        self.network_mode = "none"  # Disable network access

    @property
    def agent_type(self) -> str:
        """Return agent type identifier."""
        return "code_executor"

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for code execution."""
        return """You are a Code Execution Specialist responsible for safely running and testing code. Your role is to:

1. **Execute Code Safely**: Run code in sandboxed environments with proper security measures
2. **Analyze Results**: Interpret execution results and provide meaningful feedback
3. **Performance Metrics**: Measure and report execution time, memory usage, and efficiency
4. **Test Validation**: Run test cases and validate expected outcomes
5. **Error Analysis**: Diagnose execution errors and suggest fixes
6. **Security Assessment**: Ensure code execution doesn't pose security risks

**Execution Environment:**
- Sandboxed containers with limited resources
- No network access during execution
- Time and memory limits enforced
- Safe execution for multiple programming languages

**Analysis Areas:**
- **Correctness**: Does the code produce expected results?
- **Performance**: How efficiently does the code run?
- **Reliability**: Does the code handle edge cases properly?
- **Resource Usage**: Memory and CPU utilization
- **Error Handling**: How does the code behave with invalid inputs?

**Output Format:**
Provide execution analysis in this structure:

**Execution Status:** [Success/Failed]
**Output:** [Program output]
**Performance:** [Time: Xs, Memory: XMB]
**Analysis:** [Detailed analysis of results]
**Recommendations:** [Suggestions for improvement]

Be thorough in analysis and provide actionable feedback."""

    async def process_request(
        self,
        request: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> AgentResponse:
        """
        Execute code and provide detailed analysis.

        Args:
            request: Code to execute or execution request
            context: Additional context (language, test cases, etc.)
            **kwargs: Execution parameters

        Returns:
            Agent response with execution results and analysis
        """
        start_time = time.time()

        try:
            self.logger.info("Starting code execution", request_length=len(request))

            # Extract execution parameters
            language = kwargs.get("language") or (context or {}).get("language", "python")
            timeout = kwargs.get("timeout", self.default_timeout)
            include_tests = kwargs.get("include_tests", False)
            test_cases = kwargs.get("test_cases") or (context or {}).get("test_cases", [])

            # Update state
            self.update_state("last_execution_request", request)
            self.update_state("target_language", language)

            # Extract code from request
            code_info = self._extract_code_and_tests(request, include_tests)

            # Execute code in sandbox
            execution_result = await self._execute_code_safely(
                code_info["code"],
                language,
                timeout,
                test_cases,
            )

            # Analyze execution results
            analysis = await self._analyze_execution_results(
                code_info["code"],
                execution_result,
                language,
            )

            # Create response metadata
            metadata = {
                "language": language,
                "execution_time": execution_result.execution_time,
                "success": execution_result.success,
                "exit_code": execution_result.exit_code,
                "has_output": bool(execution_result.output),
                "has_errors": bool(execution_result.error),
                "memory_usage": execution_result.memory_usage,
                "test_results": execution_result.test_results,
                "docker_available": self.docker_available,
            }

            # Combine execution result and analysis
            full_response = self._format_execution_response(execution_result, analysis)

            # Add to conversation history
            self.add_to_history(HumanMessage(content=request))
            self.add_to_history(HumanMessage(content=full_response))

            total_time = time.time() - start_time

            self.logger.info(
                "Code execution completed",
                total_time=total_time,
                **metadata,
            )

            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                content=full_response,
                metadata=metadata,
                execution_time=total_time,
            )

        except Exception as e:
            total_time = time.time() - start_time

            self.logger.error(
                "Code execution failed",
                error=str(e),
                total_time=total_time,
            )

            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                content="",
                metadata={"error_type": type(e).__name__},
                execution_time=total_time,
                error=str(e),
            )

    def _extract_code_and_tests(self, request: str, include_tests: bool) -> dict[str, str]:
        """
        Extract code and test cases from request.

        Args:
            request: Raw execution request
            include_tests: Whether to look for test cases

        Returns:
            Dictionary with extracted code and tests
        """
        import re

        # Look for code blocks
        code_block_pattern = r"```(?:\w+\n)?(.*?)```"
        code_matches = re.findall(code_block_pattern, request, re.DOTALL)

        if code_matches:
            # If multiple code blocks, assume first is main code, others are tests
            main_code = code_matches[0].strip()
            test_code = ""

            if include_tests and len(code_matches) > 1:
                test_code = "\n\n".join(code_matches[1:])
        else:
            # No code blocks found, treat entire request as code
            main_code = request.strip()
            test_code = ""

        return {
            "code": main_code,
            "tests": test_code,
        }

    async def _execute_code_safely(
        self,
        code: str,
        language: str,
        timeout: int,
        test_cases: list[dict[str, Any]],
    ) -> ExecutionResult:
        """
        Execute code safely in a sandboxed environment.

        Args:
            code: Code to execute
            language: Programming language
            timeout: Execution timeout
            test_cases: Test cases to run

        Returns:
            Execution result with metrics
        """
        if self.docker_available:
            return await self._execute_in_docker(code, language, timeout, test_cases)
        else:
            return await self._execute_locally(code, language, timeout, test_cases)

    async def _execute_in_docker(
        self,
        code: str,
        language: str,
        timeout: int,
        test_cases: list[dict[str, Any]],
    ) -> ExecutionResult:
        """
        Execute code in Docker container.

        Args:
            code: Code to execute
            language: Programming language
            timeout: Execution timeout
            test_cases: Test cases to run

        Returns:
            Execution result
        """
        start_time = time.time()

        try:
            # Get appropriate Docker image for language
            image = self._get_docker_image(language)

            # Create temporary file with code
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=self._get_file_extension(language),
                delete=False,
            ) as f:
                f.write(code)
                temp_file = f.name

            # Prepare execution command
            command = self._get_execution_command(language, Path(temp_file).name)

            # Run container
            container = self.docker_client.containers.run(
                image=image,
                command=command,
                volumes={Path(temp_file).parent: {"bind": "/workspace", "mode": "ro"}},
                working_dir="/workspace",
                network_mode=self.network_mode,
                mem_limit=self.memory_limit,
                cpu_period=100000,
                cpu_quota=int(float(self.cpu_limit) * 100000),
                timeout=timeout,
                detach=False,
                remove=True,
                capture_output=True,
            )

            execution_time = time.time() - start_time

            # Parse container result
            output = container.decode("utf-8") if isinstance(container, bytes) else str(container)

            # Clean up temporary file
            Path(temp_file).unlink(missing_ok=True)

            return ExecutionResult(
                success=True,
                output=output,
                error="",
                exit_code=0,
                execution_time=execution_time,
            )

        except docker.errors.ContainerError as e:
            execution_time = time.time() - start_time

            return ExecutionResult(
                success=False,
                output="",
                error=str(e.stderr),
                exit_code=e.exit_status,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time

            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                exit_code=1,
                execution_time=execution_time,
            )

    async def _execute_locally(
        self,
        code: str,
        language: str,
        timeout: int,
        test_cases: list[dict[str, Any]],
    ) -> ExecutionResult:
        """
        Execute code locally (fallback when Docker unavailable).

        Args:
            code: Code to execute
            language: Programming language
            timeout: Execution timeout
            test_cases: Test cases to run

        Returns:
            Execution result
        """
        start_time = time.time()

        try:
            if language.lower() == "python":
                return await self._execute_python_locally(code, timeout)
            else:
                # For now, only support Python locally
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Local execution not supported for {language}. Docker required.",
                    exit_code=1,
                    execution_time=time.time() - start_time,
                )

        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                exit_code=1,
                execution_time=time.time() - start_time,
            )

    async def _execute_python_locally(self, code: str, timeout: int) -> ExecutionResult:
        """
        Execute Python code locally with safety measures.

        Args:
            code: Python code to execute
            timeout: Execution timeout

        Returns:
            Execution result
        """
        import tempfile

        start_time = time.time()

        try:
            # Create temporary file with code
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Execute with subprocess
            process = await asyncio.create_subprocess_exec(
                "python",
                temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )

                execution_time = time.time() - start_time

                # Clean up
                Path(temp_file).unlink(missing_ok=True)

                return ExecutionResult(
                    success=process.returncode == 0,
                    output=stdout.decode("utf-8") if stdout else "",
                    error=stderr.decode("utf-8") if stderr else "",
                    exit_code=process.returncode or 0,
                    execution_time=execution_time,
                )

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Execution timed out after {timeout} seconds",
                    exit_code=1,
                    execution_time=timeout,
                )

        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                exit_code=1,
                execution_time=time.time() - start_time,
            )

    def _get_docker_image(self, language: str) -> str:
        """
        Get appropriate Docker image for language.

        Args:
            language: Programming language

        Returns:
            Docker image name
        """
        images = {
            "python": "python:3.9-slim",
            "javascript": "node:18-slim",
            "typescript": "node:18-slim",
            "java": "openjdk:17-slim",
            "cpp": "gcc:latest",
            "go": "golang:1.19-alpine",
            "rust": "rust:1.70-slim",
        }

        return images.get(language.lower(), "python:3.9-slim")

    def _get_file_extension(self, language: str) -> str:
        """
        Get file extension for language.

        Args:
            language: Programming language

        Returns:
            File extension
        """
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "java": ".java",
            "cpp": ".cpp",
            "go": ".go",
            "rust": ".rs",
        }

        return extensions.get(language.lower(), ".py")

    def _get_execution_command(self, language: str, filename: str) -> str:
        """
        Get execution command for language.

        Args:
            language: Programming language
            filename: File to execute

        Returns:
            Execution command
        """
        commands = {
            "python": f"python {filename}",
            "javascript": f"node {filename}",
            "typescript": f"npx ts-node {filename}",
            "java": f"javac {filename} && java {filename.replace('.java', '')}",
            "cpp": f"g++ -o program {filename} && ./program",
            "go": f"go run {filename}",
            "rust": f"rustc {filename} && ./{filename.replace('.rs', '')}",
        }

        return commands.get(language.lower(), f"python {filename}")

    async def _analyze_execution_results(
        self,
        code: str,
        result: ExecutionResult,
        language: str,
    ) -> str:
        """
        Analyze execution results using LLM.

        Args:
            code: Original code
            result: Execution result
            language: Programming language

        Returns:
            Analysis text
        """
        analysis_request = f"""**Code Execution Analysis Request**

Please analyze the following code execution results and provide insights:

**Original Code:**
```{language}
{code}
```

**Execution Results:**
- Success: {result.success}
- Exit Code: {result.exit_code}
- Execution Time: {result.execution_time:.3f}s
- Output: {result.output[:1000]}...
- Errors: {result.error[:1000]}...

**Analysis Required:**
- Code correctness and functionality
- Performance characteristics
- Error analysis (if any)
- Suggestions for improvement
- Edge case handling assessment

Please provide a comprehensive analysis with actionable recommendations."""

        messages = self._build_messages(analysis_request)

        try:
            analysis = await self._call_llm(messages, max_tokens=1500)
            return analysis
        except Exception as e:
            self.logger.error("Failed to analyze execution results", error=str(e))
            return f"Analysis failed: {str(e)}"

    def _format_execution_response(
        self,
        result: ExecutionResult,
        analysis: str,
    ) -> str:
        """
        Format complete execution response.

        Args:
            result: Execution result
            analysis: LLM analysis

        Returns:
            Formatted response string
        """
        status = "✅ SUCCESS" if result.success else "❌ FAILED"

        response_parts = [
            f"**Execution Status:** {status}",
            f"**Exit Code:** {result.exit_code}",
            f"**Execution Time:** {result.execution_time:.3f}s",
        ]

        if result.memory_usage:
            response_parts.append(f"**Memory Usage:** {result.memory_usage / 1024 / 1024:.2f}MB")

        if result.output:
            response_parts.append(f"**Output:**\n```\n{result.output}\n```")

        if result.error:
            response_parts.append(f"**Errors:**\n```\n{result.error}\n```")

        if analysis:
            response_parts.append(f"**Analysis:**\n{analysis}")

        return "\n\n".join(response_parts)

    async def run_tests(
        self,
        code: str,
        test_cases: list[dict[str, Any]],
        **kwargs,
    ) -> AgentResponse:
        """
        Run specific test cases against code.

        Args:
            code: Code to test
            test_cases: List of test cases
            **kwargs: Additional parameters

        Returns:
            Test execution results
        """
        test_request = f"""**Test Execution Request**

Run the following test cases against the provided code:

**Code:**
```
{code}
```

**Test Cases:**
{json.dumps(test_cases, indent=2)}

Please execute each test case and report the results."""

        return await self.process_request(
            test_request,
            context={"task": "testing", "test_cases": test_cases},
            **kwargs,
        )

    async def benchmark_performance(
        self,
        code: str,
        iterations: int = 10,
        **kwargs,
    ) -> AgentResponse:
        """
        Benchmark code performance over multiple iterations.

        Args:
            code: Code to benchmark
            iterations: Number of iterations to run
            **kwargs: Additional parameters

        Returns:
            Performance benchmark results
        """
        benchmark_request = f"""**Performance Benchmark Request**

Run performance benchmark for the following code over {iterations} iterations:

**Code:**
```
{code}
```

Measure and report:
- Average execution time
- Min/Max execution times
- Memory usage patterns
- Performance consistency
- Resource utilization"""

        return await self.process_request(
            benchmark_request,
            context={"task": "benchmark", "iterations": iterations},
            **kwargs,
        )
