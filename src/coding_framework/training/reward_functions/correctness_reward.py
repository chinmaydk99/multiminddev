"""
Correctness reward function for code generation evaluation.

Provides binary reward based on test case execution: 1.0 if all tests pass, 0.0 otherwise.
"""

import asyncio
from typing import Any, Optional

from .base_reward import BaseRewardFunction


class CorrectnessReward(BaseRewardFunction):
    """
    Binary correctness reward function.

    Returns 1.0 if generated code passes all test cases, 0.0 otherwise.
    Uses safe code execution to evaluate test cases.
    """

    def __init__(self, weight: float = 1.0, timeout: float = 10.0, safe_execution: bool = True):
        """
        Initialize correctness reward function.

        Args:
            weight: Weight for this reward in composite calculations
            timeout: Maximum execution time per test case (seconds)
            safe_execution: Whether to use safe execution environment
        """
        super().__init__(weight=weight, normalize_range=(0.0, 1.0))

        self.timeout = timeout
        self.safe_execution = safe_execution

        # Execution statistics
        self.tests_passed = 0
        self.tests_failed = 0
        self.syntax_errors = 0
        self.runtime_errors = 0
        self.timeout_errors = 0

        self.logger.info(
            "Correctness reward function initialized",
            timeout=timeout,
            safe_execution=safe_execution,
        )

    async def _calculate_raw_reward(
        self,
        problem: str,
        generated_code: str,
        test_cases: list[dict[str, Any]],
        context: Optional[dict[str, Any]] = None,
    ) -> float:
        """
        Calculate raw correctness reward by executing test cases.

        Args:
            problem: Problem description
            generated_code: Generated code solution
            test_cases: List of test cases with 'input' and 'expected_output' fields
            context: Additional context (may contain executor_agent)

        Returns:
            1.0 if all tests pass, 0.0 otherwise
        """
        if not generated_code or not generated_code.strip():
            self.logger.debug("Empty generated code")
            return 0.0

        if not test_cases:
            self.logger.warning("No test cases provided")
            return 0.0

        # Try using CodeExecutorAgent if available in context
        if context and "executor_agent" in context and context["executor_agent"] is not None:
            try:
                return await self._evaluate_with_executor_agent(
                    generated_code, test_cases, context["executor_agent"]
                )
            except Exception as e:
                self.logger.debug(f"Executor agent failed, falling back to direct evaluation: {e}")

        # Fall back to direct evaluation
        return await self._evaluate_directly(generated_code, test_cases)

    async def _evaluate_with_executor_agent(
        self,
        generated_code: str,
        test_cases: list[dict[str, Any]],
        executor_agent: Any,
    ) -> float:
        """
        Evaluate code using CodeExecutorAgent if available.

        Args:
            generated_code: Code to evaluate
            test_cases: Test cases to run
            executor_agent: CodeExecutorAgent instance

        Returns:
            1.0 if all tests pass, 0.0 otherwise
        """
        try:
            # Use the executor agent to run the code with test cases
            execution_result = await executor_agent.process_request(
                generated_code,
                context={
                    "test_cases": test_cases,
                    "timeout": self.timeout,
                    "safe_execution": self.safe_execution,
                },
            )

            if execution_result.success:
                all_passed = execution_result.metadata.get("all_tests_passed", False)
                passed_count = execution_result.metadata.get("tests_passed", 0)
                failed_count = execution_result.metadata.get("tests_failed", 0)

                self.tests_passed += passed_count
                self.tests_failed += failed_count

                return 1.0 if all_passed else 0.0
            else:
                self.runtime_errors += 1
                return 0.0

        except Exception as e:
            self.logger.warning(f"Executor agent evaluation failed: {e}")
            # Fall back to direct evaluation
            return await self._evaluate_directly(generated_code, test_cases)

    async def _evaluate_directly(
        self,
        generated_code: str,
        test_cases: list[dict[str, Any]],
    ) -> float:
        """
        Evaluate code directly by executing test cases.

        Args:
            generated_code: Code to evaluate
            test_cases: Test cases to run

        Returns:
            1.0 if all tests pass, 0.0 otherwise
        """
        try:
            # First check for basic syntax errors
            if not self._check_syntax(generated_code):
                self.syntax_errors += 1
                return 0.0

            # Execute test cases
            passed_count = 0
            failed_count = 0

            for i, test_case in enumerate(test_cases):
                try:
                    result = await self._execute_test_case(generated_code, test_case)
                    if result:
                        passed_count += 1
                    else:
                        failed_count += 1

                except asyncio.TimeoutError:
                    self.timeout_errors += 1
                    failed_count += 1

                except Exception as e:
                    self.logger.debug(f"Test case {i} failed with error: {e}")
                    self.runtime_errors += 1
                    failed_count += 1

            # Update statistics
            self.tests_passed += passed_count
            self.tests_failed += failed_count

            # Return 1.0 if all tests passed, 0.0 otherwise
            all_passed = failed_count == 0 and passed_count > 0

            self.logger.debug(
                "Direct evaluation completed",
                passed=passed_count,
                failed=failed_count,
                all_passed=all_passed,
            )

            return 1.0 if all_passed else 0.0

        except Exception as e:
            self.logger.error(f"Direct evaluation failed: {e}")
            self.runtime_errors += 1
            return 0.0

    def _check_syntax(self, code: str) -> bool:
        """
        Check if code has valid Python syntax.

        Args:
            code: Code to check

        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            import ast

            ast.parse(code)
            return True
        except SyntaxError:
            return False
        except Exception:
            return False

    async def _execute_test_case(
        self,
        generated_code: str,
        test_case: dict[str, Any],
    ) -> bool:
        """
        Execute a single test case.

        Args:
            generated_code: Code to execute
            test_case: Test case with 'input' and 'expected_output'

        Returns:
            True if test passes, False otherwise
        """
        try:
            # Extract test case data
            test_input = test_case.get("input")
            expected_output = test_case.get("expected_output")

            if test_input is None or expected_output is None:
                return False

            # Execute with timeout
            result = await asyncio.wait_for(
                self._safe_execute(generated_code, test_input), timeout=self.timeout
            )

            # Compare result with expected output
            return self._compare_outputs(result, expected_output)

        except asyncio.TimeoutError:
            raise
        except Exception as e:
            self.logger.debug(f"Test execution error: {e}")
            return False

    async def _safe_execute(self, code: str, test_input: Any) -> Any:
        """
        Safely execute code with given input.

        Args:
            code: Code to execute
            test_input: Input for the code

        Returns:
            Execution result
        """
        # Create a restricted execution environment
        safe_globals = {
            "__builtins__": {
                "abs": abs,
                "all": all,
                "any": any,
                "bool": bool,
                "dict": dict,
                "enumerate": enumerate,
                "filter": filter,
                "float": float,
                "int": int,
                "len": len,
                "list": list,
                "map": map,
                "max": max,
                "min": min,
                "range": range,
                "reversed": reversed,
                "sorted": sorted,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "zip": zip,
            }
        }
        safe_locals = {}

        try:
            # Execute the code
            exec(code, safe_globals, safe_locals)

            # Try to find and call the main function
            # Common patterns: main function, solve function, or first function defined
            main_func = None

            for name, obj in safe_locals.items():
                if callable(obj) and not name.startswith("_"):
                    main_func = obj
                    break

            if main_func is None:
                self.logger.debug(f"No functions found in code. Available: {list(safe_locals.keys())}")
                # If no function found, try to evaluate the code directly
                return eval(code, safe_globals, safe_locals)

            # Call the function with test input
            if isinstance(test_input, (list, tuple)):
                result = main_func(*test_input)
            else:
                result = main_func(test_input)
            
            return result

        except Exception as e:
            self.logger.debug(f"Code execution failed: {e}")
            self.logger.debug(f"Code: {repr(code[:200])}...")
            raise RuntimeError(f"Execution error: {e}")

    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        """
        Compare actual output with expected output.

        Args:
            actual: Actual output from code execution
            expected: Expected output

        Returns:
            True if outputs match, False otherwise
        """
        try:
            # Handle different types of comparisons
            if type(actual) != type(expected):
                # Try converting to same type
                if isinstance(expected, str) and not isinstance(actual, str):
                    actual = str(actual)
                elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                    # Allow int/float comparison
                    pass
                else:
                    return False

            # Special handling for floating point comparison
            if isinstance(expected, float) and isinstance(actual, float):
                return abs(actual - expected) < 1e-6

            # Direct comparison
            return actual == expected

        except Exception:
            return False

    async def _get_detailed_metrics(
        self,
        problem: str,
        generated_code: str,
        test_cases: list[dict[str, Any]],
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Get detailed correctness metrics.

        Args:
            problem: Problem description
            generated_code: Generated code
            test_cases: Test cases
            context: Additional context

        Returns:
            Detailed metrics dictionary
        """
        base_metrics = await super()._get_detailed_metrics(
            problem, generated_code, test_cases, context
        )

        base_metrics.update(
            {
                "has_valid_syntax": self._check_syntax(generated_code),
                "timeout_seconds": self.timeout,
                "safe_execution": self.safe_execution,
                "total_tests_passed": self.tests_passed,
                "total_tests_failed": self.tests_failed,
                "total_syntax_errors": self.syntax_errors,
                "total_runtime_errors": self.runtime_errors,
                "total_timeout_errors": self.timeout_errors,
            }
        )

        return base_metrics

    def get_statistics(self) -> dict[str, Any]:
        """
        Get correctness reward statistics.

        Returns:
            Statistics dictionary with execution metrics
        """
        base_stats = super().get_statistics()

        total_tests = self.tests_passed + self.tests_failed
        pass_rate = (self.tests_passed / total_tests) if total_tests > 0 else 0.0

        base_stats.update(
            {
                "tests_passed": self.tests_passed,
                "tests_failed": self.tests_failed,
                "pass_rate": pass_rate,
                "syntax_errors": self.syntax_errors,
                "runtime_errors": self.runtime_errors,
                "timeout_errors": self.timeout_errors,
                "timeout_seconds": self.timeout,
            }
        )

        return base_stats
