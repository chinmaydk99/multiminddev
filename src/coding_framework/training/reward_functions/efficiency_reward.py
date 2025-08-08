"""
Efficiency reward function for code performance evaluation.

Evaluates generated code based on execution time, memory usage,
and algorithmic efficiency characteristics.
"""

import asyncio
import time
import tracemalloc
from typing import Any, Optional

from .base_reward import BaseRewardFunction


class EfficiencyReward(BaseRewardFunction):
    """
    Code efficiency reward function.

    Evaluates generated code performance based on execution time,
    memory usage, and algorithmic patterns. Higher rewards for
    more efficient solutions.
    """

    def __init__(
        self,
        weight: float = 1.0,
        max_execution_time: float = 5.0,
        max_memory_mb: float = 100.0,
        time_weight: float = 0.6,
        memory_weight: float = 0.4,
    ):
        """
        Initialize efficiency reward function.

        Args:
            weight: Weight for this reward in composite calculations
            max_execution_time: Maximum acceptable execution time (seconds)
            max_memory_mb: Maximum acceptable memory usage (MB)
            time_weight: Weight for execution time in efficiency score
            memory_weight: Weight for memory usage in efficiency score
        """
        super().__init__(weight=weight, normalize_range=(-1.0, 1.0))

        self.max_execution_time = max_execution_time
        self.max_memory_mb = max_memory_mb
        self.time_weight = time_weight
        self.memory_weight = memory_weight

        # Validate weights sum to 1.0
        if abs(time_weight + memory_weight - 1.0) > 1e-6:
            raise ValueError("Time and memory weights must sum to 1.0")

        # Efficiency metrics tracking
        self.total_execution_time = 0.0
        self.total_memory_usage = 0.0
        self.timeout_violations = 0
        self.memory_violations = 0
        self.measurements_count = 0

        self.logger.info(
            "Efficiency reward function initialized",
            max_execution_time=max_execution_time,
            max_memory_mb=max_memory_mb,
            time_weight=time_weight,
            memory_weight=memory_weight,
        )

    async def _calculate_raw_reward(
        self,
        problem: str,
        generated_code: str,
        test_cases: list[dict[str, Any]],
        context: Optional[dict[str, Any]] = None,
    ) -> float:
        """
        Calculate raw efficiency reward based on performance metrics.

        Args:
            problem: Problem description
            generated_code: Generated code solution
            test_cases: Test cases for performance evaluation
            context: Additional context

        Returns:
            Efficiency score in range [0.0, 1.0]
        """
        if not generated_code or not generated_code.strip():
            return 0.0

        if not test_cases:
            # If no test cases, evaluate with synthetic input
            test_cases = [{"input": self._generate_synthetic_input(problem)}]

        try:
            # Measure performance across test cases
            time_scores = []
            memory_scores = []

            for test_case in test_cases:
                test_input = test_case.get("input")
                if test_input is None:
                    continue

                try:
                    exec_time, memory_usage = await self._measure_performance(
                        generated_code, test_input
                    )

                    # Calculate individual scores
                    time_score = self._calculate_time_score(exec_time)
                    memory_score = self._calculate_memory_score(memory_usage)

                    time_scores.append(time_score)
                    memory_scores.append(memory_score)

                    # Update statistics
                    self.total_execution_time += exec_time
                    self.total_memory_usage += memory_usage
                    self.measurements_count += 1

                    if exec_time > self.max_execution_time:
                        self.timeout_violations += 1
                    if memory_usage > self.max_memory_mb:
                        self.memory_violations += 1

                except asyncio.TimeoutError:
                    self.timeout_violations += 1
                    time_scores.append(0.0)  # Penalty for timeout
                    memory_scores.append(0.5)  # Neutral memory score

                except Exception as e:
                    self.logger.debug(f"Performance measurement failed: {e}")
                    time_scores.append(0.3)  # Low score for errors
                    memory_scores.append(0.3)

            if not time_scores or not memory_scores:
                return 0.0

            # Calculate weighted average efficiency score
            avg_time_score = sum(time_scores) / len(time_scores)
            avg_memory_score = sum(memory_scores) / len(memory_scores)

            efficiency_score = (
                self.time_weight * avg_time_score + self.memory_weight * avg_memory_score
            )

            self.logger.debug(
                "Efficiency evaluation completed",
                avg_time_score=avg_time_score,
                avg_memory_score=avg_memory_score,
                efficiency_score=efficiency_score,
            )

            return efficiency_score

        except Exception as e:
            self.logger.error(f"Efficiency evaluation failed: {e}")
            return 0.0

    async def _measure_performance(
        self,
        generated_code: str,
        test_input: Any,
    ) -> tuple[float, float]:
        """
        Measure execution time and memory usage of generated code.

        Args:
            generated_code: Code to measure
            test_input: Input for the code

        Returns:
            Tuple of (execution_time_seconds, memory_usage_mb)
        """
        # Start memory tracing
        tracemalloc.start()

        try:
            # Execute code with timing
            start_time = time.perf_counter()

            await asyncio.wait_for(
                self._execute_code(generated_code, test_input),
                timeout=self.max_execution_time * 2,  # Allow some overhead
            )

            end_time = time.perf_counter()
            execution_time = end_time - start_time

            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            memory_usage_mb = peak / (1024 * 1024)  # Convert to MB

            return execution_time, memory_usage_mb

        finally:
            tracemalloc.stop()

    async def _execute_code(self, code: str, test_input: Any) -> Any:
        """
        Execute code with given input.

        Args:
            code: Code to execute
            test_input: Input for the code

        Returns:
            Execution result
        """
        # Create execution environment
        exec_globals = {"__builtins__": __builtins__}
        exec_locals = {}

        try:
            # Validate code is not empty and has valid syntax
            if not code or not code.strip():
                raise RuntimeError("Empty code provided")
                
            # Check syntax before execution
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                raise RuntimeError(f"Code has invalid syntax: {e}")
            
            # Execute the code
            exec(code, exec_globals, exec_locals)

            # Find and call the main function
            main_func = None
            for name, obj in exec_locals.items():
                if callable(obj) and not name.startswith("_"):
                    main_func = obj
                    break

            if main_func is None:
                raise RuntimeError("No callable function found in code")

            # Call function with input
            if isinstance(test_input, (list, tuple)):
                return main_func(*test_input)
            else:
                return main_func(test_input)

        except Exception as e:
            raise RuntimeError(f"Code execution failed: {e}")

    def _calculate_time_score(self, execution_time: float) -> float:
        """
        Calculate time efficiency score.

        Args:
            execution_time: Measured execution time in seconds

        Returns:
            Time score in range [0.0, 1.0]
        """
        if execution_time <= 0.001:  # Very fast (< 1ms)
            return 1.0
        elif execution_time <= self.max_execution_time * 0.1:  # Fast
            return 0.9
        elif execution_time <= self.max_execution_time * 0.5:  # Acceptable
            return 0.7
        elif execution_time <= self.max_execution_time:  # Slow but acceptable
            return 0.4
        else:  # Too slow
            # Exponential penalty for exceeding limit
            excess_ratio = execution_time / self.max_execution_time
            return max(0.0, 0.4 / excess_ratio)

    def _calculate_memory_score(self, memory_usage: float) -> float:
        """
        Calculate memory efficiency score.

        Args:
            memory_usage: Memory usage in MB

        Returns:
            Memory score in range [0.0, 1.0]
        """
        if memory_usage <= 1.0:  # Very low memory (< 1MB)
            return 1.0
        elif memory_usage <= self.max_memory_mb * 0.1:  # Low memory
            return 0.9
        elif memory_usage <= self.max_memory_mb * 0.5:  # Acceptable
            return 0.7
        elif memory_usage <= self.max_memory_mb:  # High but acceptable
            return 0.4
        else:  # Too high
            # Linear penalty for exceeding limit
            excess_ratio = memory_usage / self.max_memory_mb
            return max(0.0, 0.4 / excess_ratio)

    def _generate_synthetic_input(self, problem: str) -> Any:
        """
        Generate synthetic input for performance testing.

        Args:
            problem: Problem description

        Returns:
            Synthetic input data
        """
        # Simple heuristics based on problem description
        problem_lower = problem.lower()

        if any(word in problem_lower for word in ["string", "text", "char"]):
            return "example_string_input"
        elif any(word in problem_lower for word in ["list", "array", "sequence"]):
            return [1, 2, 3, 4, 5]
        elif any(word in problem_lower for word in ["number", "integer", "int"]):
            return 42
        elif any(word in problem_lower for word in ["matrix", "2d"]):
            return [[1, 2], [3, 4]]
        else:
            # Default to a simple integer
            return 10

    def _normalize_reward(self, raw_reward: float) -> float:
        """
        Normalize efficiency reward to [-1, 1] range.

        Args:
            raw_reward: Raw reward in [0, 1] range

        Returns:
            Normalized reward in [-1, 1] range
        """
        # Map [0, 1] to [-1, 1] where 0.5 maps to 0
        return (raw_reward - 0.5) * 2.0

    async def _get_detailed_metrics(
        self,
        problem: str,
        generated_code: str,
        test_cases: list[dict[str, Any]],
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Get detailed efficiency metrics.

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

        avg_execution_time = (
            self.total_execution_time / self.measurements_count
            if self.measurements_count > 0
            else 0.0
        )
        avg_memory_usage = (
            self.total_memory_usage / self.measurements_count
            if self.measurements_count > 0
            else 0.0
        )

        base_metrics.update(
            {
                "max_execution_time": self.max_execution_time,
                "max_memory_mb": self.max_memory_mb,
                "time_weight": self.time_weight,
                "memory_weight": self.memory_weight,
                "avg_execution_time": avg_execution_time,
                "avg_memory_usage_mb": avg_memory_usage,
                "total_measurements": self.measurements_count,
                "timeout_violations": self.timeout_violations,
                "memory_violations": self.memory_violations,
            }
        )

        return base_metrics

    def get_statistics(self) -> dict[str, Any]:
        """
        Get efficiency reward statistics.

        Returns:
            Statistics dictionary with performance metrics
        """
        base_stats = super().get_statistics()

        avg_execution_time = (
            self.total_execution_time / self.measurements_count
            if self.measurements_count > 0
            else 0.0
        )
        avg_memory_usage = (
            self.total_memory_usage / self.measurements_count
            if self.measurements_count > 0
            else 0.0
        )

        base_stats.update(
            {
                "measurements_count": self.measurements_count,
                "avg_execution_time": avg_execution_time,
                "avg_memory_usage_mb": avg_memory_usage,
                "timeout_violations": self.timeout_violations,
                "memory_violations": self.memory_violations,
                "max_execution_time": self.max_execution_time,
                "max_memory_mb": self.max_memory_mb,
            }
        )

        return base_stats
