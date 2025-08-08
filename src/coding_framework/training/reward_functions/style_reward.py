"""
Style reward function for code quality evaluation.

Uses existing code analysis tools and CodeReviewerAgent to evaluate
code style, complexity, and adherence to best practices.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from .base_reward import BaseRewardFunction


class StyleReward(BaseRewardFunction):
    """
    Code style reward function.

    Evaluates generated code quality using static analysis tools
    and existing CodeReviewerAgent. Considers formatting, complexity,
    type hints, and adherence to Python best practices.
    """

    def __init__(
        self,
        weight: float = 1.0,
        use_reviewer_agent: bool = True,
        use_ruff: bool = True,
        use_mypy: bool = True,
        complexity_threshold: int = 10,
    ):
        """
        Initialize style reward function.

        Args:
            weight: Weight for this reward in composite calculations
            use_reviewer_agent: Whether to use CodeReviewerAgent
            use_ruff: Whether to use Ruff for linting
            use_mypy: Whether to use MyPy for type checking
            complexity_threshold: Maximum allowed complexity score
        """
        super().__init__(weight=weight, normalize_range=(-1.0, 1.0))

        self.use_reviewer_agent = use_reviewer_agent
        self.use_ruff = use_ruff
        self.use_mypy = use_mypy
        self.complexity_threshold = complexity_threshold

        # Style metrics tracking
        self.lint_issues = 0
        self.type_errors = 0
        self.complexity_violations = 0
        self.formatting_issues = 0

        self.logger.info(
            "Style reward function initialized",
            use_reviewer_agent=use_reviewer_agent,
            use_ruff=use_ruff,
            use_mypy=use_mypy,
            complexity_threshold=complexity_threshold,
        )

    async def _calculate_raw_reward(
        self,
        problem: str,
        generated_code: str,
        test_cases: list[dict[str, Any]],
        context: Optional[dict[str, Any]] = None,
    ) -> float:
        """
        Calculate raw style reward using multiple quality metrics.

        Args:
            problem: Problem description
            generated_code: Generated code solution
            test_cases: Test cases (not used for style evaluation)
            context: Additional context (may contain reviewer_agent)

        Returns:
            Style score in range [0.0, 1.0]
        """
        if not generated_code or not generated_code.strip():
            return 0.0

        # Initialize score components
        scores = []

        # Use CodeReviewerAgent if available
        if self.use_reviewer_agent and context and "reviewer_agent" in context:
            reviewer_score = await self._evaluate_with_reviewer_agent(
                generated_code, context["reviewer_agent"]
            )
            scores.append(("reviewer", reviewer_score, 0.4))  # 40% weight

        # Use Ruff for linting
        if self.use_ruff:
            ruff_score = await self._evaluate_with_ruff(generated_code)
            scores.append(("ruff", ruff_score, 0.3))  # 30% weight

        # Use MyPy for type checking
        if self.use_mypy:
            mypy_score = await self._evaluate_with_mypy(generated_code)
            scores.append(("mypy", mypy_score, 0.2))  # 20% weight

        # Complexity analysis
        complexity_score = await self._evaluate_complexity(generated_code)
        scores.append(("complexity", complexity_score, 0.1))  # 10% weight

        # Calculate weighted average
        if not scores:
            return 0.5  # Neutral score if no evaluators available

        total_score = 0.0
        total_weight = 0.0

        for _name, score, weight in scores:
            if score is not None:
                total_score += score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.5

        final_score = total_score / total_weight

        self.logger.debug(
            "Style evaluation completed",
            scores={name: score for name, score, _ in scores},
            final_score=final_score,
        )

        return final_score

    async def _evaluate_with_reviewer_agent(
        self,
        generated_code: str,
        reviewer_agent: Any,
    ) -> Optional[float]:
        """
        Evaluate code using CodeReviewerAgent.

        Args:
            generated_code: Code to evaluate
            reviewer_agent: CodeReviewerAgent instance

        Returns:
            Style score from reviewer or None if failed
        """
        try:
            review_result = await reviewer_agent.process_request(
                generated_code,
                context={
                    "focus_areas": ["style", "complexity", "readability", "best_practices"],
                    "severity_level": "info",  # Include all issues
                },
            )

            if review_result.success:
                # Extract style score from metadata
                metadata = review_result.metadata
                style_score = metadata.get("style_score", 50.0)  # Default to 50/100

                # Convert from 0-100 to 0-1 range
                normalized_score = style_score / 100.0

                self.logger.debug(
                    "Reviewer agent evaluation",
                    raw_score=style_score,
                    normalized_score=normalized_score,
                )

                return normalized_score

        except Exception as e:
            self.logger.warning(f"Reviewer agent evaluation failed: {e}")

        return None

    async def _evaluate_with_ruff(self, generated_code: str) -> Optional[float]:
        """
        Evaluate code using Ruff linter.

        Args:
            generated_code: Code to evaluate

        Returns:
            Ruff score or None if failed
        """
        try:
            # Create temporary file for analysis
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(generated_code)
                temp_file = Path(f.name)

            try:
                # Run Ruff check
                result = subprocess.run(
                    ["ruff", "check", str(temp_file), "--output-format=json"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                # Parse Ruff output
                import json

                if result.stdout:
                    issues = json.loads(result.stdout)
                else:
                    issues = []

                # Count different types of issues
                error_count = sum(1 for issue in issues if issue.get("level") == "error")
                warning_count = sum(1 for issue in issues if issue.get("level") == "warning")

                self.lint_issues += len(issues)

                # Calculate score (fewer issues = higher score)
                # Perfect score (1.0) for 0 issues, decreasing with more issues
                total_issues = len(issues)
                if total_issues == 0:
                    score = 1.0
                else:
                    # Penalty for errors is higher than warnings
                    penalty = error_count * 0.2 + warning_count * 0.1
                    score = max(0.0, 1.0 - penalty)

                self.logger.debug(
                    "Ruff evaluation",
                    total_issues=total_issues,
                    errors=error_count,
                    warnings=warning_count,
                    score=score,
                )

                return score

            finally:
                # Clean up temporary file
                temp_file.unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            self.logger.warning("Ruff evaluation timed out")
            return 0.5
        except Exception as e:
            self.logger.warning(f"Ruff evaluation failed: {e}")
            return None

    async def _evaluate_with_mypy(self, generated_code: str) -> Optional[float]:
        """
        Evaluate code using MyPy type checker.

        Args:
            generated_code: Code to evaluate

        Returns:
            MyPy score or None if failed
        """
        try:
            # Check if code has type hints
            has_type_hints = self._check_type_hints(generated_code)

            # Create temporary file for analysis
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(generated_code)
                temp_file = Path(f.name)

            try:
                # Run MyPy check
                result = subprocess.run(
                    ["mypy", str(temp_file), "--ignore-missing-imports"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                # Count type errors
                error_lines = [line for line in result.stdout.split("\n") if "error:" in line]
                type_error_count = len(error_lines)

                self.type_errors += type_error_count

                # Calculate score
                if not has_type_hints:
                    # Penalize lack of type hints
                    score = 0.3
                elif type_error_count == 0:
                    score = 1.0
                else:
                    # Decrease score based on number of errors
                    score = max(0.0, 1.0 - (type_error_count * 0.2))

                self.logger.debug(
                    "MyPy evaluation",
                    has_type_hints=has_type_hints,
                    type_errors=type_error_count,
                    score=score,
                )

                return score

            finally:
                # Clean up temporary file
                temp_file.unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            self.logger.warning("MyPy evaluation timed out")
            return 0.5
        except Exception as e:
            self.logger.warning(f"MyPy evaluation failed: {e}")
            return None

    def _check_type_hints(self, code: str) -> bool:
        """
        Check if code contains type hints.

        Args:
            code: Code to check

        Returns:
            True if code has type hints, False otherwise
        """
        # Simple heuristic: look for type annotation patterns
        type_patterns = ["->", ":", "List[", "Dict[", "Optional[", "Union[", "Tuple["]
        return any(pattern in code for pattern in type_patterns)

    async def _evaluate_complexity(self, generated_code: str) -> float:
        """
        Evaluate code complexity using cyclomatic complexity.

        Args:
            generated_code: Code to evaluate

        Returns:
            Complexity score
        """
        try:
            # Simple complexity analysis using AST
            import ast

            class ComplexityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.complexity = 1  # Start with 1 for the function itself

                def visit_If(self, node):
                    self.complexity += 1
                    self.generic_visit(node)

                def visit_For(self, node):
                    self.complexity += 1
                    self.generic_visit(node)

                def visit_While(self, node):
                    self.complexity += 1
                    self.generic_visit(node)

                def visit_ExceptHandler(self, node):
                    self.complexity += 1
                    self.generic_visit(node)

                def visit_With(self, node):
                    self.complexity += 1
                    self.generic_visit(node)

                def visit_BoolOp(self, node):
                    self.complexity += len(node.values) - 1
                    self.generic_visit(node)

            try:
                tree = ast.parse(generated_code)
                visitor = ComplexityVisitor()
                visitor.visit(tree)
                complexity = visitor.complexity
            except SyntaxError:
                # If syntax error, return low score
                return 0.2

            # Calculate score (lower complexity = higher score)
            if complexity <= self.complexity_threshold:
                score = 1.0
            else:
                # Penalty for high complexity
                excess = complexity - self.complexity_threshold
                score = max(0.0, 1.0 - (excess * 0.1))

            if complexity > self.complexity_threshold:
                self.complexity_violations += 1

            self.logger.debug(
                "Complexity evaluation",
                complexity=complexity,
                threshold=self.complexity_threshold,
                score=score,
            )

            return score

        except Exception as e:
            self.logger.warning(f"Complexity evaluation failed: {e}")
            return 0.5

    def _normalize_reward(self, raw_reward: float) -> float:
        """
        Normalize style reward to [-1, 1] range.

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
        Get detailed style metrics.

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
                "has_type_hints": self._check_type_hints(generated_code),
                "use_reviewer_agent": self.use_reviewer_agent,
                "use_ruff": self.use_ruff,
                "use_mypy": self.use_mypy,
                "complexity_threshold": self.complexity_threshold,
                "total_lint_issues": self.lint_issues,
                "total_type_errors": self.type_errors,
                "total_complexity_violations": self.complexity_violations,
            }
        )

        return base_metrics

    def get_statistics(self) -> dict[str, Any]:
        """
        Get style reward statistics.

        Returns:
            Statistics dictionary with style metrics
        """
        base_stats = super().get_statistics()

        base_stats.update(
            {
                "lint_issues": self.lint_issues,
                "type_errors": self.type_errors,
                "complexity_violations": self.complexity_violations,
                "complexity_threshold": self.complexity_threshold,
            }
        )

        return base_stats
