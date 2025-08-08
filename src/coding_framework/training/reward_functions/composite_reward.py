"""
Composite reward function combining multiple reward metrics.

Provides weighted combination of correctness, style, and efficiency rewards
for comprehensive code evaluation in reinforcement learning training.
"""

import asyncio
from typing import Any, Optional

from .base_reward import BaseRewardFunction
from .correctness_reward import CorrectnessReward
from .efficiency_reward import EfficiencyReward
from .style_reward import StyleReward


class CompositeReward(BaseRewardFunction):
    """
    Composite reward function combining multiple evaluation metrics.

    Combines correctness, style, and efficiency rewards with configurable
    weights. Default weighting: 0.7*correctness + 0.2*style + 0.1*efficiency
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        correctness_kwargs: Optional[dict[str, Any]] = None,
        style_kwargs: Optional[dict[str, Any]] = None,
        efficiency_kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize composite reward function.

        Args:
            weights: Reward weights dict with keys: correctness, style, efficiency
            correctness_kwargs: Arguments for CorrectnessReward initialization
            style_kwargs: Arguments for StyleReward initialization
            efficiency_kwargs: Arguments for EfficiencyReward initialization
        """
        # Default weights
        default_weights = {"correctness": 0.7, "style": 0.2, "efficiency": 0.1}
        self.weights = weights or default_weights

        # Validate weights
        self._validate_weights()

        super().__init__(weight=1.0, normalize_range=(-1.0, 1.0))

        # Initialize individual reward functions
        self.correctness_reward = CorrectnessReward(
            weight=1.0,  # We'll apply weights in composite calculation
            **(correctness_kwargs or {}),
        )

        self.style_reward = StyleReward(weight=1.0, **(style_kwargs or {}))

        self.efficiency_reward = EfficiencyReward(weight=1.0, **(efficiency_kwargs or {}))

        # Composite metrics tracking
        self.component_scores = {"correctness": [], "style": [], "efficiency": []}

        self.logger.info(
            "Composite reward function initialized",
            weights=self.weights,
            correctness_kwargs=correctness_kwargs or {},
            style_kwargs=style_kwargs or {},
            efficiency_kwargs=efficiency_kwargs or {},
        )

    def _validate_weights(self) -> None:
        """Validate weight configuration."""
        required_keys = {"correctness", "style", "efficiency"}

        if not isinstance(self.weights, dict):
            raise TypeError("Weights must be a dictionary")

        if set(self.weights.keys()) != required_keys:
            raise ValueError(f"Weights must contain exactly these keys: {required_keys}")

        # Check weight values
        for key, weight in self.weights.items():
            if not isinstance(weight, (int, float)) or weight < 0:
                raise ValueError(f"Weight '{key}' must be a non-negative number")

        # Check if weights sum to approximately 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            self.logger.warning(
                f"Weights sum to {total_weight}, not 1.0. "
                "This may lead to unexpected reward scaling."
            )

    async def _calculate_raw_reward(
        self,
        problem: str,
        generated_code: str,
        test_cases: list[dict[str, Any]],
        context: Optional[dict[str, Any]] = None,
    ) -> float:
        """
        Calculate composite reward by combining individual metrics.

        Args:
            problem: Problem description
            generated_code: Generated code solution
            test_cases: Test cases for evaluation
            context: Additional context (agents, etc.)

        Returns:
            Composite reward score
        """
        if not generated_code or not generated_code.strip():
            return 0.0

        try:
            # Calculate individual rewards in parallel
            tasks = []

            # Correctness reward
            tasks.append(
                self.correctness_reward.calculate_reward(
                    problem, generated_code, test_cases, context
                )
            )

            # Style reward
            tasks.append(
                self.style_reward.calculate_reward(problem, generated_code, test_cases, context)
            )

            # Efficiency reward
            tasks.append(
                self.efficiency_reward.calculate_reward(
                    problem, generated_code, test_cases, context
                )
            )

            # Execute all reward calculations concurrently
            correctness_score, style_score, efficiency_score = await asyncio.gather(*tasks)

            # Store component scores for analysis
            self.component_scores["correctness"].append(correctness_score)
            self.component_scores["style"].append(style_score)
            self.component_scores["efficiency"].append(efficiency_score)

            # Keep limited history
            for component in self.component_scores.values():
                if len(component) > 1000:
                    component[:] = component[-500:]  # Keep last 500

            # Calculate weighted composite score
            composite_score = (
                self.weights["correctness"] * correctness_score
                + self.weights["style"] * style_score
                + self.weights["efficiency"] * efficiency_score
            )

            self.logger.debug(
                "Composite reward calculated",
                correctness=correctness_score,
                style=style_score,
                efficiency=efficiency_score,
                composite=composite_score,
                weights=self.weights,
            )

            return composite_score

        except Exception as e:
            self.logger.error(f"Composite reward calculation failed: {e}")
            return 0.0

    async def calculate_detailed_reward(
        self,
        problem: str,
        generated_code: str,
        test_cases: list[dict[str, Any]],
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Calculate composite reward with detailed breakdown.

        Args:
            problem: Problem description
            generated_code: Generated code solution
            test_cases: Test cases for evaluation
            context: Additional context

        Returns:
            Detailed reward breakdown with individual component metrics
        """
        if not generated_code or not generated_code.strip():
            return {
                "composite_reward": 0.0,
                "correctness": {"reward": 0.0, "success": False},
                "style": {"reward": 0.0, "success": False},
                "efficiency": {"reward": 0.0, "success": False},
            }

        try:
            # Calculate detailed metrics for each component
            tasks = [
                self.correctness_reward.calculate_reward_with_metrics(
                    problem, generated_code, test_cases, context
                ),
                self.style_reward.calculate_reward_with_metrics(
                    problem, generated_code, test_cases, context
                ),
                self.efficiency_reward.calculate_reward_with_metrics(
                    problem, generated_code, test_cases, context
                ),
            ]

            correctness_metrics, style_metrics, efficiency_metrics = await asyncio.gather(*tasks)

            # Extract reward scores
            correctness_score = correctness_metrics.reward
            style_score = style_metrics.reward
            efficiency_score = efficiency_metrics.reward

            # Calculate composite score
            composite_score = (
                self.weights["correctness"] * correctness_score
                + self.weights["style"] * style_score
                + self.weights["efficiency"] * efficiency_score
            )

            # Store component scores
            self.component_scores["correctness"].append(correctness_score)
            self.component_scores["style"].append(style_score)
            self.component_scores["efficiency"].append(efficiency_score)

            return {
                "composite_reward": composite_score,
                "weights": self.weights.copy(),
                "correctness": {
                    "reward": correctness_score,
                    "weight": self.weights["correctness"],
                    "weighted_reward": self.weights["correctness"] * correctness_score,
                    "success": correctness_metrics.success,
                    "execution_time": correctness_metrics.execution_time,
                    "details": correctness_metrics.details,
                },
                "style": {
                    "reward": style_score,
                    "weight": self.weights["style"],
                    "weighted_reward": self.weights["style"] * style_score,
                    "success": style_metrics.success,
                    "execution_time": style_metrics.execution_time,
                    "details": style_metrics.details,
                },
                "efficiency": {
                    "reward": efficiency_score,
                    "weight": self.weights["efficiency"],
                    "weighted_reward": self.weights["efficiency"] * efficiency_score,
                    "success": efficiency_metrics.success,
                    "execution_time": efficiency_metrics.execution_time,
                    "details": efficiency_metrics.details,
                },
            }

        except Exception as e:
            self.logger.error(f"Detailed composite reward calculation failed: {e}")
            return {
                "composite_reward": 0.0,
                "error": str(e),
                "correctness": {"reward": 0.0, "success": False},
                "style": {"reward": 0.0, "success": False},
                "efficiency": {"reward": 0.0, "success": False},
            }

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """
        Update reward weights during training.

        Args:
            new_weights: New weight configuration
        """
        old_weights = self.weights.copy()
        self.weights = new_weights

        try:
            self._validate_weights()
            self.logger.info(
                "Reward weights updated",
                old_weights=old_weights,
                new_weights=new_weights,
            )
        except (TypeError, ValueError) as e:
            # Revert to old weights on validation error
            self.weights = old_weights
            self.logger.error(f"Invalid weights, reverting to previous: {e}")
            raise

    async def _get_detailed_metrics(
        self,
        problem: str,
        generated_code: str,
        test_cases: list[dict[str, Any]],
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Get detailed composite metrics.

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

        # Calculate component statistics
        component_stats = {}
        for component, scores in self.component_scores.items():
            if scores:
                component_stats[f"{component}_avg"] = sum(scores) / len(scores)
                component_stats[f"{component}_min"] = min(scores)
                component_stats[f"{component}_max"] = max(scores)
                component_stats[f"{component}_count"] = len(scores)
            else:
                component_stats[f"{component}_avg"] = 0.0
                component_stats[f"{component}_min"] = 0.0
                component_stats[f"{component}_max"] = 0.0
                component_stats[f"{component}_count"] = 0

        base_metrics.update(
            {
                "weights": self.weights.copy(),
                "component_statistics": component_stats,
            }
        )

        return base_metrics

    def get_component_statistics(self) -> dict[str, dict[str, Any]]:
        """
        Get statistics for individual reward components.

        Returns:
            Statistics for each reward component
        """
        stats = {}

        for component in ["correctness", "style", "efficiency"]:
            component_reward = getattr(self, f"{component}_reward")
            stats[component] = component_reward.get_statistics()

        # Add composite-specific stats
        stats["composite"] = {
            "weights": self.weights.copy(),
            "total_evaluations": self.calculation_count,
        }

        # Add component score distributions
        for component, scores in self.component_scores.items():
            if scores:
                stats[component].update(
                    {
                        "score_avg": sum(scores) / len(scores),
                        "score_min": min(scores),
                        "score_max": max(scores),
                        "score_std": self._calculate_std(scores),
                    }
                )

        return stats

    def _calculate_std(self, values: list[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance**0.5

    def get_statistics(self) -> dict[str, Any]:
        """
        Get composite reward statistics.

        Returns:
            Statistics dictionary with component breakdowns
        """
        base_stats = super().get_statistics()

        # Add component-specific statistics
        component_stats = self.get_component_statistics()
        base_stats["components"] = component_stats
        base_stats["weights"] = self.weights.copy()

        return base_stats
