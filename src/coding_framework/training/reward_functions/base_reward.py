"""
Base reward function class for reinforcement learning training.

Provides abstract interface for all reward function implementations.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import structlog
from pydantic import BaseModel, Field


class RewardMetrics(BaseModel):
    """Metrics returned by reward function calculations."""

    reward: float = Field(..., description="Calculated reward in range [-1.0, 1.0]")
    execution_time: float = Field(..., description="Time taken to calculate reward")
    details: dict[str, Any] = Field(default_factory=dict, description="Detailed metrics")
    success: bool = Field(True, description="Whether reward calculation succeeded")
    error: Optional[str] = Field(None, description="Error message if calculation failed")


class BaseRewardFunction(ABC):
    """
    Abstract base class for all reward functions.

    Provides common functionality for reward calculation, normalization,
    and metrics collection for reinforcement learning training.
    """

    def __init__(self, weight: float = 1.0, normalize_range: tuple = (-1.0, 1.0)):
        """
        Initialize base reward function.

        Args:
            weight: Weight for this reward in composite calculations
            normalize_range: Target range for reward normalization
        """
        self.weight = weight
        self.normalize_range = normalize_range

        # Validation
        if not (0.0 < weight <= 1.0):
            raise ValueError("Weight must be between 0.0 and 1.0")
        if normalize_range[0] >= normalize_range[1]:
            raise ValueError("Invalid normalization range")

        self.logger = structlog.get_logger(
            component="reward_function",
            function_type=self.__class__.__name__,
        )

        # Metrics tracking
        self.calculation_count = 0
        self.total_execution_time = 0.0
        self.last_reward = 0.0
        self.reward_history: list[float] = []

        self.logger.info(
            "Reward function initialized",
            weight=weight,
            normalize_range=normalize_range,
        )

    @abstractmethod
    async def _calculate_raw_reward(
        self,
        problem: str,
        generated_code: str,
        test_cases: list[dict[str, Any]],
        context: Optional[dict[str, Any]] = None,
    ) -> float:
        """
        Calculate raw reward before normalization.

        Args:
            problem: Problem description
            generated_code: Generated code solution
            test_cases: Test cases for validation
            context: Additional context (agents, configuration, etc.)

        Returns:
            Raw reward value (will be normalized to target range)
        """
        pass

    async def calculate_reward(
        self,
        problem: str,
        generated_code: str,
        test_cases: list[dict[str, Any]],
        context: Optional[dict[str, Any]] = None,
    ) -> float:
        """
        Calculate normalized reward in target range.

        Args:
            problem: Problem description
            generated_code: Generated code solution
            test_cases: Test cases for validation
            context: Additional context (agents, configuration, etc.)

        Returns:
            Normalized reward in range [normalize_range[0], normalize_range[1]]
        """
        start_time = time.time()

        try:
            # Calculate raw reward
            raw_reward = await self._calculate_raw_reward(
                problem, generated_code, test_cases, context
            )

            # Normalize to target range
            normalized_reward = self._normalize_reward(raw_reward)

            # Apply weight
            weighted_reward = normalized_reward * self.weight

            # Update metrics
            execution_time = time.time() - start_time
            self._update_metrics(weighted_reward, execution_time)

            self.logger.debug(
                "Reward calculated",
                raw_reward=raw_reward,
                normalized_reward=normalized_reward,
                weighted_reward=weighted_reward,
                execution_time=execution_time,
            )

            return weighted_reward

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(
                "Reward calculation failed",
                error=str(e),
                execution_time=execution_time,
            )

            # Return neutral reward on error
            return 0.0

    async def calculate_reward_with_metrics(
        self,
        problem: str,
        generated_code: str,
        test_cases: list[dict[str, Any]],
        context: Optional[dict[str, Any]] = None,
    ) -> RewardMetrics:
        """
        Calculate reward with detailed metrics.

        Args:
            problem: Problem description
            generated_code: Generated code solution
            test_cases: Test cases for validation
            context: Additional context

        Returns:
            RewardMetrics with reward value and detailed information
        """
        start_time = time.time()

        try:
            # Calculate raw reward with detailed tracking
            raw_reward = await self._calculate_raw_reward(
                problem, generated_code, test_cases, context
            )

            # Normalize and weight
            normalized_reward = self._normalize_reward(raw_reward)
            weighted_reward = normalized_reward * self.weight

            execution_time = time.time() - start_time
            self._update_metrics(weighted_reward, execution_time)

            # Get detailed metrics
            details = await self._get_detailed_metrics(problem, generated_code, test_cases, context)
            details.update(
                {
                    "raw_reward": raw_reward,
                    "normalized_reward": normalized_reward,
                    "weight": self.weight,
                    "function_type": self.__class__.__name__,
                }
            )

            return RewardMetrics(
                reward=weighted_reward,
                execution_time=execution_time,
                details=details,
                success=True,
            )

        except Exception as e:
            execution_time = time.time() - start_time

            return RewardMetrics(
                reward=0.0,
                execution_time=execution_time,
                details={
                    "function_type": self.__class__.__name__,
                    "error_type": type(e).__name__,
                },
                success=False,
                error=str(e),
            )

    def _normalize_reward(self, raw_reward: float) -> float:
        """
        Normalize raw reward to target range.

        Args:
            raw_reward: Raw reward value

        Returns:
            Normalized reward
        """
        # Default implementation assumes raw_reward is in [0, 1] range
        # Subclasses can override for different raw ranges
        min_val, max_val = self.normalize_range
        return min_val + (max_val - min_val) * max(0.0, min(1.0, raw_reward))

    def _update_metrics(self, reward: float, execution_time: float) -> None:
        """
        Update internal metrics tracking.

        Args:
            reward: Calculated reward
            execution_time: Time taken for calculation
        """
        self.calculation_count += 1
        self.total_execution_time += execution_time
        self.last_reward = reward

        # Keep limited history for statistics
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-500:]  # Keep last 500

    async def _get_detailed_metrics(
        self,
        problem: str,
        generated_code: str,
        test_cases: list[dict[str, Any]],
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Get detailed metrics for debugging and analysis.

        Args:
            problem: Problem description
            generated_code: Generated code
            test_cases: Test cases
            context: Additional context

        Returns:
            Detailed metrics dictionary
        """
        # Default implementation - subclasses can override
        return {
            "problem_length": len(problem),
            "code_length": len(generated_code),
            "test_case_count": len(test_cases),
        }

    def get_statistics(self) -> dict[str, Any]:
        """
        Get reward function statistics.

        Returns:
            Statistics dictionary
        """
        if not self.reward_history:
            return {
                "calculation_count": 0,
                "avg_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "avg_execution_time": 0.0,
            }

        return {
            "calculation_count": self.calculation_count,
            "avg_reward": sum(self.reward_history) / len(self.reward_history),
            "min_reward": min(self.reward_history),
            "max_reward": max(self.reward_history),
            "last_reward": self.last_reward,
            "avg_execution_time": (
                self.total_execution_time / self.calculation_count
                if self.calculation_count > 0
                else 0.0
            ),
            "weight": self.weight,
            "normalize_range": self.normalize_range,
        }

    def reset_statistics(self) -> None:
        """Reset internal statistics and metrics."""
        self.calculation_count = 0
        self.total_execution_time = 0.0
        self.last_reward = 0.0
        self.reward_history.clear()

        self.logger.info("Reward function statistics reset")

    def __repr__(self) -> str:
        """String representation of the reward function."""
        return (
            f"{self.__class__.__name__}("
            f"weight={self.weight}, "
            f"range={self.normalize_range}, "
            f"calculations={self.calculation_count})"
        )
