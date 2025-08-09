"""
Abstract base trainer class for reinforcement learning training.

Provides common functionality for all training implementations.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import structlog
from pydantic import BaseModel

from ..agents.trainable_agent import TrainableAgent
from ..utils.config import TrainingConfig


class TrainingResults(BaseModel):
    """Standard results format for training sessions."""

    success: bool
    algorithm: str
    episodes: int
    metrics: dict[str, float]
    training_time: float
    timestamp: float = None
    error: Optional[str] = None
    checkpoint_path: Optional[str] = None

    def __init__(self, **data):
        if data.get("timestamp") is None:
            data["timestamp"] = time.time()
        super().__init__(**data)


class BaseTrainer(ABC):
    """
    Abstract base class for all training implementations.

    Provides common functionality for training agents including
    configuration management, logging, and result handling.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the base trainer.

        Args:
            config: Training configuration settings
        """
        self.config = config
        self.logger = structlog.get_logger(
            component="trainer",
            trainer_type=self.__class__.__name__,
        )

        # Training state
        self.is_training = False
        self.current_episode = 0
        self.training_metrics = {}

        self.logger.info("Trainer initialized", config=config.dict())

    @abstractmethod
    async def train_agent(
        self,
        agent: TrainableAgent,
        training_data: list[dict[str, Any]],
        validation_data: Optional[list[dict[str, Any]]] = None,
        episodes: int = 100,
        reward_function: Optional[Any] = None,
        **kwargs,
    ) -> TrainingResults:
        """
        Train an agent using the specified algorithm.

        Args:
            agent: Agent to train
            training_data: Training problem dataset
            validation_data: Optional validation dataset
            episodes: Number of training episodes
            reward_function: Reward function for training
            **kwargs: Additional training parameters

        Returns:
            Training results with metrics and status
        """
        pass

    def _initialize_training_metrics(self) -> None:
        """Initialize training metrics tracking."""
        self.training_metrics = {
            "episode_rewards": [],
            "avg_reward": 0.0,
            "best_reward": float("-inf"),
            "training_loss": [],
            "episodes_completed": 0,
            "start_time": time.time(),
        }

    def _update_episode_metrics(self, episode: int, reward: float, **metrics) -> None:
        """
        Update metrics after each training episode.

        Args:
            episode: Current episode number
            reward: Episode reward
            **metrics: Additional metrics to track
        """
        self.training_metrics["episode_rewards"].append(reward)
        self.training_metrics["episodes_completed"] = episode + 1

        # Update running statistics
        rewards = self.training_metrics["episode_rewards"]
        self.training_metrics["avg_reward"] = sum(rewards) / len(rewards)
        self.training_metrics["best_reward"] = max(self.training_metrics["best_reward"], reward)

        # Add any additional metrics
        for key, value in metrics.items():
            if key not in self.training_metrics:
                self.training_metrics[key] = []
            if isinstance(self.training_metrics[key], list):
                self.training_metrics[key].append(value)
            else:
                self.training_metrics[key] = value

        self.logger.info(
            "Episode completed",
            episode=episode,
            reward=reward,
            avg_reward=self.training_metrics["avg_reward"],
            **metrics,
        )

    def _create_training_results(
        self,
        success: bool,
        algorithm: str,
        episodes: int,
        error: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> TrainingResults:
        """
        Create standardized training results.

        Args:
            success: Whether training completed successfully
            algorithm: Algorithm used for training
            episodes: Number of episodes attempted
            error: Error message if training failed
            checkpoint_path: Path to saved checkpoint

        Returns:
            Training results object
        """
        training_time = time.time() - self.training_metrics.get("start_time", time.time())

        # Final metrics
        final_metrics = {
            "final_reward": self.training_metrics.get("avg_reward", 0.0),
            "best_reward": self.training_metrics.get("best_reward", 0.0),
            "training_time": training_time,
            "episodes_completed": self.training_metrics.get("episodes_completed", 0),
            "convergence_episode": self._find_convergence_episode(),
        }

        return TrainingResults(
            success=success,
            algorithm=algorithm,
            episodes=episodes,
            metrics=final_metrics,
            training_time=training_time,
            error=error,
            checkpoint_path=checkpoint_path,
        )

    def _find_convergence_episode(self) -> int:
        """
        Find the episode where training converged (reward stabilized).

        Returns:
            Episode number where convergence occurred
        """
        rewards = self.training_metrics.get("episode_rewards", [])
        if len(rewards) < 10:
            return len(rewards)

        # Simple convergence detection: find when reward variance becomes small
        window_size = min(10, len(rewards) // 4)
        for i in range(window_size, len(rewards)):
            recent_rewards = rewards[i - window_size : i]
            variance = sum(
                (r - sum(recent_rewards) / len(recent_rewards)) ** 2 for r in recent_rewards
            ) / len(recent_rewards)
            if variance < 0.01:  # Low variance threshold
                return i - window_size

        return len(rewards)

    async def save_checkpoint(
        self,
        episode: int,
        agent: TrainableAgent,
        metrics: dict[str, Any],
    ) -> str:
        """
        Save training checkpoint.

        Args:
            episode: Current episode number
            agent: Agent being trained
            metrics: Current training metrics

        Returns:
            Path to saved checkpoint
        """
        import json
        from pathlib import Path

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode}.json"

        checkpoint_data = {
            "episode": episode,
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type,
            "metrics": metrics,
            "timestamp": time.time(),
            "config": self.config.dict(),
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        self.logger.info("Checkpoint saved", path=str(checkpoint_path))
        return str(checkpoint_path)

    async def load_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint data
        """
        import json

        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)

        self.logger.info("Checkpoint loaded", path=checkpoint_path)
        return checkpoint_data

    def get_training_progress(self) -> dict[str, Any]:
        """
        Get current training progress and metrics.

        Returns:
            Training progress information
        """
        if not self.is_training:
            return {"status": "not_training"}

        return {
            "status": "training",
            "current_episode": self.current_episode,
            "total_episodes": getattr(self, "_total_episodes", 0),
            "progress": self.current_episode / max(getattr(self, "_total_episodes", 1), 1),
            "metrics": self.training_metrics.copy(),
        }
