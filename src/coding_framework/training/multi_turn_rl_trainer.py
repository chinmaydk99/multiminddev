"""
Multi-turn RL trainer with GRPO/DAPO support for training CUDA code generation agents.
Integrates with VERL for distributed training using Ray clusters.
"""

import torch
import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import structlog
import time
import wandb
from dataclasses import dataclass, field
import numpy as np

from ..agents.trainable_cuda_agents import (
    TrainableCUDAGeneratorAgent,
    TrainableCUDAOptimizerAgent,
    TrainableCUDATesterAgent
)
from .multi_turn_conversation import (
    MultiTurnConversationManager,
    TurnLevelRewardDistributor,
    CUDAConversationState,
    AgentRole
)
from .sft_data_preparation import SFTDataPipeline
from ..verl_integration.verl_config import VERLTrainingConfig


@dataclass
class MultiTurnRLConfig:
    """Configuration for multi-turn RL training."""
    
    # Algorithm selection
    algorithm: str = "grpo"  # grpo, dapo, ppo
    
    # Model configuration
    generator_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    optimizer_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    tester_use_model: bool = False  # Rule-based by default
    
    # Training parameters
    num_episodes: int = 10000
    episodes_per_epoch: int = 100
    
    # GRPO specific
    grpo_group_size: int = 16  # Number of responses per prompt for GRPO
    grpo_kl_coef: float = 0.0  # KL penalty coefficient
    grpo_clip_ratio_low: float = 0.2
    grpo_clip_ratio_high: float = 0.28
    
    # DAPO specific  
    dapo_use_kl_in_reward: bool = False
    dapo_loss_agg_mode: str = "token-mean"
    dapo_overlong_penalty_factor: float = 1.0
    
    # Multi-turn settings
    max_turns: int = 5
    early_termination_threshold: float = 2.0  # 2x speedup
    turn_discount_factor: float = 0.9
    immediate_reward_weight: float = 0.3
    final_reward_weight: float = 0.7
    
    # Distributed training
    num_gpus: int = 8
    num_nodes: int = 1
    ray_cluster_address: Optional[str] = None
    
    # Optimization
    learning_rate: float = 1e-6
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/multi_turn_rl"
    save_freq: int = 10  # Save every N epochs
    
    # Logging
    use_wandb: bool = True
    project_name: str = "MultiTurnCUDARL"
    experiment_name: str = "grpo_cuda_optimization"
    
    # Curriculum learning
    start_difficulty: str = "easy"
    difficulty_progression: List[str] = field(default_factory=lambda: ["easy", "medium", "hard"])
    difficulty_switch_epochs: List[int] = field(default_factory=lambda: [3, 6])


class MultiTurnRLTrainer:
    """
    Main trainer for multi-turn RL training of CUDA code generation agents.
    Supports GRPO, DAPO, and PPO algorithms with VERL integration.
    """
    
    def __init__(
        self,
        config: MultiTurnRLConfig,
        verl_config: Optional[VERLTrainingConfig] = None
    ):
        """
        Initialize multi-turn RL trainer.
        
        Args:
            config: Training configuration
            verl_config: VERL-specific configuration
        """
        self.config = config
        self.verl_config = verl_config or VERLTrainingConfig()
        self.logger = structlog.get_logger()
        
        # Initialize Ray if needed
        if not ray.is_initialized():
            if config.ray_cluster_address:
                ray.init(address=config.ray_cluster_address)
            else:
                ray.init(num_gpus=config.num_gpus)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.current_difficulty = config.start_difficulty
        
        # Initialize components (will be created in setup)
        self.generator_agent = None
        self.optimizer_agent = None
        self.tester_agent = None
        self.conversation_manager = None
        self.reward_distributor = None
        self.data_pipeline = None
        
        # Metrics tracking
        self.training_metrics = {
            "episode_rewards": [],
            "turn_rewards": [],
            "performance_improvements": [],
            "compilation_success_rates": []
        }
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                name=config.experiment_name,
                config=config.__dict__
            )
    
    async def setup(self):
        """Set up agents and training components."""
        self.logger.info("Setting up multi-turn RL trainer")
        
        # Create trainable agents
        self.generator_agent = TrainableCUDAGeneratorAgent(
            agent_id="cuda_generator",
            model_name=self.config.generator_model,
            learning_rate=self.config.learning_rate
        )
        
        self.optimizer_agent = TrainableCUDAOptimizerAgent(
            agent_id="cuda_optimizer",
            model_name=self.config.optimizer_model,
            learning_rate=self.config.learning_rate
        )
        
        self.tester_agent = TrainableCUDATesterAgent(
            agent_id="cuda_tester",
            use_trained_model=self.config.tester_use_model
        )
        
        # Create conversation manager
        self.conversation_manager = MultiTurnConversationManager(
            generator_agent=self.generator_agent,
            optimizer_agent=self.optimizer_agent,
            tester_agent=self.tester_agent,
            max_turns=self.config.max_turns,
            early_termination_threshold=self.config.early_termination_threshold
        )
        
        # Create reward distributor
        self.reward_distributor = TurnLevelRewardDistributor(
            discount_factor=self.config.turn_discount_factor,
            immediate_weight=self.config.immediate_reward_weight,
            final_weight=self.config.final_reward_weight
        )
        
        # Create data pipeline
        self.data_pipeline = SFTDataPipeline()
        
        self.logger.info("Setup complete")
    
    async def run_sft_pretraining(self, num_epochs: int = 3):
        """
        Run supervised fine-tuning before RL training.
        
        Args:
            num_epochs: Number of SFT epochs
        """
        self.logger.info("Starting SFT pretraining")
        
        # Prepare SFT data
        generator_data = await self.data_pipeline.prepare_generator_data(10000)
        optimizer_data = await self.data_pipeline.prepare_optimizer_data(8000)
        
        # Tokenize data
        generator_data = self.data_pipeline.tokenize_for_training(
            generator_data,
            self.generator_agent.tokenizer
        )
        optimizer_data = self.data_pipeline.tokenize_for_training(
            optimizer_data,
            self.optimizer_agent.tokenizer
        )
        
        # SFT training loop for generator
        self.logger.info("Training generator with SFT")
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in self._create_batches(generator_data, self.config.batch_size):
                metrics = await self.generator_agent.sft_train_step(
                    batch["input"],
                    batch["output"]
                )
                total_loss += metrics.get("sft_loss", 0)
            
            avg_loss = total_loss / len(generator_data)
            self.logger.info(f"Generator SFT Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # SFT training loop for optimizer
        self.logger.info("Training optimizer with SFT")
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in self._create_batches(optimizer_data, self.config.batch_size):
                metrics = await self.optimizer_agent.sft_train_step(
                    batch["input"],
                    batch["output"]
                )
                total_loss += metrics.get("sft_loss", 0)
            
            avg_loss = total_loss / len(optimizer_data)
            self.logger.info(f"Optimizer SFT Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        self.logger.info("SFT pretraining complete")
    
    async def train(self):
        """Main training loop for multi-turn RL."""
        self.logger.info(
            f"Starting multi-turn RL training with {self.config.algorithm}",
            num_episodes=self.config.num_episodes,
            algorithm=self.config.algorithm
        )
        
        # Setup if not done
        if self.generator_agent is None:
            await self.setup()
        
        # Optional SFT pretraining
        if self.current_epoch == 0:
            await self.run_sft_pretraining(num_epochs=2)
        
        # Main RL training loop
        for epoch in range(self.current_epoch, self.config.num_episodes // self.config.episodes_per_epoch):
            self.current_epoch = epoch
            
            # Update difficulty based on curriculum
            self._update_difficulty(epoch)
            
            # Collect episodes for this epoch
            episodes = await self._collect_episodes(self.config.episodes_per_epoch)
            
            # Process episodes based on algorithm
            if self.config.algorithm == "grpo":
                await self._train_grpo(episodes)
            elif self.config.algorithm == "dapo":
                await self._train_dapo(episodes)
            else:  # PPO
                await self._train_ppo(episodes)
            
            # Logging and checkpointing
            if (epoch + 1) % self.config.save_freq == 0:
                await self._save_checkpoint(epoch)
            
            # Log metrics
            self._log_metrics(epoch)
    
    async def _collect_episodes(self, num_episodes: int) -> List[CUDAConversationState]:
        """
        Collect episodes of multi-turn conversations.
        
        Args:
            num_episodes: Number of episodes to collect
            
        Returns:
            List of conversation states
        """
        episodes = []
        problems = self._sample_problems(num_episodes, self.current_difficulty)
        
        for i, problem in enumerate(problems):
            # Run conversation episode
            conversation_state = await self.conversation_manager.run_conversation_episode(
                problem=problem["description"],
                problem_id=problem["id"],
                difficulty=self.current_difficulty,
                target_performance=problem.get("target_speedup", 1.5)
            )
            
            episodes.append(conversation_state)
            
            # Log progress
            if (i + 1) % 10 == 0:
                self.logger.debug(f"Collected {i+1}/{num_episodes} episodes")
        
        return episodes
    
    async def _train_grpo(self, episodes: List[CUDAConversationState]):
        """
        Train agents using GRPO (Group Relative Policy Optimization).
        
        GRPO groups multiple responses and uses relative rewards within the group.
        """
        self.logger.info("Training with GRPO")
        
        # Group episodes for GRPO
        grouped_episodes = self._group_episodes_for_grpo(episodes)
        
        for group in grouped_episodes:
            # Calculate group relative advantages
            advantages = self._calculate_grpo_advantages(group)
            
            # Update each agent based on their turns
            generator_updates = []
            optimizer_updates = []
            
            for episode, episode_advantages in zip(group, advantages):
                # Distribute rewards across turns
                turn_rewards = self.reward_distributor.distribute_rewards(episode)
                
                for turn, reward, advantage in zip(episode.turns, turn_rewards, episode_advantages):
                    if turn.agent_type == AgentRole.GENERATOR:
                        generator_updates.append((turn, reward, advantage))
                    elif turn.agent_type == AgentRole.OPTIMIZER:
                        optimizer_updates.append((turn, reward, advantage))
            
            # Batch update generator
            if generator_updates:
                await self._update_agent_grpo(self.generator_agent, generator_updates)
            
            # Batch update optimizer
            if optimizer_updates:
                await self._update_agent_grpo(self.optimizer_agent, optimizer_updates)
            
            self.global_step += 1
    
    def _calculate_grpo_advantages(self, episode_group: List[CUDAConversationState]) -> List[List[float]]:
        """
        Calculate GRPO advantages using group relative rewards.
        
        Args:
            episode_group: Group of episodes for the same problem
            
        Returns:
            Advantages for each turn in each episode
        """
        # Get final rewards for the group
        final_rewards = [ep.final_reward for ep in episode_group]
        
        # Normalize rewards within group (GRPO key insight)
        mean_reward = np.mean(final_rewards)
        std_reward = np.std(final_rewards) + 1e-8
        normalized_rewards = [(r - mean_reward) / std_reward for r in final_rewards]
        
        # Calculate advantages for each episode
        all_advantages = []
        for episode, norm_reward in zip(episode_group, normalized_rewards):
            # Use normalized reward as advantage for all turns
            turn_advantages = []
            for turn in episode.turns:
                if turn.agent_type != AgentRole.TESTER:
                    # Apply turn-based discounting
                    turn_idx = turn.turn_id
                    discount = self.config.turn_discount_factor ** (len(episode.turns) - turn_idx - 1)
                    turn_advantages.append(norm_reward * discount)
                else:
                    turn_advantages.append(0.0)
            all_advantages.append(turn_advantages)
        
        return all_advantages
    
    async def _update_agent_grpo(self, agent, updates):
        """Update agent using GRPO algorithm."""
        # Aggregate updates
        all_rewards = torch.tensor([r for _, r, _ in updates])
        all_advantages = torch.tensor([a for _, _, a in updates])
        
        # Get log probs from turns
        all_log_probs = []
        for turn, _, _ in updates:
            if turn.log_probs is not None:
                all_log_probs.append(turn.log_probs)
        
        if all_log_probs:
            all_log_probs = torch.cat(all_log_probs, dim=0)
            
            # Update with GRPO-specific parameters
            metrics = agent.update_parameters(
                rewards=all_rewards,
                log_probs=all_log_probs,
                advantages=all_advantages,
                clip_ratio=(self.config.grpo_clip_ratio_low + self.config.grpo_clip_ratio_high) / 2
            )
            
            self.logger.debug(f"GRPO update metrics for {agent.agent_id}: {metrics}")
    
    async def _train_dapo(self, episodes: List[CUDAConversationState]):
        """
        Train agents using DAPO (Direct Alignment from Preferences).
        
        DAPO directly aligns model outputs with preferences without critic.
        """
        self.logger.info("Training with DAPO")
        
        for episode in episodes:
            # DAPO uses direct preference optimization
            turn_rewards = self.reward_distributor.distribute_rewards(episode)
            
            # Calculate preferences based on performance improvements
            preferences = self._calculate_dapo_preferences(episode)
            
            # Update agents based on preferences
            for turn, reward, preference in zip(episode.turns, turn_rewards, preferences):
                if turn.agent_type == AgentRole.GENERATOR:
                    await self._update_agent_dapo(self.generator_agent, turn, reward, preference)
                elif turn.agent_type == AgentRole.OPTIMIZER:
                    await self._update_agent_dapo(self.optimizer_agent, turn, reward, preference)
            
            self.global_step += 1
    
    def _calculate_dapo_preferences(self, episode: CUDAConversationState) -> List[float]:
        """Calculate DAPO preferences based on performance trajectory."""
        preferences = []
        
        for i, turn in enumerate(episode.turns):
            if turn.agent_type == AgentRole.TESTER:
                preferences.append(0.0)
                continue
            
            # Calculate preference based on performance improvement
            if i > 0 and i < len(episode.performance_history):
                prev_perf = episode.performance_history[i-1] if i > 0 else 1.0
                curr_perf = episode.performance_history[i] if i < len(episode.performance_history) else prev_perf
                improvement = (curr_perf - prev_perf) / prev_perf
                preference = np.tanh(improvement * 2)  # Scale and bound preference
            else:
                preference = 0.0
            
            preferences.append(preference)
        
        return preferences
    
    async def _update_agent_dapo(self, agent, turn, reward, preference):
        """Update agent using DAPO algorithm."""
        if turn.log_probs is None:
            return
        
        # DAPO update with preference-weighted reward
        dapo_reward = reward * (1 + preference)
        
        # Apply overlong penalty if configured
        if hasattr(turn, "output_text") and len(turn.output_text) > 2000:
            dapo_reward *= self.config.dapo_overlong_penalty_factor
        
        metrics = agent.update_parameters(
            rewards=torch.tensor([dapo_reward]),
            log_probs=turn.log_probs,
            clip_ratio=0.2  # Standard clipping for DAPO
        )
        
        self.logger.debug(f"DAPO update metrics for {agent.agent_id}: {metrics}")
    
    async def _train_ppo(self, episodes: List[CUDAConversationState]):
        """Train agents using standard PPO."""
        # Standard PPO implementation (simplified for brevity)
        for episode in episodes:
            turn_rewards = self.reward_distributor.distribute_rewards(episode)
            advantages = self.reward_distributor.calculate_advantages(turn_rewards)
            
            for turn, reward, advantage in zip(episode.turns, turn_rewards, advantages):
                if turn.agent_type == AgentRole.GENERATOR:
                    await self._update_agent_ppo(self.generator_agent, turn, reward, advantage)
                elif turn.agent_type == AgentRole.OPTIMIZER:
                    await self._update_agent_ppo(self.optimizer_agent, turn, reward, advantage)
    
    async def _update_agent_ppo(self, agent, turn, reward, advantage):
        """Update agent using PPO."""
        if turn.log_probs is None:
            return
        
        metrics = agent.update_parameters(
            rewards=torch.tensor([reward]),
            log_probs=turn.log_probs,
            advantages=torch.tensor([advantage])
        )
        
        self.logger.debug(f"PPO update metrics for {agent.agent_id}: {metrics}")
    
    def _group_episodes_for_grpo(self, episodes: List[CUDAConversationState]) -> List[List[CUDAConversationState]]:
        """Group episodes for GRPO training."""
        groups = []
        current_group = []
        
        for episode in episodes:
            current_group.append(episode)
            if len(current_group) >= self.config.grpo_group_size:
                groups.append(current_group)
                current_group = []
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _sample_problems(self, num_problems: int, difficulty: str) -> List[Dict[str, Any]]:
        """Sample training problems based on difficulty."""
        # This would load from your problem dataset
        problems = []
        for i in range(num_problems):
            problems.append({
                "id": f"{difficulty}_{i}",
                "description": f"Optimize CUDA kernel for {difficulty} problem {i}",
                "difficulty": difficulty,
                "target_speedup": 1.5 if difficulty == "easy" else (2.0 if difficulty == "medium" else 3.0)
            })
        return problems
    
    def _update_difficulty(self, epoch: int):
        """Update difficulty based on curriculum learning."""
        for i, switch_epoch in enumerate(self.config.difficulty_switch_epochs):
            if epoch == switch_epoch and i + 1 < len(self.config.difficulty_progression):
                self.current_difficulty = self.config.difficulty_progression[i + 1]
                self.logger.info(f"Switching to {self.current_difficulty} difficulty at epoch {epoch}")
    
    def _create_batches(self, dataset, batch_size):
        """Create batches from dataset."""
        for i in range(0, len(dataset), batch_size):
            yield dataset[i:i + batch_size]
    
    async def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"epoch_{epoch}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save agent checkpoints
        await self.generator_agent.save_checkpoint(str(checkpoint_path / "generator"))
        await self.optimizer_agent.save_checkpoint(str(checkpoint_path / "optimizer"))
        
        # Save training state
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "current_difficulty": self.current_difficulty,
            "training_metrics": self.training_metrics
        }
        
        torch.save(state, checkpoint_path / "training_state.pt")
        self.logger.info(f"Saved checkpoint at epoch {epoch}")
    
    def _log_metrics(self, epoch: int):
        """Log training metrics."""
        metrics = {
            "epoch": epoch,
            "global_step": self.global_step,
            "difficulty": self.current_difficulty,
            "avg_episode_reward": np.mean(self.training_metrics["episode_rewards"][-100:]) if self.training_metrics["episode_rewards"] else 0,
            "avg_performance_improvement": np.mean(self.training_metrics["performance_improvements"][-100:]) if self.training_metrics["performance_improvements"] else 0,
            "compilation_success_rate": np.mean(self.training_metrics["compilation_success_rates"][-100:]) if self.training_metrics["compilation_success_rates"] else 0
        }
        
        self.logger.info(f"Epoch {epoch} metrics: {metrics}")
        
        if self.config.use_wandb:
            wandb.log(metrics, step=self.global_step)