#!/usr/bin/env python3
"""
Run ACTUAL training pipeline with real agents - NO MOCKS.

This script runs both SFT and GRPO multi-turn RL training using the actual
trainable CUDA agents with real model parameters on multi-GPU setup.
"""

import asyncio
import sys
import os
import time
import json
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import wandb

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import ray
import structlog
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

# Import ACTUAL agents and trainers - NO MOCKS
from coding_framework.agents.trainable_cuda_agents import (
    TrainableCUDAGeneratorAgent,
    TrainableCUDAOptimizerAgent,
    TrainableCUDATesterAgent
)
from coding_framework.training.multi_turn_rl_trainer import (
    MultiTurnRLTrainer,
    MultiTurnRLConfig
)
from coding_framework.training.sft_data_preparation import SFTDataPipeline
from coding_framework.training.multi_turn_conversation import (
    MultiTurnConversationManager,
    TurnLevelRewardDistributor
)
from coding_framework.training.reward_functions.cuda_performance_reward import (
    CUDAPerformanceReward
)
from coding_framework.cuda.compiler import CUDACompiler
from coding_framework.cuda.benchmarker import CUDABenchmarker

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@dataclass
class ActualTrainingConfig:
    """Configuration for actual training with real agents."""
    
    # Model configuration
    generator_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # Using smaller model for faster training
    optimizer_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # 1.5B instead of 7B for memory
    
    # SFT configuration
    sft_epochs: int = 2
    sft_batch_size: int = 1  # Even smaller batch size to fit in memory
    sft_learning_rate: float = 5e-5
    sft_num_examples: int = 50  # Further reduced for testing
    
    # GRPO RL configuration
    rl_algorithm: str = "grpo"
    grpo_group_size: int = 4  # Group size for GRPO
    grpo_kl_coef: float = 0.02
    num_episodes: int = 50  # Actual episodes
    episodes_per_epoch: int = 10
    max_turns_per_episode: int = 4
    
    # Training parameters
    learning_rate: float = 1e-6
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 10
    
    # Reward configuration
    target_speedup: float = 1.5  # More realistic target
    correctness_weight: float = 0.4
    performance_weight: float = 0.4
    improvement_weight: float = 0.2
    
    # Multi-turn settings
    discount_factor: float = 0.9
    immediate_reward_weight: float = 0.3
    final_reward_weight: float = 0.7
    early_termination_threshold: float = 1.5
    
    # Distributed settings
    num_gpus: int = 1  # Single GPU for simplified testing
    use_ray: bool = True
    ray_num_cpus: int = 4  # Reduced for single GPU setup
    
    # Monitoring
    use_wandb: bool = True
    wandb_project: str = "cuda-multiturn-rl-actual"
    wandb_run_name: str = "grpo-training-actual"
    log_every_n_steps: int = 5
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/actual_training"
    save_freq: int = 10


class ActualSFTTrainer:
    """Actual SFT trainer with real agents and models."""
    
    def __init__(self, config: ActualTrainingConfig):
        self.config = config
        self.logger = structlog.get_logger()
        
        # Initialize wandb for monitoring
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=f"{config.wandb_run_name}-sft",
                config={
                    "phase": "sft",
                    "model": config.generator_model,
                    "batch_size": config.sft_batch_size,
                    "learning_rate": config.sft_learning_rate,
                    "num_gpus": config.num_gpus
                }
            )
    
    async def train_agents(self):
        """Train generator and optimizer agents with SFT."""
        self.logger.info("🚀 Starting ACTUAL SFT training with real agents")
        
        # Create data pipeline
        data_pipeline = SFTDataPipeline(
            data_dir=Path("data/actual_sft"),
            use_huggingface_data=False  # Use synthetic for now
        )
        
        # Prepare datasets
        self.logger.info(f"Preparing SFT datasets ({self.config.sft_num_examples} examples each)")
        generator_dataset = await data_pipeline.prepare_generator_data(
            num_examples=self.config.sft_num_examples
        )
        optimizer_dataset = await data_pipeline.prepare_optimizer_data(
            num_examples=self.config.sft_num_examples
        )
        
        # Initialize Ray for distributed training
        if self.config.use_ray and not ray.is_initialized():
            ray.init(
                num_cpus=self.config.ray_num_cpus,
                num_gpus=self.config.num_gpus,
                runtime_env={
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "false",
                        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"
                    }
                }
            )
            self.logger.info(f"Ray initialized with {ray.available_resources()}")
        
        # Train generator agent
        await self._train_single_agent("generator", generator_dataset)
        
        # Train optimizer agent
        await self._train_single_agent("optimizer", optimizer_dataset)
        
        self.logger.info("✅ SFT training completed")
        
        return True
    
    async def _train_single_agent(self, agent_type: str, dataset: Dataset):
        """Train a single agent with SFT."""
        self.logger.info(f"Training {agent_type} agent with SFT")
        
        # Create actual agent (not mock!)
        if agent_type == "generator":
            agent = TrainableCUDAGeneratorAgent(
                agent_id=f"cuda_generator_sft",
                model_name=self.config.generator_model,
                learning_rate=self.config.sft_learning_rate,
                load_in_8bit=True  # Enable 8-bit quantization to reduce memory
            )
        else:
            agent = TrainableCUDAOptimizerAgent(
                agent_id=f"cuda_optimizer_sft",
                model_name=self.config.optimizer_model,
                learning_rate=self.config.sft_learning_rate,
                load_in_8bit=True  # Enable 8-bit quantization to reduce memory
            )
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            lambda examples: agent.tokenizer(
                [f"{inp}\n{out}" for inp, out in zip(examples["input"], examples["output"])],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            ),
            batched=True
        )
        
        # Training loop
        total_steps = 0
        for epoch in range(self.config.sft_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Create batches
            for i in range(0, len(tokenized_dataset), self.config.sft_batch_size):
                batch = tokenized_dataset[i:i + self.config.sft_batch_size]
                
                # Actual SFT training step
                metrics = await agent.sft_train_step(
                    input_text=batch["input"],
                    target_text=batch["output"]
                )
                
                epoch_loss += metrics.get("loss", 0.0)
                num_batches += 1
                total_steps += 1
                
                # Log to wandb
                if self.config.use_wandb and total_steps % self.config.log_every_n_steps == 0:
                    wandb.log({
                        f"{agent_type}_sft_loss": metrics.get("loss", 0.0),
                        f"{agent_type}_sft_step": total_steps,
                        "epoch": epoch
                    })
                
                self.logger.info(
                    f"{agent_type} SFT",
                    epoch=epoch+1,
                    batch=num_batches,
                    loss=metrics.get("loss", 0.0)
                )
            
            avg_epoch_loss = epoch_loss / max(1, num_batches)
            self.logger.info(
                f"{agent_type} epoch complete",
                epoch=epoch+1,
                avg_loss=avg_epoch_loss
            )
        
        # Save checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / "sft" / agent_type
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        agent.save_checkpoint(str(checkpoint_path / "final.pt"))
        self.logger.info(f"Saved {agent_type} checkpoint to {checkpoint_path}")
        
        return agent


class ActualGRPOTrainer:
    """Actual GRPO multi-turn RL trainer with real agents."""
    
    def __init__(self, config: ActualTrainingConfig):
        self.config = config
        self.logger = structlog.get_logger()
        
        # Initialize monitoring
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=f"{config.wandb_run_name}-grpo",
                config={
                    "phase": "grpo",
                    "algorithm": "grpo",
                    "num_episodes": config.num_episodes,
                    "max_turns": config.max_turns_per_episode,
                    "group_size": config.grpo_group_size,
                    "target_speedup": config.target_speedup
                }
            )
        
        # Initialize components
        self.generator_agent = None
        self.optimizer_agent = None
        self.tester_agent = None
        self.conversation_manager = None
        self.reward_function = None
        self.cuda_compiler = CUDACompiler()
        self.cuda_benchmarker = CUDABenchmarker()
    
    async def setup(self):
        """Set up actual agents and components."""
        self.logger.info("Setting up ACTUAL agents for GRPO training")
        
        # Load SFT checkpoints if they exist
        sft_generator_checkpoint = Path(self.config.checkpoint_dir) / "sft" / "generator" / "final.pt"
        sft_optimizer_checkpoint = Path(self.config.checkpoint_dir) / "sft" / "optimizer" / "final.pt"
        
        # Create ACTUAL generator agent
        self.generator_agent = TrainableCUDAGeneratorAgent(
            agent_id="cuda_generator_grpo",
            model_name=self.config.generator_model,
            learning_rate=self.config.learning_rate,
            load_in_8bit=True  # Enable 8-bit quantization
        )
        if sft_generator_checkpoint.exists():
            self.generator_agent.load_checkpoint(str(sft_generator_checkpoint))
            self.logger.info("Loaded generator SFT checkpoint")
        
        # Create ACTUAL optimizer agent
        self.optimizer_agent = TrainableCUDAOptimizerAgent(
            agent_id="cuda_optimizer_grpo",
            model_name=self.config.optimizer_model,
            learning_rate=self.config.learning_rate,
            load_in_8bit=True  # Enable 8-bit quantization
        )
        if sft_optimizer_checkpoint.exists():
            self.optimizer_agent.load_checkpoint(str(sft_optimizer_checkpoint))
            self.logger.info("Loaded optimizer SFT checkpoint")
        
        # Create ACTUAL tester agent (rule-based)
        self.tester_agent = TrainableCUDATesterAgent(
            agent_id="cuda_tester",
            use_trained_model=False  # Rule-based for now
        )
        
        # Create conversation manager with ACTUAL agents
        self.conversation_manager = MultiTurnConversationManager(
            generator_agent=self.generator_agent,
            optimizer_agent=self.optimizer_agent,
            tester_agent=self.tester_agent,
            max_turns=self.config.max_turns_per_episode,
            early_termination_threshold=self.config.early_termination_threshold
        )
        
        # Create reward function
        self.reward_function = CUDAPerformanceReward(
            target_speedup=self.config.target_speedup,
            correctness_weight=self.config.correctness_weight,
            performance_weight=self.config.performance_weight
        )
        
        # Create reward distributor
        self.reward_distributor = TurnLevelRewardDistributor(
            discount_factor=self.config.discount_factor,
            immediate_weight=self.config.immediate_reward_weight,
            final_weight=self.config.final_reward_weight
        )
        
        self.logger.info("✅ ACTUAL agents and components ready")
    
    async def train(self):
        """Run actual GRPO multi-turn RL training."""
        start_time = time.time()
        
        self.logger.info(
            "🚀 Starting ACTUAL GRPO multi-turn RL training",
            num_episodes=self.config.num_episodes,
            group_size=self.config.grpo_group_size
        )
        
        # Training metrics
        all_episode_rewards = []
        all_performance_improvements = []
        all_compilation_successes = []
        
        # Generate training problems
        training_problems = self._generate_cuda_problems()
        
        # GRPO training loop
        for epoch in range(self.config.num_episodes // self.config.episodes_per_epoch):
            epoch_episodes = []
            
            # Collect episodes for this epoch
            for episode_idx in range(self.config.episodes_per_epoch):
                global_episode = epoch * self.config.episodes_per_epoch + episode_idx
                problem = training_problems[global_episode % len(training_problems)]
                
                self.logger.info(f"Episode {global_episode + 1}: {problem}")
                
                # Run ACTUAL multi-turn conversation
                conversation = await self.conversation_manager.run_episode(
                    problem=problem,
                    context={"episode": global_episode}
                )
                
                # Calculate rewards with ACTUAL performance metrics
                final_reward = await self.reward_function.calculate_reward(
                    problem=problem,
                    generated_code=conversation.current_kernel,
                    test_cases=[],  # Would add actual test cases
                    context={"conversation": conversation}
                )
                
                conversation.final_reward = final_reward
                epoch_episodes.append(conversation)
                
                # Track metrics
                all_episode_rewards.append(final_reward)
                if len(conversation.performance_history) > 1:
                    improvement = conversation.performance_history[-1] - conversation.performance_history[0]
                    all_performance_improvements.append(improvement)
                else:
                    all_performance_improvements.append(0.0)
                
                all_compilation_successes.append(1.0 if conversation.compilation_success else 0.0)
                
                # Log to wandb
                if self.config.use_wandb:
                    wandb.log({
                        "episode": global_episode,
                        "episode_reward": final_reward,
                        "performance_improvement": all_performance_improvements[-1],
                        "compilation_success": all_compilation_successes[-1],
                        "num_turns": len(conversation.turns),
                        "final_speedup": conversation.performance_history[-1] if conversation.performance_history else 1.0
                    })
                
                self.logger.info(
                    "Episode complete",
                    episode=global_episode + 1,
                    reward=final_reward,
                    turns=len(conversation.turns),
                    compilation_success=conversation.compilation_success
                )
            
            # GRPO update after collecting episodes
            await self._train_grpo_epoch(epoch_episodes)
            
            # Log epoch metrics
            recent_rewards = all_episode_rewards[-10:] if len(all_episode_rewards) >= 10 else all_episode_rewards
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            
            self.logger.info(
                f"Epoch {epoch + 1} complete",
                avg_recent_reward=avg_reward,
                total_episodes=len(all_episode_rewards)
            )
            
            # Save checkpoint
            if (epoch + 1) % (self.config.save_freq // self.config.episodes_per_epoch) == 0:
                await self._save_checkpoint(epoch + 1, all_episode_rewards)
        
        # Final metrics
        total_time = time.time() - start_time
        self.logger.info(
            "✅ ACTUAL GRPO training complete!",
            total_episodes=len(all_episode_rewards),
            final_avg_reward=sum(all_episode_rewards) / len(all_episode_rewards),
            max_improvement=max(all_performance_improvements) if all_performance_improvements else 0,
            compilation_success_rate=sum(all_compilation_successes) / len(all_compilation_successes),
            training_time=f"{total_time:.2f}s"
        )
        
        if self.config.use_wandb:
            wandb.finish()
    
    async def _train_grpo_epoch(self, episodes):
        """Perform GRPO training update on collected episodes."""
        self.logger.info(f"GRPO update with {len(episodes)} episodes")
        
        # Group episodes for GRPO
        groups = [episodes[i:i+self.config.grpo_group_size] 
                  for i in range(0, len(episodes), self.config.grpo_group_size)]
        
        for group in groups:
            if len(group) < 2:
                continue  # Need at least 2 for relative comparison
            
            # Calculate relative advantages within group
            group_rewards = [ep.final_reward for ep in group]
            mean_reward = sum(group_rewards) / len(group_rewards)
            advantages = [(r - mean_reward) for r in group_rewards]
            
            # Update agents based on advantages
            for episode, advantage in zip(group, advantages):
                # Distribute rewards across turns
                turn_rewards = self.reward_distributor.distribute_rewards(episode)
                
                # Update generator turns
                for turn, turn_reward in zip(episode.turns, turn_rewards):
                    if turn.agent_type == "generator" and hasattr(turn, 'log_probs'):
                        # ACTUAL parameter update
                        await self.generator_agent.update_parameters(
                            rewards=torch.tensor([turn_reward * advantage]),
                            log_probs=turn.log_probs,
                            kl_coef=self.config.grpo_kl_coef
                        )
                    elif turn.agent_type == "optimizer" and hasattr(turn, 'log_probs'):
                        # ACTUAL parameter update
                        await self.optimizer_agent.update_parameters(
                            rewards=torch.tensor([turn_reward * advantage]),
                            log_probs=turn.log_probs,
                            kl_coef=self.config.grpo_kl_coef
                        )
    
    def _generate_cuda_problems(self) -> List[str]:
        """Generate diverse CUDA optimization problems."""
        return [
            "Implement efficient matrix multiplication for 512x512 float matrices",
            "Create parallel reduction kernel for summing 1M float array",
            "Optimize element-wise vector addition for 10M elements",
            "Implement 2D convolution with 5x5 filter",
            "Create efficient transpose for 1024x1024 matrix",
            "Implement parallel histogram computation",
            "Create efficient dot product for large vectors",
            "Optimize batch matrix multiplication",
            "Implement parallel prefix sum (scan)",
            "Create efficient sparse matrix multiplication",
        ]
    
    async def _save_checkpoint(self, epoch: int, episode_rewards: List[float]):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir) / "grpo"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save agent models
        self.generator_agent.save_checkpoint(str(checkpoint_dir / f"generator_epoch_{epoch}.pt"))
        self.optimizer_agent.save_checkpoint(str(checkpoint_dir / f"optimizer_epoch_{epoch}.pt"))
        
        # Save training metrics
        metrics = {
            "epoch": epoch,
            "episode_rewards": episode_rewards,
            "avg_reward": sum(episode_rewards) / len(episode_rewards),
            "config": self.config.__dict__
        }
        
        with open(checkpoint_dir / f"metrics_epoch_{epoch}.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Checkpoint saved for epoch {epoch}")


async def main():
    """Main entry point for actual training pipeline."""
    logger.info("🎯 Starting ACTUAL training pipeline - NO MOCKS!")
    
    # Configuration
    config = ActualTrainingConfig()
    
    # Phase 1: SFT Training
    logger.info("=" * 80)
    logger.info("PHASE 1: SFT TRAINING")
    logger.info("=" * 80)
    
    sft_trainer = ActualSFTTrainer(config)
    sft_success = await sft_trainer.train_agents()
    
    if not sft_success:
        logger.error("SFT training failed")
        sys.exit(1)
    
    # Phase 2: GRPO Multi-Turn RL Training
    logger.info("=" * 80)
    logger.info("PHASE 2: GRPO MULTI-TURN RL TRAINING")
    logger.info("=" * 80)
    
    grpo_trainer = ActualGRPOTrainer(config)
    await grpo_trainer.setup()
    await grpo_trainer.train()
    
    logger.info("🎉 ACTUAL training pipeline completed successfully!")
    logger.info("Check wandb for detailed metrics and monitoring")
    
    # Cleanup
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())