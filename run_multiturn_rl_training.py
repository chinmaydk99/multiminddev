#!/usr/bin/env python3
"""
Multi-Turn RL Training Script using VERL and GRPO for AMD ROCm.
This script handles the RL phase after SFT checkpoints are available.
Requires Docker for safe HIP compilation and multi-GPU setup on AMD GPUs.
"""

import os
import sys
import json
import asyncio
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import structlog
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our VERL integration and multi-turn components
from coding_framework.training.verl_integration import MultiAgentVERLTrainer
from coding_framework.training.multi_turn_conversation import MultiTurnConversationManager
from coding_framework.training.reward_functions.hip_performance_reward import HIPPerformanceReward
from coding_framework.data.data_pipeline import HIPDataPipeline
from coding_framework.hip.compiler import HIPCompiler
from coding_framework.hip.benchmarker import HIPBenchmarker
from coding_framework.agents.trainable_hip_agents import (
    TrainableHIPGeneratorAgent,
    TrainableHIPOptimizerAgent,
    TrainableHIPTesterAgent
)

import wandb
import ray
import numpy as np

logger = structlog.get_logger("multiturn_rl")

@dataclass
class MultiTurnRLConfig:
    """Configuration for multi-turn RL training."""
    
    # Model configuration
    generator_checkpoint: str = "./sft_checkpoints/generator"
    optimizer_checkpoint: str = "./sft_checkpoints/optimizer"
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    
    # RL Training configuration
    num_episodes: int = 200
    max_turns_per_episode: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-6  # Lower LR for GRPO stability
    gamma: float = 0.99
    
    # Multi-GPU configuration
    num_gpus: int = 8
    ray_object_store_memory: int = 50000000000  # 50GB
    
    # GPU Distribution configuration (dynamic based on available GPUs)
    vllm_gpus: int = 2              # Number of GPUs for VLLM (starting from 0)
    agent_gpu_offset: int = 2       # Starting GPU for agents (after VLLM GPUs)
    auto_distribute_gpus: bool = True  # Automatically distribute based on num_gpus
    
    # VERL GRPO configuration (uses PPO trainer with adv_estimator=grpo)
    verl_algorithm: str = "grpo"  # Critic-less RL algorithm
    verl_rollout_batch_size: int = 128  # Prompts per rollout batch
    verl_train_batch_size: int = 32     # Training batch size
    verl_num_epochs: int = 4            # PPO epochs per batch
    
    # Curriculum configuration (helps with exploration efficiency)
    curriculum_enabled: bool = True      # Progressive difficulty for better sample efficiency
    start_difficulty: str = "easy"       # Begin with simpler HIP problems
    advancement_threshold: float = 0.7   # Success rate needed to advance difficulty
    
    # Paths
    output_dir: str = "./rl_checkpoints"
    wandb_project: str = "HIP-MultiTurn-RL"
    
    # Safety & Compilation
    use_docker_sandbox: bool = True
    max_compilation_time: int = 60
    target_speedup: float = 2.0
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.1

class MultiTurnRLTrainer:
    """Multi-turn RL trainer using VERL and GRPO."""
    
    def __init__(self, config: MultiTurnRLConfig):
        self.config = config
        self.logger = structlog.get_logger("multiturn_rl_trainer")
        
        # Validate checkpoints exist
        self._validate_sft_checkpoints()
        
        # Setup wandb (disabled for now - can run with local logging)
        try:
            wandb.init(
                project=config.wandb_project,
                config=config.__dict__,
                name=f"multiturn_rl_{int(time.time())}",
                mode="disabled"  # Run offline
            )
        except Exception as e:
            self.logger.warning(f"WandB init failed: {e}, continuing with local logging")
        
        # Initialize Ray for distributed training
        self._setup_ray_cluster()
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("Multi-turn RL Trainer initialized")
    
    def _calculate_gpu_distribution(self) -> Dict[str, int]:
        """Calculate dynamic GPU distribution based on available GPUs."""
        import torch
        
        # Get actual number of available GPUs
        available_gpus = torch.cuda.device_count()
        
        if available_gpus == 0:
            raise RuntimeError("No ROCm GPUs available")
        
        self.logger.info(f"Available GPUs: {available_gpus}, Configured: {self.config.num_gpus}")
        
        # Use the minimum of available and configured
        effective_gpus = min(available_gpus, self.config.num_gpus)
        
        if self.config.auto_distribute_gpus:
            # Dynamic distribution based on available GPUs
            if effective_gpus >= 8:
                # Optimal distribution for 8+ GPUs
                allocation = {
                    'vllm_start': 0,
                    'vllm_end': 1,
                    'generator': 2,
                    'optimizer': 3,
                    'tester': 4,
                    'available_for_training': list(range(5, effective_gpus))
                }
            elif effective_gpus >= 4:
                # Distribution for 4-7 GPUs
                allocation = {
                    'vllm_start': 0,
                    'vllm_end': 0,  # VLLM on single GPU
                    'generator': 1,
                    'optimizer': 2,
                    'tester': 3,
                    'available_for_training': list(range(4, effective_gpus))
                }
            elif effective_gpus >= 2:
                # Distribution for 2-3 GPUs (minimal setup)
                allocation = {
                    'vllm_start': 0,
                    'vllm_end': 0,
                    'generator': 1,
                    'optimizer': 1,  # Share GPU with generator
                    'tester': 1,     # Share GPU with generator
                    'available_for_training': []
                }
            else:
                # Single GPU setup (not recommended but functional)
                allocation = {
                    'vllm_start': 0,
                    'vllm_end': 0,
                    'generator': 0,
                    'optimizer': 0,
                    'tester': 0,
                    'available_for_training': []
                }
                self.logger.warning("Single GPU setup detected - performance will be limited")
        else:
            # Manual distribution using config parameters
            allocation = {
                'vllm_start': 0,
                'vllm_end': self.config.vllm_gpus - 1,
                'generator': self.config.agent_gpu_offset,
                'optimizer': self.config.agent_gpu_offset + 1,
                'tester': self.config.agent_gpu_offset + 2,
                'available_for_training': list(range(self.config.agent_gpu_offset + 3, effective_gpus))
            }
        
        # Validate allocation doesn't exceed available GPUs
        max_gpu_used = max(allocation['vllm_end'], allocation['generator'], 
                          allocation['optimizer'], allocation['tester'])
        
        if max_gpu_used >= effective_gpus:
            raise ValueError(f"GPU allocation requires {max_gpu_used + 1} GPUs but only {effective_gpus} available")
        
        self.logger.info("GPU Distribution", 
                        vllm_gpus=f"{allocation['vllm_start']}-{allocation['vllm_end']}",
                        generator=allocation['generator'],
                        optimizer=allocation['optimizer'], 
                        tester=allocation['tester'],
                        available_for_training=allocation['available_for_training'])
        
        return allocation
    
    def _validate_sft_checkpoints(self):
        """Validate that SFT checkpoints exist."""
        generator_path = Path(self.config.generator_checkpoint)
        optimizer_path = Path(self.config.optimizer_checkpoint)
        
        if "/" not in self.config.generator_checkpoint and not generator_path.exists():
            raise FileNotFoundError(f"Generator checkpoint not found: {generator_path}")
        if "/" not in self.config.optimizer_checkpoint and not optimizer_path.exists():
            raise FileNotFoundError(f"Optimizer checkpoint not found: {optimizer_path}")
            
        self.logger.info(
            "SFT checkpoints validated",
            generator=str(generator_path),
            optimizer=str(optimizer_path)
        )
    
    def _setup_ray_cluster(self):
        """Setup Ray cluster for distributed training."""
        if ray.is_initialized():
            ray.shutdown()
        
        ray.init(
            num_gpus=self.config.num_gpus,
            object_store_memory=self.config.ray_object_store_memory,
            ignore_reinit_error=True
        )
        
        self.logger.info(
            "Ray cluster initialized",
            num_gpus=self.config.num_gpus,
            nodes=len(ray.nodes())
        )
    
    def _initialize_components(self):
        """Initialize all training components."""
        # Data pipeline with curriculum for HIP/ROCm
        self.data_pipeline = HIPDataPipeline(
            dataset_name="SakanaAI/AI-CUDA-Engineer-Archive",  # Will convert to HIP
            cache_dir="./cache/datasets",
            curriculum_enabled=self.config.curriculum_enabled,
            initial_tier=self.config.start_difficulty
        )
        
        # HIP compilation and benchmarking for AMD ROCm
        self.compiler = HIPCompiler(use_docker=self.config.use_docker_sandbox)
        self.benchmarker = HIPBenchmarker()
        
        # Reward function for HIP
        self.reward_function = HIPPerformanceReward(
            target_speedup=self.config.target_speedup
        )
        
        # Load trained agents with dynamic GPU distribution
        gpu_allocation = self._calculate_gpu_distribution()
        
        self.generator_agent = TrainableHIPGeneratorAgent(
            model_name=self.config.generator_checkpoint,
            device=f"cuda:{gpu_allocation['generator']}"  # PyTorch ROCm uses cuda API
        )
        self.optimizer_agent = TrainableHIPOptimizerAgent(
            model_name=self.config.optimizer_checkpoint,
            device=f"cuda:{gpu_allocation['optimizer']}"  # PyTorch ROCm uses cuda API
        )
        self.tester_agent = TrainableHIPTesterAgent(
            model_name=self.config.base_model,
            device=f"cuda:{gpu_allocation['tester']}"  # PyTorch ROCm uses cuda API
        )
        
        self.logger.info("GPU allocation", **gpu_allocation)
        
        # Multi-turn conversation manager
        self.conversation_manager = MultiTurnConversationManager(
            generator_agent=self.generator_agent,
            optimizer_agent=self.optimizer_agent,
            tester_agent=self.tester_agent,
            compiler=self.compiler,
            benchmarker=self.benchmarker,
            max_turns=self.config.max_turns_per_episode,
            early_termination_threshold=self.config.target_speedup
        )
        
        # VERL trainer with complete GRPO configuration
        from coding_framework.training.verl_integration import VERLTrainingConfig
        verl_config = VERLTrainingConfig(
            algorithm=self.config.verl_algorithm,  # "grpo"
            batch_size=self.config.verl_train_batch_size,
            learning_rate=self.config.learning_rate,
            num_epochs=self.config.num_episodes // 10,
            episodes_per_epoch=10,
            max_turns_per_episode=self.config.max_turns_per_episode,
            num_gpus=self.config.num_gpus,
            checkpoint_dir=self.config.output_dir,
            
            # Complete GRPO configuration (critic-less RL)
            grpo_group_size=16,         # Group sampling size for GRPO
            grpo_kl_coef=0.02,         # KL loss coefficient
            use_kl_loss=True,          # Enable KL loss for GRPO
            kl_loss_type="low_var_kl", # KL divergence calculation method
            clip_ratio=0.2,            # GRPO clip range
            ppo_mini_batch_size=32,    # Mini-batch size for updates
            ppo_epochs=4,              # Update epochs per rollout batch
            
            # VLLM rollout configuration - reserve GPUs 0-1 for VLLM
            rollout_batch_size=self.config.verl_rollout_batch_size,
            tensor_model_parallel_size=2,  # Use 2 GPUs for VLLM (0,1)
            gpu_memory_utilization=0.7,   # Use 70% of GPU memory for VLLM
            log_prob_micro_batch_size=32,
            
            # Model paths
            generator_model=self.config.base_model,
            optimizer_model=self.config.base_model,
            ref_model=self.config.base_model
        )
        
        self.verl_trainer = MultiAgentVERLTrainer(
            config=verl_config,
            conversation_manager=self.conversation_manager,
            reward_function=self.reward_function
        )
        
        self.logger.info("All components initialized successfully")
    
    async def run_training_episode(self, episode_num: int) -> Dict[str, Any]:
        """Run a single training episode."""
        self.logger.info("Starting training episode", episode=episode_num)
        
        # Get problem from data pipeline
        difficulty = self.data_pipeline.curriculum_manager.get_current_tier()
        problems = await self.data_pipeline.get_training_batch(
            batch_size=1
        )
        problem = problems[0]
        
        # Run multi-turn conversation
        problem_dict = {
            "description": problem.problem_description,
            "difficulty": problem.difficulty,
            "reference_solution": problem.reference_solution,
            "test_cases": problem.test_cases,
            "target_performance": problem.target_performance
        }
        conversation_result = await self.conversation_manager.run_conversation(
            problem=problem_dict,
            conversation_id=f"episode_{episode_num}_{int(time.time())}"
        )
        
        # Extract training data for VERL
        training_data = {
            "problem": problem,
            "conversation": conversation_result.turns,
            "final_reward": conversation_result.final_reward,
            "success": conversation_result.conversation_success,
            "metadata": {
                "episode": episode_num,
                "difficulty": difficulty,
                "num_turns": len(conversation_result.turns)
            }
        }
        
        # Log episode metrics
        self.logger.info(
            "Episode completed",
            episode=episode_num,
            success=conversation_result.conversation_success,
            final_reward=conversation_result.final_reward,
            turns=len(conversation_result.turns),
            difficulty=difficulty
        )
        
        # Update curriculum based on success
        self.data_pipeline.curriculum_manager.record_episode_result(
            compilation_success=conversation_result.conversation_success,
            speedup=conversation_result.current_performance.get("speedup", 1.0),
            final_reward=conversation_result.final_reward
        )
        
        return training_data
    
    async def run_rl_training(self):
        """Run the complete multi-turn RL training."""
        self.logger.info("🚀 Starting Multi-Turn RL Training with GRPO")
        
        training_data = []
        episode_rewards = []
        best_avg_reward = float('-inf')
        patience_counter = 0
        
        for episode in range(self.config.num_episodes):
            # Run episode
            episode_data = await self.run_training_episode(episode)
            training_data.append(episode_data)
            episode_rewards.append(episode_data["final_reward"])
            
            # Train with VERL every few episodes
            if len(training_data) >= self.config.verl_train_batch_size:
                self.logger.info("Running VERL training update")
                
                # Train with collected data
                training_metrics = await self.verl_trainer.train_step(training_data)
                
                # Log training metrics
                metrics = {
                    "episode": episode,
                    "avg_reward": np.mean(episode_rewards[-10:]),  # Last 10 episodes
                    "training_loss": training_metrics.get("loss", 0),
                    "policy_loss": training_metrics.get("policy_loss", 0),
                    "value_loss": training_metrics.get("value_loss", 0),
                    "curriculum_tier": self.data_pipeline.curriculum_manager.get_current_tier()
                }
                try:
                    wandb.log(metrics)
                except:
                    self.logger.info("Training metrics", **metrics)
                
                # Check for improvement and early stopping
                current_avg_reward = np.mean(episode_rewards[-10:])
                if current_avg_reward > best_avg_reward + self.config.early_stopping_threshold:
                    best_avg_reward = current_avg_reward
                    patience_counter = 0
                    
                    # Save best model
                    checkpoint_path = f"{self.config.output_dir}/best_model_episode_{episode}"
                    await self._save_checkpoint(checkpoint_path)
                    self.logger.info("New best model saved", path=checkpoint_path)
                else:
                    patience_counter += 1
                
                # Early stopping check
                if patience_counter >= self.config.early_stopping_patience:
                    self.logger.info("Early stopping triggered", episode=episode)
                    break
                
                # Clear training data for next batch
                training_data = []
        
        self.logger.info("🎉 Multi-Turn RL Training Complete!")
        
        # Save final model
        final_checkpoint = f"{self.config.output_dir}/final_model"
        await self._save_checkpoint(final_checkpoint)
        
        return {
            "final_checkpoint": final_checkpoint,
            "best_avg_reward": best_avg_reward,
            "total_episodes": len(episode_rewards),
            "episode_rewards": episode_rewards
        }
    
    async def _save_checkpoint(self, path: str):
        """Save model checkpoint."""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save agent checkpoints
        self.generator_agent.save_pretrained(f"{path}/generator")
        self.optimizer_agent.save_pretrained(f"{path}/optimizer")
        
        # Save training config
        config_path = f"{path}/config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        self.logger.info("Checkpoint saved", path=path)
    
    def cleanup(self):
        """Cleanup resources."""
        if ray.is_initialized():
            ray.shutdown()
        wandb.finish()
        self.logger.info("Cleanup completed")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Multi-Turn RL Training with VERL")
    parser.add_argument("--generator-checkpoint", type=str, default="./sft_checkpoints/generator", 
                        help="Path to generator SFT checkpoint")
    parser.add_argument("--optimizer-checkpoint", type=str, default="./sft_checkpoints/optimizer",
                        help="Path to optimizer SFT checkpoint")
    parser.add_argument("--num-episodes", type=int, default=200, help="Number of training episodes")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="./rl_checkpoints", help="Output directory")
    parser.add_argument("--no-docker", action="store_true", help="Disable Docker sandbox")
    
    args = parser.parse_args()
    
    # Setup logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Create config
    config = MultiTurnRLConfig(
        generator_checkpoint=args.generator_checkpoint,
        optimizer_checkpoint=args.optimizer_checkpoint,
        num_episodes=args.num_episodes,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        use_docker_sandbox=not args.no_docker
    )
    
    # Run training
    trainer = MultiTurnRLTrainer(config)
    try:
        asyncio.run(trainer.run_rl_training())
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main()