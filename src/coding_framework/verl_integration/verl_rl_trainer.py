"""
VERL-based Reinforcement Learning Trainer for Multi-Agent CUDA Code Generation
This module properly integrates VERL's distributed training capabilities with our LangGraph orchestration.
"""

import ray
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import asyncio
import structlog

# VERL core imports - these are the actual VERL components we need
from verl import DataProto
from verl.trainer.ppo import PPOTrainer
from verl.trainer.config import PPOConfig
from verl.single_controller.ray import RayWorkerGroup, RayResourcePool
from verl.utils.dataset import RLDataset
from verl.workers.rollout.vllm_rollout import vLLMRollout
from verl.workers.actor_critic import ActorCritic

# Our components  
from ..agents.cuda_generator import CUDAGeneratorAgent
from ..agents.cuda_optimizer import CUDAOptimizerAgent
from ..agents.cuda_tester import CUDATesterAgent
from ..orchestration.cuda_workflow import CUDAKernelWorkflow
from ..training.reward_functions.cuda_performance_reward import CUDAPerformanceReward


@dataclass
class VERLTrainingConfig:
    """Configuration for VERL-based multi-turn RL training"""
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    # Distributed training settings
    num_gpus: int = 8
    num_rollout_workers: int = 4
    num_actor_workers: int = 2
    num_critic_workers: int = 2
    
    # PPO hyperparameters
    ppo_epochs: int = 4
    mini_batch_size: int = 8
    learning_rate: float = 1e-5
    kl_coef: float = 0.1
    clip_ratio: float = 0.2
    
    # Multi-turn conversation settings
    max_turns: int = 5
    turn_discount_factor: float = 0.9
    early_stop_threshold: float = 0.8
    
    # CUDA-specific reward settings
    target_speedup: float = 2.0
    correctness_weight: float = 0.4
    performance_weight: float = 0.4
    improvement_weight: float = 0.2


class VERLMultiTurnRLTrainer:
    """
    VERL-based trainer that properly uses VERL's distributed infrastructure
    for multi-turn RL training of CUDA code generation agents.
    """
    
    def __init__(self, config: VERLTrainingConfig):
        self.config = config
        self.logger = structlog.get_logger("verl_rl_trainer")
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init(num_gpus=config.num_gpus)
            self.logger.info(f"Ray initialized with {config.num_gpus} GPUs")
        
        # VERL components
        self.resource_pool = None
        self.rollout_workers = None
        self.actor_critic_workers = None
        self.ppo_trainer = None
        
        # Our multi-agent components
        self.cuda_workflow = None
        self.reward_function = CUDAPerformanceReward(
            target_speedup=config.target_speedup,
            correctness_weight=config.correctness_weight,
            performance_weight=config.performance_weight,
            improvement_weight=config.improvement_weight
        )
        
    def setup_verl_workers(self):
        """Setup VERL's distributed workers using Ray"""
        
        # Create resource pools for different worker types
        self.resource_pool = RayResourcePool(
            process_on_nodes=[self.config.num_gpus]
        )
        
        # Setup rollout workers (for generation)
        rollout_cls = ray.remote(
            num_gpus=1,
            num_cpus=4
        )(vLLMRollout)
        
        self.rollout_workers = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=rollout_cls,
            num_workers=self.config.num_rollout_workers,
            init_args={
                "model_name": self.config.model_name,
                "tensor_parallel_size": 1,
                "max_model_len": 4096,
            }
        )
        
        # Setup actor-critic workers (for training)
        actor_critic_cls = ray.remote(
            num_gpus=2,
            num_cpus=8  
        )(ActorCritic)
        
        self.actor_critic_workers = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=actor_critic_cls,
            num_workers=self.config.num_actor_workers,
            init_args={
                "model_name": self.config.model_name,
                "learning_rate": self.config.learning_rate,
            }
        )
        
        self.logger.info(
            "VERL workers initialized",
            rollout_workers=self.config.num_rollout_workers,
            actor_workers=self.config.num_actor_workers
        )
    
    async def run_multi_turn_episode(
        self,
        problem: Dict[str, Any],
        agents: Dict[str, Any]
    ) -> Tuple[List[Dict], float]:
        """
        Run a multi-turn conversation episode using VERL's rollout infrastructure
        
        Returns:
            - conversation_history: List of agent interactions
            - total_reward: Accumulated reward across turns
        """
        conversation_history = []
        total_reward = 0.0
        
        # Use VERL's rollout workers for generation
        for turn in range(self.config.max_turns):
            
            # Generate with rollout worker (distributed)
            generation_batch = DataProto(
                prompts=[problem["prompt"]],
                metadata={"turn": turn}
            )
            
            # Use VERL's distributed generation
            rollout_output = await self.rollout_workers.async_map(
                lambda worker: worker.generate(generation_batch)
            )
            
            # Process through our CUDA workflow
            workflow_result = await self.cuda_workflow.process_turn(
                generation=rollout_output,
                turn=turn,
                problem=problem
            )
            
            conversation_history.append(workflow_result)
            
            # Calculate reward for this turn
            turn_reward = await self.reward_function.calculate_reward(
                problem=problem["description"],
                generated_code=workflow_result["code"],
                test_cases=problem["test_cases"],
                context={"turn": turn}
            )
            
            # Apply turn discount
            discounted_reward = turn_reward * (self.config.turn_discount_factor ** turn)
            total_reward += discounted_reward
            
            # Early stopping if performance threshold met
            if workflow_result.get("performance", 0) >= self.config.early_stop_threshold:
                self.logger.info(f"Early stopping at turn {turn}, threshold met")
                break
        
        return conversation_history, total_reward
    
    async def train(
        self,
        cuda_generator: CUDAGeneratorAgent,
        cuda_optimizer: CUDAOptimizerAgent,
        cuda_tester: CUDATesterAgent,
        training_data: List[Dict],
        num_episodes: int = 100
    ) -> Dict[str, Any]:
        """
        Main training loop using VERL's PPO trainer with multi-turn conversations
        """
        
        # Setup VERL infrastructure
        self.setup_verl_workers()
        
        # Initialize CUDA workflow with agents
        self.cuda_workflow = CUDAKernelWorkflow(
            cuda_generator=cuda_generator,
            cuda_optimizer=cuda_optimizer,
            cuda_tester=cuda_tester,
            config={"max_turns": self.config.max_turns}
        )
        
        # Create PPO trainer config
        ppo_config = PPOConfig(
            num_ppo_epochs=self.config.ppo_epochs,
            mini_batch_size=self.config.mini_batch_size,
            learning_rate=self.config.learning_rate,
            kl_coef=self.config.kl_coef,
            clip_ratio=self.config.clip_ratio,
        )
        
        # Initialize VERL's PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            actor_rollout_workers=self.rollout_workers,
            actor_critic_workers=self.actor_critic_workers,
        )
        
        training_metrics = []
        
        for episode in range(num_episodes):
            self.logger.info(f"Starting episode {episode}/{num_episodes}")
            
            # Sample batch of problems
            batch_problems = np.random.choice(training_data, self.config.mini_batch_size)
            
            # Collect rollouts for this batch
            rollout_data = []
            rewards = []
            
            for problem in batch_problems:
                # Run multi-turn episode
                conversation, total_reward = await self.run_multi_turn_episode(
                    problem=problem,
                    agents={
                        "generator": cuda_generator,
                        "optimizer": cuda_optimizer,
                        "tester": cuda_tester
                    }
                )
                
                rollout_data.append(conversation)
                rewards.append(total_reward)
            
            # Convert to VERL DataProto format
            training_batch = self._prepare_training_batch(rollout_data, rewards)
            
            # Run PPO update using VERL's distributed training
            ppo_stats = await self.ppo_trainer.train_step(training_batch)
            
            # Log metrics
            episode_metrics = {
                "episode": episode,
                "mean_reward": np.mean(rewards),
                "max_reward": np.max(rewards),
                "min_reward": np.min(rewards),
                **ppo_stats
            }
            
            training_metrics.append(episode_metrics)
            self.logger.info("Episode completed", **episode_metrics)
            
            # Checkpoint if needed
            if episode % 10 == 0:
                await self._save_checkpoint(episode)
        
        # Cleanup Ray resources
        self.cleanup()
        
        return {
            "success": True,
            "total_episodes": num_episodes,
            "final_metrics": training_metrics[-1],
            "all_metrics": training_metrics
        }
    
    def _prepare_training_batch(
        self,
        rollout_data: List[List[Dict]],
        rewards: List[float]
    ) -> DataProto:
        """Convert our rollout data to VERL's DataProto format"""
        
        # Flatten multi-turn conversations
        all_prompts = []
        all_responses = []
        all_rewards = []
        
        for conversation, total_reward in zip(rollout_data, rewards):
            for turn_data in conversation:
                all_prompts.append(turn_data.get("prompt", ""))
                all_responses.append(turn_data.get("response", ""))
                all_rewards.append(total_reward)  # Distribute reward across turns
        
        return DataProto(
            prompts=all_prompts,
            responses=all_responses,
            rewards=torch.tensor(all_rewards),
            metadata={"batch_size": len(rollout_data)}
        )
    
    async def _save_checkpoint(self, episode: int):
        """Save model checkpoint using VERL's checkpointing"""
        checkpoint_path = f"./checkpoints/episode_{episode}"
        self.logger.info(f"Saving checkpoint to {checkpoint_path}")
        
        # Use VERL's checkpoint manager
        await self.actor_critic_workers.async_map(
            lambda worker: worker.save_checkpoint(checkpoint_path)
        )
    
    def cleanup(self):
        """Cleanup Ray resources"""
        if self.rollout_workers:
            self.rollout_workers.shutdown()
        if self.actor_critic_workers:
            self.actor_critic_workers.shutdown()
        
        if ray.is_initialized():
            ray.shutdown()
        
        self.logger.info("VERL training resources cleaned up")


async def launch_verl_training(
    agents: Dict[str, Any],
    training_data: List[Dict],
    config: Optional[VERLTrainingConfig] = None
) -> Dict[str, Any]:
    """
    Entry point for VERL-based multi-turn RL training
    
    This function properly uses VERL's distributed infrastructure for scaling
    the RL training process across multiple GPUs.
    """
    
    if config is None:
        config = VERLTrainingConfig()
    
    trainer = VERLMultiTurnRLTrainer(config)
    
    results = await trainer.train(
        cuda_generator=agents["generator"],
        cuda_optimizer=agents["optimizer"],
        cuda_tester=agents["tester"],
        training_data=training_data,
        num_episodes=100
    )
    
    return results