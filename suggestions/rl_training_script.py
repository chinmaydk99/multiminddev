#!/usr/bin/env python3
"""
Multi-Turn RL Training Script with VERL
Performs reinforcement learning on SFT-trained models using multi-turn conversations.
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
from dataclasses import dataclass, field

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import core components
from coding_framework.training.verl_integration import MultiAgentVERLTrainer, VERLTrainingConfig
from coding_framework.training.multi_turn_conversation import MultiTurnConversationManager
from coding_framework.training.reward_functions.cuda_performance_reward import CUDAPerformanceReward
from coding_framework.data.data_pipeline import CUDADataPipeline
from coding_framework.cuda.compiler import CUDACompiler
from coding_framework.cuda.benchmarker import CUDABenchmarker
from coding_framework.agents.trainable_cuda_agents import (
    TrainableCUDAGeneratorAgent,
    TrainableCUDAOptimizerAgent,
    TrainableCUDATesterAgent
)

# Import VERL dependencies
try:
    import ray
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import wandb
    VERL_DEPENDENCIES = True
except ImportError:
    VERL_DEPENDENCIES = False
    print("❌ VERL dependencies not available. Please install: pip install ray transformers peft wandb")


@dataclass
class MultiTurnRLConfig:
    """Configuration for multi-turn RL training."""
    
    # Model paths (from SFT training)
    generator_model_path: str = "./checkpoints/sft/generator/final"
    optimizer_model_path: str = "./checkpoints/sft/optimizer/final"
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    # VERL algorithm configuration
    algorithm: str = "grpo"  # grpo, dapo, ppo
    
    # Training parameters
    num_episodes: int = 1000
    max_turns_per_episode: int = 5
    batch_size: int = 256
    mini_batch_size: int = 32
    learning_rate: float = 5e-6
    
    # GRPO specific
    grpo_group_size: int = 16
    grpo_kl_coef: float = 0.02
    grpo_clip_ratio_low: float = 0.15
    grpo_clip_ratio_high: float = 0.25
    
    # Multi-turn specific
    turn_discount_factor: float = 0.9
    immediate_reward_weight: float = 0.3
    final_reward_weight: float = 0.7
    early_termination_threshold: float = 2.0  # 2x speedup
    
    # Distributed training
    num_gpus: int = 8
    num_rollout_workers: int = 4
    num_actor_workers: int = 2
    num_critic_workers: int = 2
    
    # Ray configuration
    ray_object_store_memory: int = 50000000000  # 50GB
    ray_cluster_address: Optional[str] = None
    
    # Curriculum and data
    curriculum_start_tier: str = "easy"
    curriculum_enabled: bool = True
    data_source: str = "SakanaAI/AI-CUDA-Engineer-Archive"
    
    # Reward configuration
    target_speedup: float = 2.0
    compilation_weight: float = 0.3
    correctness_weight: float = 0.3
    performance_weight: float = 0.25
    efficiency_weight: float = 0.15
    
    # Safety and compilation
    use_docker_sandbox: bool = True
    max_compilation_time: int = 60
    
    # Monitoring and checkpointing
    use_wandb: bool = True
    project_name: str = "CUDA-MultiTurn-RL"
    experiment_name: str = "verl_training"
    checkpoint_dir: str = "./checkpoints/rl"
    save_freq: int = 50  # Save every N episodes
    
    # Hardware
    cuda_visible_devices: str = "0,1,2,3,4,5,6,7"


class VERLModelLoader:
    """Loads SFT-trained models for VERL training."""
    
    def __init__(self, config: MultiTurnRLConfig):
        self.config = config
        self.logger = structlog.get_logger("verl_model_loader")
    
    def load_sft_model(self, model_path: str, agent_type: str):
        """Load SFT-trained model with LoRA adapters."""
        
        self.logger.info(f"Loading {agent_type} model from {model_path}")
        
        # Check if SFT checkpoint exists
        model_path = Path(model_path)
        if not model_path.exists():
            self.logger.warning(f"SFT checkpoint not found at {model_path}, using base model")
            return self._load_base_model()
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load LoRA adapters
            model = PeftModel.from_pretrained(base_model, str(model_path))
            
            # Merge adapters for faster inference during RL
            model = model.merge_and_unload()
            
            self.logger.info(f"Successfully loaded {agent_type} model with merged LoRA adapters")
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load SFT model: {e}")
            self.logger.info("Falling back to base model")
            return self._load_base_model()
    
    def _load_base_model(self):
        """Load base model without SFT adapters."""
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        return model, tokenizer


class CUDARLEnvironment:
    """RL environment for CUDA kernel optimization."""
    
    def __init__(self, config: MultiTurnRLConfig):
        self.config = config
        self.logger = structlog.get_logger("cuda_rl_env")
        
        # Initialize CUDA infrastructure
        self.compiler = CUDACompiler(
            use_docker=config.use_docker_sandbox,
            temp_dir="./temp/rl_compilation"
        )
        self.benchmarker = CUDABenchmarker()
        
        # Initialize data pipeline
        self.data_pipeline = CUDADataPipeline(
            dataset_name=config.data_source,
            curriculum_enabled=config.curriculum_enabled,
            initial_tier=config.curriculum_start_tier
        )
        
        # Initialize reward function
        self.reward_function = CUDAPerformanceReward(
            target_speedup=config.target_speedup,
            reward_weights={
                "compilation": config.compilation_weight,
                "correctness": config.correctness_weight,
                "performance": config.performance_weight,
                "efficiency": config.efficiency_weight
            }
        )
    
    async def get_training_episode(self):
        """Get a training episode (problem to solve)."""
        
        # Get problem from curriculum
        examples = await self.data_pipeline.get_training_batch(batch_size=1)
        if not examples:
            raise ValueError("No training examples available")
        
        return examples[0]
    
    async def evaluate_conversation(self, conversation_state):
        """Evaluate a completed multi-turn conversation."""
        
        final_reward, reward_components = await self.reward_function.calculate_reward(
            problem=conversation_state.problem_description,
            generated_code=conversation_state.current_kernel_code,
            compilation_result=conversation_state.final_compilation_result,
            benchmark_result=conversation_state.final_benchmark_result,
            context={
                "conversation_length": len(conversation_state.turns),
                "turn_number": len(conversation_state.turns),
                "max_turns": self.config.max_turns_per_episode
            }
        )
        
        return final_reward, reward_components


class MultiTurnRLTrainer:
    """Main multi-turn RL trainer using VERL."""
    
    def __init__(self, config: MultiTurnRLConfig):
        self.config = config
        self.logger = structlog.get_logger("multiturn_rl_trainer")
        
        # Setup environment
        os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
        os.environ["RAY_OBJECT_STORE_MEMORY"] = str(config.ray_object_store_memory)
        
        # Initialize components
        self.model_loader = VERLModelLoader(config)
        self.environment = CUDARLEnvironment(config)
        
        # Initialize wandb
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                name=f"{config.experiment_name}-{int(time.time())}",
                config=config.__dict__
            )
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    async def setup_ray_and_verl(self):
        """Setup Ray cluster and VERL components."""
        
        self.logger.info("Setting up Ray cluster and VERL components")
        
        # Initialize Ray
        if not ray.is_initialized():
            if self.config.ray_cluster_address:
                ray.init(address=self.config.ray_cluster_address)
            else:
                ray.init(
                    num_cpus=self.config.num_gpus * 8,
                    num_gpus=self.config.num_gpus,
                    object_store_memory=self.config.ray_object_store_memory,
                    runtime_env={
                        "env_vars": {
                            "TOKENIZERS_PARALLELISM": "false",
                            "CUDA_VISIBLE_DEVICES": self.config.cuda_visible_devices
                        }
                    }
                )
        
        self.logger.info("Ray cluster initialized", resources=ray.available_resources())
        
        # Load models
        generator_model, generator_tokenizer = self.model_loader.load_sft_model(
            self.config.generator_model_path, "generator"
        )
        
        optimizer_model, optimizer_tokenizer = self.model_loader.load_sft_model(
            self.config.optimizer_model_path, "optimizer"
        )
        
        # Initialize agents
        self.generator_agent = TrainableCUDAGeneratorAgent(
            model_name=self.config.base_model,
            model=generator_model,
            tokenizer=generator_tokenizer
        )
        
        self.optimizer_agent = TrainableCUDAOptimizerAgent(
            model_name=self.config.base_model,
            model=optimizer_model,
            tokenizer=optimizer_tokenizer
        )
        
        self.tester_agent = TrainableCUDATesterAgent(
            compiler=self.environment.compiler,
            benchmarker=self.environment.benchmarker
        )
        
        # Initialize conversation manager
        self.conversation_manager = MultiTurnConversationManager(
            generator_agent=self.generator_agent,
            optimizer_agent=self.optimizer_agent,
            tester_agent=self.tester_agent,
            compiler=self.environment.compiler,
            benchmarker=self.environment.benchmarker,
            max_turns=self.config.max_turns_per_episode,
            early_termination_threshold=self.config.early_termination_threshold
        )
        
        # Create VERL trainer configuration
        verl_config = VERLTrainingConfig(
            algorithm=self.config.algorithm,
            generator_model=self.config.base_model,
            optimizer_model=self.config.base_model,
            grpo_group_size=self.config.grpo_group_size,
            grpo_kl_coef=self.config.grpo_kl_coef,
            grpo_clip_ratio_low=self.config.grpo_clip_ratio_low,
            grpo_clip_ratio_high=self.config.grpo_clip_ratio_high,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            num_epochs=self.config.num_episodes // 100,  # Convert episodes to epochs
            episodes_per_epoch=100,
            max_turns_per_episode=self.config.max_turns_per_episode,
            turn_discount_factor=self.config.turn_discount_factor,
            num_gpus=self.config.num_gpus,
            checkpoint_dir=self.config.checkpoint_dir,
            save_freq=self.config.save_freq
        )
        
        # Initialize VERL trainer
        self.verl_trainer = MultiAgentVERLTrainer(
            config=verl_config,
            conversation_manager=self.conversation_manager,
            reward_function=self.environment.reward_function,
            data_loader=self.environment.data_pipeline
        )
        
        self.logger.info("VERL components initialized successfully")
    
    async def train(self):
        """Run the complete multi-turn RL training."""
        
        self.logger.info("Starting multi-turn RL training")
        
        try:
            # Setup Ray and VERL
            await self.setup_ray_and_verl()
            
            # Run VERL training
            training_metrics = await self.verl_trainer.train()
            
            self.logger.info("Multi-turn RL training completed", metrics=training_metrics)
            
            return training_metrics
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise
        
        finally:
            # Cleanup
            if ray.is_initialized():
                ray.shutdown()
            
            if self.config.use_wandb:
                wandb.finish()
    
    async def run_single_episode_test(self):
        """Run a single episode for testing purposes."""
        
        self.logger.info("Running single episode test")
        
        # Get a test problem
        test_problem = await self.environment.get_training_episode()
        
        # Run conversation
        conversation_result = await self.conversation_manager.run_conversation(
            problem=test_problem.__dict__,
            conversation_id="test_episode"
        )
        
        # Evaluate
        final_reward, reward_components = await self.environment.evaluate_conversation(
            conversation_result
        )
        
        self.logger.info(
            "Test episode completed",
            final_reward=final_reward,
            turns=len(conversation_result.turns),
            success=conversation_result.conversation_success,
            reward_breakdown=reward_components.__dict__
        )
        
        return conversation_result, final_reward


async def main():
    """Main RL training pipeline."""
    
    parser = argparse.ArgumentParser(description="Multi-Turn RL Training with VERL")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--generator-model", type=str, 
                       default="./checkpoints/sft/generator/final",
                       help="Path to SFT-trained generator model")
    parser.add_argument("--optimizer-model", type=str,
                       default="./checkpoints/sft/optimizer/final", 
                       help="Path to SFT-trained optimizer model")
    parser.add_argument("--num-episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--max-turns", type=int, default=5, help="Max turns per episode")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--algorithm", type=str, choices=["grpo", "dapo", "ppo"], 
                       default="grpo", help="VERL algorithm")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--test-only", action="store_true", help="Run single episode test only")
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
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logger = structlog.get_logger("rl_main")
    
    # Check dependencies
    if not VERL_DEPENDENCIES:
        logger.error("Required dependencies not available")
        return
    
    # Create config
    config = MultiTurnRLConfig(
        generator_model_path=args.generator_model,
        optimizer_model_path=args.optimizer_model,
        num_episodes=args.num_episodes,
        max_turns_per_episode=args.max_turns,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        algorithm=args.algorithm,
        use_wandb=not args.no_wandb,
        use_docker_sandbox=not args.no_docker
    )
    
    logger.info("Starting multi-turn RL training pipeline", config=config.__dict__)
    
    # Initialize trainer
    trainer = MultiTurnRLTrainer(config)
    
    if args.test_only:
        # Run single episode test
        logger.info("Running single episode test")
        await trainer.setup_ray_and_verl()
        await trainer.run_single_episode_test()
    else:
        # Run full training
        training_metrics = await trainer.train()
        logger.info("Training completed successfully", metrics=training_metrics)


if __name__ == "__main__":
    asyncio.run(main())
