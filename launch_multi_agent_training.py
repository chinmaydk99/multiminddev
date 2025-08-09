#!/usr/bin/env python
"""
Launch script for multi-agent multi-turn RL training using VERL.
This properly integrates with VERL's training infrastructure for GRPO/DAPO.

Usage:
    # For GRPO training
    python launch_multi_agent_training.py algorithm=grpo
    
    # For DAPO training
    python launch_multi_agent_training.py algorithm=dapo
    
    # With custom config
    python launch_multi_agent_training.py --config-path=config --config-name=grpo_dapo_training
"""

import os
import sys
import socket
from pathlib import Path

# Add VERL to Python path
verl_path = Path(__file__).parent / "verl"
sys.path.insert(0, str(verl_path))

import hydra
import ray
from omegaconf import OmegaConf, DictConfig
from pprint import pprint

# Import VERL components
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer
from verl.utils.device import is_cuda_available

# Import our custom multi-agent trainer
from src.coding_framework.training.verl_multi_agent_trainer import (
    MultiAgentVERLTrainer,
    create_multi_agent_trainer
)


def setup_verl_config_for_algorithm(config: DictConfig, algorithm: str):
    """
    Configure VERL settings based on the chosen algorithm.
    
    VERL uses different settings for GRPO vs DAPO vs PPO.
    """
    if algorithm == "grpo":
        # GRPO specific settings
        config.algorithm.adv_estimator = "grpo"
        config.algorithm.use_kl_in_reward = False
        config.algorithm.kl_ctrl.kl_coef = 0.0
        
        # GRPO uses multiple responses per prompt
        config.actor_rollout_ref.rollout.n = 16  # Group size
        
        # No KL loss for GRPO
        config.actor_rollout_ref.actor.use_kl_loss = False
        config.actor_rollout_ref.actor.kl_loss_coef = 0.0
        
        # GRPO clipping ratios
        config.actor_rollout_ref.actor.clip_ratio_low = 0.2
        config.actor_rollout_ref.actor.clip_ratio_high = 0.28
        
    elif algorithm == "dapo":
        # DAPO specific settings
        config.algorithm.adv_estimator = "grpo"  # DAPO uses GRPO advantage
        config.algorithm.use_kl_in_reward = False
        
        # DAPO response settings
        config.actor_rollout_ref.rollout.n = 16
        
        # DAPO loss aggregation
        config.actor_rollout_ref.actor.loss_agg_mode = "token-mean"
        
        # DAPO specific reward configuration
        config.reward_model.reward_manager = "dapo"
        
        # Overlong penalty for DAPO
        if "reward_kwargs" not in config.reward_model:
            config.reward_model.reward_kwargs = {}
        config.reward_model.reward_kwargs.overlong_buffer_cfg = {
            "enable": True,
            "len": 4096,
            "penalty_factor": 1.0,
            "log": False
        }
        
    else:  # Standard PPO
        config.algorithm.adv_estimator = "gae"
        config.algorithm.use_kl_in_reward = True
        config.actor_rollout_ref.rollout.n = 1
        config.actor_rollout_ref.actor.use_kl_loss = True
    
    return config


@ray.remote(num_cpus=1)
class MultiAgentTaskRunner:
    """
    Ray remote task runner for multi-agent training.
    This follows VERL's pattern of using a remote task runner.
    """
    
    def run(self, config: DictConfig):
        """
        Execute multi-agent training with VERL.
        
        This method runs on a Ray worker and coordinates the training.
        """
        print(f"MultiAgentTaskRunner starting on {socket.gethostname()}, PID: {os.getpid()}")
        
        # Resolve and print configuration
        OmegaConf.resolve(config)
        print("\n=== Training Configuration ===")
        pprint(OmegaConf.to_container(config, resolve=True))
        
        # Verify algorithm settings
        algorithm = config.get("algorithm_type", "grpo")
        print(f"\n=== Algorithm: {algorithm.upper()} ===")
        print(f"Advantage Estimator: {config.algorithm.adv_estimator}")
        print(f"Group Size (n): {config.actor_rollout_ref.rollout.n}")
        print(f"Use KL in Reward: {config.algorithm.use_kl_in_reward}")
        
        # Download models to local if needed
        generator_path = copy_to_local(
            config.models.generator.path,
            use_shm=config.models.generator.get("use_shm", True)
        )
        optimizer_path = copy_to_local(
            config.models.optimizer.path,
            use_shm=config.models.optimizer.get("use_shm", True)
        )
        
        print(f"\nModels downloaded:")
        print(f"  Generator: {generator_path}")
        print(f"  Optimizer: {optimizer_path}")
        
        # Update config with local paths
        config.actor_rollout_ref.model.path = generator_path
        
        # Initialize tokenizer (needed by VERL)
        tokenizer = hf_tokenizer(
            generator_path,
            trust_remote_code=config.models.generator.get("trust_remote_code", True)
        )
        
        # Create multi-agent trainer
        print("\n=== Initializing Multi-Agent Trainer ===")
        trainer = MultiAgentVERLTrainer(config)
        
        # Run training
        print("\n=== Starting Training ===")
        trainer.fit()
        
        print("\n=== Training Complete ===")


def run_multi_agent_training(config: DictConfig):
    """
    Initialize Ray and run multi-agent training.
    
    This follows VERL's pattern from main_ppo.py.
    """
    # Get algorithm type
    algorithm = config.get("algorithm_type", "grpo")
    print(f"\n{'='*60}")
    print(f"Starting Multi-Agent {algorithm.upper()} Training")
    print(f"{'='*60}\n")
    
    # Configure for specific algorithm
    config = setup_verl_config_for_algorithm(config, algorithm)
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        runtime_env = get_ppo_ray_runtime_env()
        
        # Add our custom dependencies to runtime env
        runtime_env["pip"] = runtime_env.get("pip", []) + [
            "transformers>=4.35.0",
            "torch>=2.0.0",
            "structlog"
        ]
        
        # Initialize Ray
        if config.distributed.ray_address:
            ray.init(address=config.distributed.ray_address, runtime_env=runtime_env)
        else:
            ray.init(
                num_cpus=config.distributed.get("num_cpus", 32),
                num_gpus=config.distributed.num_gpus_per_node * config.distributed.num_nodes,
                runtime_env=runtime_env
            )
        
        print(f"Ray initialized with resources: {ray.available_resources()}")
    
    # Create remote task runner
    # Check if we need profiling
    if (
        is_cuda_available() and
        config.training.get("profile_steps") is not None and
        len(config.training.get("profile_steps", [])) > 0
    ):
        # Enable NVIDIA Nsight profiling
        nsight_options = OmegaConf.to_container(
            config.training.get("controller_nsight_options", {})
        )
        runner = MultiAgentTaskRunner.options(
            runtime_env={"nsight": nsight_options}
        ).remote()
    else:
        runner = MultiAgentTaskRunner.remote()
    
    # Run training
    ray.get(runner.run.remote(config))
    
    # Optional: Save Ray timeline for performance analysis
    timeline_file = config.distributed.get("timeline_json_file")
    if timeline_file:
        ray.timeline(filename=timeline_file)
        print(f"Ray timeline saved to: {timeline_file}")


@hydra.main(config_path="config", config_name="grpo_dapo_training", version_base=None)
def main(cfg: DictConfig):
    """
    Main entry point with Hydra configuration.
    
    Args:
        cfg: Hydra configuration
    """
    # Allow algorithm override from command line
    if "algorithm" in cfg:
        cfg.algorithm_type = cfg.algorithm
    elif "algorithm_type" not in cfg:
        cfg.algorithm_type = "grpo"  # Default to GRPO
    
    # Validate algorithm choice
    valid_algorithms = ["grpo", "dapo", "ppo"]
    if cfg.algorithm_type not in valid_algorithms:
        raise ValueError(
            f"Invalid algorithm: {cfg.algorithm_type}. "
            f"Must be one of {valid_algorithms}"
        )
    
    # Run training
    run_multi_agent_training(cfg)
    
    # Cleanup
    if ray.is_initialized():
        ray.shutdown()
        print("\nRay cluster shutdown complete")


if __name__ == "__main__":
    # Example usage instructions
    if len(sys.argv) == 1:
        print("\nUsage examples:")
        print("  python launch_multi_agent_training.py algorithm_type=grpo")
        print("  python launch_multi_agent_training.py algorithm_type=dapo")
        print("  python launch_multi_agent_training.py algorithm_type=grpo distributed.num_nodes=2")
        print("\nStarting with default GRPO configuration...\n")
    
    main()