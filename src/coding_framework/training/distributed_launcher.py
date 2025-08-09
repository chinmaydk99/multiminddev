"""
Distributed training launcher that integrates VERL for RL training and Ray for SFT.
This launcher properly uses VERL's distributed capabilities for multi-turn RL training.
"""

import os
import sys
import socket
from pathlib import Path
from typing import Dict, Any, Optional, List
import ray
import hydra
from omegaconf import OmegaConf, DictConfig
import torch
import structlog
from dataclasses import dataclass
import asyncio

# Add VERL to path
verl_path = Path(__file__).parent.parent.parent.parent / "verl"
sys.path.insert(0, str(verl_path))

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer, hf_processor
from verl import DataProto

from ..agents.trainable_cuda_agents import (
    TrainableCUDAGeneratorAgent,
    TrainableCUDAOptimizerAgent,
    TrainableCUDATesterAgent
)
from .multi_turn_conversation import MultiTurnConversationManager
from .sft_data_preparation import SFTDataPipeline


@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed training."""
    
    # Training mode
    mode: str = "full"  # Options: sft_only, rl_only, full
    
    # SFT configuration
    sft:
        enabled: bool = True
        num_epochs: int = 2
        batch_size: int = 64
        learning_rate: float = 5e-5
        num_workers: int = 4
        use_ray: bool = True
    
    # RL configuration (VERL)
    rl:
        algorithm: str = "grpo"  # grpo, dapo, ppo
        use_verl: bool = True
        verl_config_path: str = "config/grpo_dapo_training.yaml"
    
    # Ray cluster configuration
    ray:
        address: Optional[str] = None  # None for local
        num_cpus: int = 32
        num_gpus: int = 8
        runtime_env: Dict[str, Any] = None
    
    # Model paths
    models:
        generator_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
        optimizer_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
        use_shm: bool = True  # Use shared memory for faster loading
    
    # Distributed settings
    distributed:
        nnodes: int = 1
        n_gpus_per_node: int = 8
        fsdp_size: int = 32
        tensor_parallel_size: int = 4
        sequence_parallel_size: int = 4


class MultiAgentVERLTrainer(RayPPOTrainer):
    """
    Custom VERL trainer for multi-agent multi-turn CUDA RL training.
    Extends VERL's RayPPOTrainer to handle multiple trainable agents.
    """
    
    def __init__(self, config: DictConfig, **kwargs):
        """Initialize multi-agent VERL trainer."""
        super().__init__(config, **kwargs)
        self.logger = structlog.get_logger()
        
        # Initialize our custom agents
        self.generator_agent = None
        self.optimizer_agent = None
        self.tester_agent = None
        self.conversation_manager = None
        
    def _init_workers(self):
        """Override to initialize our multi-agent workers."""
        super()._init_workers()
        
        # Initialize our trainable agents as VERL workers
        self._init_multi_agent_workers()
    
    def _init_multi_agent_workers(self):
        """Initialize multi-agent workers for VERL training."""
        self.logger.info("Initializing multi-agent workers for VERL")
        
        # Create worker groups for each agent
        self.generator_worker_group = self._create_agent_worker_group("generator")
        self.optimizer_worker_group = self._create_agent_worker_group("optimizer")
        
        # Tester is rule-based, no worker group needed
        self.tester_agent = TrainableCUDATesterAgent(use_trained_model=False)
    
    def _create_agent_worker_group(self, agent_type: str):
        """Create VERL worker group for an agent."""
        # Configure worker group based on agent type
        worker_config = {
            "model_path": self.config.models[f"{agent_type}_path"],
            "strategy": "fsdp2",
            "fsdp_config": self.config.actor_rollout_ref.actor.fsdp_config,
            "n_gpus_per_node": self.config.trainer.n_gpus_per_node,
            "nnodes": self.config.trainer.nnodes,
        }
        
        # Create RayWorkerGroup
        worker_group = RayWorkerGroup(
            actor_cls=ActorRolloutRefWorker,
            num_workers=self.config.trainer.nnodes,
            init_kwargs=worker_config,
            resource_specs={"num_gpus": self.config.trainer.n_gpus_per_node}
        )
        
        return worker_group
    
    def fit(self):
        """Override fit to implement multi-agent multi-turn training."""
        self.logger.info("Starting multi-agent VERL training")
        
        # Initialize conversation manager
        self.conversation_manager = MultiTurnConversationManager(
            generator_agent=self.generator_worker_group,
            optimizer_agent=self.optimizer_worker_group,
            tester_agent=self.tester_agent,
            max_turns=self.config.multi_turn.max_turns
        )
        
        # Run training loop with VERL's distributed capabilities
        super().fit()


@ray.remote(num_cpus=1)
class SFTTrainingWorker:
    """Ray worker for distributed SFT training."""
    
    def __init__(self, agent_type: str, model_path: str, device_id: int):
        """Initialize SFT training worker."""
        self.agent_type = agent_type
        self.model_path = model_path
        self.device_id = device_id
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.logger = structlog.get_logger()
        
        # Initialize agent
        if agent_type == "generator":
            self.agent = TrainableCUDAGeneratorAgent(
                model_name=model_path,
                device=self.device
            )
        elif agent_type == "optimizer":
            self.agent = TrainableCUDAOptimizerAgent(
                model_name=model_path,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def train_batch(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """Train on a batch of data."""
        metrics = {}
        
        for item in batch_data["items"]:
            step_metrics = self.agent.sft_train_step(
                input_text=item["input"],
                target_text=item["output"]
            )
            
            for key, value in step_metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
        
        # Average metrics
        avg_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
        return avg_metrics
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save agent checkpoint."""
        self.agent.save_checkpoint(checkpoint_path)


class DistributedTrainingLauncher:
    """
    Main launcher for distributed training with VERL for RL and Ray for SFT.
    """
    
    def __init__(self, config: DistributedTrainingConfig):
        """Initialize distributed training launcher."""
        self.config = config
        self.logger = structlog.get_logger()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            self._init_ray()
    
    def _init_ray(self):
        """Initialize Ray cluster."""
        runtime_env = self.config.ray.runtime_env or {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "false",
                "NCCL_DEBUG": "ERROR",
                "VLLM_LOGGING_LEVEL": "ERROR",
            }
        }
        
        if self.config.ray.address:
            ray.init(address=self.config.ray.address, runtime_env=runtime_env)
        else:
            ray.init(
                num_cpus=self.config.ray.num_cpus,
                num_gpus=self.config.ray.num_gpus,
                runtime_env=runtime_env
            )
        
        self.logger.info(
            f"Ray initialized on {socket.gethostname()}",
            resources=ray.available_resources()
        )
    
    async def run_sft_training(self):
        """Run distributed SFT training using Ray."""
        self.logger.info("Starting distributed SFT training with Ray")
        
        # Create SFT data pipeline
        data_pipeline = SFTDataPipeline()
        
        # Prepare datasets
        generator_dataset = await data_pipeline.prepare_generator_data(10000)
        optimizer_dataset = await data_pipeline.prepare_optimizer_data(8000)
        
        # Create Ray workers for distributed SFT
        num_workers = min(self.config.sft.num_workers, self.config.ray.num_gpus)
        
        # Train generator agents
        generator_workers = [
            SFTTrainingWorker.remote(
                agent_type="generator",
                model_path=self.config.models.generator_path,
                device_id=i % self.config.ray.num_gpus
            )
            for i in range(num_workers)
        ]
        
        # Distribute data and train
        batch_size = self.config.sft.batch_size
        for epoch in range(self.config.sft.num_epochs):
            # Split data among workers
            data_chunks = self._split_data(generator_dataset, num_workers)
            
            # Train in parallel
            futures = []
            for worker, chunk in zip(generator_workers, data_chunks):
                futures.append(worker.train_batch.remote({"items": chunk}))
            
            # Collect results
            results = ray.get(futures)
            avg_loss = sum(r.get("sft_loss", 0) for r in results) / len(results)
            self.logger.info(f"Generator SFT Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
        
        # Train optimizer agents (similar process)
        optimizer_workers = [
            SFTTrainingWorker.remote(
                agent_type="optimizer",
                model_path=self.config.models.optimizer_path,
                device_id=i % self.config.ray.num_gpus
            )
            for i in range(num_workers)
        ]
        
        for epoch in range(self.config.sft.num_epochs):
            data_chunks = self._split_data(optimizer_dataset, num_workers)
            futures = []
            for worker, chunk in zip(optimizer_workers, data_chunks):
                futures.append(worker.train_batch.remote({"items": chunk}))
            
            results = ray.get(futures)
            avg_loss = sum(r.get("sft_loss", 0) for r in results) / len(results)
            self.logger.info(f"Optimizer SFT Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoints
        checkpoint_dir = Path("checkpoints/sft")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for i, worker in enumerate(generator_workers):
            ray.get(worker.save_checkpoint.remote(str(checkpoint_dir / f"generator_{i}")))
        
        for i, worker in enumerate(optimizer_workers):
            ray.get(worker.save_checkpoint.remote(str(checkpoint_dir / f"optimizer_{i}")))
        
        self.logger.info("SFT training complete")
    
    def run_verl_rl_training(self):
        """Run distributed RL training using VERL."""
        self.logger.info("Starting distributed RL training with VERL")
        
        # Load VERL configuration
        verl_config = OmegaConf.load(self.config.rl.verl_config_path)
        
        # Update config with our multi-agent settings
        verl_config.algorithm.adv_estimator = self.config.rl.algorithm
        verl_config.trainer.n_gpus_per_node = self.config.distributed.n_gpus_per_node
        verl_config.trainer.nnodes = self.config.distributed.nnodes
        
        # Set model paths
        verl_config.actor_rollout_ref.model.path = self.config.models.generator_path
        
        # Create and run VERL trainer
        trainer = MultiAgentVERLTrainer(verl_config)
        trainer.fit()
        
        self.logger.info("VERL RL training complete")
    
    async def run(self):
        """Run the complete distributed training pipeline."""
        self.logger.info(
            f"Starting distributed training in {self.config.mode} mode",
            hostname=socket.gethostname(),
            pid=os.getpid()
        )
        
        try:
            # Run SFT if enabled
            if self.config.mode in ["full", "sft_only"] and self.config.sft.enabled:
                await self.run_sft_training()
            
            # Run RL if enabled
            if self.config.mode in ["full", "rl_only"] and self.config.rl.use_verl:
                self.run_verl_rl_training()
            
            self.logger.info("Distributed training complete")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        
        finally:
            # Cleanup
            if ray.is_initialized():
                ray.shutdown()
    
    def _split_data(self, dataset, num_workers):
        """Split dataset among workers."""
        chunk_size = len(dataset) // num_workers
        chunks = []
        
        for i in range(num_workers):
            start = i * chunk_size
            end = start + chunk_size if i < num_workers - 1 else len(dataset)
            chunks.append(dataset[start:end])
        
        return chunks


@hydra.main(config_path="../../../config", config_name="grpo_dapo_training", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for distributed training."""
    
    # Convert Hydra config to our config
    config = DistributedTrainingConfig()
    
    # Update config from Hydra
    if "algorithm" in cfg:
        config.rl.algorithm = cfg.algorithm.name
    
    if "models" in cfg:
        config.models.generator_path = cfg.models.generator.path
        config.models.optimizer_path = cfg.models.optimizer.path
    
    if "distributed" in cfg:
        config.distributed.nnodes = cfg.distributed.num_nodes
        config.distributed.n_gpus_per_node = cfg.distributed.num_gpus_per_node
    
    # Create launcher
    launcher = DistributedTrainingLauncher(config)
    
    # Run training
    asyncio.run(launcher.run())


if __name__ == "__main__":
    main()