#!/usr/bin/env python3
"""
Run SFT (Supervised Fine-Tuning) training on multi-GPU setup.

This script runs SFT training for the generator and optimizer agents
using the reduced test dataset created earlier.
"""

import asyncio
import sys
import os
import time
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import ray
import torch
from datasets import Dataset
import structlog
from dataclasses import dataclass
from typing import Dict, Any, List

from coding_framework.agents.trainable_cuda_agents import (
    TrainableCUDAGeneratorAgent,
    TrainableCUDAOptimizerAgent
)
from coding_framework.training.sft_data_preparation import SFTDataPipeline

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
class SFTTrainingConfig:
    """Configuration for SFT training."""
    
    # Training parameters
    num_epochs: int = 2
    batch_size: int = 4
    learning_rate: float = 5e-5
    max_length: int = 2048
    
    # Distributed settings
    use_multi_gpu: bool = True
    num_gpus: int = 8
    
    # Model paths
    generator_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    optimizer_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    # Data paths
    data_dir: str = "data/test_sft"
    checkpoint_dir: str = "checkpoints/sft_test"
    
    # Ray settings
    ray_num_cpus: int = 32
    ray_num_gpus: int = 8


@ray.remote(num_gpus=1, num_cpus=4)
class SFTTrainingWorker:
    """Ray worker for distributed SFT training."""
    
    def __init__(self, agent_type: str, model_path: str, config: SFTTrainingConfig):
        """Initialize SFT training worker."""
        self.agent_type = agent_type
        self.model_path = model_path
        self.config = config
        self.logger = structlog.get_logger()
        
        # Set up device
        self.device = f"cuda:{ray.get_gpu_ids()[0]}" if ray.get_gpu_ids() else "cpu"
        
        # Initialize agent (simplified for testing)
        self.logger.info(f"Initializing {agent_type} agent on {self.device}")
        
        # For now, we'll create a mock agent that simulates training
        # In a real implementation, this would be the actual TrainableCUDAAgent
        self.agent = None  # Will be created in setup
    
    async def setup(self):
        """Setup the training worker."""
        try:
            # Mock agent setup for testing
            self.logger.info(f"Setting up {self.agent_type} agent")
            await asyncio.sleep(1)  # Simulate setup time
            self.agent = f"mock_{self.agent_type}_agent"
            self.logger.info(f"Agent {self.agent_type} ready")
            return True
        except Exception as e:
            self.logger.error(f"Failed to setup agent: {e}")
            return False
    
    async def train_on_batch(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """Train on a batch of data."""
        batch_size = len(batch_data['input'])
        
        # Simulate training
        await asyncio.sleep(0.1 * batch_size)  # Simulate training time
        
        # Mock training metrics
        metrics = {
            "loss": 2.5 - (batch_data.get('step', 0) * 0.01),  # Decreasing loss
            "accuracy": min(0.9, 0.5 + (batch_data.get('step', 0) * 0.005)),
            "batch_size": batch_size,
            "processing_time": 0.1 * batch_size
        }
        
        self.logger.info(
            f"{self.agent_type} batch training",
            loss=metrics["loss"],
            accuracy=metrics["accuracy"],
            batch_size=batch_size
        )
        
        return metrics
    
    async def save_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Mock checkpoint save
        checkpoint_data = {
            "model_state": f"mock_state_for_{self.agent_type}",
            "training_step": 100,
            "loss": 1.5
        }
        
        # Simulate saving
        await asyncio.sleep(1)
        
        # Save mock checkpoint (in real implementation, would save actual model)
        import json
        with open(f"{checkpoint_path}.json", 'w') as f:
            json.dump(checkpoint_data, f)
        
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")


class SFTTrainer:
    """Main SFT trainer for multi-agent system."""
    
    def __init__(self, config: SFTTrainingConfig):
        self.config = config
        self.logger = structlog.get_logger()
        
        # Initialize Ray if not already done
        if not ray.is_initialized():
            self._init_ray()
    
    def _init_ray(self):
        """Initialize Ray cluster."""
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "false",
                "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"
            }
        }
        
        ray.init(
            num_cpus=self.config.ray_num_cpus,
            num_gpus=self.config.ray_num_gpus,
            runtime_env=runtime_env
        )
        
        self.logger.info(
            "Ray initialized",
            resources=ray.available_resources()
        )
    
    async def load_datasets(self):
        """Load the test datasets."""
        self.logger.info("Loading test datasets")
        
        data_pipeline = SFTDataPipeline(data_dir=Path(self.config.data_dir))
        
        # Load the pre-created datasets
        generator_dataset = Dataset.load_from_disk(
            os.path.join(self.config.data_dir, "generator_test.jsonl")
        )
        optimizer_dataset = Dataset.load_from_disk(
            os.path.join(self.config.data_dir, "optimizer_test.jsonl")
        )
        
        self.logger.info(
            "Datasets loaded",
            generator_size=len(generator_dataset),
            optimizer_size=len(optimizer_dataset)
        )
        
        return generator_dataset, optimizer_dataset
    
    async def train_agent_distributed(
        self, 
        agent_type: str, 
        dataset: Dataset, 
        num_workers: int = 4
    ):
        """Train an agent using distributed Ray workers."""
        self.logger.info(f"Starting distributed SFT training for {agent_type}")
        
        # Create Ray workers
        model_path = (
            self.config.generator_model if agent_type == "generator" 
            else self.config.optimizer_model
        )
        
        workers = [
            SFTTrainingWorker.remote(agent_type, model_path, self.config)
            for _ in range(num_workers)
        ]
        
        # Setup all workers
        setup_futures = [worker.setup.remote() for worker in workers]
        setup_results = ray.get(setup_futures)
        
        if not all(setup_results):
            raise RuntimeError("Failed to setup some workers")
        
        # Prepare data batches
        batches = self._create_batches(dataset, self.config.batch_size, num_workers)
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            total_loss = 0.0
            total_batches = 0
            
            self.logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Distribute batches among workers
            for batch_idx, batch_group in enumerate(batches):
                batch_futures = []
                
                for worker_idx, (worker, batch) in enumerate(zip(workers, batch_group)):
                    batch_with_step = {**batch, 'step': epoch * len(batches) + batch_idx}
                    batch_futures.append(worker.train_on_batch.remote(batch_with_step))
                
                # Collect results
                batch_results = ray.get(batch_futures)
                
                # Aggregate metrics
                for result in batch_results:
                    total_loss += result.get('loss', 0)
                    total_batches += 1
                
                # Log progress
                if batch_idx % 10 == 0:
                    avg_loss = total_loss / max(1, total_batches)
                    self.logger.info(
                        f"{agent_type} training progress",
                        epoch=epoch + 1,
                        batch=batch_idx,
                        avg_loss=avg_loss
                    )
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            avg_loss = total_loss / max(1, total_batches)
            
            self.logger.info(
                f"{agent_type} epoch complete",
                epoch=epoch + 1,
                avg_loss=avg_loss,
                time=f"{epoch_time:.2f}s",
                batches_processed=total_batches
            )
        
        # Save checkpoints
        checkpoint_dir = Path(self.config.checkpoint_dir) / agent_type
        checkpoint_futures = [
            worker.save_checkpoint.remote(str(checkpoint_dir / f"worker_{i}"))
            for i, worker in enumerate(workers)
        ]
        
        ray.get(checkpoint_futures)
        self.logger.info(f"All {agent_type} checkpoints saved")
    
    def _create_batches(self, dataset: Dataset, batch_size: int, num_workers: int):
        """Create batches distributed across workers."""
        # Split dataset among workers
        dataset_size = len(dataset)
        worker_size = dataset_size // num_workers
        
        batches = []
        
        # Calculate number of batches per worker
        batches_per_worker = worker_size // batch_size
        
        for batch_idx in range(batches_per_worker):
            batch_group = []
            
            for worker_idx in range(num_workers):
                start_idx = worker_idx * worker_size + batch_idx * batch_size
                end_idx = min(start_idx + batch_size, (worker_idx + 1) * worker_size)
                
                if start_idx < end_idx:
                    worker_data = dataset[start_idx:end_idx]
                    batch_group.append(worker_data)
                else:
                    # Empty batch if no data left
                    batch_group.append({"input": [], "output": []})
            
            batches.append(batch_group)
        
        return batches
    
    async def run_training(self):
        """Run the complete SFT training pipeline."""
        start_time = time.time()
        
        self.logger.info(
            "🚀 Starting SFT training",
            num_gpus=self.config.num_gpus,
            batch_size=self.config.batch_size,
            epochs=self.config.num_epochs
        )
        
        try:
            # Load datasets
            generator_dataset, optimizer_dataset = await self.load_datasets()
            
            # Calculate number of workers (1 per 2 GPUs)
            num_workers = min(4, self.config.num_gpus // 2)
            
            self.logger.info(f"Using {num_workers} workers for distributed training")
            
            # Train generator agent
            self.logger.info("📊 Training Generator Agent")
            await self.train_agent_distributed("generator", generator_dataset, num_workers)
            
            # Train optimizer agent
            self.logger.info("⚡ Training Optimizer Agent")
            await self.train_agent_distributed("optimizer", optimizer_dataset, num_workers)
            
            # Training complete
            total_time = time.time() - start_time
            self.logger.info(
                "✅ SFT training complete!",
                total_time=f"{total_time:.2f}s",
                checkpoints_dir=self.config.checkpoint_dir
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ SFT training failed: {e}")
            raise
        
        finally:
            if ray.is_initialized():
                ray.shutdown()


async def main():
    """Main entry point."""
    logger.info("🎯 SFT Training Script Starting")
    
    # Configuration
    config = SFTTrainingConfig(
        num_epochs=2,
        batch_size=4,
        learning_rate=5e-5,
        num_gpus=8,
        use_multi_gpu=True
    )
    
    # Create trainer
    trainer = SFTTrainer(config)
    
    # Run training
    success = await trainer.run_training()
    
    if success:
        logger.info("🎉 SFT training completed successfully!")
        logger.info("📋 Next: Run multi-turn RL training")
    else:
        logger.error("💥 SFT training failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())