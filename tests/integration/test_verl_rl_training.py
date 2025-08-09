#!/usr/bin/env python3
"""
Test script for VERL-based distributed RL training
This properly uses VERL's infrastructure for multi-turn RL training
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment for Ray
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"

import ray
import structlog
from coding_framework.verl_integration.verl_rl_trainer import (
    VERLMultiTurnRLTrainer,
    VERLTrainingConfig,
    launch_verl_training
)
from coding_framework.agents.cuda_generator import CUDAGeneratorAgent
from coding_framework.agents.cuda_optimizer import CUDAOptimizerAgent
from coding_framework.agents.cuda_tester import CUDATesterAgent
from coding_framework.training.cuda_data_loader import CUDADataLoader, CUDATrainingExample
from coding_framework.utils.config import load_config, AgentConfig
from coding_framework.utils.llm_interface import LLMInterface

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
)

logger = structlog.get_logger("verl_rl_test")


async def initialize_agents():
    """Initialize CUDA agents for training"""
    
    # Load configuration
    config = load_config()
    config.llm.provider = "huggingface"
    config.llm.model = "bigcode/starcoder2-3b"  # Using smaller model for testing
    
    # Create LLM interface
    llm_interface = LLMInterface(config.llm)
    await llm_interface.initialize()
    
    # Create agent configs
    agent_config = AgentConfig(
        max_retries=2,
        timeout=60,
        enable_logging=True
    )
    
    # Initialize agents
    cuda_generator = CUDAGeneratorAgent(agent_config, llm_interface)
    cuda_optimizer = CUDAOptimizerAgent(agent_config, llm_interface)
    cuda_tester = CUDATesterAgent(agent_config, llm_interface)
    
    logger.info("Agents initialized successfully")
    
    return {
        "generator": cuda_generator,
        "optimizer": cuda_optimizer,
        "tester": cuda_tester
    }


async def prepare_training_data():
    """Prepare CUDA training examples"""
    
    data_loader = CUDADataLoader()
    
    # Create some training examples
    training_examples = [
        {
            "description": "Element-wise vector addition",
            "prompt": "Generate a CUDA kernel for element-wise vector addition: C = A + B",
            "torch_reference": "torch.add(A, B)",
            "test_cases": [
                {"shape": [1024], "dtype": "float32"},
                {"shape": [2048], "dtype": "float32"}
            ],
            "target_performance": 2.0
        },
        {
            "description": "Matrix multiplication",
            "prompt": "Generate a CUDA kernel for matrix multiplication: C = A @ B",
            "torch_reference": "torch.matmul(A, B)",
            "test_cases": [
                {"shape": [512, 512], "dtype": "float32"},
                {"shape": [1024, 1024], "dtype": "float32"}
            ],
            "target_performance": 3.0
        },
        {
            "description": "Softmax operation",
            "prompt": "Generate a CUDA kernel for softmax operation along dimension 1",
            "torch_reference": "torch.softmax(x, dim=1)",
            "test_cases": [
                {"shape": [256, 128], "dtype": "float32"},
                {"shape": [512, 256], "dtype": "float32"}
            ],
            "target_performance": 2.5
        }
    ]
    
    logger.info(f"Prepared {len(training_examples)} training examples")
    return training_examples


async def test_verl_distributed_setup():
    """Test VERL distributed setup with Ray"""
    
    logger.info("Testing VERL distributed setup...")
    
    # Initialize Ray cluster
    if not ray.is_initialized():
        ray.init(num_gpus=8, num_cpus=32)
    
    # Check available resources
    resources = ray.cluster_resources()
    logger.info(
        "Ray cluster resources",
        gpus=resources.get("GPU", 0),
        cpus=resources.get("CPU", 0),
        nodes=len(ray.nodes())
    )
    
    # Test VERL configuration
    verl_config = VERLTrainingConfig(
        num_gpus=int(resources.get("GPU", 8)),
        num_rollout_workers=4,
        num_actor_workers=2,
        num_critic_workers=2,
        max_turns=3,
        mini_batch_size=2,
        ppo_epochs=2
    )
    
    logger.info("VERL configuration created", config=verl_config)
    
    # Test trainer initialization
    trainer = VERLMultiTurnRLTrainer(verl_config)
    trainer.setup_verl_workers()
    
    logger.info("VERL workers setup completed")
    
    # Cleanup
    trainer.cleanup()
    
    return True


async def test_verl_training_loop():
    """Test complete VERL training loop"""
    
    logger.info("Starting VERL training loop test...")
    
    try:
        # Initialize agents
        agents = await initialize_agents()
        
        # Prepare training data
        training_data = await prepare_training_data()
        
        # Configure VERL training
        verl_config = VERLTrainingConfig(
            num_gpus=8,
            num_rollout_workers=4,
            num_actor_workers=2,
            num_critic_workers=2,
            max_turns=3,
            mini_batch_size=2,
            ppo_epochs=2,
            learning_rate=1e-5,
            target_speedup=2.0
        )
        
        logger.info("Launching VERL training...")
        
        # Launch VERL training
        results = await launch_verl_training(
            agents=agents,
            training_data=training_data,
            config=verl_config
        )
        
        logger.info("VERL training completed", results=results)
        
        return results
        
    except Exception as e:
        logger.error(f"VERL training test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


async def main():
    """Main test runner"""
    
    print("\n" + "="*60)
    print("VERL-BASED DISTRIBUTED RL TRAINING TEST")
    print("="*60 + "\n")
    
    tests = [
        ("VERL Distributed Setup", test_verl_distributed_setup),
        ("VERL Training Loop", test_verl_training_loop)
    ]
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        print("-" * 40)
        
        try:
            result = await test_func()
            print(f"✓ {test_name} PASSED")
            
        except Exception as e:
            print(f"✗ {test_name} FAILED: {e}")
            
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())