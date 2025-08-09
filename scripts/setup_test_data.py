#!/usr/bin/env python3
"""
Setup reduced dataset for initial testing.

This script creates smaller datasets for SFT and RL phases to test the system
without using full-scale data that might take too long.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coding_framework.training.sft_data_preparation import SFTDataPipeline
from coding_framework.utils.config import TrainingConfig, Config, save_config
import structlog

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


async def setup_reduced_datasets():
    """Setup reduced datasets for testing."""
    logger.info("Setting up reduced datasets for testing")
    
    # Create data directory
    data_dir = Path("data/test_sft")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize SFT pipeline
    pipeline = SFTDataPipeline(data_dir=data_dir, use_huggingface_data=False)
    
    # Generate reduced datasets for testing
    logger.info("Preparing generator data (500 examples)")
    generator_data = await pipeline.prepare_generator_data(
        num_examples=500,
        save_path="generator_test.jsonl"
    )
    logger.info(f"Generator dataset: {len(generator_data)} examples")
    
    logger.info("Preparing optimizer data (300 examples)")
    optimizer_data = await pipeline.prepare_optimizer_data(
        num_examples=300,
        save_path="optimizer_test.jsonl"
    )
    logger.info(f"Optimizer dataset: {len(optimizer_data)} examples")
    
    logger.info("Preparing cross-agent data (100 conversations)")
    cross_agent_data = await pipeline.prepare_cross_agent_data(
        num_conversations=100,
        save_path="cross_agent_test.jsonl"
    )
    logger.info(f"Cross-agent dataset: {len(cross_agent_data)} conversations")
    
    return {
        "generator": generator_data,
        "optimizer": optimizer_data,
        "cross_agent": cross_agent_data
    }


def create_test_config():
    """Create reduced configuration for testing."""
    logger.info("Creating test configuration")
    
    # Create test training config with reduced parameters
    training_config = TrainingConfig(
        episodes=20,  # Reduced from default 100
        batch_size=4,  # Reduced from default 8
        learning_rate=2e-5,  # Slightly higher for faster learning with small data
        data_path="./data/test_sft",
        evaluation_data_path="./data/test_sft",
        checkpoint_dir="./checkpoints_test",
        save_interval=5,  # Save more frequently for testing
        log_interval=1,
        wandb_project="multimind-test",
        conversation={
            "max_turns": 3,  # Reduced from 5
            "discount_factor": 0.9,
            "early_termination_threshold": 0.8
        }
    )
    
    # Create overall config
    config = Config(training=training_config)
    
    # Save test config
    config_path = Path("config_test.yml")
    save_config(config, config_path)
    logger.info(f"Saved test configuration to {config_path}")
    
    return config


async def main():
    """Main setup function."""
    logger.info("🚀 Starting reduced dataset setup for testing")
    
    try:
        # Setup datasets
        datasets = await setup_reduced_datasets()
        
        # Create test configuration
        config = create_test_config()
        
        logger.info("✅ Successfully set up test data and configuration")
        logger.info("📊 Dataset summary:")
        for name, dataset in datasets.items():
            logger.info(f"  - {name}: {len(dataset)} examples")
        
        logger.info("📋 Next steps:")
        logger.info("  1. Run SFT training with reduced dataset")
        logger.info("  2. Run multi-turn RL with reduced episodes")
        logger.info("  3. Monitor and validate results")
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())