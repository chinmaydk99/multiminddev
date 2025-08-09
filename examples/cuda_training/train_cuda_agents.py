#!/usr/bin/env python3
"""
Complete example of CUDA multi-agent training.

This script demonstrates how to train CUDA-specialized agents using VERL
reinforcement learning with multi-turn conversations for kernel optimization.
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from coding_framework.agents.cuda_generator import CUDAGeneratorAgent
from coding_framework.agents.cuda_optimizer import CUDAOptimizerAgent
from coding_framework.agents.cuda_tester import CUDATesterAgent
from coding_framework.orchestration.supervisor import CodingSupervisor
from coding_framework.verl_integration.cuda_multi_agent_trainer import CUDAMultiAgentVERLTrainer
from coding_framework.utils.config import load_config
from coding_framework.utils.llm_interface import LLMInterface
import structlog


async def main():
    """Complete example of CUDA multi-agent training."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train CUDA agents with VERL")
    parser.add_argument("--episodes", type=int, default=10, help="Number of training episodes")
    parser.add_argument("--config", type=str, default="configs/cuda_ppo_training.yaml", 
                       help="Path to training configuration file")
    parser.add_argument("--quick-test", action="store_true", 
                       help="Run quick test with minimal episodes")
    parser.add_argument("--validate-cuda", action="store_true",
                       help="Only validate CUDA environment and exit")
    
    args = parser.parse_args()
    
    # Set up logging
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
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    logger = structlog.get_logger("cuda_training_example")
    
    try:
        # Load CUDA-optimized configuration
        config_path = Path(__file__).parent / args.config
        if not config_path.exists():
            # Use default configuration
            config = create_default_config()
            logger.warning("Config file not found, using default configuration")
        else:
            config = load_config(str(config_path))
        
        # Adjust episodes for quick test
        episodes = 3 if args.quick_test else args.episodes
        
        # Initialize CUDA trainer
        trainer = CUDAMultiAgentVERLTrainer(config.training if hasattr(config, 'training') else config)
        
        # Validate CUDA environment first
        logger.info("🔍 Validating CUDA environment...")
        cuda_validation = await trainer.validate_cuda_environment()
        
        if args.validate_cuda:
            print("\n=== CUDA Environment Validation ===")
            for key, value in cuda_validation.items():
                print(f"{key}: {value}")
            return
        
        if not cuda_validation["cuda_available"] and not args.quick_test:
            logger.error("CUDA not available - cannot run full training")
            print("❌ CUDA environment not available. Use --quick-test for mock training.")
            return
        
        # Create LLM interface
        llm_interface = LLMInterface(config.llm if hasattr(config, 'llm') else {})
        
        # Create specialized CUDA agents
        logger.info("🤖 Initializing CUDA agents...")
        
        cuda_generator = CUDAGeneratorAgent(
            config=config.agents.cuda_generator if hasattr(config, 'agents') else create_default_agent_config(),
            llm_interface=llm_interface,
            agent_id="cuda_gen_001"
        )
        
        cuda_optimizer = CUDAOptimizerAgent(
            config=config.agents.cuda_optimizer if hasattr(config, 'agents') else create_default_agent_config(),
            llm_interface=llm_interface,
            agent_id="cuda_opt_001"
        )
        
        cuda_tester = CUDATesterAgent(
            config=config.agents.cuda_tester if hasattr(config, 'agents') else create_default_agent_config(),
            llm_interface=llm_interface,
            agent_id="cuda_test_001"
        )
        
        # Verify agents are healthy
        logger.info("🏥 Performing agent health checks...")
        for agent_name, agent in [("generator", cuda_generator), ("optimizer", cuda_optimizer), ("tester", cuda_tester)]:
            health = await agent.health_check()
            if health["status"] != "healthy":
                logger.error(f"Agent {agent_name} health check failed", health=health)
                return
            logger.info(f"✅ Agent {agent_name} is healthy")
        
        # Start multi-agent CUDA training
        logger.info("🚀 Starting CUDA Multi-Agent Training...")
        logger.info(f"Episodes: {episodes}")
        logger.info(f"CUDA Available: {cuda_validation['cuda_available']}")
        logger.info(f"GPU Count: {cuda_validation['gpu_count']}")
        
        training_results = await trainer.train_cuda_agents(
            generator_agent=cuda_generator,
            optimizer_agent=cuda_optimizer,
            tester_agent=cuda_tester,
            episodes=episodes
        )
        
        # Display results
        print("\n" + "="*60)
        print("🎉 CUDA MULTI-AGENT TRAINING COMPLETE!")
        print("="*60)
        print(f"✅ Success: {training_results.get('success', False)}")
        print(f"📊 Training Type: {training_results.get('training_type', 'unknown')}")
        print(f"🔢 Episodes: {episodes}")
        print(f"⏱️  Total Training Time: {training_results.get('total_training_time', 0.0):.2f}s")
        
        metrics = training_results.get('metrics', {})
        print(f"🏆 Best Performance: {metrics.get('best_performance', 0.0):.2f}x speedup")
        
        if 'data_statistics' in training_results:
            stats = training_results['data_statistics']
            print(f"📚 Training Problems: {stats.get('total_problems', 0)}")
            print(f"📈 Mean Target Speedup: {stats.get('speedup_statistics', {}).get('mean', 0.0):.2f}x")
        
        if training_results.get('error'):
            print(f"❌ Error: {training_results['error']}")
        
        # Test trained agents on new problem
        if training_results.get('success', False):
            print("\n" + "="*60)
            print("🧪 TESTING TRAINED AGENTS")
            print("="*60)
            
            test_result = await test_trained_agents(
                cuda_generator, cuda_optimizer, cuda_tester
            )
            
            print(f"🎯 Test Result: {'✅ Success' if test_result['success'] else '❌ Failed'}")
            if test_result['success']:
                performance = test_result.get('performance', {})
                print(f"⚡ Generated Kernel Speedup: {performance.get('speedup', 'N/A')}")
                print(f"🔧 Optimizations Applied: {', '.join(performance.get('optimizations', []))}")
        
        print("\n🏁 Training example completed successfully!")
        
    except Exception as e:
        logger.error("Training example failed", error=str(e))
        print(f"❌ Training failed: {str(e)}")
        sys.exit(1)


async def test_trained_agents(generator, optimizer, tester):
    """Test trained agents on a new CUDA kernel generation problem."""
    
    test_problem = "Create a CUDA kernel for element-wise addition of two 1024x1024 matrices"
    test_context = {
        "cuda_mode": True, 
        "target_speedup": 3.0,
        "test_inputs": [
            {"shape": [1024, 1024], "dtype": "float32", "name": "A"},
            {"shape": [1024, 1024], "dtype": "float32", "name": "B"}
        ]
    }
    
    try:
        # Step 1: Generate initial kernel
        gen_response = await generator.process_request(test_problem, context=test_context)
        if not gen_response.success:
            return {"success": False, "error": "Kernel generation failed"}
        
        # Step 2: Optimize kernel
        opt_response = await optimizer.process_request(
            gen_response.content, 
            context={**test_context, "performance_feedback": "Initial implementation needs optimization"}
        )
        
        # Step 3: Test optimized kernel
        final_kernel = opt_response.content if opt_response.success else gen_response.content
        test_response = await tester.process_request(final_kernel, context=test_context)
        
        # Extract performance metrics
        performance = {
            "speedup": test_response.metadata.get("estimated_speedup", "Unknown"),
            "optimizations": opt_response.metadata.get("optimizations_applied", []) if opt_response.success else []
        }
        
        return {
            "success": True,
            "performance": performance,
            "kernel_generated": bool(gen_response.success),
            "kernel_optimized": bool(opt_response.success),
            "kernel_tested": bool(test_response.success)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def create_default_config():
    """Create default configuration for CUDA training."""
    
    class DefaultConfig:
        def __init__(self):
            self.llm = {
                "provider": "anthropic",
                "model": "claude-3-sonnet-20240229",
                "temperature": 0.7,
                "max_tokens": 4096
            }
            
            self.training = DefaultTrainingConfig()
    
    class DefaultTrainingConfig:
        def __init__(self):
            self.algorithm = "ppo"
            self.episodes = 10
            self.batch_size = 4
            self.learning_rate = 5e-6
            self.data_sources = ["kernelbench_local"]
            self.conversation = {
                "max_turns": 5,
                "discount_factor": 0.9,
                "early_termination_threshold": 0.8
            }
            self.cuda_rewards = {
                "target_speedup": 2.0,
                "correctness_weight": 0.4,
                "performance_weight": 0.4,
                "improvement_weight": 0.2
            }
    
    return DefaultConfig()


def create_default_agent_config():
    """Create default agent configuration."""
    
    class DefaultAgentConfig:
        def __init__(self):
            self.temperature = 0.7
            self.max_tokens = 2048
    
    return DefaultAgentConfig()


if __name__ == "__main__":
    asyncio.run(main())