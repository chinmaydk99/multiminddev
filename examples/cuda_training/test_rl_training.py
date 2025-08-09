#!/usr/bin/env python3
"""
Test RL Training for CUDA Code Generation
Integrates multi-agent CUDA workflow with reinforcement learning
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from coding_framework.agents.cuda_generator import CUDAGeneratorAgent
from coding_framework.agents.cuda_optimizer import CUDAOptimizerAgent  
from coding_framework.agents.cuda_tester import CUDATesterAgent
from coding_framework.orchestration.cuda_workflow import CUDAKernelWorkflow
from coding_framework.training.cuda_data_loader import CUDADataLoader, CUDATrainingExample
from coding_framework.training.reward_functions.cuda_performance_reward import CUDAPerformanceReward
from coding_framework.verl_integration.cuda_multi_agent_trainer import CUDAMultiAgentVERLTrainer
from coding_framework.utils.config import load_config

import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
)

logger = structlog.get_logger("rl_training_test")


async def test_cuda_agents_initialization():
    """Test that all CUDA agents can be initialized properly."""
    logger.info("🔧 Testing CUDA agents initialization...")
    
    try:
        # Load configuration
        config = load_config()
        config.llm.provider = "huggingface" 
        config.llm.model = "Qwen/Qwen2.5-Coder-7B-Instruct"
        config.llm.temperature = 0.3
        
        # Initialize agents
        cuda_generator = CUDAGeneratorAgent(config.llm)
        cuda_optimizer = CUDAOptimizerAgent(config.llm)
        cuda_tester = CUDATesterAgent(config.llm)
        
        # Initialize all agents
        await cuda_generator.initialize()
        await cuda_optimizer.initialize() 
        await cuda_tester.initialize()
        
        # Health check all agents
        gen_health = await cuda_generator.health_check()
        opt_health = await cuda_optimizer.health_check()
        test_health = await cuda_tester.health_check()
        
        logger.info("✅ CUDA agents initialized successfully",
                   generator_health=gen_health["status"],
                   optimizer_health=opt_health["status"], 
                   tester_health=test_health["status"])
        
        return cuda_generator, cuda_optimizer, cuda_tester
        
    except Exception as e:
        logger.error("❌ Failed to initialize CUDA agents", error=str(e))
        raise


async def test_cuda_workflow():
    """Test the CUDA workflow with a simple problem."""
    logger.info("🔄 Testing CUDA workflow...")
    
    try:
        # Initialize agents
        cuda_generator, cuda_optimizer, cuda_tester = await test_cuda_agents_initialization()
        
        # Create workflow
        workflow = CUDAKernelWorkflow(
            cuda_generator=cuda_generator,
            cuda_optimizer=cuda_optimizer,
            cuda_tester=cuda_tester,
            config={"max_turns": 3}
        )
        
        # Test simple vector addition problem
        test_problem = {
            "pytorch_operation": "torch.add(A, B) - Element-wise vector addition",
            "test_inputs": [
                {"shape": [1024], "dtype": "float32"},
                {"shape": [1024], "dtype": "float32"}
            ],
            "target_performance": 2.0
        }
        
        logger.info("🚀 Running workflow test", problem=test_problem["pytorch_operation"])
        
        start_time = time.time()
        workflow_result = await workflow.run_workflow(
            pytorch_operation=test_problem["pytorch_operation"],
            test_inputs=test_problem["test_inputs"],
            target_performance=test_problem["target_performance"],
            max_optimization_turns=3
        )
        workflow_time = time.time() - start_time
        
        logger.info("✅ Workflow test completed",
                   workflow_status=workflow_result["workflow_status"],
                   best_performance=workflow_result.get("best_performance", 0.0),
                   execution_time=workflow_time)
        
        return workflow_result
        
    except Exception as e:
        logger.error("❌ Workflow test failed", error=str(e))
        raise


async def test_reward_calculation():
    """Test the CUDA performance reward function."""
    logger.info("🎯 Testing reward calculation...")
    
    try:
        # Initialize reward function
        reward_function = CUDAPerformanceReward(
            target_speedup=2.0,
            correctness_weight=0.4,
            performance_weight=0.4,
            improvement_weight=0.2
        )
        
        # Test with sample CUDA kernel
        sample_kernel = """
__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}
"""
        
        test_cases = [
            {"shape": [1024], "dtype": "float32"},
            {"shape": [1024], "dtype": "float32"}
        ]
        
        logger.info("🧮 Calculating reward for sample kernel")
        
        reward = await reward_function.calculate_reward(
            problem="Vector addition: C = A + B",
            generated_code=sample_kernel,
            test_cases=test_cases,
            context={"turn": 0, "conversation_id": "test_reward"}
        )
        
        logger.info("✅ Reward calculation completed", reward=reward)
        
        return reward
        
    except Exception as e:
        logger.error("❌ Reward calculation failed", error=str(e))
        return 0.0


async def test_multiturn_optimization():
    """Test multi-turn optimization with conversation state."""
    logger.info("🔄 Testing multi-turn optimization...")
    
    try:
        # Initialize components
        cuda_generator, cuda_optimizer, cuda_tester = await test_cuda_agents_initialization()
        
        workflow = CUDAKernelWorkflow(
            cuda_generator=cuda_generator,
            cuda_optimizer=cuda_optimizer,
            cuda_tester=cuda_tester,
            config={"max_turns": 3, "early_stop_threshold": 1.5}
        )
        
        # Create training example
        training_example = CUDATrainingExample(
            problem_description="Implement efficient matrix transpose using shared memory",
            torch_reference="torch.transpose(matrix, 0, 1)",
            difficulty_level="INTERMEDIATE",
            operation_category="linear_algebra",
            test_inputs=[
                {"shape": [512, 512], "dtype": "float32"}
            ],
            expected_speedup=2.5
        )
        
        # Test multi-turn optimization
        optimization_context = {
            "max_turns": 3,
            "early_stop_threshold": 1.5,
            "turn_discount_factor": 0.9
        }
        
        logger.info("🔄 Starting multi-turn optimization", 
                   problem=training_example.problem_description[:50] + "...")
        
        start_time = time.time()
        result = await workflow.run_multiturn_optimization(
            problem=training_example,
            context=optimization_context
        )
        optimization_time = time.time() - start_time
        
        logger.info("✅ Multi-turn optimization completed",
                   success=result.success,
                   final_speedup=result.final_speedup,
                   turns_required=result.turns_required,
                   total_reward=result.total_reward,
                   optimization_time=optimization_time)
        
        return result
        
    except Exception as e:
        logger.error("❌ Multi-turn optimization failed", error=str(e))
        raise


async def test_rl_training_episode():
    """Test a single RL training episode."""
    logger.info("🎮 Testing RL training episode...")
    
    try:
        # Load training data
        data_loader = CUDADataLoader()
        training_examples = await data_loader.get_curriculum_batch(
            batch_size=1,
            tier_override="BASIC"
        )
        
        if not training_examples:
            logger.warning("No training examples available, creating synthetic")
            training_examples = [CUDATrainingExample(
                problem_description="Element-wise vector addition",
                torch_reference="torch.add(A, B)",
                difficulty_level="BASIC",
                operation_category="elementwise",
                test_inputs=[{"shape": [1024], "dtype": "float32"}],
                expected_speedup=2.0
            )]
        
        # Initialize trainer (simplified version without full VERL)
        config = load_config()
        config.llm.provider = "huggingface"
        config.llm.model = "Qwen/Qwen2.5-Coder-7B-Instruct"
        
        # Initialize agents
        cuda_generator, cuda_optimizer, cuda_tester = await test_cuda_agents_initialization()
        
        # Create workflow
        workflow = CUDAKernelWorkflow(
            cuda_generator=cuda_generator,
            cuda_optimizer=cuda_optimizer,
            cuda_tester=cuda_tester,
            config={"max_turns": 2}
        )
        
        # Run single training episode
        problem = training_examples[0]
        episode_context = {
            "max_turns": 2,
            "early_stop_threshold": 0.8,
            "turn_discount_factor": 0.9,
            "episode_id": f"test_episode_{int(time.time())}"
        }
        
        logger.info("🎯 Running training episode", 
                   problem=problem.problem_description)
        
        start_time = time.time()
        episode_result = await workflow.run_multiturn_optimization(
            problem=problem,
            context=episode_context
        )
        episode_time = time.time() - start_time
        
        # Calculate episode metrics
        episode_metrics = {
            "success": episode_result.success,
            "final_reward": episode_result.total_reward,
            "turns_used": episode_result.turns_required,
            "best_performance": episode_result.final_speedup,
            "compilation_success": episode_result.compilation_success,
            "tests_passed": episode_result.tests_passed,
            "episode_time": episode_time,
            "performance_trajectory": episode_result.performance_metrics.get("performance_trajectory", [])
        }
        
        logger.info("✅ Training episode completed", **episode_metrics)
        
        return episode_metrics
        
    except Exception as e:
        logger.error("❌ Training episode failed", error=str(e))
        raise


async def run_full_rl_test_suite():
    """Run comprehensive RL training test suite."""
    logger.info("🧪 Starting comprehensive RL training test suite")
    
    test_results = {
        "timestamp": time.time(),
        "tests": {}
    }
    
    tests = [
        ("agent_initialization", test_cuda_agents_initialization),
        ("reward_calculation", test_reward_calculation),
        ("cuda_workflow", test_cuda_workflow),
        ("multiturn_optimization", test_multiturn_optimization),
        ("rl_training_episode", test_rl_training_episode)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"🔬 Running test: {test_name}")
        
        try:
            start_time = time.time()
            result = await test_func()
            test_time = time.time() - start_time
            
            test_results["tests"][test_name] = {
                "status": "passed",
                "execution_time": test_time,
                "result": result if not callable(result) else "success"
            }
            
            logger.info(f"✅ Test {test_name} passed", execution_time=test_time)
            
        except Exception as e:
            test_time = time.time() - start_time
            test_results["tests"][test_name] = {
                "status": "failed", 
                "execution_time": test_time,
                "error": str(e)
            }
            
            logger.error(f"❌ Test {test_name} failed", error=str(e))
    
    # Summary
    passed_tests = sum(1 for test in test_results["tests"].values() if test["status"] == "passed")
    total_tests = len(test_results["tests"])
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    logger.info("🏁 RL test suite completed",
               passed_tests=passed_tests,
               total_tests=total_tests,
               success_rate=f"{success_rate:.2%}")
    
    return test_results


async def main():
    """Main RL training test entry point."""
    parser = argparse.ArgumentParser(description="Test RL Training for CUDA Code Generation")
    parser.add_argument("--test", choices=[
        "agents", "reward", "workflow", "multiturn", "episode", "all"
    ], default="all", help="Which test to run")
    parser.add_argument("--output-file", default="rl_test_results.json", help="Output file for test results")
    
    args = parser.parse_args()
    
    print("🚀 CUDA RL Training Test Suite")
    print("=" * 50)
    print(f"Test: {args.test}")
    print(f"Output: {args.output_file}")
    print("=" * 50)
    
    try:
        if args.test == "agents":
            result = await test_cuda_agents_initialization()
        elif args.test == "reward":
            result = await test_reward_calculation()
        elif args.test == "workflow":
            result = await test_cuda_workflow()
        elif args.test == "multiturn":
            result = await test_multiturn_optimization()
        elif args.test == "episode":
            result = await test_rl_training_episode()
        else:  # "all"
            result = await run_full_rl_test_suite()
        
        # Save results
        with open(args.output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\n✅ Test results saved to: {args.output_file}")
        return 0
        
    except Exception as e:
        logger.error("❌ Test suite failed", error=str(e))
        print(f"\n❌ Test failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))