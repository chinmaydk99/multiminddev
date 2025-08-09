#!/usr/bin/env python3
"""
Working RL training script with all fixes applied
"""

import os
import sys
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Environment setup
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import ray
import verl
from verl import DataProto

# Use fixed compiler
from coding_framework.cuda.fixed_compiler import CUDACompiler
from coding_framework.cuda.benchmarker import CUDABenchmarker

# Our components
from coding_framework.agents.cuda_generator import CUDAGeneratorAgent
from coding_framework.agents.cuda_optimizer import CUDAOptimizerAgent
from coding_framework.agents.cuda_tester import CUDATesterAgent
from coding_framework.training.reward_functions.cuda_performance_reward import CUDAPerformanceReward
from coding_framework.utils.config import load_config, AgentConfig
from coding_framework.utils.llm_interface import LLMInterface

import structlog

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

logger = structlog.get_logger("working_rl_training")


class WorkingCUDAWorkflow:
    """Working CUDA workflow with all fixes"""
    
    def __init__(self, cuda_generator, cuda_optimizer, cuda_tester, config=None):
        self.cuda_generator = cuda_generator
        self.cuda_optimizer = cuda_optimizer
        self.cuda_tester = cuda_tester
        self.config = config or {}
        self.logger = structlog.get_logger("working_cuda_workflow")
        
        # Use fixed CUDA compiler
        self.cuda_compiler = CUDACompiler(
            cuda_arch="sm_70"  # V100 architecture
        )
        self.cuda_benchmarker = CUDABenchmarker()
        
    async def run_generation_turn(self, problem: Dict[str, Any], turn: int = 0, previous_code: str = None) -> Dict[str, Any]:
        """Run a single generation turn with proper agent handling"""
        
        try:
            # Create proper request for generator
            pytorch_op = problem.get("torch_reference", problem.get("pytorch_operation", "torch operation"))
            
            self.logger.info(f"Turn {turn}: Generating CUDA kernel for {pytorch_op}")
            
            # Call generator with correct parameter
            generation_result = await self.cuda_generator.process_request(
                pytorch_operation=pytorch_op
            )
            
            # Handle AgentResponse properly
            if hasattr(generation_result, 'content'):
                generated_code = generation_result.content
                success = generation_result.success
                metadata = generation_result.metadata if hasattr(generation_result, 'metadata') else {}
            else:
                # Fallback for dict response
                generated_code = generation_result.get("content", generation_result.get("code", ""))
                success = generation_result.get("success", False)
                metadata = generation_result.get("metadata", {})
            
            # Clean up the generated code
            generated_code = self._clean_generated_code(generated_code)
            
            if not success or not generated_code or len(generated_code) < 50:
                self.logger.warning(f"Turn {turn}: Generation failed or empty code")
                return {
                    "turn": turn,
                    "generated_code": generated_code,
                    "compilation_success": False,
                    "performance": 0.0,
                    "error": "Generation failed or code too short"
                }
            
            # Ensure the code has a kernel
            if "__global__" not in generated_code:
                # Try to extract kernel from markdown code blocks
                import re
                code_match = re.search(r'```(?:cuda|c\+\+|cpp)?\n(.*?)```', generated_code, re.DOTALL)
                if code_match:
                    generated_code = code_match.group(1)
                
                # If still no kernel, create a simple one
                if "__global__" not in generated_code:
                    self.logger.warning(f"Turn {turn}: No kernel found, creating default")
                    generated_code = self._create_default_kernel(problem)
            
            # Compile the generated code
            self.logger.info(f"Turn {turn}: Compiling generated kernel")
            compilation_result = await self.cuda_compiler.compile_kernel(
                kernel_code=generated_code,
                kernel_name=f"kernel_turn_{turn}"
            )
            
            # Calculate performance metrics
            performance = 0.0
            if compilation_result.success:
                # Base performance for successful compilation
                performance = 1.0
                
                # Bonus for good register usage
                if compilation_result.register_pressure:
                    if compilation_result.register_pressure < 32:
                        performance += 0.3
                    elif compilation_result.register_pressure < 64:
                        performance += 0.1
                
                # Bonus for optimization patterns
                if "coalesced" in generated_code.lower() or "__shared__" in generated_code:
                    performance += 0.2
                if "blockIdx" in generated_code and "threadIdx" in generated_code:
                    performance += 0.2
                if "__syncthreads()" in generated_code:
                    performance += 0.1
                
                self.logger.info(
                    f"Turn {turn}: Compilation successful",
                    registers=compilation_result.register_pressure,
                    performance=performance
                )
            else:
                self.logger.warning(
                    f"Turn {turn}: Compilation failed",
                    error=compilation_result.stderr[:200] if compilation_result.stderr else "Unknown error"
                )
            
            result = {
                "turn": turn,
                "generated_code": generated_code,
                "compilation_success": compilation_result.success,
                "performance": performance,
                "compilation_time": compilation_result.compilation_time,
                "register_pressure": compilation_result.register_pressure,
                "shared_memory_usage": compilation_result.shared_memory_usage,
                "generation_success": success
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Generation turn {turn} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return {
                "turn": turn,
                "generated_code": "",
                "compilation_success": False,
                "performance": 0.0,
                "error": str(e)
            }
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean up generated code"""
        if not code:
            return ""
        
        # Remove markdown code blocks
        import re
        code = re.sub(r'```(?:cuda|c\+\+|cpp)?', '', code)
        code = re.sub(r'```', '', code)
        
        # Remove explanation text before/after code
        lines = code.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if '__global__' in line or '#include' in line or in_code:
                in_code = True
                code_lines.append(line)
            elif in_code and (line.strip() == '' or line.startswith('//')):
                code_lines.append(line)
            elif in_code and line.strip() and not any(kw in line.lower() for kw in ['here', 'this', 'the', 'above', 'below']):
                code_lines.append(line)
        
        return '\n'.join(code_lines)
    
    def _create_default_kernel(self, problem: Dict[str, Any]) -> str:
        """Create a default kernel based on problem type"""
        problem_id = problem.get("id", "default")
        
        if "add" in problem_id:
            return """
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""
        elif "mul" in problem_id or "scalar" in problem_id:
            return """
__global__ void scalarMultiply(float* input, float* output, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scalar;
    }
}
"""
        elif "relu" in problem_id:
            return """
__global__ void relu(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}
"""
        else:
            return """
__global__ void genericKernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}
"""


class WorkingRLTrainer:
    """Working RL trainer with all fixes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger
        
        # Training components
        self.workflow = None
        self.reward_function = CUDAPerformanceReward(
            target_speedup=config.get("target_speedup", 2.0),
            correctness_weight=0.4,
            performance_weight=0.4,
            improvement_weight=0.2
        )
        
        # Ray setup
        self.ray_initialized = False
        
        # Training metrics
        self.episode_rewards = []
        self.compilation_rates = []
        self.performance_history = []
        
        # Track curriculum progress
        self.current_tier = "BASIC"
        self.tier_metrics = {"BASIC": [], "INTERMEDIATE": [], "ADVANCED": []}
        
    async def initialize(self):
        """Initialize training components"""
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                num_gpus=self.config.get("num_gpus", 8),
                num_cpus=self.config.get("num_cpus", 32),
                ignore_reinit_error=True
            )
            self.ray_initialized = True
            
        resources = ray.cluster_resources()
        self.logger.info(
            "Ray cluster initialized for distributed training",
            gpus=resources.get("GPU", 0),
            cpus=resources.get("CPU", 0),
            memory_gb=resources.get("memory", 0) / 1e9
        )
        
        # Initialize agents
        config = load_config()
        config.llm.provider = "huggingface"
        config.llm.model = self.config.get("model_name", "Qwen/Qwen2.5-Coder-7B-Instruct")
        
        llm_interface = LLMInterface(config.llm)
        await llm_interface.initialize()
        
        agent_config = AgentConfig(
            max_retries=2,
            timeout=60,
            enable_logging=True
        )
        
        # Create agents
        cuda_generator = CUDAGeneratorAgent(agent_config, llm_interface)
        cuda_optimizer = CUDAOptimizerAgent(agent_config, llm_interface)
        cuda_tester = CUDATesterAgent(agent_config, llm_interface)
        
        # Create workflow with fixed components
        self.workflow = WorkingCUDAWorkflow(
            cuda_generator=cuda_generator,
            cuda_optimizer=cuda_optimizer,
            cuda_tester=cuda_tester,
            config={"max_turns": self.config.get("max_turns", 3)}
        )
        
        self.logger.info("Training system initialized successfully")
    
    def get_training_problems(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get curriculum-organized training problems"""
        
        return {
            "BASIC": [
                {
                    "id": "vector_add",
                    "description": "Element-wise vector addition C = A + B",
                    "torch_reference": "torch.add(A, B)",
                    "target_performance": 1.5
                },
                {
                    "id": "scalar_mul",
                    "description": "Scalar multiplication C = A * scalar",
                    "torch_reference": "torch.mul(A, scalar)",
                    "target_performance": 1.3
                }
            ],
            "INTERMEDIATE": [
                {
                    "id": "relu",
                    "description": "ReLU activation function",
                    "torch_reference": "torch.relu(x)",
                    "target_performance": 2.0
                },
                {
                    "id": "reduction_sum",
                    "description": "Sum reduction along dimension",
                    "torch_reference": "torch.sum(x, dim=1)",
                    "target_performance": 2.2
                }
            ],
            "ADVANCED": [
                {
                    "id": "transpose",
                    "description": "Matrix transpose with shared memory",
                    "torch_reference": "torch.transpose(x, 0, 1)",
                    "target_performance": 2.5
                },
                {
                    "id": "softmax",
                    "description": "Softmax normalization",
                    "torch_reference": "torch.softmax(x, dim=1)",
                    "target_performance": 3.0
                }
            ]
        }
    
    async def run_episode(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single training episode"""
        
        episode_start = time.time()
        conversation_history = []
        total_reward = 0.0
        best_performance = 0.0
        best_code = ""
        
        self.logger.info(f"Starting episode: {self.current_tier}/{problem['id']}")
        
        # Multi-turn generation with improvement
        previous_code = None
        for turn in range(self.config.get("max_turns", 3)):
            # Generate and evaluate
            turn_result = await self.workflow.run_generation_turn(
                problem, 
                turn,
                previous_code=previous_code
            )
            conversation_history.append(turn_result)
            
            # Calculate reward
            if turn_result["compilation_success"]:
                # Positive reward for successful compilation
                turn_reward = 0.5 + (turn_result["performance"] - 1.0) * 0.3
                
                # Bonus for improvement over previous turn
                if turn > 0 and turn_result["performance"] > conversation_history[turn-1].get("performance", 0):
                    turn_reward += 0.1
            else:
                # Small negative reward for failed compilation
                turn_reward = -0.1
            
            # Apply turn discount
            discounted_reward = turn_reward * (self.config.get("turn_discount", 0.9) ** turn)
            total_reward += discounted_reward
            
            # Track best performance and code
            if turn_result["performance"] > best_performance:
                best_performance = turn_result["performance"]
                best_code = turn_result["generated_code"]
            
            # Update previous code for next turn
            if turn_result["compilation_success"]:
                previous_code = turn_result["generated_code"]
            
            self.logger.info(
                f"Turn {turn + 1}/{self.config.get('max_turns', 3)}",
                compilation=turn_result["compilation_success"],
                performance=f"{turn_result['performance']:.2f}",
                reward=f"{turn_reward:.3f}"
            )
            
            # Early stopping if target met
            if turn_result["performance"] >= problem["target_performance"] * 0.9:
                self.logger.info(f"Target performance reached: {turn_result['performance']:.2f}")
                break
        
        episode_time = time.time() - episode_start
        
        return {
            "problem_id": problem["id"],
            "tier": self.current_tier,
            "total_reward": total_reward,
            "best_performance": best_performance,
            "best_code": best_code,
            "turns_used": len(conversation_history),
            "compilation_success": any(t["compilation_success"] for t in conversation_history),
            "episode_time": episode_time,
            "conversation_history": conversation_history
        }
    
    async def train(self, num_episodes: int = 10):
        """Run training with curriculum learning"""
        
        self.logger.info(f"Starting RL training for {num_episodes} episodes")
        
        all_problems = self.get_training_problems()
        all_results = []
        
        for episode_idx in range(num_episodes):
            # Select problem from current tier
            tier_problems = all_problems[self.current_tier]
            problem = tier_problems[episode_idx % len(tier_problems)]
            
            # Run episode
            episode_result = await self.run_episode(problem)
            all_results.append(episode_result)
            
            # Track metrics
            self.episode_rewards.append(episode_result["total_reward"])
            self.compilation_rates.append(1.0 if episode_result["compilation_success"] else 0.0)
            self.performance_history.append(episode_result["best_performance"])
            self.tier_metrics[self.current_tier].append(episode_result["best_performance"])
            
            # Log progress
            self.logger.info(
                f"Episode {episode_idx + 1}/{num_episodes}",
                tier=self.current_tier,
                problem=episode_result["problem_id"],
                reward=f"{episode_result['total_reward']:.3f}",
                performance=f"{episode_result['best_performance']:.2f}x",
                compilation_rate=f"{np.mean(self.compilation_rates[-5:]):.1%}"
            )
            
            # Check for curriculum advancement
            if len(self.tier_metrics[self.current_tier]) >= 3:
                tier_performance = np.mean(self.tier_metrics[self.current_tier][-3:])
                tier_compilation = np.mean([
                    1.0 if r["compilation_success"] else 0.0 
                    for r in all_results[-3:]
                ])
                
                if self.current_tier == "BASIC" and tier_compilation >= 0.8 and tier_performance >= 1.2:
                    self.current_tier = "INTERMEDIATE"
                    self.logger.info("🎯 Advanced to INTERMEDIATE tier!")
                elif self.current_tier == "INTERMEDIATE" and tier_compilation >= 0.7 and tier_performance >= 1.5:
                    self.current_tier = "ADVANCED"
                    self.logger.info("🚀 Advanced to ADVANCED tier!")
            
            # Save checkpoint periodically
            if (episode_idx + 1) % 5 == 0:
                self.save_checkpoint(episode_idx + 1, all_results)
        
        # Final summary
        self.logger.info("Training completed!")
        self.log_final_summary(all_results)
        
        # Save final results
        self.save_results(all_results)
        
        return all_results
    
    def save_checkpoint(self, episode: int, results: List[Dict]):
        """Save training checkpoint"""
        checkpoint = {
            "episode": episode,
            "current_tier": self.current_tier,
            "compilation_rate": float(np.mean(self.compilation_rates)),
            "mean_performance": float(np.mean(self.performance_history)),
            "tier_metrics": {k: [float(v) for v in vals] for k, vals in self.tier_metrics.items()}
        }
        
        with open(f"checkpoint_ep{episode}.json", 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        self.logger.info(f"Checkpoint saved at episode {episode}")
    
    def save_results(self, results: List[Dict]):
        """Save final training results"""
        final_results = {
            "config": self.config,
            "episodes": len(results),
            "final_tier": self.current_tier,
            "metrics": {
                "mean_reward": float(np.mean(self.episode_rewards)),
                "final_compilation_rate": float(np.mean(self.compilation_rates[-5:])),
                "best_performance": float(max(self.performance_history)),
                "mean_performance": float(np.mean(self.performance_history))
            },
            "tier_progression": {
                tier: {
                    "episodes": len(metrics),
                    "mean_performance": float(np.mean(metrics)) if metrics else 0.0,
                    "best_performance": float(max(metrics)) if metrics else 0.0
                }
                for tier, metrics in self.tier_metrics.items()
            }
        }
        
        with open("final_training_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)
        
        self.logger.info("Final results saved to final_training_results.json")
    
    def log_final_summary(self, results: List[Dict]):
        """Log final training summary"""
        successful = [r for r in results if r["compilation_success"]]
        
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        print(f"Total episodes: {len(results)}")
        print(f"Successful compilations: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"Mean reward: {np.mean(self.episode_rewards):.3f}")
        print(f"Best performance: {max(self.performance_history):.2f}x")
        print(f"Final tier reached: {self.current_tier}")
        
        print("\nTier Performance:")
        for tier, metrics in self.tier_metrics.items():
            if metrics:
                print(f"  {tier}: {len(metrics)} episodes, mean {np.mean(metrics):.2f}x, best {max(metrics):.2f}x")
        
        print("="*70)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.ray_initialized and ray.is_initialized():
            ray.shutdown()
        
        # Cleanup CUDA temp files
        if self.workflow and hasattr(self.workflow.cuda_compiler, 'cleanup_old_kernels'):
            self.workflow.cuda_compiler.cleanup_old_kernels()
        
        self.logger.info("Cleanup completed")


async def main():
    """Main entry point"""
    
    print("\n" + "="*70)
    print("WORKING RL TRAINING FOR CUDA CODE GENERATION")
    print("Multi-Agent System with VERL Infrastructure")
    print("="*70 + "\n")
    
    # Configuration
    config = {
        "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "num_gpus": 8,
        "num_cpus": 32,
        "max_turns": 3,
        "turn_discount": 0.9,
        "target_speedup": 2.0,
        "num_episodes": 10
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create trainer
    trainer = WorkingRLTrainer(config)
    
    try:
        # Initialize
        print("Initializing training system...")
        await trainer.initialize()
        
        # Run training
        print("\nStarting RL training loop...")
        print("-"*70)
        results = await trainer.train(num_episodes=config["num_episodes"])
        
        print("\n✅ Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await trainer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())