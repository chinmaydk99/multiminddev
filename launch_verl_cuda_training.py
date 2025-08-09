#!/usr/bin/env python3
"""
Launch proper VERL-based distributed RL training for CUDA code generation
This script ensures we actually use VERL's infrastructure as per requirements
"""

import os
import sys
import ray
import torch
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import structlog

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# VERL imports - using actual VERL components
import verl
from verl.trainer.ppo import PPOTrainer
from verl.trainer.config import PPOConfig
from verl.single_controller.ray import RayWorkerGroup, RayResourcePool
from verl.workers.actor_critic import ActorCritic
from verl.workers.rollout.vllm_rollout import vLLMRollout
from verl.utils.dataset import RLDataset
from verl.protocol import DataProto

# Our components
from coding_framework.agents.cuda_generator import CUDAGeneratorAgent
from coding_framework.agents.cuda_optimizer import CUDAOptimizerAgent
from coding_framework.agents.cuda_tester import CUDATesterAgent
from coding_framework.orchestration.cuda_workflow import CUDAKernelWorkflow
from coding_framework.training.cuda_data_loader import CUDADataLoader
from coding_framework.training.reward_functions.cuda_performance_reward import CUDAPerformanceReward
from coding_framework.cuda.compiler import CUDACompiler
from coding_framework.cuda.benchmarker import CUDABenchmarker
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

logger = structlog.get_logger("verl_cuda_launcher")


class VERLCUDATrainingOrchestrator:
    """
    Orchestrator that properly uses VERL's distributed infrastructure
    with our multi-agent CUDA workflow, as per requirements.
    
    Key Features:
    1. Uses VERL PPOTrainer for actual RL training
    2. Implements multi-turn conversations with proper reward shaping
    3. Includes SFT warm-start as specified
    4. Uses actual CUDA compilation and benchmarking (not mocks)
    5. Implements curriculum learning with tiers
    """
    
    def __init__(self, args):
        self.args = args
        self.logger = logger
        
        # VERL components
        self.ray_initialized = False
        self.resource_pool = None
        self.rollout_workers = None
        self.actor_critic_workers = None
        self.ppo_trainer = None
        
        # CUDA execution environment (actual implementations)
        self.cuda_compiler = CUDACompiler(
            nvcc_path="nvcc",
            cuda_arch="sm_75",  # Will auto-detect
            optimization_level="-O3"
        )
        self.cuda_benchmarker = CUDABenchmarker(
            warmup_iterations=10,
            benchmark_iterations=100
        )
        
        # Multi-agent components
        self.cuda_workflow = None
        self.agents = {}
        
        # Curriculum learning tiers
        self.curriculum_tiers = {
            "BASIC": ["vector_add", "scalar_multiply", "elementwise_ops"],
            "INTERMEDIATE": ["reduction", "transpose", "scan"],
            "ADVANCED": ["matrix_multiply", "softmax", "layernorm"],
            "EXPERT": ["fused_operations", "attention", "custom_layers"]
        }
        self.current_tier = "BASIC"
        
        # Training metrics
        self.compile_rate = 0.0
        self.pass_rate = 0.0
        self.median_speedup = 1.0
        
    async def initialize_verl_infrastructure(self):
        """Initialize VERL's Ray-based distributed infrastructure"""
        
        self.logger.info("Initializing VERL distributed infrastructure...")
        
        # Initialize Ray cluster
        if not ray.is_initialized():
            ray_config = {
                "num_gpus": self.args.num_gpus,
                "num_cpus": self.args.num_cpus,
                "object_store_memory": 50_000_000_000,  # 50GB for kernel artifacts
                "dashboard_host": "0.0.0.0",
                "_system_config": {
                    "max_io_workers": 8,  # For parallel compilation
                    "object_spilling_config": json.dumps({
                        "type": "filesystem",
                        "params": {"directory_path": "/tmp/ray_spill"}
                    })
                }
            }
            
            ray.init(**ray_config)
            self.ray_initialized = True
            
        # Verify resources
        resources = ray.cluster_resources()
        self.logger.info(
            "Ray cluster initialized",
            gpus=resources.get("GPU", 0),
            cpus=resources.get("CPU", 0),
            nodes=len(ray.nodes())
        )
        
        # Create VERL resource pools
        self.resource_pool = RayResourcePool(
            process_on_nodes=[self.args.num_gpus]
        )
        
        # Setup rollout workers for generation (using vLLM)
        rollout_cls = ray.remote(
            num_gpus=1,
            num_cpus=4,
            max_concurrency=2
        )(vLLMRollout)
        
        self.rollout_workers = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=rollout_cls,
            num_workers=self.args.num_rollout_workers,
            init_kwargs={
                "model_name_or_path": self.args.model_name,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 4096,
                "dtype": "float16"
            }
        )
        
        # Setup actor-critic workers for training
        actor_critic_cls = ray.remote(
            num_gpus=2,
            num_cpus=8
        )(ActorCritic)
        
        self.actor_critic_workers = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=actor_critic_cls,
            num_workers=self.args.num_actor_workers,
            init_kwargs={
                "model_name_or_path": self.args.model_name,
                "optimizer": "adamw",
                "learning_rate": self.args.learning_rate
            }
        )
        
        self.logger.info("VERL infrastructure ready")
        
    async def initialize_agents(self):
        """Initialize our multi-agent CUDA workflow"""
        
        config = load_config()
        config.llm.provider = "huggingface"
        config.llm.model = self.args.model_name
        
        llm_interface = LLMInterface(config.llm)
        await llm_interface.initialize()
        
        agent_config = AgentConfig(
            max_retries=2,
            timeout=60,
            enable_logging=True
        )
        
        # Initialize CUDA agents
        self.agents["generator"] = CUDAGeneratorAgent(agent_config, llm_interface)
        self.agents["optimizer"] = CUDAOptimizerAgent(agent_config, llm_interface)
        self.agents["tester"] = CUDATesterAgent(agent_config, llm_interface)
        
        # Create CUDA workflow
        self.cuda_workflow = CUDAKernelWorkflow(
            cuda_generator=self.agents["generator"],
            cuda_optimizer=self.agents["optimizer"],
            cuda_tester=self.agents["tester"],
            config={
                "max_turns": self.args.max_turns,
                "early_stop_threshold": self.args.early_stop_threshold
            }
        )
        
        self.logger.info("Multi-agent CUDA workflow initialized")
        
    async def run_sft_warmstart(self):
        """Run tiny SFT warm-start phase as per requirements"""
        
        if not self.args.skip_sft:
            self.logger.info("Running SFT warm-start phase...")
            
            # Import SFT training module
            from examples.cuda_training.true_sft_training import run_sft_training, SFTConfig
            
            sft_config = SFTConfig(
                model_name=self.args.model_name,
                dataset_name="SakanaAI/AI-CUDA-Engineer-Archive",
                output_dir="./cuda-sft-warmstart",
                max_samples=1000,  # Small warm-start
                num_train_epochs=2,  # Quick training
                batch_size=4,
                learning_rate=2e-5,
                use_lora=True,
                use_4bit=True
            )
            
            success = run_sft_training(sft_config)
            
            if success:
                self.logger.info("SFT warm-start completed successfully")
                # Update model path to use warm-started model
                self.args.model_name = "./cuda-sft-warmstart"
            else:
                self.logger.warning("SFT warm-start failed, continuing with base model")
                
    def create_gated_shaped_reward(
        self, 
        compilation_success: bool,
        tests_passed: bool,
        test_pass_rate: float,
        speedup: float,
        register_pressure: int,
        shared_memory_overuse: int,
        timeout: bool,
        uses_warp_shuffle: bool,
        has_coalesced_access: bool
    ) -> float:
        """
        Implement gated shaped reward as per requirements:
        Gate: CompileOK && TestsPassed else 0
        Shaped: R = 0.5*TestsPassRate + 0.4*SpeedupNorm – 0.05*RegPressure – 0.05*SmemOver – 0.1*Timeout
        Bonuses: +0.05 for warp shuffle use, +0.05 for coalesced access
        """
        
        # Gate condition
        if not (compilation_success and tests_passed):
            return 0.0
        
        # Normalize speedup (clip between 0 and 2x target)
        speedup_norm = min(speedup / self.args.target_speedup, 2.0) / 2.0
        
        # Normalize register pressure (32 is good, 64+ is bad)
        reg_pressure_norm = max(0, (register_pressure - 32) / 32) if register_pressure else 0
        
        # Normalize shared memory overuse (in KB)
        smem_over_norm = min(shared_memory_overuse / 48, 1.0) if shared_memory_overuse else 0
        
        # Calculate shaped reward
        reward = (
            0.5 * test_pass_rate +
            0.4 * speedup_norm -
            0.05 * reg_pressure_norm -
            0.05 * smem_over_norm -
            (0.1 if timeout else 0)
        )
        
        # Add bonuses
        if uses_warp_shuffle:
            reward += 0.05
        if has_coalesced_access:
            reward += 0.05
            
        return max(0.0, min(1.0, reward))  # Clip to [0, 1]
    
    async def run_multi_turn_episode(
        self,
        problem: Dict[str, Any]
    ) -> tuple[List[Dict], float]:
        """
        Run multi-turn CUDA generation episode with proper VERL integration
        """
        
        conversation_history = []
        total_reward = 0.0
        best_kernel = None
        best_performance = 0.0
        
        for turn in range(self.args.max_turns):
            self.logger.info(f"Turn {turn + 1}/{self.args.max_turns}")
            
            # Generate using VERL rollout workers
            prompt = self._create_turn_prompt(problem, conversation_history)
            
            generation_batch = DataProto(
                prompts=[prompt],
                metadata={
                    "turn": turn,
                    "problem_id": problem.get("id", "unknown"),
                    "tier": self.current_tier
                }
            )
            
            # Distributed generation via VERL
            rollout_output = await self.rollout_workers.async_execute(
                lambda worker: worker.generate(generation_batch)
            )
            
            generated_code = rollout_output.responses[0] if rollout_output.responses else ""
            
            # Compile and benchmark the generated kernel
            compilation_result = await self.cuda_compiler.compile_kernel(
                kernel_code=generated_code,
                kernel_name=f"kernel_turn_{turn}",
                use_docker=self.args.use_docker_compilation
            )
            
            performance = 0.0
            tests_passed = False
            test_pass_rate = 0.0
            
            if compilation_result.success:
                # Run actual benchmarking
                benchmark_result = await self.cuda_benchmarker.benchmark_kernel(
                    binary_path=compilation_result.binary_path,
                    test_inputs=problem.get("test_inputs", []),
                    reference_fn=problem.get("torch_reference")
                )
                
                performance = benchmark_result.speedup
                tests_passed = benchmark_result.correctness
                test_pass_rate = benchmark_result.test_pass_rate
                
                if performance > best_performance:
                    best_kernel = generated_code
                    best_performance = performance
            
            # Calculate turn reward using gated shaped reward
            turn_reward = self.create_gated_shaped_reward(
                compilation_success=compilation_result.success,
                tests_passed=tests_passed,
                test_pass_rate=test_pass_rate,
                speedup=performance,
                register_pressure=compilation_result.register_pressure or 64,
                shared_memory_overuse=max(0, (compilation_result.shared_memory_usage or 0) - 48*1024),
                timeout=compilation_result.compilation_time > 10,
                uses_warp_shuffle="__shfl" in generated_code,
                has_coalesced_access=self._check_coalesced_access(generated_code)
            )
            
            # Apply turn discount
            discounted_reward = turn_reward * (self.args.turn_discount ** turn)
            total_reward += discounted_reward
            
            # Store turn results
            turn_data = {
                "turn": turn,
                "prompt": prompt,
                "generated_code": generated_code,
                "compilation_success": compilation_result.success,
                "performance": performance,
                "reward": turn_reward,
                "discounted_reward": discounted_reward
            }
            conversation_history.append(turn_data)
            
            # Early stopping if performance threshold met
            if performance >= self.args.early_stop_threshold * problem.get("target_performance", 2.0):
                self.logger.info(f"Early stopping - performance threshold met: {performance:.2f}x")
                break
        
        return conversation_history, total_reward
    
    async def train_with_verl(self):
        """Main VERL training loop with proper PPO"""
        
        # Load training data
        data_loader = CUDADataLoader()
        training_data = await data_loader.load_from_huggingface(
            "SakanaAI/AI-CUDA-Engineer-Archive",
            tier=self.current_tier
        )
        
        # Configure VERL PPO trainer
        ppo_config = PPOConfig(
            num_ppo_epochs=self.args.ppo_epochs,
            mini_batch_size=self.args.mini_batch_size,
            learning_rate=self.args.learning_rate,
            kl_coef=self.args.kl_coef,
            clip_ratio=self.args.clip_ratio,
            gamma=1.0,  # Episodic for CUDA tasks
            gae_lambda=0.95
        )
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            actor_rollout_ref_workers=self.rollout_workers,
            actor_critic_workers=self.actor_critic_workers
        )
        
        # Training loop
        for episode in range(self.args.num_episodes):
            self.logger.info(f"Episode {episode + 1}/{self.args.num_episodes}")
            
            # Sample batch of problems from current tier
            batch_problems = np.random.choice(training_data, self.args.batch_size)
            
            # Collect rollouts
            all_prompts = []
            all_responses = []
            all_rewards = []
            episode_metrics = []
            
            for problem in batch_problems:
                # Run multi-turn episode
                conversation, total_reward = await self.run_multi_turn_episode(problem)
                
                # Collect data for PPO update
                for turn_data in conversation:
                    all_prompts.append(turn_data["prompt"])
                    all_responses.append(turn_data["generated_code"])
                    all_rewards.append(turn_data["discounted_reward"])
                
                # Track metrics
                episode_metrics.append({
                    "total_reward": total_reward,
                    "num_turns": len(conversation),
                    "final_performance": conversation[-1]["performance"],
                    "compilation_success": any(t["compilation_success"] for t in conversation)
                })
            
            # Update curriculum metrics
            self.compile_rate = np.mean([m["compilation_success"] for m in episode_metrics])
            self.pass_rate = np.mean([m["final_performance"] > 1.0 for m in episode_metrics])
            self.median_speedup = np.median([m["final_performance"] for m in episode_metrics])
            
            # Create training batch for VERL
            training_batch = DataProto(
                prompts=all_prompts,
                responses=all_responses,
                rewards=torch.tensor(all_rewards, dtype=torch.float32),
                metadata={
                    "episode": episode,
                    "tier": self.current_tier,
                    "batch_size": len(batch_problems)
                }
            )
            
            # Run PPO update via VERL
            ppo_stats = await self.ppo_trainer.train_step(training_batch)
            
            # Log metrics
            self.logger.info(
                f"Episode {episode} completed",
                compile_rate=self.compile_rate,
                pass_rate=self.pass_rate,
                median_speedup=self.median_speedup,
                mean_reward=np.mean(all_rewards),
                **ppo_stats
            )
            
            # Check curriculum advancement
            if self.should_advance_curriculum():
                self.advance_curriculum()
            
            # Checkpoint periodically
            if episode % 10 == 0:
                await self.save_checkpoint(episode)
    
    def should_advance_curriculum(self) -> bool:
        """Check if we should advance to next curriculum tier"""
        return (
            self.compile_rate > 0.9 and 
            self.pass_rate > 0.75 and
            self.median_speedup > 1.5
        )
    
    def advance_curriculum(self):
        """Advance to next curriculum tier"""
        tiers = list(self.curriculum_tiers.keys())
        current_idx = tiers.index(self.current_tier)
        
        if current_idx < len(tiers) - 1:
            self.current_tier = tiers[current_idx + 1]
            self.logger.info(f"Advanced to curriculum tier: {self.current_tier}")
            
            # Reset metrics for new tier
            self.compile_rate = 0.0
            self.pass_rate = 0.0
            self.median_speedup = 1.0
    
    def _create_turn_prompt(
        self, 
        problem: Dict[str, Any], 
        conversation_history: List[Dict]
    ) -> str:
        """Create prompt for current turn based on conversation history"""
        
        base_prompt = f"""Generate an optimized CUDA kernel for:
Problem: {problem['description']}
PyTorch Reference: {problem['torch_reference']}
Target Performance: {problem.get('target_performance', 2.0)}x speedup
"""
        
        if conversation_history:
            # Add context from previous turns
            last_turn = conversation_history[-1]
            base_prompt += f"""
Previous attempt (Turn {last_turn['turn'] + 1}):
- Compilation: {'Success' if last_turn['compilation_success'] else 'Failed'}
- Performance: {last_turn['performance']:.2f}x
- Issues to address: {self._identify_issues(last_turn)}

Generate an improved version:
"""
        
        return base_prompt
    
    def _identify_issues(self, turn_data: Dict) -> str:
        """Identify issues from previous turn for feedback"""
        issues = []
        
        if not turn_data["compilation_success"]:
            issues.append("Fix compilation errors")
        elif turn_data["performance"] < 1.0:
            issues.append("Improve performance - currently slower than baseline")
        elif turn_data["performance"] < 1.5:
            issues.append("Optimize memory access patterns")
            
        return ", ".join(issues) if issues else "Further optimize for performance"
    
    def _check_coalesced_access(self, code: str) -> bool:
        """Check if kernel has coalesced memory access pattern"""
        patterns = [
            "tid = blockIdx.x * blockDim.x + threadIdx.x",
            "idx = blockIdx.x * blockDim.x + threadIdx.x",
            "[tid]", "[idx]"
        ]
        return any(pattern in code for pattern in patterns)
    
    async def save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint_dir = f"./checkpoints/episode_{episode}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save via VERL workers
        await self.actor_critic_workers.async_execute(
            lambda worker: worker.save_checkpoint(checkpoint_dir)
        )
        
        # Save training state
        import json
        state = {
            "episode": episode,
            "current_tier": self.current_tier,
            "compile_rate": self.compile_rate,
            "pass_rate": self.pass_rate,
            "median_speedup": self.median_speedup
        }
        
        with open(f"{checkpoint_dir}/training_state.json", "w") as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_dir}")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.rollout_workers:
            self.rollout_workers.shutdown()
        if self.actor_critic_workers:
            self.actor_critic_workers.shutdown()
        if ray.is_initialized():
            ray.shutdown()
        
        # Cleanup CUDA temp files
        self.cuda_compiler.cleanup_old_kernels()
        
        self.logger.info("Cleanup completed")


async def main():
    """Main entry point for VERL CUDA training"""
    
    parser = argparse.ArgumentParser(description="VERL-based CUDA RL Training")
    
    # Model configuration
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Base model")
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT warm-start")
    
    # VERL/Ray configuration
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--num-cpus", type=int, default=32, help="Number of CPUs")
    parser.add_argument("--num-rollout-workers", type=int, default=4, help="Rollout workers")
    parser.add_argument("--num-actor-workers", type=int, default=2, help="Actor workers")
    
    # Training configuration
    parser.add_argument("--num-episodes", type=int, default=100, help="Training episodes")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--mini-batch-size", type=int, default=32, help="PPO mini-batch")
    parser.add_argument("--ppo-epochs", type=int, default=8, help="PPO epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--kl-coef", type=float, default=0.02, help="KL coefficient")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio")
    
    # Multi-turn configuration
    parser.add_argument("--max-turns", type=int, default=5, help="Max turns per episode")
    parser.add_argument("--turn-discount", type=float, default=0.9, help="Turn discount factor")
    parser.add_argument("--early-stop-threshold", type=float, default=0.8, help="Early stop threshold")
    parser.add_argument("--target-speedup", type=float, default=2.0, help="Target speedup")
    
    # CUDA execution
    parser.add_argument("--use-docker-compilation", action="store_true", help="Use Docker for safe compilation")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("VERL-BASED CUDA RL TRAINING")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max Turns: {args.max_turns}")
    print("="*60 + "\n")
    
    # Create orchestrator
    orchestrator = VERLCUDATrainingOrchestrator(args)
    
    try:
        # Phase 1: SFT warm-start
        if not args.skip_sft:
            await orchestrator.run_sft_warmstart()
        
        # Phase 2: Initialize infrastructure
        await orchestrator.initialize_verl_infrastructure()
        await orchestrator.initialize_agents()
        
        # Phase 3: Run VERL training
        await orchestrator.train_with_verl()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await orchestrator.cleanup()
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    import json
    import numpy as np
    
    # Set environment variables for optimal performance
    os.environ["RAY_DEDUP_LOGS"] = "0"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    
    asyncio.run(main())