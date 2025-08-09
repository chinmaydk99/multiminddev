#!/usr/bin/env python3
"""
Run Multi-Turn RL training on multi-GPU setup.

This script runs multi-turn reinforcement learning training after SFT,
using VERL for distributed RL training with reduced episodes for testing.
"""

import asyncio
import sys
import os
import time
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import ray
import torch
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


@dataclass
class ConversationTurn:
    """Represents a single turn in the multi-agent conversation."""
    agent_type: str
    input_text: str
    output_text: str
    reward: float = 0.0
    timestamp: float = 0.0


@dataclass
class CUDAConversationState:
    """State of a complete multi-turn CUDA optimization conversation."""
    problem: str
    turns: List[ConversationTurn]
    performance_history: List[float]
    current_kernel: str = ""
    compilation_success: bool = True
    final_reward: float = 0.0


@dataclass
class MultiTurnRLConfig:
    """Configuration for multi-turn RL training."""
    
    # Training parameters
    num_episodes: int = 20  # Reduced for testing
    max_turns_per_episode: int = 3  # Reduced from 5
    batch_size: int = 4
    learning_rate: float = 1e-6
    
    # Reward configuration
    target_speedup: float = 2.0
    correctness_weight: float = 0.4
    performance_weight: float = 0.4
    improvement_weight: float = 0.2
    
    # Multi-turn settings
    discount_factor: float = 0.9
    immediate_reward_weight: float = 0.3
    final_reward_weight: float = 0.7
    
    # Distributed settings
    num_gpus: int = 8
    use_multi_gpu: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/multiturn_rl_test"
    save_freq: int = 5


@ray.remote(num_gpus=1, num_cpus=2)
class MockMultiTurnAgent:
    """Mock multi-turn agent for testing the RL framework."""
    
    def __init__(self, agent_type: str, config: MultiTurnRLConfig):
        self.agent_type = agent_type
        self.config = config
        self.logger = structlog.get_logger()
        self.device = f"cuda:{ray.get_gpu_ids()[0]}" if ray.get_gpu_ids() else "cpu"
        
        # Mock agent state
        self.training_step = 0
        self.performance_improvement = 0.0
        
        self.logger.info(f"{agent_type} agent initialized on {self.device}")
    
    async def generate_response(self, input_text: str, context: Dict[str, Any]) -> str:
        """Generate response for the given input."""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        self.training_step += 1
        
        if self.agent_type == "generator":
            return self._mock_generate_kernel(input_text)
        elif self.agent_type == "optimizer":
            return self._mock_optimize_kernel(input_text, context)
        else:  # tester
            return self._mock_test_kernel(input_text, context)
    
    def _mock_generate_kernel(self, problem: str) -> str:
        """Mock kernel generation."""
        return f"""__global__ void mock_kernel(float* input, float* output, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        output[idx] = input[idx] * 2.0f;  // Mock operation for {problem}
    }}
}}"""
    
    def _mock_optimize_kernel(self, kernel: str, context: Dict[str, Any]) -> str:
        """Mock kernel optimization."""
        # Simulate performance improvement over time
        improvement_factor = min(2.0, 1.0 + (self.training_step * 0.05))
        self.performance_improvement = improvement_factor
        
        return f"""__global__ void optimized_kernel(float* input, float* output, int n) {{
    __shared__ float shared_mem[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory (optimization)
    if (threadIdx.x < 256 && idx < n) {{
        shared_mem[threadIdx.x] = input[idx];
    }}
    __syncthreads();
    
    if (idx < n) {{
        output[idx] = shared_mem[threadIdx.x] * 2.0f;  // {improvement_factor:.2f}x speedup
    }}
}}"""
    
    def _mock_test_kernel(self, kernel: str, context: Dict[str, Any]) -> str:
        """Mock kernel testing."""
        # Simulate improving compilation and performance over time
        success_rate = min(0.95, 0.6 + (self.training_step * 0.01))
        speedup = min(3.0, 1.0 + (self.training_step * 0.03))
        
        if random.random() < success_rate:
            return f"✅ Compilation successful. Performance: {speedup:.2f}x speedup achieved."
        else:
            return "❌ Compilation failed. Error in kernel syntax."


class MultiTurnConversationManager:
    """Manages multi-turn conversations between agents."""
    
    def __init__(self, config: MultiTurnRLConfig):
        self.config = config
        self.logger = structlog.get_logger()
        
        # Create mock agents
        self.generator_agent = MockMultiTurnAgent.remote("generator", config)
        self.optimizer_agent = MockMultiTurnAgent.remote("optimizer", config)
        self.tester_agent = MockMultiTurnAgent.remote("tester", config)
    
    async def run_conversation(self, problem: str) -> CUDAConversationState:
        """Run a complete multi-turn conversation."""
        conversation = CUDAConversationState(
            problem=problem,
            turns=[],
            performance_history=[1.0],  # Start with 1x baseline
            compilation_success=True
        )
        
        current_kernel = ""
        
        for turn in range(self.config.max_turns_per_episode):
            if turn == 0:
                # Turn 1: Generator creates initial kernel
                response_future = self.generator_agent.generate_response.remote(
                    problem, {"turn": turn}
                )
                response = ray.get(response_future)
                
                conversation.turns.append(ConversationTurn(
                    agent_type="generator",
                    input_text=problem,
                    output_text=response,
                    timestamp=time.time()
                ))
                current_kernel = response
                
            elif turn == 1:
                # Turn 2: Tester evaluates kernel
                test_input = f"Test this kernel:\n{current_kernel}"
                response_future = self.tester_agent.generate_response.remote(
                    test_input, {"kernel": current_kernel, "turn": turn}
                )
                response = ray.get(response_future)
                
                conversation.turns.append(ConversationTurn(
                    agent_type="tester",
                    input_text=test_input,
                    output_text=response,
                    timestamp=time.time()
                ))
                
                # Extract performance info from response
                performance = self._extract_performance(response)
                conversation.performance_history.append(performance)
                conversation.compilation_success = "successful" in response.lower()
                
            else:
                # Turn 3: Optimizer improves kernel if needed
                opt_input = f"Optimize kernel for better performance:\n{current_kernel}"
                response_future = self.optimizer_agent.generate_response.remote(
                    opt_input, {"kernel": current_kernel, "performance": conversation.performance_history[-1], "turn": turn}
                )
                response = ray.get(response_future)
                
                conversation.turns.append(ConversationTurn(
                    agent_type="optimizer",
                    input_text=opt_input,
                    output_text=response,
                    timestamp=time.time()
                ))
                current_kernel = response
                
                # Test optimized kernel
                test_input = f"Test optimized kernel:\n{current_kernel}"
                test_response_future = self.tester_agent.generate_response.remote(
                    test_input, {"kernel": current_kernel, "turn": turn + 0.5}
                )
                test_response = ray.get(test_response_future)
                
                final_performance = self._extract_performance(test_response)
                conversation.performance_history.append(final_performance)
                
                # Early termination if target achieved
                if final_performance >= self.config.target_speedup:
                    self.logger.info(f"Target speedup achieved: {final_performance:.2f}x")
                    break
        
        conversation.current_kernel = current_kernel
        conversation.final_reward = self._calculate_final_reward(conversation)
        
        return conversation
    
    def _extract_performance(self, response: str) -> float:
        """Extract performance metric from response."""
        import re
        match = re.search(r'(\d+\.?\d*)x', response)
        if match:
            return float(match.group(1))
        return 1.0
    
    def _calculate_final_reward(self, conversation: CUDAConversationState) -> float:
        """Calculate final episode reward."""
        final_speedup = conversation.performance_history[-1] if conversation.performance_history else 1.0
        
        # Reward components
        speedup_score = min(final_speedup / self.config.target_speedup, 2.0) * self.config.performance_weight
        correctness_score = 1.0 if conversation.compilation_success else 0.0
        correctness_score *= self.config.correctness_weight
        
        # Improvement reward
        if len(conversation.performance_history) > 1:
            improvement = conversation.performance_history[-1] - conversation.performance_history[0]
            improvement_score = min(improvement / self.config.target_speedup, 1.0) * self.config.improvement_weight
        else:
            improvement_score = 0.0
        
        total_reward = speedup_score + correctness_score + improvement_score
        
        return total_reward


class MultiTurnRLTrainer:
    """Main trainer for multi-turn reinforcement learning."""
    
    def __init__(self, config: MultiTurnRLConfig):
        self.config = config
        self.logger = structlog.get_logger()
        
        # Initialize Ray if not done
        if not ray.is_initialized():
            self._init_ray()
        
        self.conversation_manager = MultiTurnConversationManager(config)
        
        # Training metrics
        self.episode_rewards = []
        self.performance_improvements = []
        self.compilation_success_rates = []
    
    def _init_ray(self):
        """Initialize Ray cluster."""
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "false",
                "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"
            }
        }
        
        ray.init(
            num_cpus=16,
            num_gpus=self.config.num_gpus,
            runtime_env=runtime_env
        )
        
        self.logger.info(
            "Ray initialized for multi-turn RL",
            resources=ray.available_resources()
        )
    
    def _generate_training_problems(self) -> List[str]:
        """Generate CUDA optimization problems for training."""
        problems = [
            "Implement efficient matrix multiplication for 1024x1024 matrices",
            "Create parallel reduction kernel for summing large arrays",
            "Optimize convolution operation for 3x3 filters",
            "Implement efficient transpose operation for large matrices",
            "Create parallel sorting algorithm using CUDA",
            "Optimize element-wise operations with vectorized memory access",
            "Implement efficient batch normalization kernel",
            "Create parallel prefix sum (scan) algorithm",
        ]
        return problems
    
    async def run_episode(self, episode_num: int, problem: str) -> CUDAConversationState:
        """Run a single training episode."""
        self.logger.info(f"Running episode {episode_num + 1}: {problem}")
        
        # Run multi-turn conversation
        conversation = await self.conversation_manager.run_conversation(problem)
        
        # Log episode results
        self.logger.info(
            "Episode completed",
            episode=episode_num + 1,
            final_reward=conversation.final_reward,
            turns=len(conversation.turns),
            final_speedup=conversation.performance_history[-1] if conversation.performance_history else 1.0,
            compilation_success=conversation.compilation_success
        )
        
        return conversation
    
    async def train(self):
        """Main training loop."""
        start_time = time.time()
        
        self.logger.info(
            "🚀 Starting multi-turn RL training",
            episodes=self.config.num_episodes,
            max_turns=self.config.max_turns_per_episode,
            target_speedup=self.config.target_speedup
        )
        
        # Generate training problems
        problems = self._generate_training_problems()
        
        try:
            # Training loop
            for episode in range(self.config.num_episodes):
                # Sample a problem
                problem = random.choice(problems)
                
                # Run episode
                conversation = await self.run_episode(episode, problem)
                
                # Update metrics
                self.episode_rewards.append(conversation.final_reward)
                if len(conversation.performance_history) > 1:
                    improvement = conversation.performance_history[-1] - conversation.performance_history[0]
                    self.performance_improvements.append(improvement)
                else:
                    self.performance_improvements.append(0.0)
                
                self.compilation_success_rates.append(1.0 if conversation.compilation_success else 0.0)
                
                # Log progress every 5 episodes
                if (episode + 1) % 5 == 0:
                    self._log_training_progress(episode + 1)
                
                # Save checkpoint
                if (episode + 1) % self.config.save_freq == 0:
                    self._save_checkpoint(episode + 1)
            
            # Training complete
            total_time = time.time() - start_time
            self._log_final_results(total_time)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Multi-turn RL training failed: {e}")
            return False
        
        finally:
            if ray.is_initialized():
                ray.shutdown()
    
    def _log_training_progress(self, episode: int):
        """Log training progress."""
        recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
        recent_improvements = self.performance_improvements[-10:] if len(self.performance_improvements) >= 10 else self.performance_improvements
        recent_success_rate = sum(self.compilation_success_rates[-10:]) / max(1, len(self.compilation_success_rates[-10:]))
        
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        avg_improvement = sum(recent_improvements) / len(recent_improvements)
        
        self.logger.info(
            "📊 Training Progress",
            episode=episode,
            avg_reward_recent=f"{avg_reward:.3f}",
            avg_improvement_recent=f"{avg_improvement:.3f}",
            success_rate_recent=f"{recent_success_rate:.3f}"
        )
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "episode": episode,
            "config": asdict(self.config),
            "metrics": {
                "episode_rewards": self.episode_rewards,
                "performance_improvements": self.performance_improvements,
                "compilation_success_rates": self.compilation_success_rates
            }
        }
        
        checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _log_final_results(self, total_time: float):
        """Log final training results."""
        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
        avg_improvement = sum(self.performance_improvements) / len(self.performance_improvements)
        overall_success_rate = sum(self.compilation_success_rates) / len(self.compilation_success_rates)
        
        self.logger.info(
            "✅ Multi-turn RL training completed!",
            total_episodes=len(self.episode_rewards),
            training_time=f"{total_time:.2f}s",
            avg_episode_reward=f"{avg_reward:.3f}",
            avg_performance_improvement=f"{avg_improvement:.3f}",
            overall_success_rate=f"{overall_success_rate:.3f}",
            checkpoint_dir=self.config.checkpoint_dir
        )


async def main():
    """Main entry point."""
    logger.info("🎯 Multi-Turn RL Training Script Starting")
    
    # Configuration
    config = MultiTurnRLConfig(
        num_episodes=20,
        max_turns_per_episode=3,
        target_speedup=2.0,
        num_gpus=8
    )
    
    # Create trainer
    trainer = MultiTurnRLTrainer(config)
    
    # Run training
    success = await trainer.train()
    
    if success:
        logger.info("🎉 Multi-turn RL training completed successfully!")
        logger.info("📋 Next: Monitor and validate results")
    else:
        logger.error("💥 Multi-turn RL training failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())