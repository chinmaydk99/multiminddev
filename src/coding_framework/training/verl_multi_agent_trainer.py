"""
Multi-agent trainer that properly integrates with VERL's GRPO/DAPO training.
VERL uses RayPPOTrainer as the base class but supports GRPO/DAPO through configuration.
"""

import os
import sys
import socket
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
import ray
from omegaconf import OmegaConf, DictConfig
import structlog
import numpy as np
from dataclasses import dataclass
import json

# Add VERL to path
verl_path = Path(__file__).parent.parent.parent.parent / "verl"
sys.path.insert(0, str(verl_path))

# Import VERL components
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer  # Base trainer for all algorithms
from verl.trainer.ppo.core_algos import compute_advantage
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker

# For DAPO-specific implementation
from verl.recipe.dapo.dapo_ray_trainer import RayDAPOTrainer

from ..agents.trainable_cuda_agents import (
    TrainableCUDAGeneratorAgent,
    TrainableCUDAOptimizerAgent,
    TrainableCUDATesterAgent
)
from .multi_turn_conversation import (
    MultiTurnConversationManager,
    TurnLevelRewardDistributor,
    CUDAConversationState,
    AgentRole
)


class MultiAgentVERLTrainer(RayPPOTrainer):
    """
    Custom VERL trainer for multi-agent CUDA code generation.
    
    Important: RayPPOTrainer is the base class for ALL algorithms in VERL.
    The actual algorithm (PPO/GRPO/DAPO) is determined by:
    - algorithm.adv_estimator: 'grpo' for GRPO, 'gae' for PPO
    - Additional algorithm-specific parameters
    
    For GRPO:
    - algorithm.adv_estimator = 'grpo'
    - algorithm.use_kl_in_reward = False
    - actor_rollout_ref.rollout.n = 16 (group size)
    
    For DAPO:
    - Extends RayDAPOTrainer instead (which itself extends RayPPOTrainer)
    - Has specific reward calculation and advantage estimation
    """
    
    def __init__(self, config: DictConfig, **kwargs):
        """
        Initialize multi-agent VERL trainer.
        
        Args:
            config: Hydra configuration with algorithm settings
        """
        super().__init__(config, **kwargs)
        self.logger = structlog.get_logger()
        
        # Verify algorithm configuration
        self.algorithm_type = config.algorithm.get("adv_estimator", "gae")
        self.logger.info(
            f"Initializing MultiAgentVERLTrainer with algorithm: {self.algorithm_type}",
            use_kl_in_reward=config.algorithm.get("use_kl_in_reward", True),
            group_size=config.actor_rollout_ref.rollout.get("n", 1)
        )
        
        # Multi-agent components
        self.generator_agent = None
        self.optimizer_agent = None
        self.tester_agent = None
        self.conversation_manager = None
        self.reward_distributor = None
        
        # Multi-turn configuration
        self.max_turns = config.get("multi_turn", {}).get("max_turns", 5)
        self.turn_discount = config.get("multi_turn", {}).get("turn_discount_factor", 0.9)
        
    def _init_workers(self):
        """
        Override to properly initialize multi-agent workers.
        
        VERL manages workers through RayWorkerGroup for distributed training.
        We need to adapt our agents to work within VERL's worker infrastructure.
        """
        # Call parent to initialize standard VERL workers
        super()._init_workers()
        
        # Additionally initialize our multi-agent specific components
        self._init_multi_agent_components()
    
    def _init_multi_agent_components(self):
        """Initialize multi-agent components that work with VERL's infrastructure."""
        self.logger.info("Initializing multi-agent components")
        
        # Initialize conversation manager with VERL worker groups
        self.conversation_manager = VERLMultiTurnManager(
            actor_rollout_wg=self.actor_rollout_wg,  # VERL's worker group
            config=self.config
        )
        
        # Initialize reward distributor for multi-turn credit assignment
        self.reward_distributor = TurnLevelRewardDistributor(
            discount_factor=self.turn_discount,
            immediate_weight=0.3,
            final_weight=0.7
        )
        
        # Tester agent (rule-based, doesn't need VERL worker)
        self.tester_agent = TrainableCUDATesterAgent(use_trained_model=False)
    
    def _generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Override sequence generation to implement multi-turn conversations.
        
        This is where VERL generates responses. We extend it to handle
        multi-turn conversations between agents.
        
        Args:
            prompts: Input prompts in VERL's DataProto format
            
        Returns:
            Generated sequences with multi-turn conversation data
        """
        self.logger.debug(f"Generating multi-turn sequences for {len(prompts)} prompts")
        
        # For GRPO, we need to generate multiple responses per prompt
        if self.algorithm_type == "grpo":
            return self._generate_grpo_sequences(prompts)
        else:
            return self._generate_standard_sequences(prompts)
    
    def _generate_grpo_sequences(self, prompts: DataProto) -> DataProto:
        """
        Generate sequences for GRPO with multiple responses per prompt.
        
        GRPO requires generating N responses per prompt (group size).
        The responses are then ranked relatively within each group.
        """
        all_sequences = []
        group_size = self.config.actor_rollout_ref.rollout.n
        
        for prompt_data in prompts:
            prompt_text = prompt_data.get("prompt")
            
            # Generate multiple conversation episodes for the same prompt
            group_conversations = []
            for _ in range(group_size):
                # Run multi-turn conversation
                conversation = self._run_multi_turn_conversation(prompt_text)
                group_conversations.append(conversation)
            
            # Process group for GRPO advantages
            processed_group = self._process_grpo_group(group_conversations)
            all_sequences.extend(processed_group)
        
        return DataProto(all_sequences)
    
    def _generate_standard_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences for standard PPO/DAPO training."""
        all_sequences = []
        
        for prompt_data in prompts:
            prompt_text = prompt_data.get("prompt")
            
            # Run single multi-turn conversation
            conversation = self._run_multi_turn_conversation(prompt_text)
            
            # Convert to VERL format
            sequence_data = self._convert_conversation_to_verl(conversation)
            all_sequences.append(sequence_data)
        
        return DataProto(all_sequences)
    
    def _run_multi_turn_conversation(self, prompt: str) -> CUDAConversationState:
        """
        Run a complete multi-turn conversation for CUDA optimization.
        
        This coordinates the Generator -> Tester -> Optimizer -> Tester flow.
        """
        # Use VERL's actor_rollout_wg to generate responses
        conversation_state = CUDAConversationState(
            problem=prompt,
            problem_id=f"cuda_prob_{np.random.randint(10000)}",
            max_turns=self.max_turns
        )
        
        for turn_idx in range(self.max_turns):
            if turn_idx == 0:
                # Generator creates initial kernel
                response = self._generate_with_verl_worker(prompt, "generator")
                conversation_state.add_turn(
                    agent_type=AgentRole.GENERATOR,
                    input_text=prompt,
                    output_text=response["text"],
                    log_probs=response.get("log_probs"),
                    token_ids=response.get("token_ids")
                )
                conversation_state.current_kernel = response["text"]
                
            elif turn_idx % 2 == 1:
                # Tester evaluates
                test_result = self.tester_agent.test_kernel(
                    conversation_state.current_kernel
                )
                conversation_state.add_turn(
                    agent_type=AgentRole.TESTER,
                    input_text="Test kernel",
                    output_text=str(test_result),
                    immediate_reward=self._calculate_test_reward(test_result)
                )
                
                # Update performance
                speedup = test_result.get("performance", {}).get("speedup", 1.0)
                conversation_state.update_kernel(
                    conversation_state.current_kernel,
                    speedup
                )
                
            else:
                # Optimizer improves kernel
                opt_prompt = f"Optimize this kernel:\n{conversation_state.current_kernel}"
                response = self._generate_with_verl_worker(opt_prompt, "optimizer")
                conversation_state.add_turn(
                    agent_type=AgentRole.OPTIMIZER,
                    input_text=opt_prompt,
                    output_text=response["text"],
                    log_probs=response.get("log_probs"),
                    token_ids=response.get("token_ids")
                )
                conversation_state.current_kernel = response["text"]
            
            # Check early termination
            if conversation_state.should_terminate_early():
                break
        
        # Calculate final reward
        conversation_state.final_reward = self._calculate_final_reward(conversation_state)
        conversation_state.episode_complete = True
        
        return conversation_state
    
    def _generate_with_verl_worker(self, prompt: str, agent_type: str) -> Dict[str, Any]:
        """
        Generate response using VERL's worker infrastructure.
        
        This properly integrates with VERL's distributed workers.
        """
        # Prepare data in VERL format
        data = DataProto([{
            "prompt": prompt,
            "agent_type": agent_type
        }])
        
        # Use VERL's actor_rollout_wg to generate
        # This handles distributed generation across GPUs
        output = ray.get(
            self.actor_rollout_wg.generate_sequences(
                data,
                temperature=self.config.actor_rollout_ref.rollout.temperature,
                top_p=self.config.actor_rollout_ref.rollout.top_p,
                top_k=self.config.actor_rollout_ref.rollout.top_k,
                max_new_tokens=self.config.data.max_response_length
            )
        )
        
        return {
            "text": output.data[0].get("response", ""),
            "log_probs": output.data[0].get("log_probs"),
            "token_ids": output.data[0].get("token_ids")
        }
    
    def _process_grpo_group(
        self,
        conversations: List[CUDAConversationState]
    ) -> List[Dict[str, Any]]:
        """
        Process a group of conversations for GRPO training.
        
        GRPO uses relative rewards within each group rather than absolute rewards.
        This is the key difference from standard PPO.
        """
        # Get final rewards for the group
        final_rewards = [conv.final_reward for conv in conversations]
        
        # Calculate GRPO advantages (relative within group)
        mean_reward = np.mean(final_rewards)
        std_reward = np.std(final_rewards) + 1e-8
        
        processed = []
        for conv, final_reward in zip(conversations, final_rewards):
            # GRPO advantage: normalized reward within group
            grpo_advantage = (final_reward - mean_reward) / std_reward
            
            # Distribute advantages across turns
            turn_advantages = self._distribute_grpo_advantages(conv, grpo_advantage)
            
            # Convert to VERL format with GRPO advantages
            verl_data = self._convert_conversation_to_verl(
                conv,
                override_advantages=turn_advantages
            )
            processed.append(verl_data)
        
        return processed
    
    def _distribute_grpo_advantages(
        self,
        conversation: CUDAConversationState,
        group_advantage: float
    ) -> List[float]:
        """Distribute GRPO group advantage across turns."""
        advantages = []
        
        for turn in conversation.turns:
            if turn.agent_type == AgentRole.TESTER:
                advantages.append(0.0)  # Tester doesn't get trained
            else:
                # Apply turn-based discounting to group advantage
                turn_discount = self.turn_discount ** (
                    len(conversation.turns) - turn.turn_id - 1
                )
                advantages.append(group_advantage * turn_discount)
        
        return advantages
    
    def _calculate_test_reward(self, test_result: Dict[str, Any]) -> float:
        """Calculate immediate reward from test results."""
        reward = 0.0
        
        if test_result["compilation"]["success"]:
            reward += 0.3
        
        if test_result.get("correctness", {}).get("passed", False):
            reward += 0.4
        
        speedup = test_result.get("performance", {}).get("speedup", 1.0)
        reward += min(speedup - 1.0, 1.0) * 0.3
        
        return reward
    
    def _calculate_final_reward(self, conversation: CUDAConversationState) -> float:
        """Calculate final episode reward for CUDA optimization."""
        # Performance component
        best_speedup = conversation.best_performance
        target_speedup = 1.5  # Target from config
        speedup_reward = min(best_speedup / target_speedup, 2.0) * 0.4
        
        # Compilation success rate
        compilations = [
            r.success for r in conversation.compilation_results
        ]
        compile_reward = (sum(compilations) / len(compilations)) * 0.3 if compilations else 0
        
        # Efficiency (fewer turns is better)
        efficiency_reward = max(0, (self.max_turns - conversation.num_turns) / self.max_turns) * 0.2
        
        # Improvement over baseline
        if len(conversation.performance_history) >= 2:
            improvement = conversation.performance_history[-1] - conversation.performance_history[0]
            improvement_reward = min(improvement, 1.0) * 0.1
        else:
            improvement_reward = 0
        
        return speedup_reward + compile_reward + efficiency_reward + improvement_reward
    
    def _convert_conversation_to_verl(
        self,
        conversation: CUDAConversationState,
        override_advantages: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Convert conversation to VERL's expected data format."""
        # Combine all turns into a single sequence for VERL
        full_prompt = conversation.problem
        full_response = ""
        all_log_probs = []
        all_token_ids = []
        
        for turn in conversation.turns:
            if turn.agent_type != AgentRole.TESTER:
                full_response += f"\n[{turn.agent_type.value}]: {turn.output_text}"
                if turn.log_probs is not None:
                    all_log_probs.append(turn.log_probs)
                if turn.token_ids is not None:
                    all_token_ids.append(turn.token_ids)
        
        # Concatenate log probs and token ids
        if all_log_probs:
            combined_log_probs = torch.cat(all_log_probs, dim=0)
        else:
            combined_log_probs = None
        
        if all_token_ids:
            combined_token_ids = torch.cat(all_token_ids, dim=0)
        else:
            combined_token_ids = None
        
        # Prepare VERL data format
        verl_data = {
            "prompt": full_prompt,
            "response": full_response,
            "reward": conversation.final_reward,
            "log_probs": combined_log_probs,
            "token_ids": combined_token_ids,
            "metadata": {
                "num_turns": conversation.num_turns,
                "best_performance": conversation.best_performance,
                "conversation_id": conversation.problem_id
            }
        }
        
        # Add advantages if provided (for GRPO)
        if override_advantages is not None:
            verl_data["advantages"] = override_advantages
        
        return verl_data


class VERLMultiTurnManager:
    """
    Manager for multi-turn conversations that works with VERL's infrastructure.
    """
    
    def __init__(self, actor_rollout_wg, config):
        """
        Initialize with VERL's worker group.
        
        Args:
            actor_rollout_wg: VERL's RayWorkerGroup for distributed generation
            config: Training configuration
        """
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.logger = structlog.get_logger()


def create_multi_agent_trainer(config_path: str, algorithm: str = "grpo"):
    """
    Factory function to create the appropriate trainer based on algorithm.
    
    Args:
        config_path: Path to configuration file
        algorithm: Algorithm to use (grpo, dapo, ppo)
        
    Returns:
        Configured multi-agent trainer
    """
    # Load configuration
    config = OmegaConf.load(config_path)
    
    # Set algorithm-specific parameters
    if algorithm == "grpo":
        config.algorithm.adv_estimator = "grpo"
        config.algorithm.use_kl_in_reward = False
        config.actor_rollout_ref.rollout.n = 16  # Group size
        trainer_class = MultiAgentVERLTrainer
        
    elif algorithm == "dapo":
        # DAPO has its own trainer in VERL
        config.algorithm.adv_estimator = "grpo"  # DAPO uses GRPO estimator
        config.algorithm.use_kl_in_reward = False
        # Import DAPO-specific trainer
        from verl.recipe.dapo.dapo_ray_trainer import RayDAPOTrainer
        
        # Create custom DAPO trainer for multi-agent
        class MultiAgentDAPOTrainer(RayDAPOTrainer):
            def __init__(self, config, **kwargs):
                super().__init__(config, **kwargs)
                # Add multi-agent components
                self.conversation_manager = None
                self.reward_distributor = None
            
            # Override methods as needed for multi-agent
        
        trainer_class = MultiAgentDAPOTrainer
        
    else:  # Standard PPO
        config.algorithm.adv_estimator = "gae"
        config.algorithm.use_kl_in_reward = True
        trainer_class = MultiAgentVERLTrainer
    
    return trainer_class(config)