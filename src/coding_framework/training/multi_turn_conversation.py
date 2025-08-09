"""
Multi-turn conversation state management for reinforcement learning training.
Handles conversation flow, state tracking, and reward distribution across turns.
"""

import torch
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import structlog
import numpy as np
from enum import Enum


class AgentRole(Enum):
    """Agent roles in the conversation."""
    GENERATOR = "generator"
    OPTIMIZER = "optimizer"
    TESTER = "tester"


@dataclass
class ConversationTurn:
    """Represents a single turn in a multi-agent conversation."""
    turn_id: int
    agent_type: AgentRole
    input_text: str
    output_text: str
    log_probs: Optional[torch.Tensor] = None
    token_ids: Optional[torch.Tensor] = None
    immediate_reward: float = 0.0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "turn_id": self.turn_id,
            "agent_type": self.agent_type.value,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "immediate_reward": self.immediate_reward,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            # Tensors stored as lists for JSON serialization
            "log_probs": self.log_probs.tolist() if self.log_probs is not None else None,
            "token_ids": self.token_ids.tolist() if self.token_ids is not None else None,
        }


@dataclass
class CompilationResult:
    """Result of kernel compilation attempt."""
    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None


@dataclass
class CUDAConversationState:
    """
    Tracks state across a multi-turn CUDA optimization conversation.
    
    This maintains the complete conversation history, performance trajectory,
    and rewards for training agents through RL.
    """
    problem: str  # Original PyTorch operation or CUDA problem
    problem_id: str
    difficulty: str = "medium"  # easy, medium, hard
    turns: List[ConversationTurn] = field(default_factory=list)
    current_kernel: Optional[str] = None
    best_kernel: Optional[str] = None
    performance_history: List[float] = field(default_factory=list)
    compilation_results: List[CompilationResult] = field(default_factory=list)
    final_reward: float = 0.0
    episode_complete: bool = False
    max_turns: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_turns(self) -> int:
        """Get number of turns in conversation."""
        return len(self.turns)
    
    @property
    def current_performance(self) -> float:
        """Get current performance metric."""
        return self.performance_history[-1] if self.performance_history else 0.0
    
    @property
    def best_performance(self) -> float:
        """Get best performance achieved."""
        return max(self.performance_history) if self.performance_history else 0.0
    
    def add_turn(
        self,
        agent_type: AgentRole,
        input_text: str,
        output_text: str,
        log_probs: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        immediate_reward: float = 0.0
    ) -> ConversationTurn:
        """Add a new turn to the conversation."""
        turn = ConversationTurn(
            turn_id=self.num_turns,
            agent_type=agent_type,
            input_text=input_text,
            output_text=output_text,
            log_probs=log_probs,
            token_ids=token_ids,
            immediate_reward=immediate_reward
        )
        self.turns.append(turn)
        return turn
    
    def update_kernel(self, kernel_code: str, performance: float) -> None:
        """Update current kernel and track performance."""
        self.current_kernel = kernel_code
        self.performance_history.append(performance)
        
        # Update best kernel if this is better
        if performance >= self.best_performance:
            self.best_kernel = kernel_code
    
    def should_terminate_early(self) -> bool:
        """Determine if conversation should terminate early."""
        # Terminate if we've reached max turns
        if self.num_turns >= self.max_turns:
            return True
        
        # Terminate if we've achieved excellent performance
        if self.current_performance > 2.0:  # 2x speedup
            return True
        
        # Terminate if performance has plateaued
        if len(self.performance_history) >= 3:
            recent_perf = self.performance_history[-3:]
            if max(recent_perf) - min(recent_perf) < 0.05:  # Less than 5% change
                return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "problem": self.problem,
            "problem_id": self.problem_id,
            "difficulty": self.difficulty,
            "turns": [turn.to_dict() for turn in self.turns],
            "current_kernel": self.current_kernel,
            "best_kernel": self.best_kernel,
            "performance_history": self.performance_history,
            "final_reward": self.final_reward,
            "episode_complete": self.episode_complete,
            "num_turns": self.num_turns,
            "best_performance": self.best_performance,
            "metadata": self.metadata
        }


class MultiTurnConversationManager:
    """
    Manages multi-turn conversations between agents for CUDA optimization.
    
    Coordinates agent interactions, tracks state, and handles reward calculation.
    """
    
    def __init__(
        self,
        generator_agent,
        optimizer_agent,
        tester_agent,
        max_turns: int = 5,
        early_termination_threshold: float = 0.8
    ):
        """
        Initialize conversation manager.
        
        Args:
            generator_agent: CUDA generator agent
            optimizer_agent: CUDA optimizer agent
            tester_agent: CUDA tester agent
            max_turns: Maximum turns per episode
            early_termination_threshold: Performance threshold for early termination
        """
        self.generator_agent = generator_agent
        self.optimizer_agent = optimizer_agent
        self.tester_agent = tester_agent
        self.max_turns = max_turns
        self.early_termination_threshold = early_termination_threshold
        
        self.logger = structlog.get_logger()
        
    async def run_conversation_episode(
        self,
        problem: str,
        problem_id: str,
        difficulty: str = "medium",
        target_performance: float = 1.5
    ) -> CUDAConversationState:
        """
        Run a complete multi-turn conversation episode.
        
        Args:
            problem: CUDA optimization problem description
            problem_id: Unique problem identifier
            difficulty: Problem difficulty level
            target_performance: Target speedup to achieve
            
        Returns:
            Complete conversation state with all turns and rewards
        """
        # Initialize conversation state
        state = CUDAConversationState(
            problem=problem,
            problem_id=problem_id,
            difficulty=difficulty,
            max_turns=self.max_turns
        )
        
        self.logger.info(
            "Starting conversation episode",
            problem_id=problem_id,
            difficulty=difficulty,
            target_performance=target_performance
        )
        
        try:
            # Episode flow: Generator -> Tester -> Optimizer -> Tester -> ...
            while not state.should_terminate_early():
                turn_num = state.num_turns
                
                if turn_num == 0:
                    # Turn 1: Generator creates initial kernel
                    await self._generator_turn(state, problem)
                    
                elif turn_num % 2 == 1:
                    # Odd turns: Tester evaluates kernel
                    await self._tester_turn(state)
                    
                    # Check if we should continue optimizing
                    if state.current_performance >= target_performance:
                        self.logger.info(
                            "Target performance achieved",
                            current_performance=state.current_performance,
                            target=target_performance
                        )
                        break
                        
                else:
                    # Even turns: Optimizer improves kernel
                    await self._optimizer_turn(state)
            
            # Calculate final reward
            state.final_reward = self._calculate_final_reward(state, target_performance)
            state.episode_complete = True
            
            self.logger.info(
                "Conversation episode complete",
                problem_id=problem_id,
                num_turns=state.num_turns,
                final_performance=state.current_performance,
                final_reward=state.final_reward
            )
            
        except Exception as e:
            self.logger.error(f"Error in conversation episode: {e}")
            state.episode_complete = True
            state.final_reward = 0.0
        
        return state
    
    async def _generator_turn(self, state: CUDAConversationState, problem: str) -> None:
        """Execute generator agent turn."""
        self.logger.debug("Executing generator turn")
        
        # Generate initial kernel
        result = await self.generator_agent.generate_cuda_kernel(
            operation_description=problem,
            tensor_info=state.metadata.get("tensor_info"),
            performance_hints=state.metadata.get("performance_hints")
        )
        
        # Create turn
        turn = state.add_turn(
            agent_type=AgentRole.GENERATOR,
            input_text=problem,
            output_text=result["kernel_code"],
            log_probs=result.get("log_probs"),
            token_ids=result.get("token_ids"),
            immediate_reward=self._calculate_immediate_reward_generator(result)
        )
        
        # Update state
        state.update_kernel(result["kernel_code"], 1.0)  # Baseline performance
        turn.metadata["kernel_name"] = result.get("kernel_name")
        turn.metadata["is_valid_syntax"] = result.get("is_valid_syntax", False)
    
    async def _optimizer_turn(self, state: CUDAConversationState) -> None:
        """Execute optimizer agent turn."""
        self.logger.debug("Executing optimizer turn")
        
        # Get performance analysis from last test
        last_test = state.compilation_results[-1] if state.compilation_results else None
        performance_analysis = {
            "current_speedup": state.current_performance,
            "compilation_success": last_test.success if last_test else False,
            "errors": last_test.errors if last_test else []
        }
        
        # Optimize kernel
        result = await self.optimizer_agent.optimize_kernel(
            kernel_code=state.current_kernel,
            performance_analysis=performance_analysis,
            optimization_targets=["shared_memory", "memory_coalescing"]
        )
        
        # Create turn
        turn = state.add_turn(
            agent_type=AgentRole.OPTIMIZER,
            input_text=f"Optimize kernel with current performance: {state.current_performance}x",
            output_text=result["optimized_code"],
            log_probs=result.get("log_probs"),
            token_ids=result.get("token_ids"),
            immediate_reward=self._calculate_immediate_reward_optimizer(result)
        )
        
        # Update kernel (performance will be updated after testing)
        state.current_kernel = result["optimized_code"]
        turn.metadata["applied_optimizations"] = result.get("applied_optimizations", [])
        turn.metadata["optimization_score"] = result.get("optimization_score", 0.0)
    
    async def _tester_turn(self, state: CUDAConversationState) -> None:
        """Execute tester agent turn."""
        self.logger.debug("Executing tester turn")
        
        # Test current kernel
        result = await self.tester_agent.test_kernel(
            kernel_code=state.current_kernel,
            test_inputs=state.metadata.get("test_inputs"),
            performance_target=state.metadata.get("target_performance")
        )
        
        # Create compilation result
        compilation = CompilationResult(
            success=result["compilation"]["success"],
            errors=result["compilation"].get("errors", []),
            warnings=result["compilation"].get("warnings", []),
            execution_time_ms=result["performance"].get("execution_time_ms"),
            memory_usage_mb=result["performance"].get("memory_usage_mb")
        )
        state.compilation_results.append(compilation)
        
        # Calculate performance improvement
        speedup = result["performance"].get("speedup", 1.0)
        state.update_kernel(state.current_kernel, speedup)
        
        # Create turn
        turn = state.add_turn(
            agent_type=AgentRole.TESTER,
            input_text="Test and profile kernel",
            output_text=result.get("test_report", ""),
            immediate_reward=self._calculate_immediate_reward_tester(result)
        )
        
        turn.metadata["test_results"] = result
    
    def _calculate_immediate_reward_generator(self, result: Dict[str, Any]) -> float:
        """Calculate immediate reward for generator turn."""
        reward = 0.0
        
        # Reward for valid syntax
        if result.get("is_valid_syntax", False):
            reward += 0.3
        
        # Reward for having kernel name
        if result.get("kernel_name"):
            reward += 0.1
        
        # Penalty for validation errors
        errors = result.get("validation_errors", [])
        reward -= 0.1 * len(errors)
        
        return max(reward, -1.0)
    
    def _calculate_immediate_reward_optimizer(self, result: Dict[str, Any]) -> float:
        """Calculate immediate reward for optimizer turn."""
        reward = 0.0
        
        # Reward based on optimization score
        reward += result.get("optimization_score", 0.0) * 0.5
        
        # Reward for applied optimizations
        applied = result.get("applied_optimizations", [])
        reward += 0.1 * len(applied)
        
        return min(reward, 1.0)
    
    def _calculate_immediate_reward_tester(self, result: Dict[str, Any]) -> float:
        """Calculate immediate reward for tester turn."""
        reward = 0.0
        
        # Reward for successful compilation
        if result["compilation"]["success"]:
            reward += 0.2
        
        # Reward for correctness
        if result.get("correctness", {}).get("passed", False):
            reward += 0.3
        
        return reward
    
    def _calculate_final_reward(
        self,
        state: CUDAConversationState,
        target_performance: float
    ) -> float:
        """
        Calculate final episode reward based on achieved performance.
        
        Args:
            state: Final conversation state
            target_performance: Target speedup
            
        Returns:
            Final reward value
        """
        # Base reward components
        speedup_score = min(state.best_performance / target_performance, 2.0) * 0.4
        
        # Correctness score (based on compilation success)
        compilations = [r.success for r in state.compilation_results]
        correctness_score = (sum(compilations) / len(compilations)) * 0.3 if compilations else 0.0
        
        # Efficiency score (fewer turns is better)
        efficiency_score = max(0, (self.max_turns - state.num_turns) / self.max_turns) * 0.2
        
        # Improvement score (reward for progressive improvement)
        if len(state.performance_history) >= 2:
            improvement = state.performance_history[-1] - state.performance_history[0]
            improvement_score = min(improvement / target_performance, 1.0) * 0.1
        else:
            improvement_score = 0.0
        
        final_reward = speedup_score + correctness_score + efficiency_score + improvement_score
        
        self.logger.debug(
            "Final reward calculation",
            speedup_score=speedup_score,
            correctness_score=correctness_score,
            efficiency_score=efficiency_score,
            improvement_score=improvement_score,
            final_reward=final_reward
        )
        
        return final_reward


class TurnLevelRewardDistributor:
    """
    Distributes final episode reward across conversation turns for credit assignment.
    """
    
    def __init__(
        self,
        discount_factor: float = 0.9,
        immediate_weight: float = 0.3,
        final_weight: float = 0.7
    ):
        """
        Initialize reward distributor.
        
        Args:
            discount_factor: Discount for future rewards
            immediate_weight: Weight for immediate turn rewards
            final_weight: Weight for final episode reward
        """
        self.discount_factor = discount_factor
        self.immediate_weight = immediate_weight
        self.final_weight = final_weight
        self.logger = structlog.get_logger()
    
    def distribute_rewards(
        self,
        conversation_state: CUDAConversationState
    ) -> List[float]:
        """
        Distribute final episode reward across turns.
        
        Args:
            conversation_state: Complete conversation state
            
        Returns:
            List of rewards for each turn
        """
        turn_rewards = []
        final_reward = conversation_state.final_reward
        num_turns = len(conversation_state.turns)
        
        for turn_idx, turn in enumerate(conversation_state.turns):
            # Skip tester turns (rule-based, not trained)
            if turn.agent_type == AgentRole.TESTER:
                turn_rewards.append(0.0)
                continue
            
            # Calculate discounted final reward component
            turns_remaining = num_turns - turn_idx - 1
            discounted_final = final_reward * (self.discount_factor ** turns_remaining)
            
            # Combine immediate and final rewards
            turn_reward = (
                self.immediate_weight * turn.immediate_reward +
                self.final_weight * discounted_final
            )
            
            # Apply role-specific scaling
            if turn.agent_type == AgentRole.GENERATOR:
                # Generators get higher reward for good initial solutions
                turn_reward *= 1.1
            elif turn.agent_type == AgentRole.OPTIMIZER:
                # Optimizers get reward scaled by performance improvement
                if turn_idx > 0:
                    perf_before = conversation_state.performance_history[turn_idx - 1]
                    perf_after = conversation_state.performance_history[turn_idx]
                    improvement_multiplier = max(0.5, min(2.0, perf_after / perf_before))
                    turn_reward *= improvement_multiplier
            
            turn_rewards.append(turn_reward)
        
        self.logger.debug(
            "Rewards distributed across turns",
            num_turns=num_turns,
            turn_rewards=turn_rewards,
            final_reward=final_reward
        )
        
        return turn_rewards
    
    def calculate_advantages(
        self,
        rewards: List[float],
        values: Optional[List[float]] = None,
        gamma: float = 0.99,
        lam: float = 0.95
    ) -> torch.Tensor:
        """
        Calculate GAE advantages for PPO training.
        
        Args:
            rewards: List of rewards
            values: Value estimates (if available)
            gamma: Discount factor
            lam: GAE lambda
            
        Returns:
            Advantage estimates
        """
        if values is None:
            # Simple advantage calculation without value function
            advantages = []
            discounted_reward = 0
            for reward in reversed(rewards):
                discounted_reward = reward + gamma * discounted_reward
                advantages.append(discounted_reward)
            advantages.reverse()
            advantages = torch.tensor(advantages, dtype=torch.float32)
        else:
            # GAE with value function
            advantages = torch.zeros(len(rewards))
            last_advantage = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + gamma * next_value - values[t]
                advantages[t] = last_advantage = delta + gamma * lam * last_advantage
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages