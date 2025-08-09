from typing import Dict, Any, List, Optional
import asyncio
import time
import structlog

from .multi_agent_trainer import MultiAgentVERLTrainer
from ..agents.cuda_generator import CUDAGeneratorAgent
from ..agents.cuda_optimizer import CUDAOptimizerAgent
from ..agents.cuda_tester import CUDATesterAgent
from ..training.cuda_data_loader import CUDATrainingDataLoader
from ..training.reward_functions.cuda_performance_reward import CUDAPerformanceReward
from ..utils.config import TrainingConfig
from ..orchestration.cuda_workflow import CUDAKernelWorkflow


class CUDAMultiAgentVERLTrainer(MultiAgentVERLTrainer):
    """VERL trainer specialized for CUDA kernel generation with multi-agent conversations."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        
        # CUDA-specific components
        self.cuda_reward = CUDAPerformanceReward(
            target_speedup=config.cuda_rewards.get("target_speedup", 2.0),
            correctness_weight=config.cuda_rewards.get("correctness_weight", 0.4),
            performance_weight=config.cuda_rewards.get("performance_weight", 0.4),
            improvement_weight=config.cuda_rewards.get("improvement_weight", 0.2)
        )
        self.cuda_data_loader = CUDATrainingDataLoader(
            dataset_sources=config.data_sources,
            max_problems_per_source=1000
        )
        
        # Multi-turn conversation configuration
        conversation_config = config.conversation if hasattr(config, 'conversation') else {}
        self.max_conversation_turns = conversation_config.get("max_turns", 5)
        self.conversation_discount_factor = conversation_config.get("discount_factor", 0.9)
        self.early_termination_threshold = conversation_config.get("early_termination_threshold", 0.8)
        
        # CUDA workflow for orchestrating multi-agent interactions
        self.cuda_workflow = None  # Will be initialized with agents
        
        self.logger = structlog.get_logger("cuda_multi_agent_trainer")
        self.logger.info(
            "CUDA multi-agent trainer initialized",
            max_conversation_turns=self.max_conversation_turns,
            conversation_discount_factor=self.conversation_discount_factor,
            early_termination_threshold=self.early_termination_threshold
        )
    
    async def train_cuda_agents(
        self,
        generator_agent: CUDAGeneratorAgent,
        optimizer_agent: CUDAOptimizerAgent, 
        tester_agent: CUDATesterAgent,
        episodes: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """Train CUDA agents using multi-turn conversational RL."""
        
        start_time = time.time()
        
        try:
            # Initialize CUDA workflow with agents
            self.cuda_workflow = CUDAKernelWorkflow(
                cuda_generator=generator_agent,
                cuda_optimizer=optimizer_agent,
                cuda_tester=tester_agent,
                config=self.config.dict() if hasattr(self.config, 'dict') else {}
            )
            
            # Register all CUDA agents
            self.register_agent(generator_agent, is_primary=True)
            self.register_agent(optimizer_agent)
            self.register_agent(tester_agent)
            
            # Load CUDA-specific training data
            train_data, val_data = await self.cuda_data_loader.load_cuda_problems()
            
            if not train_data:
                raise ValueError("No training data available")
            
            self.logger.info(
                "Loaded CUDA training data",
                train_problems=len(train_data),
                val_problems=len(val_data)
            )
            
            # Start VERL training with multi-turn conversations
            training_results = await self.train_agent(
                agent=generator_agent,
                training_data=train_data,
                validation_data=val_data,
                episodes=episodes,
                reward_function=self.cuda_reward,
                conversation_config={
                    "max_turns": self.max_conversation_turns,
                    "discount_factor": self.conversation_discount_factor,
                    "other_agents": [optimizer_agent, tester_agent],
                    "early_termination_threshold": self.early_termination_threshold
                },
                **kwargs
            )
            
            total_time = time.time() - start_time
            
            self.logger.info(
                "CUDA multi-agent training completed",
                success=training_results.get("success", False),
                episodes=episodes,
                total_time=total_time,
                best_performance=training_results.get("metrics", {}).get("best_performance", 0.0)
            )
            
            return {
                **training_results,
                "training_type": "cuda_multi_agent",
                "total_training_time": total_time,
                "data_statistics": self.cuda_data_loader.get_problem_statistics(train_data)
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"CUDA training failed: {str(e)}"
            
            self.logger.error(
                "CUDA multi-agent training failed",
                error=error_msg,
                episodes=episodes,
                total_time=total_time
            )
            
            return {
                "success": False,
                "error": error_msg,
                "training_type": "cuda_multi_agent",
                "total_training_time": total_time,
                "episodes_completed": 0
            }
    
    async def _run_multi_turn_episode(
        self,
        problem: Dict[str, Any],
        conversation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run single training episode with multi-turn agent conversation."""
        
        # Initialize conversation state
        conversation_state = {
            "problem": problem["pytorch_operation"],
            "test_inputs": problem["test_inputs"],
            "target_performance": problem["expected_speedup"],
            "turns": [],
            "performance_history": [],
            "best_performance": 0.0,
            "conversation_id": f"episode_{int(time.time())}"
        }
        
        max_turns = conversation_config["max_turns"]
        agents = [self.agents["cuda_generator"]] + conversation_config["other_agents"]
        early_termination_threshold = conversation_config.get("early_termination_threshold", 0.8)
        
        total_reward = 0.0
        episode_start_time = time.time()
        
        try:
            # Use CUDA workflow to orchestrate the conversation
            if self.cuda_workflow:
                workflow_result = await self.cuda_workflow.run_workflow(
                    pytorch_operation=conversation_state["problem"],
                    test_inputs=conversation_state["test_inputs"],
                    target_performance=conversation_state["target_performance"],
                    max_optimization_turns=max_turns
                )
                
                # Extract conversation results from workflow
                if workflow_result.get("workflow_status") == "completed":
                    total_reward = self._calculate_workflow_reward(workflow_result)
                    conversation_state.update({
                        "turns": workflow_result.get("messages", []),
                        "performance_history": workflow_result.get("performance_history", []),
                        "best_performance": workflow_result.get("best_performance", 0.0)
                    })
                else:
                    total_reward = -0.5  # Penalty for incomplete workflow
            else:
                # Fallback: manual multi-turn conversation
                total_reward = await self._run_manual_conversation(
                    conversation_state, agents, max_turns, early_termination_threshold
                )
            
            episode_time = time.time() - episode_start_time
            
            self.logger.debug(
                "Multi-turn episode completed",
                total_reward=total_reward,
                best_performance=conversation_state["best_performance"],
                num_turns=len(conversation_state["turns"]),
                episode_time=episode_time
            )
            
            return {
                "conversation_state": conversation_state,
                "total_reward": total_reward,
                "best_turn_reward": max(conversation_state["performance_history"]) if conversation_state["performance_history"] else 0.0,
                "num_turns": len(conversation_state["turns"]),
                "episode_time": episode_time,
                "early_termination": conversation_state["best_performance"] >= early_termination_threshold
            }
            
        except Exception as e:
            episode_time = time.time() - episode_start_time
            self.logger.error(
                "Multi-turn episode failed",
                error=str(e),
                episode_time=episode_time
            )
            
            return {
                "conversation_state": conversation_state,
                "total_reward": -1.0,
                "best_turn_reward": 0.0,
                "num_turns": 0,
                "episode_time": episode_time,
                "error": str(e)
            }
    
    def _calculate_workflow_reward(self, workflow_result: Dict[str, Any]) -> float:
        """Calculate reward from workflow results."""
        best_performance = workflow_result.get("best_performance", 0.0)
        target_performance = workflow_result.get("target_performance", 2.0)
        
        # Base reward from performance
        performance_reward = min(best_performance / target_performance, 2.0) - 1.0
        
        # Bonus for workflow completion
        if workflow_result.get("workflow_status") == "completed":
            performance_reward += 0.2
        
        # Bonus for achieving target
        if best_performance >= target_performance:
            performance_reward += 0.3
        
        return max(-1.0, min(performance_reward, 1.0))
    
    async def _run_manual_conversation(
        self,
        conversation_state: Dict[str, Any],
        agents: List,
        max_turns: int,
        early_termination_threshold: float
    ) -> float:
        """Fallback manual conversation implementation."""
        
        total_reward = 0.0
        
        for turn in range(max_turns):
            # Select agent for current turn (rotate or based on state)
            current_agent = agents[turn % len(agents)]
            
            try:
                # Generate agent response based on conversation history
                response = await current_agent.process_request(
                    self._format_conversation_prompt(conversation_state, current_agent),
                    context=conversation_state
                )
                
                # Add turn to conversation
                conversation_state["turns"].append({
                    "agent": current_agent.agent_type,
                    "response": response.content,
                    "metadata": response.metadata,
                    "success": response.success
                })
                
                # Calculate turn reward
                turn_reward = await self.cuda_reward.calculate_reward(
                    problem=conversation_state["problem"],
                    generated_code=self._serialize_conversation(conversation_state),
                    test_cases=conversation_state["test_inputs"],
                    context={
                        "turn": turn,
                        "previous_performance": conversation_state["best_performance"],
                        "conversation_id": conversation_state["conversation_id"]
                    }
                )
                
                # Apply discount factor for multi-turn reward
                discounted_reward = turn_reward * (self.conversation_discount_factor ** turn)
                total_reward += discounted_reward
                
                # Update conversation state
                conversation_state["performance_history"].append(turn_reward)
                if turn_reward > conversation_state["best_performance"]:
                    conversation_state["best_performance"] = turn_reward
                
                # Early termination if excellent performance achieved
                if turn_reward > early_termination_threshold:
                    self.logger.info(
                        "Early termination triggered",
                        turn=turn,
                        performance=turn_reward,
                        threshold=early_termination_threshold
                    )
                    break
                    
            except Exception as e:
                self.logger.error(f"Turn {turn} failed", error=str(e))
                total_reward -= 0.1  # Small penalty for failed turns
                break
        
        return total_reward
    
    def _format_conversation_prompt(
        self, 
        conversation_state: Dict[str, Any], 
        current_agent
    ) -> str:
        """Format prompt for agent based on conversation history."""
        
        base_prompt = f"Problem: {conversation_state['problem']}\n\n"
        
        if conversation_state["turns"]:
            base_prompt += "Previous conversation:\n"
            for turn in conversation_state["turns"][-3:]:  # Last 3 turns for context
                base_prompt += f"Agent {turn['agent']}: {turn['response'][:500]}...\n\n"
        
        # Agent-specific prompts
        if hasattr(current_agent, 'agent_type'):
            if current_agent.agent_type == "cuda_optimizer":
                base_prompt += f"Current performance: {conversation_state['best_performance']:.2f}x\n"
                base_prompt += f"Target performance: {conversation_state['target_performance']}x\n"
                base_prompt += "Please optimize the kernel for better performance.\n"
            elif current_agent.agent_type == "cuda_tester":
                base_prompt += "Please analyze and test the current kernel implementation.\n"
        
        return base_prompt
    
    def _serialize_conversation(self, conversation_state: Dict[str, Any]) -> str:
        """Serialize conversation state for reward calculation."""
        
        serialized = f"Problem: {conversation_state['problem']}\n\n"
        
        for i, turn in enumerate(conversation_state["turns"]):
            serialized += f"Turn {i+1} - Agent {turn['agent']}:\n"
            serialized += f"{turn['response']}\n"
            if turn.get('metadata'):
                serialized += f"Metadata: {turn['metadata']}\n"
            serialized += "\n---\n\n"
        
        return serialized
    
    async def validate_cuda_environment(self) -> Dict[str, Any]:
        """Validate CUDA development environment."""
        
        validation_results = {
            "cuda_available": False,
            "nvcc_available": False,
            "torch_cuda": False,
            "gpu_count": 0,
            "cuda_version": None,
            "errors": []
        }
        
        try:
            # Check PyTorch CUDA availability
            import torch
            validation_results["torch_cuda"] = torch.cuda.is_available()
            validation_results["gpu_count"] = torch.cuda.device_count()
            if torch.cuda.is_available():
                validation_results["cuda_version"] = torch.version.cuda
                validation_results["cuda_available"] = True
        except Exception as e:
            validation_results["errors"].append(f"PyTorch CUDA check failed: {str(e)}")
        
        try:
            # Check nvcc compiler availability
            import subprocess
            result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
            validation_results["nvcc_available"] = result.returncode == 0
        except Exception as e:
            validation_results["errors"].append(f"NVCC check failed: {str(e)}")
        
        self.logger.info("CUDA environment validation completed", **validation_results)
        
        return validation_results