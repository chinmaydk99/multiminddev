"""
Multi-Agent VERL Trainer that integrates VERL's distributed training capabilities
with our multi-agent code generation framework.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
import structlog

from ..agents.trainable_agent import TrainableAgent
from ..training.base_trainer import BaseTrainer, TrainingResults
from ..utils.config import TrainingConfig
from .verl_config import VERLTrainingConfig
from .verl_coordinator import VERLCoordinator


class MultiAgentVERLTrainer(BaseTrainer):
    """
    Multi-Agent VERL Trainer that leverages VERL's distributed training 
    capabilities while coordinating multiple specialized agents.
    
    This trainer integrates:
    - VERL's distributed PPO/GRPO/ReMax algorithms
    - Ray cluster management and FSDP2 backends  
    - vLLM/SGLang inference engines
    - Multi-agent coordination (Generator, Reviewer, Executor)
    - Multi-turn conversation handling
    """
    
    def __init__(
        self, 
        config: TrainingConfig, 
        verl_config: Optional[VERLTrainingConfig] = None
    ):
        super().__init__(config)
        
        # VERL-specific configuration
        self.verl_config = verl_config or VERLTrainingConfig()
        
        # VERL coordinator for distributed training
        self.verl_coordinator = VERLCoordinator(self.verl_config)
        
        # Agent registry
        self.agents: Dict[str, TrainableAgent] = {}
        self.primary_agent_id: Optional[str] = None
        
        # Training state
        self.distributed_initialized = False
        
        self.logger.info(
            "Multi-Agent VERL Trainer initialized",
            algorithm=self.verl_config.distributed.algorithm,
            num_gpus=self.verl_config.distributed.num_gpus,
            enable_multi_turn=self.verl_config.multi_agent.enable_multi_turn
        )
        
    async def initialize_distributed_training(self) -> None:
        """Initialize VERL distributed training environment."""
        
        if self.distributed_initialized:
            return
            
        self.logger.info("Initializing VERL distributed training")
        
        try:
            # Initialize VERL coordinator
            await self.verl_coordinator.initialize()
            
            # Register agents with VERL coordinator
            for agent_id, agent in self.agents.items():
                role = self._get_agent_role(agent)
                self.verl_coordinator.register_agent(agent, role)
                
            self.distributed_initialized = True
            self.logger.info("VERL distributed training initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed training: {e}")
            raise
            
    def register_agent(self, agent: TrainableAgent, is_primary: bool = False) -> None:
        """Register an agent for multi-agent training."""
        
        agent_id = agent.agent_id
        self.agents[agent_id] = agent
        
        if is_primary or not self.primary_agent_id:
            self.primary_agent_id = agent_id
            
        self.logger.info(
            "Agent registered",
            agent_id=agent_id,
            agent_type=agent.agent_type,
            is_primary=is_primary
        )
        
    def _get_agent_role(self, agent: TrainableAgent) -> Optional[str]:
        """Determine agent role based on agent type."""
        
        agent_type = agent.agent_type.lower()
        
        if "generator" in agent_type or "code" in agent_type:
            return "generator"
        elif "reviewer" in agent_type or "review" in agent_type:
            return "reviewer"  
        elif "executor" in agent_type or "execute" in agent_type:
            return "executor"
        else:
            return None
            
    async def train_agent(
        self,
        agent: TrainableAgent,
        training_data: List[Dict[str, Any]], 
        validation_data: Optional[List[Dict[str, Any]]] = None,
        episodes: int = 100,
        reward_function: Optional[Any] = None,
        **kwargs,
    ) -> TrainingResults:
        """
        Train agents using VERL distributed training with multi-agent coordination.
        
        Args:
            agent: Primary agent to train (usually Generator)
            training_data: Training problem dataset
            validation_data: Optional validation dataset
            episodes: Number of training episodes/epochs
            reward_function: Custom reward function (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Training results with VERL metrics and distributed training info
        """
        
        self.is_training = True
        self._total_episodes = episodes
        self._initialize_training_metrics()
        
        start_time = time.time()
        
        try:
            # Register the primary agent if not already registered
            if agent.agent_id not in self.agents:
                self.register_agent(agent, is_primary=True)
                
            # Initialize distributed training environment
            await self.initialize_distributed_training()
            
            self.logger.info(
                "Starting VERL multi-agent distributed training",
                primary_agent_id=agent.agent_id,
                num_agents=len(self.agents),
                episodes=episodes,
                algorithm=self.verl_config.distributed.algorithm,
                num_gpus=self.verl_config.distributed.num_gpus
            )
            
            # Prepare training data for VERL format
            train_data_path = await self._prepare_training_data(training_data)
            eval_data_path = await self._prepare_validation_data(validation_data) if validation_data else None
            
            # Start VERL distributed training
            verl_results = await self.verl_coordinator.start_distributed_training(
                train_data_path=train_data_path,
                eval_data_path=eval_data_path,
                num_epochs=episodes,
                **kwargs
            )
            
            # Process VERL training results
            final_metrics = self._process_verl_results(verl_results)
            
            # Update our training metrics
            self.training_metrics.update(final_metrics.metrics or {})
            
            self.is_training = False
            training_time = time.time() - start_time
            
            self.logger.info(
                "VERL multi-agent distributed training completed",
                success=verl_results.success,
                training_time=training_time,
                final_reward=final_metrics.metrics.get("final_reward", 0.0),
                algorithm=self.verl_config.distributed.algorithm
            )
            
            return TrainingResults(
                success=verl_results.success,
                algorithm=f"verl_{self.verl_config.distributed.algorithm}",
                episodes=episodes,
                metrics={
                    **final_metrics.metrics,
                    "distributed_training": True,
                    "num_gpus": self.verl_config.distributed.num_gpus,
                    "num_agents": len(self.agents),
                    "multi_turn_enabled": self.verl_config.multi_agent.enable_multi_turn,
                },
                training_time=training_time,
                checkpoint_path=verl_results.checkpoint_path,
                error=verl_results.error if not verl_results.success else None
            )
            
        except Exception as e:
            self.is_training = False
            training_time = time.time() - start_time
            error_msg = f"VERL multi-agent training failed: {str(e)}"
            
            self.logger.error(error_msg, error=str(e))
            
            return TrainingResults(
                success=False,
                algorithm=f"verl_{self.verl_config.distributed.algorithm}",
                episodes=episodes,
                training_time=training_time,
                error=error_msg
            )
            
    async def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> str:
        """Prepare training data in VERL's expected JSONL format."""
        
        import tempfile
        import json
        
        # Create temporary file for training data
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        
        try:
            for item in training_data:
                # Convert to VERL's multi-turn format
                verl_item = {
                    "prompt": item.get("problem", ""),
                    "conversations": [],  # Multi-turn conversations will be generated during training
                    "metadata": {
                        "problem_id": item.get("id", ""),
                        "difficulty": item.get("difficulty", "medium"),
                        "test_cases": item.get("test_cases", []),
                        "expected_solution": item.get("solution", ""),
                    }
                }
                
                temp_file.write(json.dumps(verl_item) + '\n')
                
            temp_file.flush()
            
            self.logger.info(
                "Training data prepared for VERL",
                num_problems=len(training_data),
                data_path=temp_file.name
            )
            
            return temp_file.name
            
        except Exception as e:
            temp_file.close()
            raise e
        finally:
            temp_file.close()
            
    async def _prepare_validation_data(
        self, 
        validation_data: List[Dict[str, Any]]
    ) -> str:
        """Prepare validation data in VERL's expected format."""
        
        import tempfile
        import json
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        
        try:
            for item in validation_data:
                verl_item = {
                    "prompt": item.get("problem", ""),
                    "expected_output": item.get("solution", ""),
                    "test_cases": item.get("test_cases", []),
                    "metadata": {
                        "problem_id": item.get("id", ""),
                        "difficulty": item.get("difficulty", "medium"),
                    }
                }
                
                temp_file.write(json.dumps(verl_item) + '\n')
                
            temp_file.flush()
            
            self.logger.info(
                "Validation data prepared for VERL",
                num_problems=len(validation_data),
                data_path=temp_file.name
            )
            
            return temp_file.name
            
        except Exception as e:
            temp_file.close()
            raise e
        finally:
            temp_file.close()
            
    def _process_verl_results(self, verl_results: TrainingResults) -> TrainingResults:
        """Process and enhance VERL training results with multi-agent context."""
        
        if not verl_results.success:
            return verl_results
            
        # Enhance metrics with multi-agent information
        enhanced_metrics = verl_results.metrics.copy() if verl_results.metrics else {}
        
        # Add agent-specific metrics
        enhanced_metrics.update({
            "multi_agent_coordination": {
                "total_agents": len(self.agents),
                "agent_roles": {
                    agent_id: self._get_agent_role(agent) 
                    for agent_id, agent in self.agents.items()
                },
                "primary_agent": self.primary_agent_id,
                "coordination_strategy": self.verl_config.multi_agent.coordination_strategy,
            },
            "distributed_training_info": {
                "algorithm": self.verl_config.distributed.algorithm,
                "num_gpus": self.verl_config.distributed.num_gpus,
                "num_nodes": self.verl_config.distributed.num_nodes,
                "strategy": self.verl_config.distributed.strategy,
                "vllm_enabled": True,
                "ray_cluster": self.verl_config.distributed.ray_cluster_address or "local",
            },
            "multi_turn_settings": {
                "enabled": self.verl_config.multi_agent.enable_multi_turn,
                "max_turns": self.verl_config.multi_agent.max_turns,
                "reward_aggregation": self.verl_config.multi_agent.conversation_reward_aggregation,
            }
        })
        
        return TrainingResults(
            success=verl_results.success,
            algorithm=verl_results.algorithm,
            episodes=verl_results.episodes,
            metrics=enhanced_metrics,
            training_time=verl_results.training_time,
            checkpoint_path=verl_results.checkpoint_path,
            error=verl_results.error
        )
        
    async def get_training_progress(self) -> Dict[str, Any]:
        """Get enhanced training progress including VERL distributed metrics."""
        
        base_progress = super().get_training_progress()
        
        if not self.is_training or not self.distributed_initialized:
            return base_progress
            
        # Add VERL-specific progress information
        base_progress.update({
            "distributed_training": {
                "coordinator_active": self.verl_coordinator.training_active,
                "num_registered_agents": len(self.agents),
                "algorithm": self.verl_config.distributed.algorithm,
                "cluster_resources": "available" if self.distributed_initialized else "initializing",
            }
        })
        
        return base_progress
        
    async def shutdown(self) -> None:
        """Shutdown VERL coordinator and clean up resources."""
        
        self.logger.info("Shutting down Multi-Agent VERL Trainer")
        
        if self.distributed_initialized:
            await self.verl_coordinator.shutdown()
            
        self.agents.clear()
        self.distributed_initialized = False
        
        self.logger.info("Multi-Agent VERL Trainer shutdown complete")