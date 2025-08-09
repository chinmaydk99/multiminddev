import asyncio
import ray
from typing import Dict, Any, List, Optional
import structlog
from pathlib import Path
import json
import time

from .verl_config import VERLTrainingConfig
from ..agents.base_agent import BaseAgent
from ..training.base_trainer import TrainingResults


class VERLCoordinator:
    """
    Coordinates VERL distributed training with multi-agent code generation workflow.
    
    This class serves as the bridge between our multi-agent framework and VERL's
    distributed training capabilities, handling the orchestration of distributed
    training while maintaining our agent-based architecture.
    """
    
    def __init__(self, config: VERLTrainingConfig):
        self.config = config
        self.logger = structlog.get_logger(component="verl_coordinator")
        
        # VERL training components (will be initialized during setup)
        self.verl_trainer = None
        self.ray_initialized = False
        
        # Agent registry
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_roles = {
            "generator": self.config.multi_agent.generator_agent_id,
            "reviewer": self.config.multi_agent.reviewer_agent_id,
            "executor": self.config.multi_agent.executor_agent_id,
        }
        
        # Training state
        self.training_active = False
        self.current_experiment = None
        
    async def initialize(self) -> None:
        """Initialize VERL coordinator and distributed training environment."""
        
        self.logger.info("Initializing VERL coordinator")
        
        try:
            # Initialize Ray cluster if not already initialized
            await self._setup_ray_cluster()
            
            # Initialize VERL trainer with our configuration
            await self._setup_verl_trainer()
            
            # Setup monitoring and logging
            await self._setup_monitoring()
            
            self.logger.info("VERL coordinator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize VERL coordinator: {e}")
            raise
            
    async def _setup_ray_cluster(self) -> None:
        """Initialize Ray cluster for distributed training."""
        
        if not ray.is_initialized():
            ray_config = {
                "address": self.config.distributed.ray_cluster_address or "auto",
                "namespace": self.config.distributed.ray_namespace,
                "runtime_env": {
                    "pip": [
                        "verl>=0.1.0",
                        "torch>=2.1.0", 
                        "transformers>=4.35.0",
                        "vllm>=0.2.0",
                    ]
                }
            }
            
            self.logger.info("Initializing Ray cluster", **ray_config)
            ray.init(**ray_config)
            self.ray_initialized = True
        
        # Validate cluster resources
        cluster_resources = ray.cluster_resources()
        total_gpus = cluster_resources.get("GPU", 0)
        
        if total_gpus < self.config.distributed.num_gpus:
            self.logger.warning(
                f"Insufficient GPUs: need {self.config.distributed.num_gpus}, "
                f"available {total_gpus}"
            )
            
        self.logger.info("Ray cluster ready", cluster_resources=cluster_resources)
        
    async def _setup_verl_trainer(self) -> None:
        """Initialize VERL trainer with distributed configuration."""
        
        try:
            # Import VERL components (would be actual imports in real implementation)
            # For now, we'll create a mock that follows VERL's interface
            
            self.logger.info("Setting up VERL trainer")
            
            # Convert our config to VERL's format
            verl_config = self.config.to_verl_config()
            
            # In real implementation, this would be:
            # from verl.trainer.ppo import PPOTrainer  # or GRPO, ReMax
            # self.verl_trainer = PPOTrainer(config=verl_config)
            
            # Mock VERL trainer for now
            self.verl_trainer = MockVERLDistributedTrainer(verl_config)
            
            self.logger.info("VERL trainer initialized", algorithm=self.config.algorithm)
            
        except Exception as e:
            self.logger.error(f"Failed to setup VERL trainer: {e}")
            raise
            
    async def _setup_monitoring(self) -> None:
        """Setup monitoring and experiment tracking."""
        
        # WandB integration
        if self.config.wandb_project:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.distributed.experiment_name,
                    config=self.config.dict()
                )
                self.logger.info("WandB initialized", project=self.config.wandb_project)
            except ImportError:
                self.logger.warning("WandB not available, skipping initialization")
                
        # MLflow integration
        if self.config.mlflow_tracking_uri:
            try:
                import mlflow
                mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
                mlflow.set_experiment(self.config.distributed.experiment_name)
                self.logger.info("MLflow initialized", uri=self.config.mlflow_tracking_uri)
            except ImportError:
                self.logger.warning("MLflow not available, skipping initialization")
                
    def register_agent(self, agent: BaseAgent, role: Optional[str] = None) -> None:
        """Register an agent for multi-agent training."""
        
        agent_id = agent.agent_id
        self.agents[agent_id] = agent
        
        if role and role in self.agent_roles:
            self.agent_roles[role] = agent_id
            
        self.logger.info(
            "Agent registered",
            agent_id=agent_id,
            agent_type=agent.agent_type,
            role=role
        )
        
    async def start_distributed_training(
        self,
        train_data_path: str,
        eval_data_path: Optional[str] = None,
        num_epochs: int = 10,
        **kwargs
    ) -> TrainingResults:
        """Start distributed VERL training with multi-agent coordination."""
        
        if self.training_active:
            raise RuntimeError("Training already in progress")
            
        self.training_active = True
        start_time = time.time()
        
        try:
            self.logger.info(
                "Starting distributed VERL training",
                num_epochs=num_epochs,
                train_data_path=train_data_path,
                algorithm=self.config.algorithm
            )
            
            # Prepare training data for VERL
            train_data = await self._prepare_training_data(train_data_path)
            eval_data = await self._prepare_evaluation_data(eval_data_path) if eval_data_path else None
            
            # Setup multi-agent reward function
            reward_function = self._create_multi_agent_reward_function()
            
            # Run VERL training with our multi-agent setup
            training_results = await self._run_verl_training(
                train_data=train_data,
                eval_data=eval_data,
                reward_function=reward_function,
                num_epochs=num_epochs,
                **kwargs
            )
            
            # Save final results
            results_path = await self._save_training_results(training_results)
            
            total_time = time.time() - start_time
            
            self.logger.info(
                "Distributed VERL training completed successfully",
                total_time=total_time,
                results_path=results_path,
                final_reward=training_results.get("final_reward", 0.0)
            )
            
            return TrainingResults(
                success=True,
                algorithm=self.config.algorithm,
                episodes=num_epochs,
                metrics=training_results,
                training_time=total_time,
                checkpoint_path=results_path
            )
            
        except Exception as e:
            self.logger.error(f"Distributed VERL training failed: {e}")
            return TrainingResults(
                success=False,
                algorithm=self.config.algorithm,
                episodes=num_epochs,
                metrics={},  # Required field
                training_time=time.time() - start_time,
                error=str(e)
            )
        finally:
            self.training_active = False
            
    async def _prepare_training_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Prepare training data in VERL's expected format from Hugging Face datasets."""
        
        self.logger.info("Loading training data from Hugging Face", dataset=data_path)
        
        try:
            from datasets import load_dataset
            
            # Load training dataset from Hugging Face (NOT HumanEval to avoid eval hacking)
            if data_path in ["mbpp", "google-research-datasets/mbpp"]:
                dataset = load_dataset("google-research-datasets/mbpp", split="train")
                self.logger.info(f"Loaded MBPP training dataset with {len(dataset)} examples")
            elif data_path in ["codeparrot/github-code-clean", "github-code"]:
                dataset = load_dataset("codeparrot/github-code-clean", split="train", streaming=True)
                # Take first 10000 examples to avoid memory issues
                dataset = dataset.take(10000)
                self.logger.info("Loaded GitHub Code dataset (first 10k examples)")
            else:
                # Try to load as custom dataset
                dataset = load_dataset(data_path, split="train")
                self.logger.info(f"Loaded custom dataset with {len(dataset)} examples")
            
            training_data = []
            
            for i, item in enumerate(dataset):
                try:
                    # Convert different dataset formats to VERL's multi-turn format
                    if "text" in item:  # MBPP format
                        verl_item = {
                            "prompt": item.get("text", ""),
                            "conversations": [],
                            "metadata": {
                                "problem_id": item.get("task_id", f"mbpp_{i}"),
                                "difficulty": "medium",
                                "test_cases": item.get("test_list", []),
                                "canonical_solution": item.get("code", "")
                            }
                        }
                    elif "prompt" in item:  # HumanEval-like format
                        verl_item = {
                            "prompt": item.get("prompt", ""),
                            "conversations": [],
                            "metadata": {
                                "problem_id": item.get("task_id", f"problem_{i}"),
                                "difficulty": "medium",
                                "test_cases": item.get("test", ""),
                                "canonical_solution": item.get("canonical_solution", "")
                            }
                        }
                    elif "content" in item:  # GitHub code format
                        verl_item = {
                            "prompt": f"# Complete this code:\n{item.get('content', '')[:500]}",
                            "conversations": [],
                            "metadata": {
                                "problem_id": f"github_{i}",
                                "difficulty": "medium",
                                "language": item.get("language", "python"),
                                "canonical_solution": item.get("content", "")
                            }
                        }
                    else:
                        # Generic format
                        verl_item = {
                            "prompt": str(item).get("prompt", str(item)[:500]),
                            "conversations": [],
                            "metadata": {
                                "problem_id": f"generic_{i}",
                                "difficulty": "medium"
                            }
                        }
                    
                    training_data.append(verl_item)
                    
                except Exception as e:
                    self.logger.warning(f"Skipping invalid data item {i}: {e}")
                    continue
                    
            self.logger.info(f"Prepared {len(training_data)} training examples from Hugging Face dataset")
            return training_data
            
        except ImportError:
            self.logger.error("datasets library not installed. Install with: pip install datasets")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load dataset from Hugging Face: {e}")
            raise
        
    async def _prepare_evaluation_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Prepare evaluation data in VERL's expected format from Hugging Face."""
        
        self.logger.info("Loading evaluation data from Hugging Face", dataset=data_path)
        
        try:
            from datasets import load_dataset
            
            # Load evaluation dataset (same as training for now)
            if data_path in ["humaneval", "openai/HumanEval", "./data/eval_problems.jsonl"]:
                dataset = load_dataset("openai/HumanEval", split="test")
                self.logger.info(f"Loaded HumanEval evaluation dataset with {len(dataset)} examples")
            else:
                dataset = load_dataset(data_path, split="test")
                self.logger.info(f"Loaded custom evaluation dataset with {len(dataset)} examples")
            
            eval_data = []
            
            for i, item in enumerate(dataset):
                try:
                    eval_item = {
                        "prompt": item.get("prompt", ""),
                        "expected_output": item.get("canonical_solution", ""),
                        "test_cases": item.get("test", ""),
                        "metadata": {
                            "task_id": item.get("task_id", f"eval_{i}"),
                            "entry_point": item.get("entry_point", "")
                        }
                    }
                    eval_data.append(eval_item)
                except Exception as e:
                    self.logger.warning(f"Skipping invalid eval item {i}: {e}")
                    continue
                    
            self.logger.info(f"Prepared {len(eval_data)} evaluation examples from Hugging Face")
            return eval_data
            
        except Exception as e:
            self.logger.error(f"Failed to load evaluation dataset: {e}")
            raise
        
    def _create_multi_agent_reward_function(self):
        """Create reward function that coordinates multiple agents."""
        
        from .verl_reward_adapter import VERLRewardAdapter
        
        # Get agents for reward calculation
        generator_agent = self.agents.get(self.agent_roles["generator"])
        reviewer_agent = self.agents.get(self.agent_roles["reviewer"])
        executor_agent = self.agents.get(self.agent_roles["executor"])
        
        if not generator_agent:
            raise ValueError("Generator agent not registered")
            
        return VERLRewardAdapter(
            generator_agent=generator_agent,
            reviewer_agent=reviewer_agent,
            executor_agent=executor_agent,
            reward_weights=self.config.multi_agent.reward_weights,
            logger=self.logger
        )
        
    async def _run_verl_training(
        self,
        train_data: List[Dict[str, Any]],
        eval_data: Optional[List[Dict[str, Any]]],
        reward_function: Any,
        num_epochs: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Run the actual VERL distributed training."""
        
        self.logger.info("Running VERL distributed training")
        
        # In real implementation, this would call VERL's training methods
        # For now, simulate distributed training
        
        training_metrics = {
            "epochs_completed": 0,
            "average_reward": 0.0,
            "best_reward": 0.0,
            "training_loss": [],
            "evaluation_scores": []
        }
        
        for epoch in range(num_epochs):
            self.logger.info(f"Training epoch {epoch + 1}/{num_epochs}")
            
            # Simulate epoch training with VERL
            epoch_reward = await self._simulate_verl_epoch(
                train_data, reward_function, epoch
            )
            
            # Update metrics
            training_metrics["epochs_completed"] = epoch + 1
            training_metrics["training_loss"].append(0.1 / (epoch + 1))  # Decreasing loss
            training_metrics["average_reward"] = epoch_reward
            training_metrics["best_reward"] = max(
                training_metrics["best_reward"], epoch_reward
            )
            
            # Run evaluation if data provided
            if eval_data and (epoch + 1) % self.config.eval_frequency == 0:
                eval_score = await self._run_evaluation(eval_data, reward_function)
                training_metrics["evaluation_scores"].append(eval_score)
                
        training_metrics["final_reward"] = training_metrics["average_reward"]
        
        self.logger.info("VERL distributed training completed", **training_metrics)
        return training_metrics
        
    async def _simulate_verl_epoch(
        self, 
        train_data: List[Dict[str, Any]], 
        reward_function: Any, 
        epoch: int
    ) -> float:
        """Simulate a VERL training epoch (placeholder for real VERL integration)."""
        
        # In real implementation, this would be handled by VERL's training loop
        total_reward = 0.0
        batch_size = min(32, len(train_data))
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            
            # Simulate batch processing
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Simulate reward calculation
            batch_reward = 0.5 + (epoch * 0.1)  # Increasing reward over epochs
            total_reward += batch_reward
            
        return total_reward / max(len(train_data) // batch_size, 1)
        
    async def _run_evaluation(
        self, 
        eval_data: List[Dict[str, Any]], 
        reward_function: Any
    ) -> float:
        """Run evaluation on the provided dataset."""
        
        self.logger.info("Running evaluation")
        
        # Simulate evaluation
        eval_score = 0.6 + (len(eval_data) * 0.001)  # Mock evaluation score
        
        self.logger.info("Evaluation completed", eval_score=eval_score)
        return eval_score
        
    async def _save_training_results(self, results: Dict[str, Any]) -> str:
        """Save training results to checkpoint directory."""
        
        checkpoint_dir = Path(self.config.distributed.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        results_file = checkpoint_dir / f"training_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "config": self.config.dict(),
                "results": results,
                "timestamp": timestamp,
                "experiment": self.config.distributed.experiment_name
            }, f, indent=2)
            
        return str(results_file)
        
    async def shutdown(self) -> None:
        """Gracefully shutdown VERL coordinator."""
        
        self.logger.info("Shutting down VERL coordinator")
        
        if self.training_active:
            self.logger.warning("Training still active during shutdown")
            
        if self.ray_initialized:
            ray.shutdown()
            
        self.logger.info("VERL coordinator shutdown complete")


class MockVERLDistributedTrainer:
    """Mock VERL trainer for development and testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = structlog.get_logger(component="mock_verl_trainer")
        self.logger.info("Mock VERL trainer initialized", algorithm=config.get("algorithm"))
        
    async def train(self, **kwargs) -> Dict[str, Any]:
        """Mock training method."""
        self.logger.info("Mock VERL training started")
        await asyncio.sleep(1)  # Simulate training time
        return {"status": "completed", "mock": True}