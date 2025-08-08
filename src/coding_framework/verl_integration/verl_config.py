from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import structlog


class VERLDistributedConfig(BaseModel):
    """Configuration for VERL distributed training."""
    
    # Algorithm selection
    algorithm: str = Field(default="ppo", description="VERL algorithm: ppo, grpo, remax")
    
    # Distributed training configuration
    num_gpus: int = Field(default=4, description="Total number of GPUs for training")
    num_nodes: int = Field(default=1, description="Number of nodes in the cluster")
    gpus_per_node: int = Field(default=4, description="Number of GPUs per node")
    
    # Ray configuration (leveraging VERL's Ray integration)
    ray_cluster_address: Optional[str] = Field(default=None, description="Ray cluster address")
    ray_namespace: str = Field(default="verl_coding", description="Ray namespace")
    
    # FSDP2 configuration (leveraging VERL's FSDP2 support)
    strategy: str = Field(default="fsdp2", description="Training strategy: fsdp2, megatron")
    fsdp_offload_policy: bool = Field(default=True, description="Enable FSDP2 CPU offloading")
    fsdp_sharding_strategy: str = Field(default="FULL_SHARD", description="FSDP sharding strategy")
    
    # Model configuration
    model_name: str = Field(default="Qwen/Qwen2.5-Coder-7B-Instruct", description="Base model")
    max_length: int = Field(default=2048, description="Maximum sequence length")
    
    # Training parameters
    rollout_batch_size: int = Field(default=512, description="Rollout batch size")
    train_batch_size: int = Field(default=128, description="Training batch size")
    learning_rate: float = Field(default=1e-5, description="Learning rate")
    gradient_accumulation_steps: int = Field(default=8, description="Gradient accumulation steps")
    
    # VERL-specific parameters
    rollout_ref_num_nodes: int = Field(default=1, description="Nodes for rollout reference")
    rollout_ref_num_gpus_per_node: int = Field(default=4, description="GPUs per node for rollout")
    actor_num_nodes: int = Field(default=1, description="Nodes for actor")
    actor_num_gpus_per_node: int = Field(default=4, description="GPUs per node for actor")
    critic_num_nodes: int = Field(default=1, description="Nodes for critic")
    critic_num_gpus_per_node: int = Field(default=2, description="GPUs per node for critic")
    reward_model_num_nodes: int = Field(default=1, description="Nodes for reward model")
    reward_model_num_gpus_per_node: int = Field(default=2, description="GPUs per node for reward model")
    
    # Inference engine configuration
    vllm_tensor_parallel_size: int = Field(default=4, description="vLLM tensor parallel size")
    vllm_max_model_len: int = Field(default=4096, description="vLLM max model length")
    vllm_gpu_memory_utilization: float = Field(default=0.9, description="vLLM GPU memory utilization")
    
    # Checkpointing and logging
    checkpoint_dir: str = Field(default="./verl_checkpoints", description="Checkpoint directory")
    experiment_name: str = Field(default="coding_verl_experiment", description="Experiment name")
    save_freq: int = Field(default=100, description="Checkpoint save frequency")
    
    # Resource limits
    memory_limit_gb: int = Field(default=80, description="Memory limit per GPU in GB")
    timeout_seconds: int = Field(default=3600, description="Training timeout in seconds")


class MultiAgentVERLConfig(BaseModel):
    """Configuration for multi-agent coordination with VERL."""
    
    # Agent configuration
    generator_agent_id: str = Field(default="generator", description="Generator agent ID")
    reviewer_agent_id: str = Field(default="reviewer", description="Reviewer agent ID") 
    executor_agent_id: str = Field(default="executor", description="Executor agent ID")
    
    # Multi-turn conversation configuration
    max_turns: int = Field(default=5, description="Maximum conversation turns")
    turn_timeout: int = Field(default=120, description="Timeout per turn in seconds")
    
    # Reward function configuration
    reward_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "correctness": 0.6,
            "code_quality": 0.2, 
            "efficiency": 0.1,
            "review_score": 0.1
        },
        description="Reward component weights"
    )
    
    # Agent coordination strategy
    coordination_strategy: str = Field(
        default="sequential", 
        description="Agent coordination: sequential, parallel, adaptive"
    )
    
    # VERL multi-turn settings
    enable_multi_turn: bool = Field(default=True, description="Enable multi-turn RL")
    conversation_reward_aggregation: str = Field(
        default="final_turn", 
        description="Reward aggregation: final_turn, cumulative, discounted"
    )
    discount_factor: float = Field(default=0.95, description="Discount factor for multi-turn rewards")
    
    # Agent switching logic
    enable_agent_switching: bool = Field(default=True, description="Enable dynamic agent switching")
    switching_threshold: float = Field(default=0.7, description="Quality threshold for agent switching")
    max_agent_switches: int = Field(default=3, description="Maximum agent switches per conversation")


class VERLTrainingConfig(BaseModel):
    """Combined VERL training configuration."""
    
    distributed: VERLDistributedConfig = Field(default_factory=VERLDistributedConfig)
    multi_agent: MultiAgentVERLConfig = Field(default_factory=MultiAgentVERLConfig)
    
    # Training dataset configuration
    train_data_path: str = Field(default="./data/coding_problems.jsonl", description="Training data path")
    eval_data_path: str = Field(default="./data/eval_problems.jsonl", description="Evaluation data path")
    data_preprocessing: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_problem_length": 1024,
            "filter_duplicates": True,
            "augment_data": True
        },
        description="Data preprocessing configuration"
    )
    
    # Evaluation configuration
    eval_frequency: int = Field(default=50, description="Evaluation frequency (steps)")
    eval_batch_size: int = Field(default=64, description="Evaluation batch size")
    eval_benchmarks: List[str] = Field(
        default_factory=lambda: ["humaneval", "mbpp", "bigcodebench"],
        description="Evaluation benchmarks to run"
    )
    
    # Monitoring configuration
    wandb_project: Optional[str] = Field(default="verl-coding-framework", description="WandB project")
    mlflow_tracking_uri: Optional[str] = Field(default=None, description="MLflow tracking URI")
    log_level: str = Field(default="INFO", description="Logging level")
    enable_profiling: bool = Field(default=False, description="Enable performance profiling")
    
    def to_verl_config(self) -> Dict[str, Any]:
        """Convert to VERL's configuration format."""
        
        logger = structlog.get_logger(component="verl_config_converter")
        
        verl_config = {
            # Algorithm configuration
            "algorithm": self.distributed.algorithm,
            
            # Model configuration
            "actor_rollout_ref": {
                "model": {
                    "path": self.distributed.model_name,
                    "max_length": self.distributed.max_length,
                },
                "ref": {
                    "strategy": self.distributed.strategy,
                },
                "actor": {
                    "strategy": self.distributed.strategy,
                    "fsdp_config": {
                        "offload_policy": self.distributed.fsdp_offload_policy,
                    }
                },
                "rollout": {
                    "num_nodes": self.distributed.rollout_ref_num_nodes,
                    "num_gpus_per_node": self.distributed.rollout_ref_num_gpus_per_node,
                    "vllm_config": {
                        "tensor_parallel_size": self.distributed.vllm_tensor_parallel_size,
                        "max_model_len": self.distributed.vllm_max_model_len,
                        "gpu_memory_utilization": self.distributed.vllm_gpu_memory_utilization,
                    }
                }
            },
            
            # Critic configuration
            "critic": {
                "strategy": self.distributed.strategy,
                "num_nodes": self.distributed.critic_num_nodes,
                "num_gpus_per_node": self.distributed.critic_num_gpus_per_node,
            },
            
            # Reward model configuration
            "reward_model": {
                "strategy": self.distributed.strategy,
                "num_nodes": self.distributed.reward_model_num_nodes,
                "num_gpus_per_node": self.distributed.reward_model_num_gpus_per_node,
            },
            
            # Ray configuration
            "ray": {
                "cluster_address": self.distributed.ray_cluster_address,
                "namespace": self.distributed.ray_namespace,
            },
            
            # Training configuration
            "train": {
                "batch_size": self.distributed.train_batch_size,
                "rollout_batch_size": self.distributed.rollout_batch_size,
                "learning_rate": self.distributed.learning_rate,
                "gradient_accumulation_steps": self.distributed.gradient_accumulation_steps,
            },
            
            # Checkpointing
            "checkpoint": {
                "output_dir": self.distributed.checkpoint_dir,
                "save_freq": self.distributed.save_freq,
            },
            
            # Experiment tracking
            "experiment": {
                "name": self.distributed.experiment_name,
                "project": self.wandb_project,
            },
            
            # Multi-agent specific configuration (custom extension)
            "multi_agent": {
                "enable": True,
                "max_turns": self.multi_agent.max_turns,
                "coordination_strategy": self.multi_agent.coordination_strategy,
                "reward_weights": self.multi_agent.reward_weights,
                "enable_agent_switching": self.multi_agent.enable_agent_switching,
            }
        }
        
        logger.info("VERL configuration generated", config_keys=list(verl_config.keys()))
        return verl_config