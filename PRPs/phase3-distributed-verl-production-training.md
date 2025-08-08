# PRP: Phase 3 - Distributed VERL Training & Production-Ready RL Pipeline

**Version:** 1.1 (Updated to leverage native VERL capabilities)  
**Date:** 2025-01-08  
**Status:** Ready for Implementation  
**Priority:** High  

## Executive Summary

**UPDATED APPROACH**: Leverage VERL's native distributed multi-turn RL capabilities instead of building from scratch. VERL already provides Ray-orchestrated distributed training, FSDP2/Megatron backends, vLLM/SGLang inference engines, and advanced RL algorithms (GRPO, ReMax). Our focus shifts to integrating VERL's production features with our multi-agent coding framework and adding missing evaluation/deployment components.

**Key Deliverable:** Integrate VERL's production-grade distributed training capabilities with our coding framework, implement missing evaluation benchmarks, and add production deployment automation.

## Problem Statement

The current Phase 2 framework has:
✅ Basic VERL PPO training pipeline  
✅ Simple reward function system (correctness, style, efficiency)  
✅ Mock training implementation with proper interfaces  
✅ Basic configuration and checkpointing  

**VERL Already Provides (Available to leverage):**
✅ **Distributed training support** - Ray cluster orchestration with FSDP2/Megatron backends  
✅ **Advanced RL algorithms** - GRPO, ReMax, PPO with 3D-HybridEngine  
✅ **Multi-turn RL capabilities** - Native support for conversational/coding tasks  
✅ **High-performance inference** - vLLM and SGLang integration  
✅ **Memory optimization** - FSDP2 with CPU offloading and gradient accumulation  
✅ **Fault tolerance** - Built-in Ray resilience and checkpoint management  

**Still Missing (Our implementation focus):**
❌ **Code generation benchmarks** - HumanEval, MBPP, BigCodeBench evaluation  
❌ **Production deployment** - Model serving, A/B testing, gradual rollouts  
❌ **Comprehensive monitoring** - WandB/MLflow integration with VERL metrics  
❌ **Multi-agent coordination** - Integration of Generator, Reviewer, Executor agents with VERL  

**Gap:** Need to integrate VERL's distributed capabilities with our multi-agent coding framework and add production evaluation/deployment infrastructure.

## Requirements Analysis

### Functional Requirements

1. **VERL Integration & Configuration**
   - Configure VERL for multi-agent code generation workflow
   - Set up Ray cluster with VERL's distributed training orchestration
   - Configure FSDP2 backend with appropriate sharding strategies for coding models
   - Integrate vLLM/SGLang inference engines for fast response generation
   - Configure VERL's advanced algorithms (GRPO, ReMax, PPO) for coding tasks

2. **Multi-Agent VERL Coordination**
   - Adapt Generator, Reviewer, Executor agents to work with VERL training loops
   - Implement multi-turn conversation handling for iterative code refinement
   - Design reward functions compatible with VERL's multi-turn framework
   - Configure agent switching and coordination within VERL's distributed environment

3. **Code Generation Evaluation Framework** (New Implementation)
   - **HumanEval** integration (164 problems, pass@k metrics)
   - **MBPP** support (1000 basic problems with 3-shot prompting)
   - **BigCodeBench** evaluation (1140 complex tasks with function calls)
   - **CodeContests** benchmarking (competitive programming problems)
   - Integration with VERL's training loop for continuous evaluation
   - Custom evaluation metrics for multi-agent code generation workflow

4. **Production Deployment Infrastructure** (New Implementation)
   - Model serving API with VERL-trained model endpoints
   - A/B testing framework for comparing VERL algorithm performance
   - Gradual rollout strategies with automatic rollback capabilities
   - Integration with VERL's checkpoint and model management system

5. **Enhanced Monitoring & Observability** (Extension of VERL)
   - WandB/MLflow integration with VERL's existing metrics
   - Custom dashboards for multi-agent training progress
   - Code generation quality metrics and trends
   - Resource utilization monitoring for distributed VERL training

### Non-Functional Requirements

1. **Scalability**: Support training across 2-100+ GPUs with linear scaling efficiency
2. **Reliability**: <1% training failure rate with automatic recovery mechanisms
3. **Performance**: 3-5x training speed improvement through distributed optimization
4. **Maintainability**: Clean separation of concerns with modular algorithm implementations
5. **Observability**: Complete training visibility with metrics, logs, and performance traces
6. **Cost Efficiency**: Optimal resource utilization with spot instance support

### Success Criteria

- [ ] **Distributed Training**: Successfully train agents across 4+ GPUs with 80%+ scaling efficiency
- [ ] **Algorithm Performance**: GRPO/RLOO outperform PPO baseline by 15-20% on evaluation benchmarks
- [ ] **Production Readiness**: Zero-downtime deployment with automated A/B testing and rollback
- [ ] **Benchmark Performance**: Achieve competitive scores on HumanEval (>50%), MBPP (>60%), BigCodeBench (>40%)
- [ ] **Training Stability**: 95%+ training completion rate with comprehensive fault tolerance
- [ ] **Cost Optimization**: 40-60% cost reduction through efficient resource management

## Technical Research Findings

### VERL Distributed Training Capabilities

Based on research of VERL documentation and GitHub repository:

**Key Distributed Features:**
- **Ray Integration**: Native support for Ray distributed computing with automatic task scheduling
- **FSDP Backend**: PyTorch Fully Sharded Data Parallel for memory-efficient large model training
- **Multi-Backend Support**: Compatible with Megatron-LM, vLLM, SGLang for various deployment scenarios
- **3D-HybridEngine**: Efficient resource utilization across different cluster sizes
- **Multinode Training**: Manual launch, SkyPilot, Slurm, and dstack deployment options

**Scaling Architecture:**
```python
# VERL distributed training pattern
python3 -m verl.trainer.main_ppo \
    data.train_files=data/train.parquet \
    actor_rollout_ref.model.path=model_path \
    algorithm.kl_ctrl.kl_coef=0.001 \
    ray.num_workers=4 \
    ray.resources_per_worker.gpu=1
```

### Advanced RL Algorithms Research

#### GRPO (Group Relative Policy Optimization)
**Algorithm Overview**: Uses group-based advantage estimation instead of value functions
- **Memory Efficiency**: No critic network required, 50% memory reduction vs PPO
- **Performance**: Used by DeepSeek-Math and DeepSeek-R1 for mathematical reasoning
- **Implementation**: Available in VERL framework with similar training loop to PPO
- **Key Advantage**: Self-contained baselines using multi-sample group statistics

**GRPO Training Pattern:**
```python
class GRPOTrainer(VERLTrainer):
    def __init__(self, config):
        # No critic network needed
        self.group_size = config.group_size  # Multiple samples per prompt
        
    async def calculate_advantages(self, responses, rewards):
        # Use leave-one-out group baseline
        group_baselines = self._calculate_group_baselines(rewards)
        advantages = rewards - group_baselines
        return advantages
```

#### RLOO (REINFORCE Leave One Out)  
**Algorithm Overview**: Simplifies PPO by eliminating critic and using multi-sample advantages
- **Performance**: 2-3x faster than PPO, 50-70% less VRAM usage
- **Availability**: Implemented in Hugging Face TRL library
- **Robustness**: More stable than PPO with better noise resistance
- **Implementation**: Only requires policy, reference, and reward models

**RLOO Integration Pattern:**
```python
from trl import RLOOTrainer, RLOOConfig

class RLOOVERLTrainer(BaseTrainer):
    def __init__(self, config):
        self.rloo_config = RLOOConfig(
            num_rollouts=config.num_rollouts,  # K samples per prompt
            advantage_estimator="leave_one_out"
        )
```

#### DAPO (Direct Advantage Policy Optimization)
**Algorithm Overview**: Step-level RL with critic functions for dense reward signals
- **Approach**: Predicts reasoning accuracy at each generation step
- **Benefits**: Addresses sparse reward problem in code generation
- **Use Case**: Particularly effective for multi-step reasoning tasks
- **Implementation**: Requires step-level reward annotation and critic training

### Production Infrastructure Research

#### Ray Cluster Management
**Key Capabilities:**
- **Autoscaling**: Dynamic resource allocation based on training load
- **Fault Tolerance**: Automatic node failure detection and recovery
- **Resource Management**: GPU, CPU, memory allocation with priority queuing
- **Multi-Cloud Support**: AWS, GCP, Kubernetes deployment options

**Best Practices:**
- Use Ray Train for distributed ML training coordination
- Implement graceful degradation for node failures
- Configure appropriate resource requests and limits
- Enable checkpoint synchronization across nodes

#### PyTorch FSDP Integration
**Memory Optimization:**
- **Parameter Sharding**: Distribute model parameters across GPUs
- **Gradient Accumulation**: Efficient gradient synchronization
- **Mixed Precision**: bfloat16 computation with float32 accumulation
- **Activation Checkpointing**: Trade compute for memory efficiency

**FSDP Configuration:**
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def setup_fsdp_model(model):
    # Shard model parameters across GPUs
    for layer in model.layers:
        fully_shard(layer)
    fully_shard(model)
    return model
```

#### Production Monitoring Stack
**WandB Integration:**
- **Experiment Tracking**: Hyperparameter sweeps and model comparison
- **Real-time Monitoring**: Training metrics, resource utilization, model artifacts
- **Collaborative Analysis**: Team dashboards and report sharing
- **Model Registry**: Version control and deployment tracking

**MLflow Capabilities:**
- **Lifecycle Management**: End-to-end ML pipeline orchestration
- **Model Serving**: REST API deployment with load balancing
- **A/B Testing**: Model comparison framework with statistical significance
- **Reproducibility**: Experiment reproducibility with environment tracking

### Code Generation Benchmarks Analysis

#### HumanEval Benchmark
- **Structure**: 164 programming problems with function signatures and docstrings
- **Evaluation**: pass@k metric with k ∈ {1, 10, 100}
- **Usage**: Standard baseline for code generation model comparison
- **Integration**: JSON-lines format with automated test execution

#### MBPP (Mostly Basic Python Problems)  
- **Scale**: 1,000 entry-level Python programming problems
- **Splits**: Training (601-974), validation (511-600), test (11-510)
- **Format**: Task description + solution + 3 test cases
- **Prompting**: 3-shot prompting with expert programmer context

#### BigCodeBench
- **Complexity**: 1,140 software-engineering-oriented tasks
- **Features**: Function calls, complex instructions, practical applications
- **Modes**: Complete (docstring completion) vs Instruct (natural language)
- **Adoption**: Used by major AI research teams (Meta, DeepSeek, Amazon)

### Existing Codebase Integration Points

#### Configuration System Extension
Current `TrainingConfig` supports basic VERL parameters. Extension needed:
```python
class DistributedTrainingConfig(BaseModel):
    # Ray cluster configuration
    ray_cluster: RayConfig = Field(default_factory=RayConfig)
    
    # FSDP configuration  
    fsdp: FSDPConfig = Field(default_factory=FSDPConfig)
    
    # Advanced algorithms
    algorithm_configs: dict[str, BaseModel] = Field(default_factory=dict)
    
    # Evaluation configuration
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
```

#### Agent Architecture Compatibility
Existing `BaseAgent` pattern supports training mode switching:
```python
# Current pattern - extend for distributed training
agent.update_state("training_mode", True)
agent.update_state("distributed_rank", rank)
agent.update_state("world_size", world_size)
```

#### CLI Integration Pattern
Current `train` command structure supports algorithm selection:
```bash
# Extend existing pattern
coding-framework train --algorithm grpo --nodes 4 --gpus-per-node 2
coding-framework evaluate --benchmark humaneval --model-path ./checkpoints/
coding-framework deploy --model-path ./models/best --strategy gradual
```

## Implementation Blueprint

### Phase 3A: Distributed Training Infrastructure (Week 1-3)

#### Task 1: Ray Cluster Management System
**File:** `src/coding_framework/distributed/__init__.py`
```python
from .ray_cluster import RayClusterManager
from .fsdp_config import FSDPConfig, setup_fsdp_model
from .checkpoint_manager import DistributedCheckpointManager
```

**File:** `src/coding_framework/distributed/ray_cluster.py`
```python
import ray
from ray import train
from typing import Optional, Dict, Any

class RayClusterManager:
    """Manages Ray cluster initialization and resource allocation."""
    
    def __init__(self, config: RayClusterConfig):
        self.config = config
        self.cluster_info = None
        
    async def initialize_cluster(self) -> Dict[str, Any]:
        """Initialize Ray cluster with autoscaling configuration."""
        cluster_config = {
            "head_node_type": self.config.head_node_type,
            "worker_node_types": self.config.worker_node_types,
            "min_workers": self.config.min_workers,
            "max_workers": self.config.max_workers,
            "idle_timeout_minutes": self.config.idle_timeout,
        }
        
        if not ray.is_initialized():
            ray.init(
                address=self.config.cluster_address or "auto",
                runtime_env={"pip": self.config.dependencies}
            )
            
        self.cluster_info = ray.cluster_resources()
        return self.cluster_info
        
    async def scale_cluster(self, target_nodes: int) -> None:
        """Dynamically scale cluster based on training needs."""
        # Implement autoscaling logic
        pass
        
    async def health_check(self) -> Dict[str, Any]:
        """Monitor cluster health and node status."""
        return {
            "nodes_alive": len(ray.nodes()),
            "resources_available": ray.available_resources(),
            "active_tasks": len(ray.util.list_tasks()),
        }
```

#### Task 2: FSDP Integration for Memory Efficiency  
**File:** `src/coding_framework/distributed/fsdp_config.py`
```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

class FSDPConfig(BaseModel):
    """Configuration for FSDP distributed training."""
    
    mixed_precision: bool = Field(default=True)
    cpu_offload: bool = Field(default=False)
    backward_prefetch: str = Field(default="backward_pre")
    sharding_strategy: str = Field(default="full_shard")
    auto_wrap_policy: Optional[str] = Field(default="transformer")

def setup_fsdp_model(model: torch.nn.Module, config: FSDPConfig) -> FSDP:
    """Configure model for FSDP distributed training."""
    
    # Mixed precision policy
    mixed_precision_policy = None
    if config.mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    
    # Auto-wrap policy for transformer models
    auto_wrap_policy = None
    if config.auto_wrap_policy == "transformer":
        auto_wrap_policy = transformer_auto_wrap_policy
    
    # Configure FSDP
    fsdp_model = FSDP(
        model,
        mixed_precision=mixed_precision_policy,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device(),
        cpu_offload=config.cpu_offload,
    )
    
    return fsdp_model
```

#### Task 3: Distributed Checkpoint Management
**File:** `src/coding_framework/distributed/checkpoint_manager.py`
```python
import torch.distributed.checkpoint as dist_cp
from pathlib import Path
from typing import Dict, Any, Optional

class DistributedCheckpointManager:
    """Manages distributed training checkpoints with fault tolerance."""
    
    def __init__(self, checkpoint_dir: str, config: CheckpointConfig):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    async def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer, 
        epoch: int,
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save distributed training checkpoint."""
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}"
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model state using distributed checkpointing
        model_path = checkpoint_path / "model"
        dist_cp.save_state_dict(
            state_dict=model.state_dict(),
            storage_writer=dist_cp.FileSystemWriter(str(model_path)),
        )
        
        # Save optimizer state
        optimizer_path = checkpoint_path / "optimizer"  
        dist_cp.save_state_dict(
            state_dict=optimizer.state_dict(),
            storage_writer=dist_cp.FileSystemWriter(str(optimizer_path)),
        )
        
        # Save metadata
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "epoch": epoch,
                "metrics": metrics,
                "config": self.config.dict(),
                "timestamp": time.time(),
                **(metadata or {})
            }, f, indent=2)
            
        return str(checkpoint_path)
        
    async def load_checkpoint(
        self, 
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, Any]:
        """Load distributed checkpoint with recovery validation."""
        
        checkpoint_dir = Path(checkpoint_path)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        # Load model state
        model_path = checkpoint_dir / "model"
        dist_cp.load_state_dict(
            state_dict=model.state_dict(),
            storage_reader=dist_cp.FileSystemReader(str(model_path)),
        )
        
        # Load optimizer state  
        optimizer_path = checkpoint_dir / "optimizer"
        dist_cp.load_state_dict(
            state_dict=optimizer.state_dict(), 
            storage_reader=dist_cp.FileSystemReader(str(optimizer_path)),
        )
        
        # Load metadata
        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
            
        return metadata
```

### Phase 3B: Advanced RL Algorithms Implementation (Week 3-5)

#### Task 4: GRPO Trainer Implementation
**File:** `src/coding_framework/training/algorithms/grpo_trainer.py`
```python
import asyncio
import torch
from typing import List, Dict, Any, Optional
from ...training.base_trainer import BaseTrainer, TrainingResults
from ...agents.base_agent import BaseAgent

class GRPOTrainer(BaseTrainer):
    """
    Group Relative Policy Optimization trainer.
    
    Uses group-based advantage estimation instead of value functions
    for memory-efficient reinforcement learning training.
    """
    
    def __init__(self, config: TrainingConfig, grpo_config: Optional[GRPOConfig] = None):
        super().__init__(config)
        self.grpo_config = grpo_config or GRPOConfig()
        
        # GRPO specific parameters
        self.group_size = self.grpo_config.group_size
        self.temperature = self.grpo_config.temperature
        self.kl_coef = self.grpo_config.kl_coef
        
        self.logger.info(
            "GRPO trainer initialized", 
            group_size=self.group_size,
            temperature=self.temperature
        )
        
    async def train_agent(
        self,
        agent: BaseAgent,
        training_data: List[Dict[str, Any]], 
        validation_data: Optional[List[Dict[str, Any]]] = None,
        episodes: int = 100,
        reward_function: Optional[Any] = None,
        **kwargs
    ) -> TrainingResults:
        """Train agent using GRPO algorithm."""
        
        self.is_training = True
        self._total_episodes = episodes
        self._initialize_training_metrics()
        
        try:
            # Initialize distributed training if configured
            if self.config.distributed:
                await self._initialize_distributed_training()
                
            # Training loop with group-based optimization
            for episode in range(episodes):
                self.current_episode = episode
                
                # Sample batch of problems
                episode_problems = self._sample_training_batch(training_data)
                
                # Generate multiple responses per problem (group sampling)
                group_responses = await self._generate_response_groups(
                    agent, episode_problems
                )
                
                # Calculate rewards for all responses
                group_rewards = await self._calculate_group_rewards(
                    episode_problems, group_responses, reward_function
                )
                
                # GRPO advantage calculation using group baselines
                advantages = self._calculate_grpo_advantages(group_rewards)
                
                # Policy update using GRPO loss
                policy_loss = await self._update_policy_grpo(
                    agent, group_responses, advantages
                )
                
                # Update metrics
                episode_reward = torch.mean(group_rewards).item()
                self._update_episode_metrics(
                    episode, episode_reward, 
                    policy_loss=policy_loss,
                    group_size=self.group_size
                )
                
                # Validation and checkpointing
                if (episode + 1) % self.config.save_interval == 0:
                    await self.save_checkpoint(episode, agent, self.training_metrics)
                    
                if (episode + 1) % self.config.log_interval == 0:
                    await self._log_training_progress(episode, validation_data, reward_function)
            
            # Final evaluation
            final_metrics = await self._evaluate_training(agent, validation_data, reward_function)
            
            return self._create_training_results(
                success=True,
                algorithm="grpo", 
                episodes=episodes,
                checkpoint_path=await self.save_checkpoint(episodes-1, agent, final_metrics)
            )
            
        except Exception as e:
            self.logger.error(f"GRPO training failed: {e}")
            return self._create_training_results(
                success=False,
                algorithm="grpo",
                episodes=episodes,
                error=str(e)
            )
        finally:
            self.is_training = False
            
    def _calculate_grpo_advantages(self, group_rewards: torch.Tensor) -> torch.Tensor:
        """Calculate GRPO advantages using leave-one-out group baselines."""
        
        # Reshape rewards to (batch_size, group_size)  
        batch_size = group_rewards.size(0) // self.group_size
        rewards = group_rewards.view(batch_size, self.group_size)
        
        # Calculate leave-one-out baselines for each response
        advantages = torch.zeros_like(rewards)
        
        for i in range(self.group_size):
            # Use mean of other responses as baseline
            mask = torch.ones(self.group_size, dtype=torch.bool)
            mask[i] = False
            
            baselines = torch.mean(rewards[:, mask], dim=1, keepdim=True)
            advantages[:, i] = rewards[:, i] - baselines.squeeze()
            
        return advantages.view(-1)  # Flatten back to original shape
        
    async def _generate_response_groups(
        self, 
        agent: BaseAgent, 
        problems: List[Dict[str, Any]]
    ) -> List[List[str]]:
        """Generate multiple responses per problem for group-based training."""
        
        group_responses = []
        
        for problem_data in problems:
            problem = problem_data["problem"]
            responses = []
            
            # Generate group_size responses for each problem
            for _ in range(self.group_size):
                response = await agent.process_request(
                    problem, 
                    context={
                        "training_mode": True,
                        "temperature": self.temperature,
                        "episode": self.current_episode
                    }
                )
                
                if response.success:
                    responses.append(response.content)
                else:
                    # Fallback for failed generations
                    responses.append("# Failed to generate solution")
                    
            group_responses.append(responses)
            
        return group_responses
```

#### Task 5: RLOO Trainer Implementation  
**File:** `src/coding_framework/training/algorithms/rloo_trainer.py`
```python
import torch
from typing import List, Dict, Any, Optional
from ...training.base_trainer import BaseTrainer, TrainingResults

class RLOOTrainer(BaseTrainer):
    """
    REINFORCE Leave One Out trainer.
    
    Simplified RL training using multi-sample advantages without
    critic networks for improved memory efficiency and speed.
    """
    
    def __init__(self, config: TrainingConfig, rloo_config: Optional[RLOOConfig] = None):
        super().__init__(config)
        self.rloo_config = rloo_config or RLOOConfig()
        
        # RLOO specific parameters
        self.num_rollouts = self.rloo_config.num_rollouts
        self.sequence_level = self.rloo_config.sequence_level
        self.kl_penalty_coef = self.rloo_config.kl_penalty_coef
        
        # No critic network needed for RLOO
        self.requires_critic = False
        
        self.logger.info(
            "RLOO trainer initialized",
            num_rollouts=self.num_rollouts,
            sequence_level=self.sequence_level
        )
        
    async def train_agent(
        self,
        agent: BaseAgent,
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None, 
        episodes: int = 100,
        reward_function: Optional[Any] = None,
        **kwargs
    ) -> TrainingResults:
        """Train agent using RLOO algorithm."""
        
        self.is_training = True
        self._total_episodes = episodes
        self._initialize_training_metrics()
        
        try:
            # Initialize reference policy for KL penalty
            reference_policy = await self._initialize_reference_policy(agent)
            
            # Training loop
            for episode in range(episodes):
                self.current_episode = episode
                
                # Sample problems
                episode_problems = self._sample_training_batch(training_data)
                
                # Generate multiple rollouts per problem
                rollouts = await self._generate_rollouts(agent, episode_problems)
                
                # Calculate rewards 
                rewards = await self._calculate_rollout_rewards(
                    episode_problems, rollouts, reward_function
                )
                
                # RLOO advantage estimation using leave-one-out
                advantages = self._calculate_rloo_advantages(rewards)
                
                # Calculate KL penalty against reference policy
                kl_penalty = await self._calculate_kl_penalty(
                    agent, reference_policy, rollouts
                )
                
                # REINFORCE policy update
                policy_loss = await self._update_policy_reinforce(
                    agent, rollouts, advantages, kl_penalty
                )
                
                # Update metrics
                episode_reward = torch.mean(rewards).item()
                self._update_episode_metrics(
                    episode, episode_reward,
                    policy_loss=policy_loss,
                    kl_penalty=kl_penalty.item()
                )
                
                # Checkpointing and validation
                if (episode + 1) % self.config.save_interval == 0:
                    await self.save_checkpoint(episode, agent, self.training_metrics)
                    
            # Final evaluation and results
            final_metrics = await self._evaluate_training(agent, validation_data, reward_function)
            
            return self._create_training_results(
                success=True,
                algorithm="rloo",
                episodes=episodes, 
                checkpoint_path=await self.save_checkpoint(episodes-1, agent, final_metrics)
            )
            
        except Exception as e:
            self.logger.error(f"RLOO training failed: {e}")
            return self._create_training_results(
                success=False, 
                algorithm="rloo",
                episodes=episodes,
                error=str(e)
            )
        finally:
            self.is_training = False
            
    def _calculate_rloo_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Calculate RLOO advantages using leave-one-out baselines."""
        
        # Reshape to (batch_size, num_rollouts)
        batch_size = rewards.size(0) // self.num_rollouts
        reward_matrix = rewards.view(batch_size, self.num_rollouts)
        
        advantages = torch.zeros_like(reward_matrix)
        
        # Leave-one-out advantage calculation
        for i in range(self.num_rollouts):
            # Use mean of other rollouts as baseline
            other_rewards = torch.cat([
                reward_matrix[:, :i], 
                reward_matrix[:, i+1:]
            ], dim=1)
            
            baseline = torch.mean(other_rewards, dim=1, keepdim=True)
            advantages[:, i] = reward_matrix[:, i] - baseline.squeeze()
            
        return advantages.view(-1)  # Flatten back
        
    async def _generate_rollouts(
        self,
        agent: BaseAgent,
        problems: List[Dict[str, Any]]
    ) -> List[List[str]]:
        """Generate multiple rollouts per problem."""
        
        rollouts = []
        
        for problem_data in problems:
            problem = problem_data["problem"]
            problem_rollouts = []
            
            # Generate num_rollouts completions
            for rollout_idx in range(self.num_rollouts):
                response = await agent.process_request(
                    problem,
                    context={
                        "training_mode": True,
                        "rollout_idx": rollout_idx,
                        "episode": self.current_episode
                    }
                )
                
                if response.success:
                    problem_rollouts.append(response.content)
                else:
                    problem_rollouts.append("# Generation failed")
                    
            rollouts.append(problem_rollouts)
            
        return rollouts
```

#### Task 6: DAPO Trainer Implementation
**File:** `src/coding_framework/training/algorithms/dapo_trainer.py`
```python
import torch
from typing import List, Dict, Any, Optional
from ...training.base_trainer import BaseTrainer, TrainingResults

class DAPOTrainer(BaseTrainer):
    """
    Direct Advantage Policy Optimization trainer.
    
    Implements step-level RL with critic functions for dense
    reward signals in multi-step reasoning tasks.
    """
    
    def __init__(self, config: TrainingConfig, dapo_config: Optional[DAPOConfig] = None):
        super().__init__(config)  
        self.dapo_config = dapo_config or DAPOConfig()
        
        # DAPO specific parameters
        self.step_level_rewards = self.dapo_config.step_level_rewards
        self.critic_learning_rate = self.dapo_config.critic_learning_rate
        self.advantage_normalization = self.dapo_config.advantage_normalization
        
        # Initialize critic network for step-level predictions
        self.critic_network = None
        self.critic_optimizer = None
        
        self.logger.info(
            "DAPO trainer initialized",
            step_level_rewards=self.step_level_rewards,
            critic_lr=self.critic_learning_rate
        )
        
    async def train_agent(
        self,
        agent: BaseAgent, 
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None,
        episodes: int = 100,
        reward_function: Optional[Any] = None,
        **kwargs
    ) -> TrainingResults:
        """Train agent using DAPO algorithm."""
        
        self.is_training = True
        self._total_episodes = episodes
        self._initialize_training_metrics()
        
        try:
            # Initialize critic network for step-level value estimation
            await self._initialize_critic_network(agent)
            
            for episode in range(episodes):
                self.current_episode = episode
                
                # Sample training problems
                episode_problems = self._sample_training_batch(training_data)
                
                # Generate step-by-step solutions
                step_solutions = await self._generate_step_solutions(
                    agent, episode_problems
                )
                
                # Calculate step-level rewards using critic predictions
                step_rewards = await self._calculate_step_rewards(
                    episode_problems, step_solutions, reward_function
                )
                
                # Update critic network with step-level targets
                critic_loss = await self._update_critic(step_solutions, step_rewards)
                
                # Calculate advantages using critic predictions  
                advantages = await self._calculate_step_advantages(
                    step_solutions, step_rewards
                )
                
                # Policy update using step-level advantages
                policy_loss = await self._update_policy_dapo(
                    agent, step_solutions, advantages
                )
                
                # Update episode metrics
                episode_reward = torch.mean(step_rewards).item()
                self._update_episode_metrics(
                    episode, episode_reward,
                    policy_loss=policy_loss,
                    critic_loss=critic_loss
                )
                
                # Periodic evaluation and checkpointing
                if (episode + 1) % self.config.save_interval == 0:
                    checkpoint_data = {
                        **self.training_metrics,
                        "critic_state": self.critic_network.state_dict(),
                        "critic_optimizer_state": self.critic_optimizer.state_dict()
                    }
                    await self.save_checkpoint(episode, agent, checkpoint_data)
                    
            # Final evaluation
            final_metrics = await self._evaluate_training(agent, validation_data, reward_function)
            
            return self._create_training_results(
                success=True,
                algorithm="dapo",
                episodes=episodes,
                checkpoint_path=await self.save_checkpoint(episodes-1, agent, final_metrics)
            )
            
        except Exception as e:
            self.logger.error(f"DAPO training failed: {e}")
            return self._create_training_results(
                success=False,
                algorithm="dapo", 
                episodes=episodes,
                error=str(e)
            )
        finally:
            self.is_training = False
            
    async def _initialize_critic_network(self, agent: BaseAgent) -> None:
        """Initialize critic network for step-level value estimation."""
        
        # Create critic network architecture
        self.critic_network = StepLevelCritic(
            input_dim=self.dapo_config.critic_input_dim,
            hidden_dim=self.dapo_config.critic_hidden_dim,
            output_dim=1  # Single value prediction per step
        )
        
        self.critic_optimizer = torch.optim.Adam(
            self.critic_network.parameters(),
            lr=self.critic_learning_rate
        )
        
        self.logger.info("Critic network initialized for step-level value estimation")
        
    async def _generate_step_solutions(
        self,
        agent: BaseAgent,
        problems: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Generate step-by-step solutions with intermediate states."""
        
        step_solutions = []
        
        for problem_data in problems:
            problem = problem_data["problem"]
            
            # Generate solution with step-level breakdown
            response = await agent.process_request(
                problem,
                context={
                    "training_mode": True,
                    "step_level_generation": True,
                    "return_intermediate_steps": True,
                    "episode": self.current_episode
                }
            )
            
            if response.success and hasattr(response, 'intermediate_steps'):
                step_solutions.append(response.intermediate_steps)
            else:
                # Fallback to single-step solution
                step_solutions.append([{
                    "step": 0,
                    "content": response.content if response.success else "# Failed",
                    "reasoning": "Complete solution"
                }])
                
        return step_solutions
        
    async def _calculate_step_advantages(
        self,
        step_solutions: List[List[Dict[str, Any]]],
        step_rewards: torch.Tensor
    ) -> torch.Tensor:
        """Calculate step-level advantages using critic predictions."""
        
        advantages = []
        
        for solution_steps in step_solutions:
            solution_advantages = []
            
            # Get critic predictions for each step
            step_values = []
            for step_data in solution_steps:
                step_embedding = self._embed_step(step_data)
                step_value = self.critic_network(step_embedding)
                step_values.append(step_value)
                
            step_values = torch.stack(step_values)
            
            # Calculate advantages using temporal difference
            for i, step_value in enumerate(step_values):
                if i < len(step_values) - 1:
                    # TD advantage: reward + gamma * next_value - current_value  
                    next_value = step_values[i + 1]
                    advantage = step_rewards[i] + 0.99 * next_value - step_value
                else:
                    # Final step advantage
                    advantage = step_rewards[i] - step_value
                    
                solution_advantages.append(advantage)
                
            advantages.extend(solution_advantages)
            
        return torch.stack(advantages)

class StepLevelCritic(torch.nn.Module):
    """Critic network for step-level value estimation in DAPO."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(), 
            torch.nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, step_embedding: torch.Tensor) -> torch.Tensor:
        return self.network(step_embedding)
```

### Phase 3C: Comprehensive Evaluation Framework (Week 5-6)

#### Task 7: Benchmark Integration System
**File:** `src/coding_framework/evaluation/__init__.py`
```python
from .benchmark_manager import BenchmarkManager
from .evaluators import HumanEvalEvaluator, MBPPEvaluator, BigCodeBenchEvaluator
from .metrics import PassAtKMetric, ExecutionSuccessMetric, CodeQualityMetric
```

**File:** `src/coding_framework/evaluation/benchmark_manager.py`
```python
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

class BenchmarkManager:
    """Manages multiple code generation benchmarks and evaluation."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.evaluators = {
            "humaneval": HumanEvalEvaluator(config.humaneval),
            "mbpp": MBPPEvaluator(config.mbpp), 
            "bigcodebench": BigCodeBenchEvaluator(config.bigcodebench)
        }
        self.logger = structlog.get_logger(component="benchmark_manager")
        
    async def run_comprehensive_evaluation(
        self,
        agent: BaseAgent,
        benchmarks: List[str] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Run evaluation across multiple benchmarks."""
        
        benchmarks = benchmarks or list(self.evaluators.keys())
        results = {}
        
        self.logger.info(
            "Starting comprehensive evaluation",
            benchmarks=benchmarks,
            agent_id=agent.agent_id
        )
        
        # Run evaluations in parallel for efficiency
        evaluation_tasks = []
        for benchmark_name in benchmarks:
            if benchmark_name in self.evaluators:
                evaluator = self.evaluators[benchmark_name]
                task = asyncio.create_task(
                    evaluator.evaluate_agent(agent),
                    name=f"eval_{benchmark_name}"
                )
                evaluation_tasks.append((benchmark_name, task))
                
        # Collect results
        for benchmark_name, task in evaluation_tasks:
            try:
                benchmark_results = await task
                results[benchmark_name] = benchmark_results
                self.logger.info(
                    f"{benchmark_name} evaluation completed",
                    **{f"{benchmark_name}_score": benchmark_results.get("pass_at_1", 0)}
                )
            except Exception as e:
                self.logger.error(f"{benchmark_name} evaluation failed: {e}")
                results[benchmark_name] = {"error": str(e), "success": False}
                
        # Calculate aggregate metrics
        results["aggregate"] = self._calculate_aggregate_metrics(results)
        
        # Save results if requested
        if save_results:
            await self._save_evaluation_results(results, agent.agent_id)
            
        return results
        
    def _calculate_aggregate_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate performance metrics across benchmarks."""
        
        aggregate = {
            "total_problems": 0,
            "total_solved": 0,
            "weighted_pass_at_1": 0.0,
            "benchmark_scores": {}
        }
        
        # Weights for different benchmarks (can be configured)
        benchmark_weights = {
            "humaneval": 0.4,
            "mbpp": 0.3, 
            "bigcodebench": 0.3
        }
        
        total_weight = 0
        weighted_score = 0
        
        for benchmark_name, result in results.items():
            if benchmark_name != "aggregate" and isinstance(result, dict):
                if result.get("success", False):
                    pass_at_1 = result.get("pass_at_1", 0)
                    weight = benchmark_weights.get(benchmark_name, 0.1)
                    
                    weighted_score += pass_at_1 * weight
                    total_weight += weight
                    
                    aggregate["benchmark_scores"][benchmark_name] = pass_at_1
                    aggregate["total_problems"] += result.get("total_problems", 0)
                    aggregate["total_solved"] += result.get("solved_problems", 0)
                    
        # Calculate final weighted score
        if total_weight > 0:
            aggregate["weighted_pass_at_1"] = weighted_score / total_weight
            
        return aggregate
        
    async def _save_evaluation_results(
        self, 
        results: Dict[str, Any], 
        agent_id: str
    ) -> str:
        """Save evaluation results with timestamp."""
        
        output_dir = Path(self.config.results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        results_file = output_dir / f"evaluation_{agent_id}_{timestamp}.json"
        
        evaluation_data = {
            "agent_id": agent_id,
            "timestamp": timestamp,
            "evaluation_config": self.config.dict(),
            "results": results
        }
        
        with open(results_file, "w") as f:
            json.dump(evaluation_data, f, indent=2)
            
        self.logger.info("Evaluation results saved", path=str(results_file))
        return str(results_file)
```

**File:** `src/coding_framework/evaluation/evaluators/humaneval_evaluator.py`
```python
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List

class HumanEvalEvaluator:
    """Evaluator for HumanEval benchmark."""
    
    def __init__(self, config: HumanEvalConfig):
        self.config = config
        self.problems = self._load_problems()
        self.logger = structlog.get_logger(component="humaneval_evaluator")
        
    def _load_problems(self) -> List[Dict[str, Any]]:
        """Load HumanEval problems from dataset."""
        
        problems_file = Path(self.config.dataset_path) / "HumanEval.jsonl"
        if not problems_file.exists():
            raise FileNotFoundError(f"HumanEval dataset not found: {problems_file}")
            
        problems = []
        with open(problems_file) as f:
            for line in f:
                problems.append(json.loads(line.strip()))
                
        self.logger.info(f"Loaded {len(problems)} HumanEval problems")
        return problems
        
    async def evaluate_agent(self, agent: BaseAgent) -> Dict[str, Any]:
        """Evaluate agent on HumanEval benchmark."""
        
        results = {
            "benchmark": "humaneval",
            "total_problems": len(self.problems),
            "solved_problems": 0,
            "pass_at_1": 0.0,
            "pass_at_10": 0.0,
            "pass_at_100": 0.0,
            "problem_results": []
        }
        
        # Generate solutions for all problems
        self.logger.info("Generating solutions for HumanEval problems")
        
        for i, problem in enumerate(self.problems):
            problem_id = problem["task_id"]
            prompt = problem["prompt"] 
            canonical_solution = problem["canonical_solution"]
            test_cases = problem["test"]
            
            try:
                # Generate multiple solutions for pass@k evaluation
                solutions = []
                for k in range(min(self.config.max_samples, 100)):
                    response = await agent.process_request(
                        prompt,
                        context={
                            "evaluation_mode": True,
                            "benchmark": "humaneval",
                            "problem_id": problem_id,
                            "sample_idx": k
                        }
                    )
                    
                    if response.success:
                        solutions.append(response.content)
                    else:
                        solutions.append(f"# Generation failed: {response.error}")
                        
                # Execute solutions and calculate pass rates
                execution_results = await self._execute_solutions(
                    problem_id, solutions, test_cases
                )
                
                pass_counts = sum(execution_results)
                problem_pass_at_1 = 1.0 if pass_counts > 0 else 0.0
                
                # Store problem-level results
                problem_result = {
                    "task_id": problem_id,
                    "solutions_generated": len(solutions),
                    "solutions_passed": pass_counts,
                    "pass_at_1": problem_pass_at_1,
                    "execution_results": execution_results
                }
                
                results["problem_results"].append(problem_result)
                
                if problem_pass_at_1 > 0:
                    results["solved_problems"] += 1
                    
                # Log progress
                if (i + 1) % 10 == 0:
                    current_pass_rate = results["solved_problems"] / (i + 1)
                    self.logger.info(
                        f"HumanEval progress: {i+1}/{len(self.problems)}",
                        current_pass_rate=current_pass_rate
                    )
                    
            except Exception as e:
                self.logger.error(f"Error evaluating problem {problem_id}: {e}")
                problem_result = {
                    "task_id": problem_id,
                    "error": str(e),
                    "pass_at_1": 0.0
                }
                results["problem_results"].append(problem_result)
                
        # Calculate final pass@k metrics
        results["pass_at_1"] = results["solved_problems"] / results["total_problems"]
        results["pass_at_10"] = self._calculate_pass_at_k(results["problem_results"], 10)
        results["pass_at_100"] = self._calculate_pass_at_k(results["problem_results"], 100)
        
        results["success"] = True
        
        self.logger.info(
            "HumanEval evaluation completed",
            pass_at_1=results["pass_at_1"],
            solved_problems=results["solved_problems"],
            total_problems=results["total_problems"]
        )
        
        return results
        
    async def _execute_solutions(
        self, 
        problem_id: str, 
        solutions: List[str], 
        test_cases: str
    ) -> List[bool]:
        """Execute solutions against test cases safely."""
        
        execution_results = []
        
        for i, solution in enumerate(solutions):
            try:
                # Create temporary file for solution
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    # Write solution + test cases
                    test_code = f"{solution}\n\n{test_cases}\ncheck({problem_id.split('/')[-1]})"
                    f.write(test_code)
                    temp_file = f.name
                    
                # Execute in sandboxed environment
                result = subprocess.run(
                    ["python", temp_file],
                    timeout=self.config.execution_timeout,
                    capture_output=True,
                    text=True
                )
                
                # Check if execution was successful
                execution_success = (result.returncode == 0 and "AssertionError" not in result.stderr)
                execution_results.append(execution_success)
                
                # Clean up
                Path(temp_file).unlink(missing_ok=True)
                
            except subprocess.TimeoutExpired:
                execution_results.append(False)
            except Exception as e:
                self.logger.warning(f"Execution error for {problem_id} solution {i}: {e}")
                execution_results.append(False)
                
        return execution_results
        
    def _calculate_pass_at_k(self, problem_results: List[Dict[str, Any]], k: int) -> float:
        """Calculate pass@k metric across all problems."""
        
        total_problems = len(problem_results)
        if total_problems == 0:
            return 0.0
            
        pass_at_k_sum = 0.0
        
        for result in problem_results:
            solutions_passed = result.get("solutions_passed", 0)
            solutions_generated = result.get("solutions_generated", 0)
            
            if solutions_generated >= k:
                # Probability that at least one of k solutions passes
                pass_at_k_prob = 1.0 - (
                    math.comb(solutions_generated - solutions_passed, k) / 
                    math.comb(solutions_generated, k)
                ) if solutions_passed < solutions_generated else 1.0
            else:
                # Fewer solutions than k, use what we have
                pass_at_k_prob = 1.0 if solutions_passed > 0 else 0.0
                
            pass_at_k_sum += pass_at_k_prob
            
        return pass_at_k_sum / total_problems
```

### Phase 3D: Production Deployment & Monitoring (Week 6-7)

#### Task 8: Model Serving Infrastructure
**File:** `src/coding_framework/deployment/model_server.py`
```python
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional, List
import uvicorn
from pydantic import BaseModel

class ModelServingRequest(BaseModel):
    """Request schema for model serving."""
    problem: str
    context: Optional[Dict[str, Any]] = None
    model_version: Optional[str] = "latest"
    timeout: int = 30

class ModelServingResponse(BaseModel):
    """Response schema for model serving."""
    solution: str
    success: bool
    model_version: str
    execution_time: float
    metadata: Dict[str, Any]

class ModelServer:
    """Production model serving with load balancing and monitoring."""
    
    def __init__(self, config: ModelServerConfig):
        self.config = config
        self.app = FastAPI(title="Coding Framework Model Server")
        self.model_registry = {}
        self.performance_metrics = {}
        
        # Setup routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup FastAPI routes for model serving."""
        
        @self.app.post("/generate", response_model=ModelServingResponse)
        async def generate_code(
            request: ModelServingRequest,
            background_tasks: BackgroundTasks
        ):
            """Generate code solution for given problem."""
            
            start_time = time.time()
            
            try:
                # Get model version
                model_version = request.model_version
                if model_version not in self.model_registry:
                    model_version = "latest"
                    
                agent = self.model_registry[model_version]
                
                # Generate solution
                response = await agent.process_request(
                    request.problem,
                    context=request.context or {}
                )
                
                execution_time = time.time() - start_time
                
                # Log metrics in background
                background_tasks.add_task(
                    self._log_serving_metrics,
                    model_version, execution_time, response.success
                )
                
                return ModelServingResponse(
                    solution=response.content,
                    success=response.success,
                    model_version=model_version,
                    execution_time=execution_time,
                    metadata=response.metadata or {}
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                background_tasks.add_task(
                    self._log_serving_metrics,
                    model_version, execution_time, False
                )
                
                raise HTTPException(
                    status_code=500,
                    detail=f"Generation failed: {str(e)}"
                )
                
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "models_loaded": len(self.model_registry),
                "uptime": time.time() - self.start_time
            }
            
        @self.app.get("/metrics")
        async def get_metrics():
            """Get serving metrics."""
            return self.performance_metrics
            
        @self.app.post("/models/{model_version}/load")
        async def load_model(model_version: str, model_path: str):
            """Load a new model version."""
            
            try:
                # Load agent from checkpoint
                agent = await self._load_agent_from_checkpoint(model_path)
                self.model_registry[model_version] = agent
                
                return {"message": f"Model {model_version} loaded successfully"}
                
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load model: {str(e)}"
                )
                
    async def start_server(self):
        """Start the model serving server."""
        
        self.start_time = time.time()
        
        # Load default models
        await self._load_default_models()
        
        # Start server
        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            workers=self.config.workers
        )
        
        server = uvicorn.Server(config)
        await server.serve()
        
    async def _load_default_models(self):
        """Load default model versions."""
        
        for model_config in self.config.default_models:
            try:
                agent = await self._load_agent_from_checkpoint(model_config.path)
                self.model_registry[model_config.version] = agent
                
                self.logger.info(
                    f"Loaded model {model_config.version}",
                    path=model_config.path
                )
                
            except Exception as e:
                self.logger.error(
                    f"Failed to load model {model_config.version}: {e}"
                )
                
    async def _log_serving_metrics(
        self, 
        model_version: str, 
        execution_time: float, 
        success: bool
    ):
        """Log serving performance metrics."""
        
        if model_version not in self.performance_metrics:
            self.performance_metrics[model_version] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_execution_time": 0.0,
                "average_execution_time": 0.0
            }
            
        metrics = self.performance_metrics[model_version]
        metrics["total_requests"] += 1
        metrics["total_execution_time"] += execution_time
        
        if success:
            metrics["successful_requests"] += 1
            
        metrics["average_execution_time"] = (
            metrics["total_execution_time"] / metrics["total_requests"]
        )
```

#### Task 9: A/B Testing Framework
**File:** `src/coding_framework/deployment/ab_testing.py`
```python
import random
import asyncio
from typing import Dict, Any, List, Optional
import numpy as np
from scipy import stats

class ABTestManager:
    """Manages A/B testing for model deployment."""
    
    def __init__(self, config: ABTestConfig):
        self.config = config
        self.active_tests = {}
        self.test_results = {}
        self.logger = structlog.get_logger(component="ab_test_manager")
        
    async def create_ab_test(
        self,
        test_name: str,
        control_model: str,
        treatment_model: str,
        traffic_split: float = 0.5,
        success_metric: str = "correctness",
        min_samples: int = 100,
        significance_level: float = 0.05
    ) -> str:
        """Create a new A/B test."""
        
        test_config = {
            "test_name": test_name,
            "control_model": control_model,
            "treatment_model": treatment_model,
            "traffic_split": traffic_split,
            "success_metric": success_metric,
            "min_samples": min_samples,
            "significance_level": significance_level,
            "start_time": time.time(),
            "status": "active",
            "control_results": [],
            "treatment_results": []
        }
        
        self.active_tests[test_name] = test_config
        
        self.logger.info(
            "A/B test created",
            test_name=test_name,
            control_model=control_model,
            treatment_model=treatment_model,
            traffic_split=traffic_split
        )
        
        return test_name
        
    async def route_request(
        self, 
        test_name: str, 
        request: ModelServingRequest
    ) -> str:
        """Route request to control or treatment group."""
        
        if test_name not in self.active_tests:
            raise ValueError(f"Test {test_name} not found")
            
        test_config = self.active_tests[test_name]
        
        # Determine group assignment
        if random.random() < test_config["traffic_split"]:
            return test_config["treatment_model"]
        else:
            return test_config["control_model"]
            
    async def log_result(
        self,
        test_name: str,
        model_version: str,
        request: ModelServingRequest,
        response: ModelServingResponse,
        success_metrics: Dict[str, float]
    ):
        """Log result for A/B test analysis."""
        
        if test_name not in self.active_tests:
            return
            
        test_config = self.active_tests[test_name]
        
        result_data = {
            "timestamp": time.time(),
            "request": request.dict(),
            "response": response.dict(),
            "metrics": success_metrics
        }
        
        # Determine which group this result belongs to
        if model_version == test_config["control_model"]:
            test_config["control_results"].append(result_data)
        elif model_version == test_config["treatment_model"]:
            test_config["treatment_results"].append(result_data)
            
        # Check if we should analyze results
        total_results = (
            len(test_config["control_results"]) + 
            len(test_config["treatment_results"])
        )
        
        if total_results % 50 == 0:  # Analyze every 50 results
            await self._analyze_test_results(test_name)
            
    async def _analyze_test_results(self, test_name: str) -> Dict[str, Any]:
        """Analyze A/B test results for statistical significance."""
        
        test_config = self.active_tests[test_name]
        control_results = test_config["control_results"] 
        treatment_results = test_config["treatment_results"]
        
        if len(control_results) < 10 or len(treatment_results) < 10:
            return {"status": "insufficient_data"}
            
        # Extract success metric values
        metric_name = test_config["success_metric"]
        control_values = [
            result["metrics"].get(metric_name, 0) 
            for result in control_results
        ]
        treatment_values = [
            result["metrics"].get(metric_name, 0)
            for result in treatment_results  
        ]
        
        # Perform statistical test
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
        
        # Calculate confidence interval for difference
        pooled_std = np.sqrt(
            ((len(control_values) - 1) * np.var(control_values, ddof=1) +
             (len(treatment_values) - 1) * np.var(treatment_values, ddof=1)) /
            (len(control_values) + len(treatment_values) - 2)
        )
        
        se_diff = pooled_std * np.sqrt(1/len(control_values) + 1/len(treatment_values))
        margin_error = stats.t.ppf(1 - test_config["significance_level"]/2, 
                                  len(control_values) + len(treatment_values) - 2) * se_diff
        
        difference = treatment_mean - control_mean
        ci_lower = difference - margin_error
        ci_upper = difference + margin_error
        
        # Determine significance
        is_significant = p_value < test_config["significance_level"]
        
        analysis_result = {
            "test_name": test_name,
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "difference": difference,
            "percent_change": (difference / control_mean * 100) if control_mean != 0 else 0,
            "p_value": p_value,
            "t_statistic": t_stat,
            "confidence_interval": [ci_lower, ci_upper],
            "is_significant": is_significant,
            "sample_sizes": {
                "control": len(control_values),
                "treatment": len(treatment_values)
            }
        }
        
        # Store analysis result
        self.test_results[test_name] = analysis_result
        
        # Log significant results
        if is_significant:
            self.logger.info(
                "Significant A/B test result detected",
                **analysis_result
            )
            
            # Check if we should stop the test
            if (len(control_results) >= test_config["min_samples"] and 
                len(treatment_results) >= test_config["min_samples"]):
                await self._conclude_test(test_name, analysis_result)
                
        return analysis_result
        
    async def _conclude_test(
        self, 
        test_name: str, 
        analysis_result: Dict[str, Any]
    ):
        """Conclude A/B test and make deployment decision."""
        
        test_config = self.active_tests[test_name]
        test_config["status"] = "concluded"
        test_config["conclusion_time"] = time.time()
        test_config["final_analysis"] = analysis_result
        
        # Make deployment recommendation
        if analysis_result["is_significant"]:
            if analysis_result["difference"] > 0:
                recommendation = "deploy_treatment"
                message = f"Treatment model significantly outperforms control by {analysis_result['percent_change']:.2f}%"
            else:
                recommendation = "keep_control"  
                message = f"Control model significantly outperforms treatment by {abs(analysis_result['percent_change']):.2f}%"
        else:
            recommendation = "inconclusive"
            message = "No significant difference detected between models"
            
        test_config["recommendation"] = recommendation
        test_config["recommendation_message"] = message
        
        self.logger.info(
            "A/B test concluded",
            test_name=test_name,
            recommendation=recommendation,
            message=message
        )
        
    async def get_test_status(self, test_name: str) -> Dict[str, Any]:
        """Get current status of A/B test."""
        
        if test_name not in self.active_tests:
            raise ValueError(f"Test {test_name} not found")
            
        test_config = self.active_tests[test_name].copy()
        
        # Add current analysis if available
        if test_name in self.test_results:
            test_config["current_analysis"] = self.test_results[test_name]
            
        return test_config
```

#### Task 10: Monitoring & Observability Integration
**File:** `src/coding_framework/monitoring/training_dashboard.py`  
```python
import wandb
import mlflow
from typing import Dict, Any, Optional, List
import asyncio

class TrainingDashboard:
    """Comprehensive training monitoring with WandB and MLflow integration."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.wandb_run = None
        self.mlflow_run = None
        self.logger = structlog.get_logger(component="training_dashboard")
        
    async def initialize_monitoring(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """Initialize monitoring systems."""
        
        # Initialize WandB
        if self.config.wandb_enabled:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                tags=list(tags.values()) if tags else None,
                config=self.config.dict(),
                resume="allow"
            )
            
        # Initialize MLflow
        if self.config.mlflow_enabled:
            mlflow.set_experiment(experiment_name)
            self.mlflow_run = mlflow.start_run(run_name=run_name)
            
            # Log tags and config
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
                    
            mlflow.log_params(self.config.dict())
            
        self.logger.info(
            "Training monitoring initialized",
            experiment_name=experiment_name,
            wandb_enabled=self.config.wandb_enabled,
            mlflow_enabled=self.config.mlflow_enabled
        )
        
    async def log_training_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None
    ):
        """Log training metrics to monitoring systems."""
        
        # Add step/epoch to metrics
        if step is not None:
            metrics["step"] = step
        if epoch is not None:
            metrics["epoch"] = epoch
            
        # Log to WandB
        if self.wandb_run:
            wandb.log(metrics, step=step)
            
        # Log to MLflow
        if self.mlflow_run:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
                    
    async def log_system_metrics(self):
        """Log system resource usage metrics."""
        
        try:
            import psutil
            import GPUtil
            
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            system_metrics = {
                "system/cpu_percent": cpu_percent,
                "system/memory_percent": memory.percent,
                "system/memory_used_gb": memory.used / (1024**3),
                "system/memory_available_gb": memory.available / (1024**3)
            }
            
            # GPU metrics
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    system_metrics[f"system/gpu_{i}_utilization"] = gpu.load * 100
                    system_metrics[f"system/gpu_{i}_memory_percent"] = gpu.memoryUtil * 100
                    system_metrics[f"system/gpu_{i}_temperature"] = gpu.temperature
            except:
                pass  # GPU monitoring optional
                
            await self.log_training_metrics(system_metrics)
            
        except Exception as e:
            self.logger.warning(f"Failed to log system metrics: {e}")
            
    async def log_model_artifacts(
        self,
        model_path: str,
        artifact_name: str,
        artifact_type: str = "model",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log model artifacts and checkpoints."""
        
        # Log to WandB
        if self.wandb_run:
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                metadata=metadata
            )
            artifact.add_file(model_path)
            self.wandb_run.log_artifact(artifact)
            
        # Log to MLflow
        if self.mlflow_run:
            mlflow.log_artifact(model_path, artifact_path="models")
            
    async def create_custom_dashboard(
        self,
        dashboard_config: Dict[str, Any]
    ):
        """Create custom monitoring dashboard."""
        
        if self.wandb_run:
            # Create custom WandB dashboard
            wandb.define_metric("epoch")
            wandb.define_metric("training/loss", step_metric="epoch")
            wandb.define_metric("training/reward", step_metric="epoch")
            wandb.define_metric("validation/*", step_metric="epoch")
            
            # Log dashboard configuration
            wandb.config.update({"dashboard": dashboard_config})
            
    async def finalize_monitoring(self, final_metrics: Dict[str, Any]):
        """Finalize monitoring session."""
        
        # Log final metrics
        await self.log_training_metrics(final_metrics)
        
        # Finish WandB run
        if self.wandb_run:
            wandb.finish()
            
        # End MLflow run
        if self.mlflow_run:
            mlflow.end_run()
            
        self.logger.info("Training monitoring finalized")
```

## Critical Implementation Gotchas

### Distributed Training Complexity
**Issue:** Ray cluster coordination and FSDP memory management can be complex and error-prone.
**Solution:**
- Start with 2-node distributed training before scaling to larger clusters
- Implement comprehensive health checking: `ray.cluster_resources()` monitoring
- Use gradual FSDP sharding: start with `ShardingStrategy.SHARD_GRAD_OP` before `FULL_SHARD`
- Implement automatic fallback to single-node training on cluster failures

### Advanced Algorithm Stability
**Issue:** GRPO, RLOO, DAPO algorithms can have different convergence characteristics than PPO.
**Solution:**
- **GRPO**: Ensure group_size ≥ 4 for stable baselines, monitor group reward variance
- **RLOO**: Use num_rollouts ≥ 8 for reliable advantage estimation, implement reward clipping
- **DAPO**: Carefully tune critic learning rate (typically 10x higher than policy), use step-level reward smoothing

### Memory Management at Scale  
**Issue:** Large models with advanced algorithms can exceed GPU memory limits.
**Solution:**
- Implement gradient accumulation with FSDP: `accumulation_steps = target_batch_size // micro_batch_size`
- Use mixed precision training: `autocast()` with `torch.bfloat16`
- Configure CPU offloading for FSDP: `cpu_offload=CPUOffload(offload_params=True)`
- Monitor memory usage: `torch.cuda.memory_summary()` after each training step

### Evaluation Benchmark Integration
**Issue:** Different benchmarks have varying formats, execution requirements, and evaluation metrics.
**Solution:**
- Standardize evaluation interface: common `EvaluatorBase` class with `evaluate_agent()` method
- Implement secure code execution: Docker containers with resource limits and network isolation
- Handle benchmark-specific quirks: HumanEval uses `check()` function, MBPP uses assert statements
- Cache benchmark datasets and precompute problem embeddings for faster evaluation

### Production Deployment Reliability
**Issue:** Model serving failures and A/B testing statistical power requirements.
**Solution:**  
- Implement graceful degradation: fallback to previous model version on serving failures
- Use proper statistical power calculation: minimum 100 samples per group for 80% power
- Configure health checks with retries: 3 failed health checks before model rotation
- Implement circuit breaker pattern: temporary fallback on high error rates

### Resource Cost Optimization
**Issue:** Distributed training and evaluation can be expensive without proper optimization.
**Solution:**
- Use spot instances with automatic fallback to on-demand: AWS Spot Fleet or GCP Preemptible
- Implement training job scheduling: priority queues and resource sharing across experiments
- Optimize evaluation frequency: run comprehensive benchmarks weekly, quick validation daily
- Cache model inference results: Redis cache for repeated evaluation problems

## Validation Gates

### Distributed Training Validation
```bash
# Test Ray cluster setup and scaling
uv run python -c "
import ray
ray.init()
print('Ray cluster:', ray.cluster_resources())
assert ray.cluster_resources()['CPU'] >= 8, 'Insufficient CPU resources'
print('✅ Ray cluster validation passed')
"

# Test FSDP model sharding
uv run python examples/distributed_training/test_fsdp.py --gpus 2 --batch-size 4
# Must complete without OOM errors and show memory reduction vs standard training
```

### Algorithm Implementation Validation
```bash
# Test GRPO trainer implementation
uv run pytest tests/training/test_grpo_trainer.py -v
# Must show group advantage calculation correctness and memory efficiency vs PPO

# Test RLOO trainer implementation  
uv run pytest tests/training/test_rloo_trainer.py -v
# Must demonstrate faster training speed and lower memory usage

# Test DAPO trainer with step-level rewards
uv run pytest tests/training/test_dapo_trainer.py -v  
# Must show step-level advantage calculation and critic network training
```

### Evaluation Framework Validation
```bash
# Test benchmark integration
uv run python -m coding_framework.evaluation.test_benchmarks --quick-test
# Must successfully load and evaluate on subset of HumanEval, MBPP, BigCodeBench

# Test evaluation metrics calculation
uv run pytest tests/evaluation/test_metrics.py -v --cov=src/coding_framework/evaluation
# Must achieve >95% test coverage and accurate pass@k calculations
```

### Production Deployment Validation
```bash
# Test model serving API
uv run python examples/deployment/test_model_server.py --port 8000 --timeout 30
# Must handle concurrent requests with <500ms average latency

# Test A/B testing framework
uv run python examples/deployment/test_ab_testing.py --control-model ppo --treatment-model grpo
# Must correctly route traffic and calculate statistical significance
```

### Monitoring Integration Validation
```bash
# Test WandB integration
export WANDB_PROJECT=coding-framework-test
uv run python examples/monitoring/test_wandb_integration.py --episodes 5
# Must successfully log metrics, artifacts, and create dashboard

# Test MLflow tracking
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run python examples/monitoring/test_mlflow_integration.py --episodes 5
# Must log parameters, metrics, and model artifacts successfully
```

### End-to-End Integration Test
```bash
# Full Phase 3 integration test
uv run python examples/phase3/integration_test.py \
  --algorithm grpo \
  --nodes 2 \
  --episodes 20 \
  --benchmark humaneval \
  --enable-monitoring \
  --enable-ab-testing

# Success criteria:
# - Distributed training completes successfully
# - Algorithm achieves >baseline performance
# - Evaluation runs without errors  
# - A/B test framework captures results
# - Monitoring dashboards show complete metrics
```

## Risk Assessment

### High Risk Items
1. **Distributed Training Complexity** - Ray/FSDP integration challenges
   - **Mitigation**: Start with 2-node setup, comprehensive testing, automatic fallbacks
2. **Algorithm Implementation Correctness** - Advanced RL algorithms have subtle requirements  
   - **Mitigation**: Reference implementations, extensive unit tests, algorithm comparison studies
3. **Production Scale Reliability** - Complex monitoring and deployment pipeline
   - **Mitigation**: Staged rollout, comprehensive health checks, circuit breaker patterns

### Medium Risk Items  
1. **Resource Cost Overrun** - Distributed training can be expensive
   - **Mitigation**: Spot instance usage, resource monitoring, cost budgets with alerts
2. **Evaluation Framework Accuracy** - Benchmark integration correctness
   - **Mitigation**: Cross-validation with official implementations, manual verification

### Low Risk Items
1. **Configuration Extension** - Well-established Pydantic configuration system
2. **Monitoring Integration** - Mature WandB/MLflow libraries with extensive documentation

## Dependencies

### Internal Dependencies
- Phase 2 VERL training foundation (extends existing VERLPPOTrainer)
- Existing BaseAgent architecture (no breaking changes)
- Configuration system (extends TrainingConfig)
- CLI framework (adds new commands)

### External Dependencies
```toml
# Add to pyproject.toml [project.optional-dependencies.phase3]
"ray[default,train,tune]>=2.8.0",        # Distributed computing
"torch>=2.1.0",                          # PyTorch with FSDP support  
"transformers>=4.35.0",                  # HuggingFace models
"accelerate>=0.24.0",                    # FSDP integration
"datasets>=2.15.0",                      # Benchmark datasets
"wandb>=0.16.0",                         # Experiment tracking
"mlflow>=2.8.0",                         # ML lifecycle management
"fastapi>=0.104.0",                      # Model serving API
"uvicorn>=0.24.0",                       # ASGI server
"scipy>=1.11.0",                         # Statistical analysis
"GPUtil>=1.4.0",                         # GPU monitoring
```

### System Dependencies
- **Docker**: For secure code execution and model serving
- **NVIDIA Drivers**: For GPU training (CUDA 11.8+)
- **Ray Cluster**: Multi-node distributed training (optional for single-node)

## Implementation Tasks Summary

### Phase 3A: Distributed Training Infrastructure (Week 1-3)
- [ ] **Task 1**: Ray Cluster Management System
- [ ] **Task 2**: FSDP Integration for Memory Efficiency  
- [ ] **Task 3**: Distributed Checkpoint Management

### Phase 3B: Advanced RL Algorithms (Week 3-5)
- [ ] **Task 4**: GRPO Trainer Implementation
- [ ] **Task 5**: RLOO Trainer Implementation
- [ ] **Task 6**: DAPO Trainer Implementation

### Phase 3C: Comprehensive Evaluation Framework (Week 5-6)
- [ ] **Task 7**: Benchmark Integration System (HumanEval, MBPP, BigCodeBench)

### Phase 3D: Production Deployment & Monitoring (Week 6-7)
- [ ] **Task 8**: Model Serving Infrastructure
- [ ] **Task 9**: A/B Testing Framework  
- [ ] **Task 10**: Monitoring & Observability Integration

## Success Metrics

### Quantitative Success Criteria
- [ ] **Distributed Scaling**: 80%+ efficiency scaling from 1 to 4 GPUs
- [ ] **Algorithm Performance**: GRPO/RLOO achieve 15-20% improvement over PPO baseline
- [ ] **Benchmark Scores**: HumanEval >50%, MBPP >60%, BigCodeBench >40%  
- [ ] **Training Reliability**: 95%+ training completion rate with fault tolerance
- [ ] **Deployment Uptime**: 99.9% model serving availability
- [ ] **Cost Optimization**: 40-60% cost reduction through resource efficiency

### Qualitative Success Criteria  
- [ ] **Code Quality**: All new code follows existing architectural patterns
- [ ] **Documentation**: Comprehensive examples and deployment guides
- [ ] **Maintainability**: Clean modular design with clear separation of concerns
- [ ] **Production Readiness**: Zero-downtime deployment with monitoring and rollback
- [ ] **Developer Experience**: Simple CLI commands for complex distributed operations

## Future Extensibility

This Phase 3 implementation provides foundation for:
- **Phase 4**: Multi-agent RL training (training reviewer and executor agents)
- **Phase 5**: Human-in-the-loop learning with real-time feedback integration  
- **Phase 6**: Auto-scaling production clusters with Kubernetes orchestration
- **Phase 7**: Advanced safety constraints and red-team evaluation frameworks
- **Phase 8**: Domain-specific model variants (web development, data science, systems programming)

## PRP Confidence Score: 9/10

**High Confidence Factors:**
- Comprehensive research of distributed training patterns and advanced RL algorithms
- Detailed understanding of production ML infrastructure requirements
- Clear integration points with existing Phase 2 codebase architecture  
- Thorough validation gates covering all major system components
- Specific implementation blueprints with concrete code examples
- Risk mitigation strategies for known distributed training challenges

**Minor Risk Factors:**
- First-time implementation of advanced RL algorithms in production environment (-0.5 points)
- Complex distributed training coordination across multiple system components (-0.5 points)

**Overall Assessment:** This PRP provides comprehensive context and detailed implementation guidance for successful one-pass implementation of production-grade distributed VERL training with advanced algorithms, comprehensive evaluation, and deployment automation.