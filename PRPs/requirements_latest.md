# Multi-Agent Multi-Turn RL for CUDA Code Generation
## Requirements Document v1.0

---

**Project**: MultiMindDev CUDA Optimization Framework  
**Document Type**: Technical Requirements Specification  
**Version**: 1.0  
**Date**: August 2025  
**Author**: MultiMindDev Team  

---

## Table of Contents

1. [Project Vision](#project-vision)
2. [Core Transformation Objectives](#core-transformation-objectives)
3. [Agent Architecture Requirements](#agent-architecture-requirements)
4. [SFT Training Stages](#sft-training-stages)
5. [Multi-Turn RL Framework](#multi-turn-rl-framework)
6. [VERL Integration Specifications](#verl-integration-specifications)
7. [Implementation Phases](#implementation-phases)
8. [Success Criteria](#success-criteria)
9. [Technical Resources & Documentation](#technical-resources--documentation)
10. [Research Papers & References](#research-papers--references)

---

## Project Vision

Transform the current LangGraph-orchestrated multi-agent system into a **true multi-turn reinforcement learning framework** where individual agents own trainable models and improve through collaborative experience. The end goal is a system of specialized, trained models that work together to generate optimized CUDA kernels through multi-turn conversations.

### Current vs Target Architecture

| Current State | Target State |
|---------------|--------------|
| Agents wrap external LLMs (Claude/GPT) | Agents own trainable model parameters |
| LangGraph orchestrates conversations | Direct agent-to-agent communication |
| No parameter updates or learning | VERL-based RL training with improving performance |
| Sophisticated prompt engineering | True multi-turn reinforcement learning |

---

## Core Transformation Objectives

### 1. Architectural Shift: From Orchestration to Training

**Remove Dependencies**:
- ❌ LangGraph for agent coordination
- ❌ External LLM APIs (Claude/GPT/Anthropic)
- ❌ `LLMInterface` wrapper pattern

**Add Capabilities**:
- ✅ Owned transformer model parameters per agent
- ✅ VERL-based PPO training infrastructure
- ✅ Multi-turn conversation learning
- ✅ Collaborative reward optimization

### 2. Agent Specialization Through Training

**Replace**: Generic `BaseAgent` wrappers  
**With**: `TrainableAgent` classes with specialized capabilities

**Core Training Philosophy**:
- Each agent starts with domain-specific SFT
- Agents learn collaboration through multi-turn RL
- Performance improves through experience across episodes
- Reward signals drive specialization and cooperation

---

## Agent Architecture Requirements

### Base TrainableAgent Class

```python
class TrainableAgent:
    """Base class for agents with owned, trainable parameters"""
    
    def __init__(self, model_name: str, agent_type: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.optimizer = AdamW(self.model.parameters())
        
    async def generate_with_log_probs(self, prompt: str) -> Tuple[str, torch.Tensor]:
        """Generate response while tracking log probabilities for RL training"""
        pass
        
    def update_parameters(self, rewards: torch.Tensor, log_probs: torch.Tensor):
        """Update model parameters using VERL PPO"""
        pass
```

### Agent Specializations

#### 1. CUDA Generator Agent
- **Purpose**: Create initial CUDA kernels from PyTorch operation descriptions
- **Base Model**: `Qwen/Qwen2.5-Coder-7B-Instruct`
- **Specialization**: Syntactic correctness, compilability, CUDA patterns
- **Input**: PyTorch operation descriptions, tensor shapes, data types
- **Output**: Complete CUDA C++ kernel implementations
- **Key Capabilities**:
  - Thread indexing and grid sizing
  - Memory access pattern generation
  - Boundary checking and error handling
  - Kernel launch parameter calculation

#### 2. CUDA Optimizer Agent
- **Purpose**: Apply performance optimizations to existing kernels
- **Base Model**: `Qwen/Qwen2.5-Coder-7B-Instruct` (separate instance)
- **Specialization**: Memory coalescing, shared memory, register optimization
- **Input**: Existing kernel code, performance analysis, optimization targets
- **Output**: Performance-optimized kernel variants
- **Key Capabilities**:
  - Shared memory utilization
  - Memory coalescing improvements
  - Vectorized memory access (float4, etc.)
  - Loop unrolling and instruction optimization

#### 3. CUDA Tester Agent
- **Purpose**: Compile, test, and profile kernels for correctness and performance
- **Implementation**: Rule-based system + optional small trained model
- **Specialization**: Compilation testing, profiling, issue identification
- **Input**: CUDA kernel code, test specifications
- **Output**: Compilation results, performance metrics, issue reports
- **Key Capabilities**:
  - nvcc compilation with error parsing
  - Functional correctness validation
  - Performance benchmarking and profiling
  - Memory usage analysis

---

## SFT Training Stages

### Stage 1: Individual Agent SFT (Phase 1)

#### Generator Agent SFT
**Objective**: Learn basic CUDA kernel generation patterns
**Duration**: 2-3 epochs
**Dataset Size**: ~10,000 examples

**Data Format**:
```json
{
  "input": "Generate CUDA kernel for: torch.add(a, b)\nShapes: a=[1024], b=[1024]\nDtype: float32",
  "output": "#include <cuda_runtime.h>\n__global__ void add_kernel(float* a, float* b, float* c, int n) {\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx < n) {\n        c[idx] = a[idx] + b[idx];\n    }\n}"
}
```

**Data Sources**:
- SakanaAI/AI-CUDA-Engineer-Archive (filtered for generation examples)
- Synthetic PyTorch → CUDA pairs
- Template-based kernel generation

**Success Criteria**:
- 90%+ compilation success rate
- 80%+ functional correctness on test cases
- Proper CUDA syntax and structure patterns

#### Optimizer Agent SFT
**Objective**: Learn performance optimization patterns
**Duration**: 2-3 epochs
**Dataset Size**: ~8,000 examples

**Data Format**:
```json
{
  "input": "Optimize this CUDA kernel:\n[SLOW_KERNEL]\nIssues: Poor memory coalescing, no shared memory usage\nTarget: 2x speedup",
  "output": "[OPTIMIZED_KERNEL_WITH_SHARED_MEMORY_AND_COALESCING]"
}
```

**Data Sources**:
- Paired unoptimized/optimized kernel examples
- Performance analysis reports → optimization pairs
- Manual optimization examples from CUDA best practices

**Success Criteria**:
- 70%+ performance improvement on optimization tasks
- Correct application of optimization techniques
- Maintained functional correctness after optimization

#### Tester Agent SFT (Optional)
**Objective**: Learn to identify performance issues and testing patterns
**Duration**: 1-2 epochs (if using trained model)
**Dataset Size**: ~5,000 examples

**Alternative**: Rule-based implementation using nvcc, profiling tools

**Data Sources**:
- Kernel code → compilation result pairs
- Performance profiling data → issue identification
- Test case generation examples

### Stage 2: Cross-Agent Compatibility Training (Phase 1.5)

**Objective**: Ensure agents can work together effectively
**Duration**: 1 epoch
**Method**: Joint training on multi-turn conversations

**Process**:
1. Run multi-turn conversations using SFT-trained agents
2. Collect conversation data with standardized input/output formats
3. Fine-tune agents on successful collaboration patterns
4. Ensure consistent communication protocols

---

## Multi-Turn RL Framework

### Conversation Structure

```python
@dataclass
class CUDAConversationState:
    """Tracks state across multi-turn conversation"""
    problem: str                              # Original PyTorch operation
    turns: List[ConversationTurn]            # Agent interactions
    current_kernel: Optional[str]            # Latest kernel version
    performance_history: List[float]         # Speedup progression
    compilation_results: List[CompilationResult]  # Technical feedback
    final_reward: float                      # Episode outcome

@dataclass
class ConversationTurn:
    agent_type: str                          # "generator", "optimizer", "tester"
    input_text: str                          # Prompt to agent
    output_text: str                         # Agent response
    log_probs: Optional[torch.Tensor]        # For RL training
    immediate_reward: float                  # Turn-level feedback
    timestamp: float
```

### Episode Flow

```
Episode Start: Sample CUDA optimization problem
├─ Turn 1: Generator creates initial kernel
├─ Turn 2: Tester evaluates kernel (compilation, correctness, performance)
├─ Turn 3: Optimizer improves kernel (if needed)
├─ Turn 4: Tester validates optimized kernel
└─ Episode End: Calculate final reward and distribute across turns
```

### Reward Distribution Strategy

#### Final Reward Calculation
```python
def calculate_final_reward(self, conversation_state: CUDAConversationState) -> float:
    """Calculate episode reward based on final kernel performance"""
    
    # Core performance metrics
    speedup_score = min(final_speedup / target_speedup, 2.0) * 0.4
    correctness_score = 1.0 if all_tests_pass else 0.0 * 0.3
    efficiency_score = memory_efficiency * register_efficiency * 0.2
    compilation_score = 1.0 if compiles_successfully else 0.0 * 0.1
    
    return speedup_score + correctness_score + efficiency_score + compilation_score
```

#### Turn-Level Credit Assignment
```python
def distribute_rewards_across_turns(
    self, 
    final_reward: float, 
    conversation_state: CUDAConversationState
) -> List[float]:
    """Distribute final episode reward across conversation turns"""
    
    turn_rewards = []
    discount_factor = 0.9
    
    for turn_idx, turn in enumerate(conversation_state.turns):
        if turn.agent_type in ["generator", "optimizer"]:  # Only trained agents
            # Immediate reward component
            immediate_reward = self.calculate_immediate_reward(turn)
            
            # Discounted final reward component
            turns_remaining = len(conversation_state.turns) - turn_idx - 1
            discounted_final = final_reward * (discount_factor ** turns_remaining)
            
            # Weighted combination: 30% immediate, 70% final outcome
            turn_reward = 0.3 * immediate_reward + 0.7 * discounted_final
            turn_rewards.append(turn_reward)
        else:
            turn_rewards.append(0.0)  # Tester doesn't get trained
    
    return turn_rewards
```

### Training Loop Architecture

```python
async def run_multi_turn_training(self, num_episodes: int = 10000):
    """Main multi-turn RL training loop"""
    
    for episode in range(num_episodes):
        # Sample problem from curriculum
        problem = await self.data_loader.sample_problem()
        
        # Run multi-turn conversation
        conversation_state = await self.run_conversation_episode(problem)
        
        # Calculate rewards
        final_reward = self.calculate_final_reward(conversation_state)
        turn_rewards = self.distribute_rewards_across_turns(final_reward, conversation_state)
        
        # Extract training data for each agent
        generator_data = self.extract_agent_data(conversation_state, "generator")
        optimizer_data = self.extract_agent_data(conversation_state, "optimizer")
        
        # Update agent parameters with VERL
        if generator_data.has_data():
            await self.verl_trainer.train_step(
                agent=self.generator_agent,
                prompts=generator_data.prompts,
                responses=generator_data.responses,
                rewards=generator_data.rewards,
                old_log_probs=generator_data.log_probs
            )
            
        if optimizer_data.has_data():
            await self.verl_trainer.train_step(
                agent=self.optimizer_agent,
                prompts=optimizer_data.prompts,
                responses=optimizer_data.responses,
                rewards=optimizer_data.rewards,
                old_log_probs=optimizer_data.log_probs
            )
        
        # Periodic evaluation and checkpointing
        if episode % 100 == 0:
            await self.evaluate_agents(episode)
            await self.save_checkpoint(episode)
```

---

## VERL Integration Specifications

### VERL Configuration

```python
@dataclass
class VERLTrainingConfig:
    """VERL-specific training configuration"""
    
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    max_sequence_length: int = 2048
    
    # Distributed training
    num_gpus: int = 8
    num_rollout_workers: int = 4
    num_actor_workers: int = 2
    num_critic_workers: int = 2
    
    # PPO hyperparameters
    ppo_epochs: int = 4
    mini_batch_size: int = 8
    learning_rate: float = 1e-5
    kl_coef: float = 0.02
    clip_ratio: float = 0.2
    
    # Multi-turn specific
    max_turns_per_episode: int = 5
    turn_discount_factor: float = 0.9
    early_termination_threshold: float = 0.8
```

### Ray Cluster Setup

```yaml
# ray_cluster.yaml
cluster_name: cuda-multi-agent-rl

provider:
    type: aws
    region: us-west-2

head_node:
    InstanceType: p3.8xlarge  # 4 V100 GPUs
    ImageId: ami-0abcdef1234567890  # CUDA-enabled AMI

worker_nodes:
    InstanceType: p3.8xlarge
    ImageId: ami-0abcdef1234567890
    MinWorkers: 2
    MaxWorkers: 4

setup_commands:
    - pip install verl ray torch transformers
    - pip install nvidia-ml-py3 pynvml
```

### Training Infrastructure Requirements

**Computational Resources**:
- **Minimum**: 4x GPU setup (V100/A100/RTX 4090)
- **Recommended**: 8x GPU distributed setup
- **Memory**: 32GB+ RAM per node
- **Storage**: 1TB+ SSD for models and datasets

**Software Stack**:
- VERL v0.2.0+ (multi-turn RL support)
- Ray v2.0+ (distributed computing)
- PyTorch v2.0+ (model training)
- CUDA Toolkit 11.8+ (kernel compilation)
- Docker (containerized execution)

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
**Objective**: Transform architecture from orchestration to trainable agents

**Tasks**:
- [ ] Design and implement `TrainableAgent` base class
- [ ] Remove LangGraph dependencies
- [ ] Remove external LLM API dependencies
- [ ] Implement individual agent SFT training pipeline
- [ ] Create CUDA Generator Agent with specialized SFT
- [ ] Create CUDA Optimizer Agent with specialized SFT
- [ ] Build conversation state management framework

**Deliverables**:
- Trainable agent architecture
- SFT-trained Generator and Optimizer agents
- Conversation state tracking system
- Basic multi-turn conversation capability (without RL)

### Phase 2: Multi-Turn RL Integration (Weeks 3-4)
**Objective**: Implement VERL-based multi-turn RL training

**Tasks**:
- [ ] Integrate VERL PPO trainer with agent architecture
- [ ] Implement episode collection and reward calculation
- [ ] Build turn-level credit assignment system
- [ ] Create distributed training pipeline with Ray
- [ ] Implement training loop with curriculum progression
- [ ] Add comprehensive logging and monitoring

**Deliverables**:
- VERL-integrated training pipeline
- Multi-turn RL training capability
- Distributed training infrastructure
- Training monitoring and evaluation systems

### Phase 3: Training and Optimization (Weeks 5-6)
**Objective**: Execute large-scale training and optimize performance

**Tasks**:
- [ ] Run large-scale multi-turn RL training (10,000+ episodes)
- [ ] Implement curriculum learning and difficulty progression
- [ ] Optimize training stability and convergence
- [ ] Develop comprehensive evaluation benchmarks
- [ ] Compare against baseline systems
- [ ] Implement advanced reward shaping techniques

**Deliverables**:
- Trained multi-agent system
- Performance benchmarks and evaluation
- Training optimization and stability improvements
- Comparative analysis with existing tools

### Phase 4: Deployment and Production (Weeks 7-8)
**Objective**: Prepare production-ready deployment

**Tasks**:
- [ ] Package trained agents as unified ensemble
- [ ] Build production API and containerization
- [ ] Implement monitoring and performance tracking
- [ ] Create documentation and usage guides
- [ ] Prepare open-source release
- [ ] Submit research paper

**Deliverables**:
- Production-ready deployment package
- API documentation and usage guides
- Open-source repository release
- Research paper submission

---

## Success Criteria

### Technical Objectives

**Learning Convergence**:
- [ ] Agents show measurable performance improvement over training episodes
- [ ] Training loss decreases consistently over time
- [ ] Episode rewards increase with statistical significance

**Collaboration Quality**:
- [ ] Multi-turn conversations produce better results than single-turn approaches
- [ ] Agents develop complementary specializations
- [ ] Early termination rates improve (agents learn when to stop)

**Performance Benchmarks**:
- [ ] System matches or exceeds existing CUDA optimization tools
- [ ] Generated kernels achieve target speedup ratios (2x minimum)
- [ ] Compilation success rate > 95%
- [ ] Functional correctness rate > 90%

### Research Contributions

**Novel Architecture**:
- [ ] First demonstration of multi-turn RL for collaborative code generation
- [ ] Successful training of specialized programming agents
- [ ] Effective credit assignment in multi-agent conversations

**Practical Impact**:
- [ ] Deployable system for real-world CUDA optimization
- [ ] Open-source framework for multi-agent code generation
- [ ] Reproducible training methodology

---

## Technical Resources & Documentation

### VERL Framework
- **Main Repository**: https://github.com/volcengine/verl
- **Documentation**: https://verl.readthedocs.io/en/latest/
- **Multi-Turn RL Guide**: https://verl.readthedocs.io/en/latest/tutorials/multi_turn_rl.html
- **PPO Implementation**: https://verl.readthedocs.io/en/latest/algorithms/ppo.html

### Ray Distributed Computing
- **Ray Core Documentation**: https://docs.ray.io/en/latest/ray-core/walkthrough.html
- **Ray Train**: https://docs.ray.io/en/latest/train/train.html
- **Distributed PyTorch**: https://docs.ray.io/en/latest/train/examples/pytorch/pytorch_quick_start.html

### Multi-Agent RL Frameworks
- **OpenAI Multi-Agent**: https://openai.com/research/emergent-tool-use
- **DeepMind MADDPG**: https://github.com/deepmind/maddpg
- **RLLib Multi-Agent**: https://docs.ray.io/en/latest/rllib/rllib-concepts.html#multi-agent-and-hierarchical

### CUDA Programming Resources
- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **CUDA Best Practices**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **NSight Compute Profiler**: https://docs.nvidia.com/nsight-compute/
- **CUDA Samples**: https://github.com/NVIDIA/cuda-samples

### Code Generation & LLMs
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers/
- **CodeT5/CodeBERT**: https://github.com/salesforce/CodeT5
- **StarCoder Models**: https://huggingface.co/bigcode/starcoder2-15b

---

## Research Papers & References

### Multi-Agent Reinforcement Learning
1. **"Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"** (2017)
   - Authors: Lowe et al.
   - Link: https://arxiv.org/abs/1706.02275
   - Key Insight: MADDPG algorithm for multi-agent training

2. **"Emergent Tool Use From Multi-Agent Autocurricula"** (2019)
   - Authors: Baker et al. (OpenAI)
   - Link: https://arxiv.org/abs/1909.07528
   - Key Insight: Emergent behaviors in multi-agent environments

3. **"The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"** (2022)
   - Authors: Yu et al.
   - Link: https://arxiv.org/abs/2103.01955
   - Key Insight: PPO effectiveness in multi-agent settings

### Multi-Turn Dialogue and RL
4. **"Multi-Turn Dialogue Response Generation via Mutual Information Maximization"** (2021)
   - Authors: Meng et al.
   - Link: https://arxiv.org/abs/2106.10041
   - Key Insight: Multi-turn conversation optimization

5. **"Training Language Models to Follow Instructions with Human Feedback"** (2022)
   - Authors: Ouyang et al. (OpenAI)
   - Link: https://arxiv.org/abs/2203.02155
   - Key Insight: RLHF for instruction following

### Code Generation with RL
6. **"CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning"** (2022)
   - Authors: Le et al.
   - Link: https://arxiv.org/abs/2207.01780
   - Key Insight: RL for code generation improvement

7. **"Competition-level code generation with AlphaCode"** (2022)
   - Authors: Li et al. (DeepMind)
   - Link: https://arxiv.org/abs/2203.07814
   - Key Insight: Large-scale code generation approaches

8. **"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"** (2020)
   - Authors: Raffel et al.
   - Link: https://arxiv.org/abs/1910.10683
   - Key Insight: T5 architecture for code tasks

### CUDA and GPU Computing
9. **"Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"** (2021)
   - Authors: Tillet et al.
   - Link: https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf
   - Key Insight: Automated GPU kernel generation

10. **"TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems"** (2015)
    - Authors: Abadi et al.
    - Link: https://arxiv.org/abs/1603.04467
    - Key Insight: GPU computation optimization

### Multi-Agent Code Generation (Novel)
11. **"CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation"** (2021)
    - Authors: Wang et al.
    - Link: https://arxiv.org/abs/2109.00859
    - Key Insight: Code understanding for collaborative generation

12. **"Program Synthesis with Large Language Models"** (2021)
    - Authors: Austin et al.
    - Link: https://arxiv.org/abs/2108.07732
    - Key Insight: LLM program synthesis capabilities

---

## Additional Resources

### Datasets
- **SakanaAI/AI-CUDA-Engineer-Archive**: https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive
- **CodeSearchNet**: https://github.com/github/CodeSearchNet
- **The Stack**: https://huggingface.co/datasets/bigcode/the-stack
- **HumanEval**: https://github.com/openai/human-eval

### Tools and Frameworks
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
- **NSight Systems**: https://developer.nvidia.com/nsight-systems
- **Docker CUDA**: https://hub.docker.com/r/nvidia/cuda
- **Weights & Biases**: https://wandb.ai/site

### Community and Support
- **VERL Discord**: https://discord.gg/verl-community
- **Ray Community**: https://discuss.ray.io/
- **CUDA Programming Forums**: https://forums.developer.nvidia.com/c/gpu-accelerated-libraries/cuda-programming-and-performance/111

---

*This requirements document serves as a comprehensive guide for transforming the current multi-agent orchestration system into a true multi-turn reinforcement learning framework for CUDA code generation. The implementation should follow the phases outlined above while leveraging the research and technical resources provided for optimal results.*