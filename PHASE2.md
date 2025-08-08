FEATURE: VERL Training Pipeline Integration & Basic Reward Functions
Based on your Phase 1 foundation, Phase 2 should focus on implementing the core VERL training pipeline with basic reward functions to make your multi-agent system actually learn and improve. This bridges the gap between your excellent orchestration framework and the reinforcement learning capabilities that differentiate your project.
EXAMPLES:
Training Pipeline Examples Structure:
examples/
├── verl_training/
│   ├── README.md                     # Training pipeline overview and setup
│   ├── basic_training.py             # Simple PPO training example
│   ├── reward_functions/
│   │   ├── __init__.py
│   │   ├── correctness_reward.py     # Binary pass/fail reward
│   │   ├── style_reward.py           # Code style scoring
│   │   ├── efficiency_reward.py      # Performance-based rewards
│   │   └── composite_reward.py       # Multi-factor reward combination
│   ├── training_data/
│   │   ├── simple_problems.json      # 50-100 basic coding problems
│   │   ├── leetcode_easy.json        # Curated easy LeetCode problems
│   │   └── synthetic_problems.py     # Problem generation utilities
│   ├── evaluation/
│   │   ├── benchmark_suite.py        # Evaluation metrics and testing
│   │   ├── baseline_comparison.py    # Compare against non-RL baseline
│   │   └── training_metrics.py       # Track training progress
│   └── configs/
│       ├── ppo_basic.yaml           # Simple PPO configuration
│       ├── distributed_training.yaml # Ray cluster setup
│       └── reward_weights.yaml      # Reward function parameters
Example Training Session:
python# examples/verl_training/basic_training.py
async def train_coding_agents():
    """Example of basic VERL training loop"""
    # 1. Load training problems (50-100 simple problems)
    # 2. Initialize VERL PPO trainer with basic reward function
    # 3. Train for 100 episodes with correctness + style rewards
    # 4. Evaluate on held-out test set
    # 5. Save checkpoint and training metrics
Reward Function Examples:
python# examples/verl_training/reward_functions/correctness_reward.py
class CorrectnessReward(BaseRewardFunction):
    """Binary reward: 1.0 if code passes all tests, 0.0 otherwise"""
    
# examples/verl_training/reward_functions/composite_reward.py  
class CompositeReward(BaseRewardFunction):
    """Weighted combination: 0.7*correctness + 0.2*style + 0.1*efficiency"""
DOCUMENTATION:
VERL Integration Documentation:

VERL Core Documentation: https://verl.readthedocs.io/en/latest/
VERL GitHub Examples: https://github.com/volcengine/verl/tree/main/examples
VERL PPO Recipe: https://github.com/volcengine/verl/tree/main/recipe/ppo
VERL FSDP Backend: https://verl.readthedocs.io/en/latest/start/installation.html#fsdp-backend
Ray Integration Guide: https://docs.ray.io/en/latest/train/getting-started.html

Training Data Sources:

LeetCode API Documentation: For problem scraping and test case generation
HumanEval Dataset: https://github.com/openai/human-eval (coding benchmarks)
CodeContests Dataset: https://github.com/google-deepmind/code_contests
MBPP Dataset: https://github.com/google-research/google-research/tree/master/mbpp

Reward Function Research:

CodeT5 Paper: Reward function design for code generation
AlphaCode Paper: Multi-objective reward functions
CodeRL Paper: Reinforcement learning for code generation specifically

Ray/Distributed Training:

Ray Train Documentation: https://docs.ray.io/en/latest/train/train.html
VERL Distributed Training: https://verl.readthedocs.io/en/latest/workers/ray.html
PyTorch FSDP Guide: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html

OTHER CONSIDERATIONS:
Critical VERL Integration Gotchas:

Memory Management: VERL training is memory-intensive. Your current config has max_tokens: 2048 but VERL episodes need sequence length planning. Consider:

Gradient accumulation steps for large batches
Model sharding with FSDP
Sequence length limits during training vs inference


Reward Function Stability:

Sparse vs Dense Rewards: Start with binary correctness, gradually add dense signals
Reward Scaling: Normalize rewards to [-1, 1] range to prevent training instability
Reward Clipping: Prevent extreme rewards from destabilizing training


Training Data Quality:

Problem Diversity: Balance simple/complex problems to prevent overfitting
Test Case Coverage: Ensure test cases cover edge cases, not just happy path
Ground Truth Solutions: Need multiple correct solutions per problem for reward calculation



Agent-Specific Training Considerations:
Code Generator Agent Training:

Action Space: Token-level generation with VERL PPO
State Representation: Problem description + partial code context
Training Objective: Maximize composite reward (correctness + style + efficiency)
Exploration: Temperature scheduling during training

Code Reviewer Agent Training:

Training Mode: Supervised learning on review quality scores initially
Reward Signal: Correlation between review suggestions and final code improvement
Data Requirements: Need human-annotated code reviews for initial training

Code Executor Agent Training:

Minimal RL: Primarily rule-based with learning for timeout/resource optimization
Safety First: No RL training on security-critical execution decisions
Performance Learning: Learn optimal resource allocation for different code types

Infrastructure Requirements:

Ray Cluster Setup: Your current architecture supports this, but need actual cluster configuration
Model Checkpointing: VERL training needs frequent checkpoints due to instability
Monitoring Integration: W&B integration for training metrics (already in your config)
Data Pipeline: Efficient loading of training problems during distributed training

Integration with Existing Architecture:

Supervisor Modification: Current CodingSupervisor.train_agents() is a placeholder - needs actual VERL integration
Agent State Management: Training mode vs inference mode for agents
Configuration Extension: Your current config system needs VERL-specific parameters
Workflow Adaptation: Training workflows vs inference workflows in LangGraph

Phase 2 Success Criteria:

Functional Training Loop: Can train Code Generator with basic PPO for 100 episodes
Measurable Improvement: Training shows improvement in correctness reward over episodes
Baseline Comparison: Trained agent outperforms non-RL baseline on held-out problems
Reproducible Results: Training runs are deterministic and results can be reproduced
Resource Efficiency: Training completes within reasonable time/memory constraints

Technical Debt to Address:

Docker Executor Integration: Currently has fallbacks - needs full Docker support for training safety
LLM Interface Robustness: Training requires more robust error handling than inference
State Serialization: VERL needs to serialize/deserialize agent states during training
Metrics Collection: Training requires more granular metrics than current performance tracking

This Phase 2 focus will transform your excellent orchestration framework into a true learning system, setting up the foundation for the advanced features planned in subsequent phases.