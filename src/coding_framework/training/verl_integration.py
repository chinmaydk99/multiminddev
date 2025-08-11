import asyncio
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
import structlog
import torch

# VERL imports
try:
    from verl import (
        ActorRolloutRef,
        CriticRef,
        DataProto,
        RayWorkerGroup,
        RewardModelRef,
        VERLTrainer,
    )
    from verl.trainer.dapo import DAPOTrainer
    from verl.trainer.grpo import GRPOTrainer
    from verl.trainer.ppo import PPOTrainer
    from verl.utils.reward_score import get_reward_score
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False
    structlog.get_logger().warning("VERL not available, using mock implementation")

# Dataset imports
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# ==============================================================================
# DATA PIPELINE
# ==============================================================================

class SakanaDataLoader:
    """Loader for SakanaAI CUDA dataset with curriculum support."""

    def __init__(
        self,
        dataset_name: str = "SakanaAI/AI-CUDA-Engineer-Archive",
        cache_dir: Optional[str] = None,
        curriculum_enabled: bool = True,
        fallback_data_path: Optional[str] = None
    ):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.curriculum_enabled = curriculum_enabled
        self.fallback_data_path = fallback_data_path
        self.logger = structlog.get_logger()

        # Load and process dataset
        self._load_dataset()
        self._process_curriculum()

    def _load_dataset(self):
        """Load the SakanaAI dataset."""
        if DATASETS_AVAILABLE:
            try:
                self.dataset = load_dataset(
                    self.dataset_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
                self.logger.info(
                    "SakanaAI dataset loaded",
                    num_train=len(self.dataset["train"]) if "train" in self.dataset else 0,
                    num_test=len(self.dataset["test"]) if "test" in self.dataset else 0
                )
                return
            except Exception as e:
                self.logger.warning(f"Failed to load SakanaAI dataset: {e}")

        # Fallback to local data
        if self.fallback_data_path and os.path.exists(self.fallback_data_path):
            self._load_fallback_data()
        else:
            self._create_synthetic_data()

    def _load_fallback_data(self):
        """Load fallback data from local files."""
        with open(self.fallback_data_path) as f:
            data = json.load(f)

        self.dataset = {"train": data.get("train", []), "test": data.get("test", [])}
        self.logger.info("Fallback dataset loaded", num_train=len(self.dataset["train"]))

    def _create_synthetic_data(self):
        """Create synthetic CUDA problems for testing."""
        synthetic_problems = [
            {
                "problem_description": "Implement element-wise vector addition for two vectors of size N.",
                "cuda_kernel": "__global__ void vector_add(float* a, float* b, float* c, int n) { int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx < n) c[idx] = a[idx] + b[idx]; }",
                "difficulty": "easy",
                "shapes": [[1024], [4096], [8192]],
                "target_performance": {"speedup": 1.5}
            },
            {
                "problem_description": "Implement matrix multiplication using shared memory optimization.",
                "cuda_kernel": "__global__ void matmul(float* A, float* B, float* C, int N) { __shared__ float As[16][16]; __shared__ float Bs[16][16]; /* implementation */ }",
                "difficulty": "medium",
                "shapes": [[512, 512], [1024, 1024]],
                "target_performance": {"speedup": 2.0}
            },
            {
                "problem_description": "Implement 2D convolution with optimized memory access patterns.",
                "cuda_kernel": "__global__ void conv2d(float* input, float* kernel, float* output, int H, int W) { /* complex implementation */ }",
                "difficulty": "hard",
                "shapes": [[256, 256, 3], [512, 512, 3]],
                "target_performance": {"speedup": 3.0}
            }
        ]

        self.dataset = {"train": synthetic_problems, "test": []}
        self.logger.info("Synthetic dataset created", num_problems=len(synthetic_problems))

    def _process_curriculum(self):
        """Process dataset for curriculum learning."""
        if not self.curriculum_enabled:
            return

        train_data = self.dataset["train"] if "train" in self.dataset else []

        self.curriculum_data = {
            "easy": [],
            "medium": [],
            "hard": []
        }

        for example in train_data:
            difficulty = self._infer_difficulty(example)
            self.curriculum_data[difficulty].append(example)

        self.logger.info(
            "Curriculum data processed",
            easy=len(self.curriculum_data["easy"]),
            medium=len(self.curriculum_data["medium"]),
            hard=len(self.curriculum_data["hard"])
        )

    def _infer_difficulty(self, example: Dict[str, Any]) -> str:
        """Infer difficulty from example characteristics."""
        if "difficulty" in example:
            return example["difficulty"]

        problem_desc = example.get("problem_description", "").lower()
        kernel_code = example.get("cuda_kernel", "")

        # Simple heuristics for difficulty classification
        if any(keyword in problem_desc for keyword in ["vector", "element-wise", "simple", "basic"]):
            return "easy"
        elif any(keyword in problem_desc for keyword in ["matrix", "reduction", "transpose"]):
            return "medium"
        elif any(keyword in problem_desc for keyword in ["convolution", "fft", "sort", "graph"]):
            return "hard"

        if kernel_code:
            if "__shared__" in kernel_code or "atomicAdd" in kernel_code:
                return "hard"
            elif "blockIdx" in kernel_code and "threadIdx" in kernel_code:
                return "medium"

        return "medium"

    async def sample_problem(
        self,
        difficulty: Optional[str] = None,
        curriculum_tier: Optional[str] = None
    ) -> Dict[str, Any]:
        """Sample a problem from the dataset."""
        if self.curriculum_enabled and curriculum_tier:
            difficulty = curriculum_tier

        if difficulty and difficulty in self.curriculum_data:
            candidates = self.curriculum_data[difficulty]
        else:
            candidates = list(self.dataset["train"]) if "train" in self.dataset else []

        if not candidates:
            raise ValueError(f"No problems available for difficulty: {difficulty}")

        example = random.choice(candidates)

        return {
            "description": example.get("problem_description", ""),
            "difficulty": difficulty or self._infer_difficulty(example),
            "reference_solution": example.get("cuda_kernel", ""),
            "test_cases": self._generate_test_cases(example),
            "target_performance": self._extract_target_performance(example),
            "baseline_performance": example.get("baseline_performance", {}),
            "metadata": {
                "original_example": example,
                "dataset": self.dataset_name
            }
        }

    def _generate_test_cases(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases for the problem."""
        shapes = example.get("shapes", [])
        if not shapes:
            difficulty = self._infer_difficulty(example)
            if difficulty == "easy":
                shapes = [[1024], [4096]]
            elif difficulty == "medium":
                shapes = [[1024, 1024], [2048, 2048]]
            else:
                shapes = [[512, 512, 512], [1024, 1024, 256]]

        test_cases = []
        for shape in shapes:
            test_cases.append({
                "input_shapes": [shape, shape] if len(shape) <= 2 else [shape],
                "dtype": torch.float32,
                "grid_dims": self._calculate_grid_dims(shape),
                "block_dims": self._calculate_block_dims(shape)
            })

        return test_cases

    def _calculate_grid_dims(self, shape: List[int]) -> Tuple[int, int, int]:
        """Calculate appropriate grid dimensions for shape."""
        if len(shape) == 1:
            return (min(32, (shape[0] + 255) // 256), 1, 1)
        elif len(shape) == 2:
            return (min(32, (shape[0] + 15) // 16), min(32, (shape[1] + 15) // 16), 1)
        else:
            return (min(16, (shape[0] + 7) // 8), min(16, (shape[1] + 7) // 8), min(8, (shape[2] + 7) // 8))

    def _calculate_block_dims(self, shape: List[int]) -> Tuple[int, int, int]:
        """Calculate appropriate block dimensions for shape."""
        if len(shape) == 1:
            return (256, 1, 1)
        elif len(shape) == 2:
            return (16, 16, 1)
        else:
            return (8, 8, 8)

    def _extract_target_performance(self, example: Dict[str, Any]) -> Dict[str, float]:
        """Extract target performance metrics."""
        target = example.get("target_performance", {})
        if target:
            return target

        difficulty = self._infer_difficulty(example)
        if difficulty == "easy":
            return {"speedup": 1.5}
        elif difficulty == "medium":
            return {"speedup": 2.0}
        else:
            return {"speedup": 3.0}

# ==============================================================================
# CURRICULUM LEARNING SYSTEM
# ==============================================================================

@dataclass
class CurriculumTier:
    """Definition of a curriculum learning tier."""
    name: str
    difficulty: str
    success_threshold: float = 0.7  # Success rate needed to advance
    min_episodes: int = 50  # Minimum episodes before considering advancement
    max_episodes: int = 500  # Maximum episodes in this tier
    performance_target: float = 2.0  # Target speedup for this tier
    unlock_requirements: List[str] = field(default_factory=list)  # Previous tiers that must be completed

@dataclass
class CurriculumProgress:
    """Tracking progress through curriculum."""
    current_tier: str = "easy"
    tier_episodes: int = 0
    tier_successes: int = 0
    tier_start_time: float = field(default_factory=time.time)
    unlocked_tiers: List[str] = field(default_factory=lambda: ["easy"])
    performance_history: deque = field(default_factory=lambda: deque(maxlen=100))

class CurriculumManager:
    """Manages curriculum learning progression."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = structlog.get_logger()

        # Define curriculum tiers
        self.tiers = {
            "easy": CurriculumTier(
                name="easy",
                difficulty="easy",
                success_threshold=0.7,
                min_episodes=30,
                max_episodes=200,
                performance_target=1.5
            ),
            "medium": CurriculumTier(
                name="medium",
                difficulty="medium",
                success_threshold=0.6,
                min_episodes=50,
                max_episodes=300,
                performance_target=2.0,
                unlock_requirements=["easy"]
            ),
            "hard": CurriculumTier(
                name="hard",
                difficulty="hard",
                success_threshold=0.5,
                min_episodes=100,
                max_episodes=500,
                performance_target=3.0,
                unlock_requirements=["easy", "medium"]
            )
        }

        self.progress = CurriculumProgress()

    def get_current_difficulty(self) -> str:
        """Get current curriculum difficulty."""
        return self.tiers[self.progress.current_tier].difficulty

    def record_episode_result(self, conversation_result) -> bool:
        """Record episode result and check for tier advancement."""
        self.progress.tier_episodes += 1

        if conversation_result.conversation_success:
            self.progress.tier_successes += 1

        # Record performance
        final_speedup = conversation_result.current_performance.get("speedup", 0.0)
        self.progress.performance_history.append(final_speedup)

        # Check for tier advancement
        return self._check_tier_advancement()

    def _check_tier_advancement(self) -> bool:
        """Check if agent should advance to next tier."""
        current_tier = self.tiers[self.progress.current_tier]

        # Must meet minimum episode requirement
        if self.progress.tier_episodes < current_tier.min_episodes:
            return False

        # Check success rate
        success_rate = self.progress.tier_successes / self.progress.tier_episodes
        if success_rate < current_tier.success_threshold:
            # Force advancement if maximum episodes reached
            if self.progress.tier_episodes >= current_tier.max_episodes:
                self.logger.warning(
                    "Forcing tier advancement due to max episodes",
                    current_tier=self.progress.current_tier,
                    success_rate=success_rate,
                    episodes=self.progress.tier_episodes
                )
                return self._advance_tier()
            return False

        # Check performance target
        if len(self.progress.performance_history) >= 10:
            recent_avg_performance = np.mean(list(self.progress.performance_history)[-10:])
            if recent_avg_performance < current_tier.performance_target:
                return False

        # Find next available tier
        return self._advance_tier()

    def _advance_tier(self) -> bool:
        """Advance to next available tier."""
        next_tier = self._find_next_tier()
        if next_tier:
            self.logger.info(
                "Advancing curriculum tier",
                from_tier=self.progress.current_tier,
                to_tier=next_tier,
                episodes=self.progress.tier_episodes,
                successes=self.progress.tier_successes
            )

            self.progress.current_tier = next_tier
            self.progress.tier_episodes = 0
            self.progress.tier_successes = 0
            self.progress.tier_start_time = time.time()

            if next_tier not in self.progress.unlocked_tiers:
                self.progress.unlocked_tiers.append(next_tier)

            return True
        return False

    def _find_next_tier(self) -> Optional[str]:
        """Find next available tier to advance to."""
        tier_order = ["easy", "medium", "hard"]
        current_idx = tier_order.index(self.progress.current_tier)

        for i in range(current_idx + 1, len(tier_order)):
            candidate = tier_order[i]
            tier = self.tiers[candidate]

            # Check if all requirements are unlocked
            if all(req in self.progress.unlocked_tiers for req in tier.unlock_requirements):
                return candidate

        return None

# ==============================================================================
# SAFETY FEATURES
# ==============================================================================

class CUDASafetyAnalyzer:
    """Advanced safety analysis for generated CUDA code."""

    def __init__(self):
        self.logger = structlog.get_logger()
        self.dangerous_patterns = [
            (r"while\s*\(\s*1\s*\)", "Infinite loop detected"),
            (r"for\s*\([^;]*;[^;]*;[^)]*\)\s*{[^}]*}", "Potentially unbounded loop"),
            (r"recursion|recursive", "Recursion in CUDA kernel (stack overflow risk)"),
            (r"malloc\s*\(", "Dynamic memory allocation in kernel"),
            (r"printf\s*\(.*very.*long", "Excessive printf output"),
            (r"__syncthreads\s*\(\s*\)\s*;\s*if", "Divergent __syncthreads (deadlock risk)"),
            (r"atomicAdd.*while", "Potential atomic operation in loop (performance)"),
            (r"\*\s*\(\s*\(\s*\w+\s*\*\s*\)\s*0\s*\)", "Null pointer dereference")
        ]

    def analyze_code_safety(self, code: str) -> Dict[str, Any]:
        """Perform comprehensive safety analysis on CUDA code."""
        import re

        safety_report = {
            "is_safe": True,
            "warnings": [],
            "critical_issues": [],
            "performance_concerns": [],
            "memory_safety": True,
            "thread_safety": True
        }

        # Check for dangerous patterns
        for pattern, description in self.dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                if "deadlock" in description.lower() or "overflow" in description.lower():
                    safety_report["critical_issues"].append(description)
                    safety_report["is_safe"] = False
                elif "performance" in description.lower():
                    safety_report["performance_concerns"].append(description)
                else:
                    safety_report["warnings"].append(description)

        # Memory safety analysis
        malloc_count = len(re.findall(r'malloc\s*\(', code))
        free_count = len(re.findall(r'free\s*\(', code))
        if malloc_count > 0 and free_count < malloc_count:
            safety_report["memory_safety"] = False
            safety_report["critical_issues"].append("Potential memory leak detected")
            safety_report["is_safe"] = False

        # Thread safety analysis
        if "__syncthreads" in code:
            # Check for potential divergent execution paths
            if_count = len(re.findall(r'if\s*\(', code))
            syncthreads_count = len(re.findall(r'__syncthreads\s*\(\s*\)', code))
            if syncthreads_count > 0 and if_count > syncthreads_count:
                safety_report["thread_safety"] = False
                safety_report["warnings"].append("Potential thread divergence with __syncthreads")

        # Resource usage analysis
        shared_memory_usage = self._analyze_shared_memory_usage(code)
        if shared_memory_usage > 48000:  # 48KB limit for many GPUs
            safety_report["warnings"].append(f"High shared memory usage: {shared_memory_usage} bytes")

        return safety_report

    def _analyze_shared_memory_usage(self, code: str) -> int:
        """Estimate shared memory usage from code."""
        import re

        total_usage = 0

        # Look for shared memory declarations
        shared_patterns = [
            r'__shared__\s+\w+\s+\w+\[\s*(\d+)\s*\]',
            r'__shared__\s+\w+\s+\w+\[\s*(\d+)\s*\]\s*\[\s*(\d+)\s*\]'
        ]

        for pattern in shared_patterns:
            matches = re.findall(pattern, code)
            for match in matches:
                if isinstance(match, tuple):
                    size = int(match[0]) * int(match[1]) if len(match) > 1 else int(match[0])
                else:
                    size = int(match)
                total_usage += size * 4  # Assume 4 bytes per element

        return total_usage

class ResourceLimiter:
    """Enforce resource limits during CUDA execution."""

    def __init__(
        self,
        max_execution_time: float = 30.0,  # seconds
        max_memory_usage: int = 2 * 1024**3,  # 2GB
        max_gpu_memory: int = 1 * 1024**3  # 1GB GPU memory
    ):
        self.max_execution_time = max_execution_time
        self.max_memory_usage = max_memory_usage
        self.max_gpu_memory = max_gpu_memory
        self.logger = structlog.get_logger()

    async def execute_with_limits(self, execution_func, *args, **kwargs):
        """Execute function with resource limits."""

        start_time = time.time()
        initial_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        try:
            # Set up timeout
            result = await asyncio.wait_for(
                execution_func(*args, **kwargs),
                timeout=self.max_execution_time
            )

            # Check memory usage
            if torch.cuda.is_available():
                current_gpu_memory = torch.cuda.memory_allocated()
                if current_gpu_memory - initial_gpu_memory > self.max_gpu_memory:
                    self.logger.warning(
                        "GPU memory limit exceeded",
                        used=current_gpu_memory - initial_gpu_memory,
                        limit=self.max_gpu_memory
                    )

            execution_time = time.time() - start_time
            self.logger.debug(
                "Execution completed within limits",
                execution_time=execution_time,
                gpu_memory_used=torch.cuda.memory_allocated() - initial_gpu_memory if torch.cuda.is_available() else 0
            )

            return result

        except asyncio.TimeoutError:
            self.logger.error(
                "Execution timeout",
                max_time=self.max_execution_time,
                elapsed=time.time() - start_time
            )
            raise RuntimeError(f"Execution exceeded time limit of {self.max_execution_time} seconds")

        finally:
            # Cleanup GPU memory if possible
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# ==============================================================================
# GOTCHAS AND PITFALLS HANDLING
# ==============================================================================

class CUDAPitfallDetector:
    """Detect and handle common CUDA pitfalls."""

    def __init__(self):
        self.logger = structlog.get_logger()

        self.pitfall_patterns = {
            "bank_conflicts": {
                "pattern": r"shared_mem\[\s*threadIdx\.x\s*\*\s*(\d+)\s*\]",
                "check": self._check_bank_conflicts,
                "fix_suggestion": "Use padding or different access patterns to avoid bank conflicts"
            },
            "race_conditions": {
                "pattern": r"(\w+)\[\s*threadIdx\.x\s*\]\s*=.*\1\[\s*threadIdx\.x\s*\]",
                "check": self._check_race_conditions,
                "fix_suggestion": "Use atomic operations or proper synchronization"
            },
            "divergent_branches": {
                "pattern": r"if\s*\(\s*threadIdx\.x\s*[<>%]\s*\d+\s*\)",
                "check": self._check_divergent_branches,
                "fix_suggestion": "Minimize thread divergence by restructuring conditionals"
            },
            "uncoalesced_access": {
                "pattern": r"\w+\[\s*threadIdx\.x\s*\*\s*\w+\s*\+\s*\w+\s*\]",
                "check": self._check_uncoalesced_access,
                "fix_suggestion": "Ensure memory accesses are coalesced for better performance"
            }
        }

    def analyze_pitfalls(self, code: str, benchmark_result=None) -> Dict[str, Any]:
        """Analyze code for common CUDA pitfalls."""
        import re

        pitfall_report = {
            "detected_pitfalls": [],
            "performance_impact": "low",
            "fix_suggestions": [],
            "code_quality_score": 1.0
        }

        quality_penalty = 0.0

        for pitfall_name, pitfall_info in self.pitfall_patterns.items():
            pattern = pitfall_info["pattern"]
            matches = re.findall(pattern, code)

            if matches:
                severity = pitfall_info["check"](code, matches, benchmark_result)

                pitfall_report["detected_pitfalls"].append({
                    "name": pitfall_name,
                    "severity": severity,
                    "matches": len(matches),
                    "suggestion": pitfall_info["fix_suggestion"]
                })

                pitfall_report["fix_suggestions"].append(pitfall_info["fix_suggestion"])

                # Apply quality penalty
                penalty = {"low": 0.1, "medium": 0.2, "high": 0.4}[severity]
                quality_penalty += penalty

        # Determine overall performance impact
        high_severity_count = sum(1 for p in pitfall_report["detected_pitfalls"] if p["severity"] == "high")
        if high_severity_count >= 2:
            pitfall_report["performance_impact"] = "high"
        elif high_severity_count >= 1 or len(pitfall_report["detected_pitfalls"]) >= 3:
            pitfall_report["performance_impact"] = "medium"

        pitfall_report["code_quality_score"] = max(0.0, 1.0 - quality_penalty)

        return pitfall_report

    def _check_bank_conflicts(self, code: str, matches: List, benchmark_result) -> str:
        """Check severity of bank conflicts."""
        if benchmark_result and hasattr(benchmark_result, 'memory_bandwidth_gb_s'):
            if benchmark_result.memory_bandwidth_gb_s < 200:  # Low bandwidth indicates issues
                return "high"

        # Check for stride patterns that commonly cause bank conflicts
        stride_values = [int(m) for m in matches if m.isdigit()]
        if any(stride % 32 == 0 and stride != 32 for stride in stride_values):
            return "high"

        return "medium"

    def _check_race_conditions(self, code: str, matches: List, benchmark_result) -> str:
        """Check severity of race conditions."""
        # Race conditions are always high severity
        if "__syncthreads" not in code and "atomic" not in code.lower():
            return "high"
        return "medium"

    def _check_divergent_branches(self, code: str, matches: List, benchmark_result) -> str:
        """Check severity of divergent branches."""
        if len(matches) > 2:
            return "high"
        elif len(matches) > 0:
            return "medium"
        return "low"

    def _check_uncoalesced_access(self, code: str, matches: List, benchmark_result) -> str:
        """Check severity of uncoalesced memory access."""
        if benchmark_result and hasattr(benchmark_result, 'memory_bandwidth_gb_s'):
            if benchmark_result.memory_bandwidth_gb_s < 100:
                return "high"
        return "medium"

# ==============================================================================
# AGENT SPECIALIZATION
# ==============================================================================

class AgentSpecializer:
    """Handles specialized training for different agent types."""

    def __init__(self):
        self.logger = structlog.get_logger()

        # Agent-specific prompt templates and training data
        self.agent_configs = {
            "generator": {
                "focus": "correctness_and_basic_performance",
                "reward_weights": {
                    "compilation": 0.4,
                    "correctness": 0.4,
                    "performance": 0.15,
                    "efficiency": 0.05
                },
                "prompt_style": "creative_generation",
                "temperature": 0.7
            },
            "optimizer": {
                "focus": "performance_optimization",
                "reward_weights": {
                    "compilation": 0.2,
                    "correctness": 0.2,
                    "performance": 0.5,
                    "efficiency": 0.1
                },
                "prompt_style": "optimization_focused",
                "temperature": 0.5
            }
        }

    def customize_training_data(
        self,
        agent_type: str,
        conversation_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Customize training data for specific agent type."""

        if agent_type not in self.agent_configs:
            return conversation_data

        config = self.agent_configs[agent_type]
        customized_data = []

        for item in conversation_data:
            if item.get("agent_type") == agent_type:
                # Adjust reward based on agent focus
                original_reward = item.get("reward", 0.0)

                # Apply agent-specific reward weighting
                if agent_type == "generator":
                    # Boost compilation and correctness rewards
                    if "compilation_success" in item.get("context", {}):
                        original_reward *= 1.2
                elif agent_type == "optimizer":
                    # Boost performance improvement rewards
                    if item.get("performance_improvement", 0) > 0:
                        original_reward *= 1.3

                item["reward"] = original_reward
                customized_data.append(item)

        return customized_data

# ==============================================================================
# ENHANCED VERL CONFIGURATION AND TRAINER
# ==============================================================================

@dataclass
class VERLTrainingConfig:
    """Enhanced configuration for VERL multi-agent training."""

    # Algorithm selection
    algorithm: str = "grpo"  # grpo, dapo, ppo

    # Model paths
    generator_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    optimizer_model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

    # GRPO specific
    grpo_group_size: int = 16
    grpo_kl_coef: float = 0.0
    grpo_clip_ratio_low: float = 0.2
    grpo_clip_ratio_high: float = 0.28

    # DAPO specific
    dapo_use_kl_in_reward: bool = False
    dapo_loss_agg_mode: str = "token-mean"

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-6
    num_epochs: int = 100
    episodes_per_epoch: int = 100

    # Multi-turn specific
    max_turns_per_episode: int = 5
    turn_discount_factor: float = 0.9

    # Curriculum learning
    curriculum_enabled: bool = True
    curriculum_config: Dict[str, Any] = field(default_factory=dict)

    # Safety features
    safety_checks_enabled: bool = True
    resource_limits_enabled: bool = True
    max_execution_time: float = 30.0
    max_gpu_memory: int = 1 * 1024**3

    # Distributed training
    num_gpus: int = 8
    num_nodes: int = 1
    ray_cluster_address: Optional[str] = None

    # Data and checkpointing
    data_path: str = "data/cuda_problems"
    fallback_data_path: str = "data/synthetic_problems.json"
    checkpoint_dir: str = "checkpoints/verl_training"
    save_freq: int = 10

    # Agent specialization
    agent_specialization_enabled: bool = True

class MultiAgentVERLTrainer:
    """Enhanced VERL trainer with curriculum learning, safety, and specialization."""

    def __init__(
        self,
        config: VERLTrainingConfig,
        conversation_manager,
        reward_function
    ):
        self.config = config
        self.conversation_manager = conversation_manager
        self.reward_function = reward_function
        self.logger = structlog.get_logger()

        # Initialize components
        self.data_loader = SakanaDataLoader(
            cache_dir=config.data_path,
            curriculum_enabled=config.curriculum_enabled,
            fallback_data_path=config.fallback_data_path
        )

        self.curriculum_manager = CurriculumManager(config.curriculum_config) if config.curriculum_enabled else None
        self.safety_analyzer = CUDASafetyAnalyzer() if config.safety_checks_enabled else None
        self.resource_limiter = ResourceLimiter(
            max_execution_time=config.max_execution_time,
            max_gpu_memory=config.max_gpu_memory
        ) if config.resource_limits_enabled else None
        self.pitfall_detector = CUDAPitfallDetector()
        self.agent_specializer = AgentSpecializer() if config.agent_specialization_enabled else None

        # Initialize Ray and VERL components
        self.actor_rollout_wg = None
        self.critic_wg = None
        self.trainer = None

        self._setup_distributed_training()

    def _setup_distributed_training(self):
        """Initialize Ray cluster and VERL worker groups."""
        if not VERL_AVAILABLE:
            self.logger.warning("VERL not available, using mock implementation")
            self.trainer = MockVERLTrainer()
            return

        # Initialize Ray cluster
        if not ray.is_initialized():
            if self.config.ray_cluster_address:
                ray.init(address=self.config.ray_cluster_address)
            else:
                ray.init(
                    num_cpus=self.config.num_gpus * 8,
                    num_gpus=self.config.num_gpus,
                    object_store_memory=10 * 1024**3,
                    runtime_env={
                        "env_vars": {
                            "TOKENIZERS_PARALLELISM": "false",
                            "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(self.config.num_gpus)))
                        }
                    }
                )

        self._setup_verl_workers()
        self._setup_verl_trainer()

    def _setup_verl_workers(self):
        """Setup VERL worker groups for distributed training."""
        if not VERL_AVAILABLE:
            return

        self.actor_rollout_wg = RayWorkerGroup(
            world_size=self.config.num_gpus,
            actor_rollout_ref=ActorRolloutRef(
                model=self.config.generator_model,
                tokenizer=self.config.generator_model,
                rollout=RolloutConfig(
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    max_new_tokens=1024,
                    do_sample=True
                )
            )
        )

        if self.config.algorithm in ["ppo", "grpo"]:
            self.critic_wg = RayWorkerGroup(
                world_size=min(4, self.config.num_gpus),
                critic_ref=CriticRef(
                    model=self.config.optimizer_model,
                    tokenizer=self.config.optimizer_model
                )
            )

    def _setup_verl_trainer(self):
        """Initialize VERL trainer based on algorithm."""
        if not VERL_AVAILABLE:
            return

        if self.config.algorithm == "grpo":
            self.trainer = GRPOTrainer(
                actor_rollout_wg=self.actor_rollout_wg,
                group_size=self.config.grpo_group_size,
                kl_coef=self.config.grpo_kl_coef,
                clip_ratio_low=self.config.grpo_clip_ratio_low,
                clip_ratio_high=self.config.grpo_clip_ratio_high,
                learning_rate=self.config.learning_rate
            )
        elif self.config.algorithm == "dapo":
            self.trainer = DAPOTrainer(
                actor_rollout_wg=self.actor_rollout_wg,
                use_kl_in_reward=self.config.dapo_use_kl_in_reward,
                loss_agg_mode=self.config.dapo_loss_agg_mode,
                learning_rate=self.config.learning_rate
            )
        elif self.config.algorithm == "ppo":
            self.trainer = PPOTrainer(
                actor_rollout_wg=self.actor_rollout_wg,
                critic_wg=self.critic_wg,
                learning_rate=self.config.learning_rate
            )

    async def train(self) -> Dict[str, Any]:
        """Enhanced training loop with curriculum learning and safety."""
        training_metrics = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "average_reward": 0.0,
            "average_speedup": 0.0,
            "compilation_success_rate": 0.0,
            "curriculum_tier": "easy",
            "safety_violations": 0,
            "pitfalls_detected": 0
        }

        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")

            # Get current curriculum difficulty
            current_difficulty = self.curriculum_manager.get_current_difficulty() if self.curriculum_manager else "medium"

            epoch_metrics = await self._train_epoch(current_difficulty)

            # Update curriculum based on results
            if self.curriculum_manager:
                for conv_result in epoch_metrics.get("conversation_results", []):
                    tier_advanced = self.curriculum_manager.record_episode_result(conv_result)
                    if tier_advanced:
                        self.logger.info("Curriculum tier advanced", new_tier=self.curriculum_manager.progress.current_tier)

            # Update training metrics
            training_metrics.update(epoch_metrics)
            training_metrics["curriculum_tier"] = current_difficulty

            self.logger.info("Epoch completed", epoch=epoch + 1, **epoch_metrics)

            # Save checkpoint
            if (epoch + 1) % self.config.save_freq == 0:
                await self._save_checkpoint(epoch + 1, training_metrics)

        return training_metrics

    async def _train_epoch(self, difficulty: str) -> Dict[str, Any]:
        """Train for one epoch with enhanced safety and analysis."""
        epoch_conversations = []
        epoch_rewards = []
        safety_violations = 0
        pitfalls_detected = 0

        for episode_idx in range(self.config.episodes_per_epoch):
            try:
                # Sample problem with curriculum difficulty
                problem = await self.data_loader.sample_problem(difficulty=difficulty)

                # Run conversation with resource limits if enabled
                if self.resource_limiter:
                    conversation = await self.resource_limiter.execute_with_limits(
                        self.conversation_manager.run_conversation,
                        problem,
                        f"epoch_{episode_idx}_{int(time.time())}"
                    )
                else:
                    conversation = await self.conversation_manager.run_conversation(
                        problem,
                        f"epoch_{episode_idx}_{int(time.time())}"
                    )

                # Safety analysis on final code
                if self.safety_analyzer and conversation.current_kernel_code:
                    safety_report = self.safety_analyzer.analyze_code_safety(conversation.current_kernel_code)
                    if not safety_report["is_safe"]:
                        safety_violations += 1
                        self.logger.warning("Safety violation detected",
                                          conversation_id=conversation.conversation_id,
                                          issues=safety_report["critical_issues"])

                # Pitfall analysis
                if conversation.current_kernel_code:
                    last_turn = conversation.turns[-1] if conversation.turns else None
                    benchmark_result = last_turn.context.get("benchmark_result") if last_turn else None

                    pitfall_report = self.pitfall_detector.analyze_pitfalls(
                        conversation.current_kernel_code,
                        benchmark_result
                    )

                    if pitfall_report["detected_pitfalls"]:
                        pitfalls_detected += len(pitfall_report["detected_pitfalls"])
                        self.logger.debug("Pitfalls detected",
                                        conversation_id=conversation.conversation_id,
                                        pitfalls=pitfall_report["detected_pitfalls"])

                epoch_conversations.append(conversation)
                epoch_rewards.append(conversation.final_reward)

            except Exception as e:
                self.logger.error(f"Episode {episode_idx} failed: {e}")
                continue

        # Convert conversations to VERL training format
        training_data = self._convert_conversations_to_verl_format(epoch_conversations)

        # Run VERL training step
        training_loss = 0.0
        if training_data:
            training_loss = await self._verl_training_step(training_data)

        return {
            "episodes": len(epoch_conversations),
            "successful_episodes": sum(1 for c in epoch_conversations if c.conversation_success),
            "average_reward": np.mean(epoch_rewards) if epoch_rewards else 0.0,
            "training_loss": training_loss,
            "safety_violations": safety_violations,
            "pitfalls_detected": pitfalls_detected,
            "conversation_results": epoch_conversations
        }

    def _convert_conversations_to_verl_format(self, conversations: List) -> Optional[Dict[str, Any]]:
        """Convert conversations to VERL format with agent specialization."""
        if self.config.algorithm == "grpo":
            return self._convert_for_grpo(conversations)
        elif self.config.algorithm == "dapo":
            return self._convert_for_dapo(conversations)
        elif self.config.algorithm == "ppo":
            return self._convert_for_ppo(conversations)
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")

    def _convert_for_grpo(self, conversations: List) -> Optional[Dict[str, Any]]:
        """Convert conversations for GRPO training with specialization."""
        grouped_data = {"generator": [], "optimizer": []}

        for conv in conversations:
            conv_data = self._extract_conversation_training_data(conv)
            if conv_data:
                for item in conv_data:
                    agent_type = item.get("agent_type")
                    if agent_type in grouped_data:
                        grouped_data[agent_type].append(item)

        # Apply agent specialization if enabled
        if self.agent_specializer:
            for agent_type in grouped_data:
                grouped_data[agent_type] = self.agent_specializer.customize_training_data(
                    agent_type, grouped_data[agent_type]
                )

        return grouped_data if any(grouped_data.values()) else None

    def _convert_for_dapo(self, conversations: List) -> Optional[Dict[str, Any]]:
        """Convert conversations for DAPO training."""
        training_data = []
        for conv in conversations:
            conv_data = self._extract_conversation_training_data(conv)
            if conv_data:
                training_data.extend(conv_data)

        return {"data": training_data, "type": "dapo"} if training_data else None

    def _convert_for_ppo(self, conversations: List) -> Optional[Dict[str, Any]]:
        """Convert conversations for PPO training."""
        training_data = []
        for conv in conversations:
            conv_data = self._extract_conversation_training_data(conv)
            if conv_data:
                training_data.extend(conv_data)

        return {"data": training_data, "type": "ppo"} if training_data else None

    def _extract_conversation_training_data(self, conversation) -> List[Dict[str, Any]]:
        """Extract enhanced training data from conversation."""
        training_data = []
        turn_rewards = self._calculate_turn_rewards(conversation)

        for turn, reward in zip(conversation.turns, turn_rewards):
            if turn.agent_type.value in ["generator", "optimizer"]:
                training_data.append({
                    "prompt": turn.prompt,
                    "response": turn.response,
                    "reward": reward,
                    "log_probs": turn.log_probs,
                    "token_ids": turn.token_ids,
                    "agent_type": turn.agent_type.value,
                    "turn_number": turn.turn_number,
                    "conversation_id": conversation.conversation_id,
                    "compilation_success": turn.compilation_success,
                    "performance_metrics": turn.performance_metrics,
                    "context": turn.context
                })

        return training_data

    def _calculate_turn_rewards(self, conversation) -> List[float]:
        """Calculate turn rewards with enhanced factors."""
        turn_rewards = []
        final_reward = conversation.final_reward

        for i, turn in enumerate(conversation.turns):
            if turn.agent_type.value in ["generator", "optimizer"]:
                immediate_reward = 0.0

                if turn.compilation_success:
                    immediate_reward += 0.3

                speedup = turn.performance_metrics.get("speedup", 0)
                if speedup > 1.0:
                    immediate_reward += 0.2 * min(speedup / 2.0, 1.0)

                # Discounted final reward
                turns_remaining = len(conversation.turns) - i - 1
                discounted_final = final_reward * (self.config.turn_discount_factor ** turns_remaining)

                total_reward = 0.3 * immediate_reward + 0.7 * discounted_final
                turn_rewards.append(total_reward)
            else:
                turn_rewards.append(0.0)

        return turn_rewards

    async def _verl_training_step(self, training_data: Dict[str, Any]) -> float:
        """Execute VERL training step."""
        try:
            if self.config.algorithm == "grpo":
                loss = await self.trainer.train_step_grpo(training_data)
            elif self.config.algorithm == "dapo":
                loss = await self.trainer.train_step_dapo(training_data)
            elif self.config.algorithm == "ppo":
                loss = await self.trainer.train_step_ppo(training_data)

            return loss.item() if hasattr(loss, 'item') else float(loss)

        except Exception as e:
            self.logger.error(f"VERL training step failed: {e}")
            return 0.0

    async def _save_checkpoint(self, epoch: int, metrics: Dict[str, Any]):
        """Save enhanced checkpoint with curriculum state."""
        checkpoint_path = f"{self.config.checkpoint_dir}/epoch_{epoch}"
        os.makedirs(checkpoint_path, exist_ok=True)

        try:
            # Save VERL model states
            if hasattr(self.trainer, 'save_checkpoint'):
                await self.trainer.save_checkpoint(checkpoint_path)

            # Save curriculum state
            curriculum_state = None
            if self.curriculum_manager:
                curriculum_state = {
                    "current_tier": self.curriculum_manager.progress.current_tier,
                    "tier_episodes": self.curriculum_manager.progress.tier_episodes,
                    "tier_successes": self.curriculum_manager.progress.tier_successes,
                    "unlocked_tiers": self.curriculum_manager.progress.unlocked_tiers,
                    "performance_history": list(self.curriculum_manager.progress.performance_history)
                }

            # Save metadata
            metadata = {
                "epoch": epoch,
                "config": self.config.__dict__,
                "metrics": metrics,
                "curriculum_state": curriculum_state
            }

            with open(f"{checkpoint_path}/metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Enhanced checkpoint saved at {checkpoint_path}")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

# ==============================================================================
# MOCK IMPLEMENTATIONS
# ==============================================================================

class MockVERLTrainer:
    """Mock VERL trainer for when VERL is not available."""

    async def train_step_grpo(self, training_data):
        return torch.tensor(0.1)

    async def train_step_dapo(self, training_data):
        return torch.tensor(0.1)

    async def train_step_ppo(self, training_data):
        return torch.tensor(0.1)

    async def save_checkpoint(self, path):
        pass

class RolloutConfig:
    """Mock rollout configuration."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
