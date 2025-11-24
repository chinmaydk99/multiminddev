"""
Main data pipeline for HIP RL training on AMD ROCm.
Integrates dataset loading, curriculum management, and training data preparation.
"""

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
import torch

from .curriculum_manager import CurriculumManager
from .sakana_loader import SakanaDataLoader


@dataclass
class TestCase:
    """Test case for HIP kernel evaluation."""
    input_shapes: List[List[int]]
    dtype: str = "float32"
    grid_dims: Tuple[int, int, int] = (1, 1, 1)
    block_dims: Tuple[int, int, int] = (256, 1, 1)  # AMD wavefront friendly
    input_tensors: Optional[List[torch.Tensor]] = None
    expected_output: Optional[torch.Tensor] = None

    def generate_inputs(self):
        """Generate random input tensors if not provided."""
        if self.input_tensors is None:
            self.input_tensors = []
            for shape in self.input_shapes:
                if self.dtype == "float32":
                    tensor = torch.randn(shape, dtype=torch.float32)
                elif self.dtype == "int32":
                    tensor = torch.randint(0, 100, shape, dtype=torch.int32)
                else:
                    tensor = torch.randn(shape)
                self.input_tensors.append(tensor)
        return self.input_tensors


@dataclass
class TrainingExample:
    """Complete training example for HIP kernel generation on AMD ROCm."""
    problem_id: str
    problem_description: str
    difficulty: str
    reference_solution: Optional[str] = None
    test_cases: List[TestCase] = field(default_factory=list)
    target_performance: Dict[str, float] = field(default_factory=dict)
    baseline_performance: Dict[str, float] = field(default_factory=dict)
    torch_reference: Optional[str] = None
    operation_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_generator_prompt(self) -> str:
        """Convert to prompt for generator agent."""
        prompt = f"""Generate an optimized HIP kernel for AMD ROCm for the following problem:

Problem: {self.problem_description}

Requirements:
- Target speedup: {self.target_performance.get('speedup', 2.0)}x over PyTorch
- Operation type: {self.operation_type or 'general'}
- Difficulty: {self.difficulty}

Test cases will use the following configurations:"""

        for i, test_case in enumerate(self.test_cases[:2]):  # Show first 2 test cases
            prompt += f"\n- Test {i+1}: Input shapes {test_case.input_shapes}, Grid {test_case.grid_dims}, Block {test_case.block_dims}"

        if self.torch_reference:
            prompt += f"\n\nPyTorch reference implementation:\n```python\n{self.torch_reference}\n```"

        prompt += "\n\nGenerate the HIP kernel code:"
        return prompt

    def to_optimizer_prompt(self, initial_kernel: str) -> str:
        """Convert to prompt for optimizer agent."""
        return f"""Optimize this HIP kernel for AMD ROCm for better performance:

Current kernel:
```cpp
{initial_kernel}
```

Problem context: {self.problem_description}
Current difficulty: {self.difficulty}
Target speedup: {self.target_performance.get('speedup', 2.0)}x

Apply optimization techniques such as:
- LDS (Local Data Share) memory usage
- Memory coalescing
- VGPR (Vector Register) optimization for occupancy
- Loop unrolling
- Wavefront-level primitives

Generate the optimized kernel:"""


class HIPDataPipeline:
    """
    Main data pipeline for HIP RL training on AMD ROCm.
    Coordinates dataset loading, curriculum management, and batch preparation.
    """

    def __init__(
        self,
        dataset_name: str = "SakanaAI/AI-CUDA-Engineer-Archive",
        cache_dir: str = "./cache/datasets",
        curriculum_enabled: bool = True,
        initial_tier: str = "easy"
    ):
        """
        Initialize data pipeline.
        
        Args:
            dataset_name: Name of the dataset to load (will convert to HIP)
            cache_dir: Directory for caching datasets
            curriculum_enabled: Whether to use curriculum learning
            initial_tier: Starting difficulty tier
        """
        self.logger = structlog.get_logger("hip_data_pipeline")

        # Initialize components
        self.data_loader = SakanaDataLoader(
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            curriculum_enabled=curriculum_enabled,
            convert_cuda_to_hip=True  # Enable CUDA to HIP conversion
        )

        self.curriculum_manager = CurriculumManager(
            initial_tier=initial_tier
        ) if curriculum_enabled else None

        self.curriculum_enabled = curriculum_enabled

        # Cache for generated examples
        self.example_cache = {}
        self.cache_size = 100

        self.logger.info(
            "HIP data pipeline initialized",
            dataset=dataset_name,
            curriculum_enabled=curriculum_enabled,
            initial_tier=initial_tier if curriculum_enabled else None
        )

    async def get_training_batch(
        self,
        batch_size: int = 32,
        use_cache: bool = True
    ) -> List[TrainingExample]:
        """
        Get a batch of training examples.
        
        Args:
            batch_size: Number of examples to return
            use_cache: Whether to use cached examples
            
        Returns:
            List of training examples
        """
        batch = []

        # Determine difficulty based on curriculum
        if self.curriculum_enabled and self.curriculum_manager:
            tier_info = self.curriculum_manager.get_tier_info()
            difficulty = tier_info["difficulty"]
            operations = tier_info["operations"]
        else:
            difficulty = None
            operations = None

        for i in range(batch_size):
            # Check cache
            cache_key = f"{difficulty}_{i % self.cache_size}"
            if use_cache and cache_key in self.example_cache:
                example = self.example_cache[cache_key]
            else:
                # Generate new example
                example = await self._create_training_example(
                    difficulty=difficulty,
                    operations=operations
                )

                # Cache it
                if use_cache:
                    self.example_cache[cache_key] = example

            batch.append(example)

        self.logger.debug(
            "Training batch prepared",
            batch_size=batch_size,
            difficulty=difficulty,
            cached=use_cache
        )

        return batch

    async def _create_training_example(
        self,
        difficulty: Optional[str] = None,
        operations: Optional[List[str]] = None
    ) -> TrainingExample:
        """Create a single training example."""

        # Sample problem from dataset
        problem_data = await self.data_loader.sample_problem(difficulty=difficulty)

        # Determine operation type
        if operations:
            operation_type = random.choice(operations)
        else:
            operation_type = self._infer_operation_type(problem_data["description"])

        # Create test cases
        test_cases = []
        for test_config in problem_data["test_cases"]:
            test_case = TestCase(
                input_shapes=test_config["input_shapes"],
                dtype=test_config.get("dtype", "float32"),
                grid_dims=test_config.get("grid_dims", (1, 1, 1)),
                block_dims=test_config.get("block_dims", (256, 1, 1))
            )
            test_case.generate_inputs()
            test_cases.append(test_case)

        # Create training example
        example = TrainingExample(
            problem_id=f"{difficulty}_{operation_type}_{random.randint(1000, 9999)}",
            problem_description=problem_data["description"],
            difficulty=problem_data["difficulty"],
            reference_solution=problem_data.get("reference_solution"),
            test_cases=test_cases,
            target_performance=problem_data["target_performance"],
            baseline_performance=problem_data.get("baseline_performance", {}),
            torch_reference=self._generate_torch_reference(operation_type),
            operation_type=operation_type,
            metadata=problem_data.get("metadata", {})
        )

        return example

    def _infer_operation_type(self, description: str) -> str:
        """Infer operation type from problem description."""
        desc_lower = description.lower()

        if "add" in desc_lower or "sum" in desc_lower:
            return "vector_add"
        elif "multiply" in desc_lower or "product" in desc_lower:
            return "element_multiply"
        elif "reduction" in desc_lower:
            return "reduction"
        elif "transpose" in desc_lower:
            return "transpose"
        elif "matrix" in desc_lower and "multiply" in desc_lower:
            return "matrix_multiply"
        elif "convolution" in desc_lower or "conv" in desc_lower:
            return "convolution"
        else:
            return "general"

    def _generate_torch_reference(self, operation_type: str) -> str:
        """Generate PyTorch reference implementation."""

        references = {
            "vector_add": """
def vector_add(a, b):
    return a + b
""",
            "element_multiply": """
def element_multiply(a, b):
    return a * b
""",
            "reduction": """
def reduction_sum(x):
    return torch.sum(x)
""",
            "transpose": """
def transpose(x):
    return x.T
""",
            "matrix_multiply": """
def matrix_multiply(a, b):
    return torch.matmul(a, b)
""",
            "convolution": """
def conv2d(input, kernel):
    return torch.nn.functional.conv2d(input, kernel)
"""
        }

        return references.get(operation_type, "# PyTorch reference not available")

    def record_training_result(
        self,
        example_id: str,
        compilation_success: bool,
        speedup: float,
        final_reward: float
    ):
        """Record result of training on an example."""

        if self.curriculum_manager:
            self.curriculum_manager.record_episode_result(
                compilation_success=compilation_success,
                speedup=speedup,
                final_reward=final_reward
            )

        self.logger.debug(
            "Training result recorded",
            example_id=example_id,
            compilation_success=compilation_success,
            speedup=speedup,
            reward=final_reward
        )

    def get_curriculum_status(self) -> Dict[str, Any]:
        """Get current curriculum learning status."""

        if not self.curriculum_manager:
            return {"curriculum_enabled": False}

        return {
            "curriculum_enabled": True,
            "current_tier": self.curriculum_manager.get_current_tier(),
            "tier_info": self.curriculum_manager.get_tier_info(),
            "progress_summary": self.curriculum_manager.get_progress_summary()
        }

    def should_stop_training(self) -> bool:
        """Check if training should stop."""

        if self.curriculum_manager:
            return self.curriculum_manager.should_stop_training()

        return False

    async def prepare_evaluation_set(
        self,
        num_problems: int = 100,
        difficulty_distribution: Optional[Dict[str, float]] = None
    ) -> List[TrainingExample]:
        """
        Prepare a set of problems for evaluation.
        
        Args:
            num_problems: Number of evaluation problems
            difficulty_distribution: Distribution of difficulties (e.g., {"easy": 0.3, "medium": 0.4, "hard": 0.3})
            
        Returns:
            List of evaluation examples
        """
        if difficulty_distribution is None:
            difficulty_distribution = {"easy": 0.3, "medium": 0.4, "hard": 0.3}

        eval_set = []

        for difficulty, proportion in difficulty_distribution.items():
            num_for_difficulty = int(num_problems * proportion)

            for _ in range(num_for_difficulty):
                example = await self._create_training_example(difficulty=difficulty)
                eval_set.append(example)

        self.logger.info(
            "Evaluation set prepared",
            num_problems=len(eval_set),
            distribution=difficulty_distribution
        )

        return eval_set


# Alias for backward compatibility
CUDADataPipeline = HIPDataPipeline
