import os
import json
from typing import List, Dict, Tuple, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import structlog

if TYPE_CHECKING:
    from .curriculum_manager import CUDACurriculumManager

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class CUDATrainingExample:
    """Training example for CUDA kernel generation."""
    problem_description: str
    torch_reference: str  # PyTorch reference implementation
    test_inputs: List[Dict[str, Any]]  # Input tensor specifications
    expected_speedup_range: Tuple[float, float] = (1.0, 10.0)
    difficulty_level: str = "medium"  # easy, medium, hard, expert
    operation_category: str = "general"  # elementwise, reduction, linear_algebra, etc.
    expected_output_shape: Optional[Tuple[int, ...]] = None
    test_data: Optional[List[torch.Tensor]] = None  # Actual test tensors
    metadata: Dict[str, Any] = field(default_factory=dict)


class CUDADataLoader:
    """Enhanced CUDA data loader with curriculum learning support."""
    
    def __init__(self, curriculum_manager: Optional['CUDACurriculumManager'] = None):
        """
        Initialize data loader with curriculum support.
        
        Args:
            curriculum_manager: Optional curriculum manager for tier-based loading
        """
        self.dataset = None
        self.curriculum_manager = curriculum_manager
        self.logger = structlog.get_logger("cuda_curriculum_loader")
        
        # Operation mapping from curriculum tiers to operation types
        self.tier_operations = {
            "BASIC": ["vector_add", "scalar_multiply", "element_wise"],
            "INTERMEDIATE": ["reduction", "transpose", "matrix_vector", "convolution_1d"],
            "ADVANCED": ["matrix_multiply", "softmax", "layer_norm", "conv2d"],
            "EXPERT": ["fused_attention", "custom_layers", "optimized_gemm", "flash_attention"]
        }
        
        # Load dataset if available
        if DATASETS_AVAILABLE:
            try:
                self.dataset = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive")
                self.logger.info("Loaded SakanaAI CUDA dataset successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load SakanaAI dataset: {e}")
                self.dataset = None
    
    async def get_curriculum_batch(
        self, 
        batch_size: int,
        tier_override: Optional[str] = None
    ) -> List[CUDATrainingExample]:
        """
        Get a batch of training examples for current curriculum tier.
        
        Args:
            batch_size: Number of examples to return
            tier_override: Optional tier override (for testing/validation)
            
        Returns:
            List of training examples appropriate for current tier
        """
        # Determine target tier
        if tier_override:
            tier_name = tier_override
            operations = self.tier_operations.get(tier_override.upper(), self.tier_operations["BASIC"])
        elif self.curriculum_manager:
            tier_info = self.curriculum_manager.get_current_tier_info()
            tier_name = tier_info["tier"]
            operations = tier_info["operations"]
        else:
            tier_name = "BASIC"
            operations = self.tier_operations["BASIC"]
        
        self.logger.debug(
            "Loading curriculum batch",
            tier=tier_name,
            operations=operations,
            batch_size=batch_size
        )
        
        if self.dataset is not None:
            return await self._get_batch_from_dataset(tier_name, operations, batch_size)
        else:
            return await self._get_synthetic_batch(tier_name, operations, batch_size)
    
    async def _get_batch_from_dataset(
        self, 
        tier_name: str,
        operations: List[str], 
        batch_size: int
    ) -> List[CUDATrainingExample]:
        """Get batch from SakanaAI dataset filtered by curriculum tier."""
        examples = []
        
        try:
            # Filter dataset by operation type and difficulty
            filtered_data = self.dataset["train"].filter(
                lambda x: any(op_type in x.get("operation_type", "").lower() 
                             for op_type in operations)
            )
            
            # Create training examples with tier-appropriate complexity
            complexity_limit = self._get_complexity_limit(tier_name)
            
            for i, item in enumerate(filtered_data.select(range(min(batch_size * 2, len(filtered_data))))):
                # Filter by complexity if kernel code is available
                kernel_code = item.get("cuda_kernel", item.get("kernel_code", ""))
                if kernel_code and len(kernel_code.split('\n')) > complexity_limit:
                    continue  # Skip overly complex kernels for current tier
                
                example = CUDATrainingExample(
                    problem_description=item.get("problem_description", item.get("description", "")),
                    torch_reference=item.get("torch_implementation", item.get("torch_reference", "")),
                    test_inputs=self._parse_test_inputs(item.get("test_data", item.get("test_inputs", []))),
                    expected_speedup_range=(
                        item.get("min_speedup", 1.0),
                        item.get("max_speedup", item.get("performance_targets", self._get_target_speedup(tier_name)))
                    ),
                    difficulty_level=self._map_tier_to_difficulty(tier_name),
                    operation_category=item.get("category", operations[0] if operations else "general"),
                    metadata={
                        "source": "sakana_ai",
                        "task_id": item.get("task_id", f"curriculum_{tier_name}_{i}"),
                        "tier": tier_name,
                        "complexity": len(kernel_code.split('\n')) if kernel_code else 0,
                        "original_item": item
                    }
                )
                examples.append(example)
                
                if len(examples) >= batch_size:
                    break
                
        except Exception as e:
            self.logger.error(f"Error loading from dataset: {e}")
            # Fallback to synthetic examples
            return await self._get_synthetic_batch(tier_name, operations, batch_size)
        
        # If we don't have enough examples, pad with synthetic ones
        if len(examples) < batch_size:
            synthetic_examples = await self._get_synthetic_batch(
                tier_name, operations, batch_size - len(examples)
            )
            examples.extend(synthetic_examples)
        
        return examples[:batch_size]
    
    async def _get_synthetic_batch(
        self, 
        tier_name: str,
        operations: List[str], 
        batch_size: int
    ) -> List[CUDATrainingExample]:
        """Generate synthetic training examples for curriculum tier."""
        examples = []
        
        # Tier-appropriate synthetic templates with progressive complexity
        synthetic_templates = self._get_tier_templates(tier_name)
        
        for i in range(batch_size):
            op_type = operations[i % len(operations)]
            if op_type in synthetic_templates:
                template = synthetic_templates[op_type]
                
                # Scale complexity based on tier
                scaled_template = self._scale_template_complexity(template, tier_name)
                
                example = CUDATrainingExample(
                    problem_description=scaled_template["description"],
                    torch_reference=scaled_template["torch_ref"],
                    test_inputs=scaled_template["inputs"],
                    expected_speedup_range=scaled_template["speedup"],
                    difficulty_level=self._map_tier_to_difficulty(tier_name),
                    operation_category=op_type,
                    metadata={
                        "source": "synthetic",
                        "curriculum_tier": tier_name,
                        "template_id": op_type,
                        "complexity_scaled": True
                    }
                )
                examples.append(example)
        
        return examples
    
    def _get_tier_templates(self, tier_name: str) -> Dict[str, Dict[str, Any]]:
        """Get synthetic templates appropriate for tier."""
        
        base_templates = {
            "vector_add": {
                "description": "Implement element-wise addition of two vectors",
                "torch_ref": "torch.add(a, b)",
                "inputs": [{"shape": [1024], "dtype": "float32", "name": "a"}, 
                          {"shape": [1024], "dtype": "float32", "name": "b"}],
                "speedup": (1.2, 2.0)
            },
            "scalar_multiply": {
                "description": "Multiply vector by scalar value",
                "torch_ref": "torch.mul(x, scalar)",
                "inputs": [{"shape": [512], "dtype": "float32", "name": "x"}],
                "speedup": (1.1, 1.5)
            },
            "element_wise": {
                "description": "Element-wise operations on tensors",
                "torch_ref": "torch.relu(x)",
                "inputs": [{"shape": [256, 256], "dtype": "float32", "name": "x"}],
                "speedup": (1.3, 2.2)
            },
            "reduction": {
                "description": "Sum reduction along specified dimension",
                "torch_ref": "torch.sum(x, dim=1)",
                "inputs": [{"shape": [512, 1024], "dtype": "float32", "name": "x"}],
                "speedup": (1.8, 3.5)
            },
            "transpose": {
                "description": "Matrix transpose operation",
                "torch_ref": "torch.transpose(x, 0, 1)",
                "inputs": [{"shape": [1024, 1024], "dtype": "float32", "name": "x"}],
                "speedup": (1.5, 2.8)
            },
            "matrix_vector": {
                "description": "Matrix-vector multiplication",
                "torch_ref": "torch.mv(A, x)",
                "inputs": [{"shape": [512, 512], "dtype": "float32", "name": "A"},
                          {"shape": [512], "dtype": "float32", "name": "x"}],
                "speedup": (2.0, 4.0)
            },
            "convolution_1d": {
                "description": "1D convolution operation",
                "torch_ref": "torch.conv1d(input, weight)",
                "inputs": [{"shape": [1, 32, 128], "dtype": "float32", "name": "input"},
                          {"shape": [64, 32, 5], "dtype": "float32", "name": "weight"}],
                "speedup": (1.8, 3.2)
            },
            "matrix_multiply": {
                "description": "General matrix multiplication",
                "torch_ref": "torch.mm(A, B)",
                "inputs": [{"shape": [512, 512], "dtype": "float32", "name": "A"},
                          {"shape": [512, 512], "dtype": "float32", "name": "B"}],
                "speedup": (2.5, 5.0)
            },
            "softmax": {
                "description": "Softmax activation function",
                "torch_ref": "torch.softmax(x, dim=1)",
                "inputs": [{"shape": [128, 512], "dtype": "float32", "name": "x"}],
                "speedup": (1.5, 3.0)
            },
            "layer_norm": {
                "description": "Layer normalization",
                "torch_ref": "torch.layer_norm(x, normalized_shape)",
                "inputs": [{"shape": [64, 256], "dtype": "float32", "name": "x"}],
                "speedup": (2.0, 4.5)
            },
            "conv2d": {
                "description": "2D convolution operation",
                "torch_ref": "torch.conv2d(input, weight)",
                "inputs": [{"shape": [1, 32, 32, 32], "dtype": "float32", "name": "input"},
                          {"shape": [64, 32, 3, 3], "dtype": "float32", "name": "weight"}],
                "speedup": (2.2, 4.5)
            },
            "fused_attention": {
                "description": "Fused multi-head attention mechanism",
                "torch_ref": "torch.nn.functional.scaled_dot_product_attention(q, k, v)",
                "inputs": [{"shape": [1, 8, 128, 64], "dtype": "float32", "name": "q"},
                          {"shape": [1, 8, 128, 64], "dtype": "float32", "name": "k"},
                          {"shape": [1, 8, 128, 64], "dtype": "float32", "name": "v"}],
                "speedup": (3.0, 8.0)
            },
            "custom_layers": {
                "description": "Custom neural network layer implementation",
                "torch_ref": "custom_layer(x, weights, bias)",
                "inputs": [{"shape": [32, 512], "dtype": "float32", "name": "x"},
                          {"shape": [512, 1024], "dtype": "float32", "name": "weights"},
                          {"shape": [1024], "dtype": "float32", "name": "bias"}],
                "speedup": (4.0, 10.0)
            },
            "optimized_gemm": {
                "description": "Optimized general matrix multiplication",
                "torch_ref": "torch.mm(A, B) # with specific optimizations",
                "inputs": [{"shape": [1024, 1024], "dtype": "float32", "name": "A"},
                          {"shape": [1024, 1024], "dtype": "float32", "name": "B"}],
                "speedup": (5.0, 15.0)
            },
            "flash_attention": {
                "description": "Memory-efficient attention computation",
                "torch_ref": "flash_attn_func(q, k, v)",
                "inputs": [{"shape": [1, 16, 256, 128], "dtype": "float32", "name": "q"},
                          {"shape": [1, 16, 256, 128], "dtype": "float32", "name": "k"},
                          {"shape": [1, 16, 256, 128], "dtype": "float32", "name": "v"}],
                "speedup": (6.0, 20.0)
            }
        }
        
        return base_templates
    
    def _scale_template_complexity(
        self, 
        template: Dict[str, Any], 
        tier_name: str
    ) -> Dict[str, Any]:
        """Scale template complexity based on curriculum tier."""
        
        complexity_scales = {
            "BASIC": 0.5,
            "INTERMEDIATE": 1.0,
            "ADVANCED": 1.5,
            "EXPERT": 2.0
        }
        
        scale = complexity_scales.get(tier_name, 1.0)
        scaled_template = template.copy()
        
        # Scale input tensor sizes
        scaled_inputs = []
        for inp in template["inputs"]:
            scaled_inp = inp.copy()
            if "shape" in scaled_inp:
                original_shape = scaled_inp["shape"]
                scaled_shape = []
                for dim in original_shape:
                    # Scale dimension but keep reasonable bounds
                    scaled_dim = max(32, min(4096, int(dim * scale)))
                    scaled_shape.append(scaled_dim)
                scaled_inp["shape"] = scaled_shape
            scaled_inputs.append(scaled_inp)
        
        scaled_template["inputs"] = scaled_inputs
        
        # Scale expected speedup based on tier
        min_speedup, max_speedup = template["speedup"]
        tier_multipliers = {
            "BASIC": 0.8,
            "INTERMEDIATE": 1.0,
            "ADVANCED": 1.2,
            "EXPERT": 1.5
        }
        multiplier = tier_multipliers.get(tier_name, 1.0)
        scaled_template["speedup"] = (
            min_speedup * multiplier,
            max_speedup * multiplier
        )
        
        return scaled_template
    
    def _get_complexity_limit(self, tier_name: str) -> int:
        """Get line count complexity limit for tier."""
        limits = {
            "BASIC": 100,
            "INTERMEDIATE": 200,
            "ADVANCED": 300,
            "EXPERT": 500
        }
        return limits.get(tier_name, 200)
    
    def _get_target_speedup(self, tier_name: str) -> float:
        """Get target speedup for tier."""
        targets = {
            "BASIC": 2.0,
            "INTERMEDIATE": 3.0,
            "ADVANCED": 5.0,
            "EXPERT": 10.0
        }
        return targets.get(tier_name, 2.0)
    
    def _map_tier_to_difficulty(self, tier_name: str) -> str:
        """Map curriculum tier to difficulty level."""
        mapping = {
            "BASIC": "easy",
            "INTERMEDIATE": "medium", 
            "ADVANCED": "hard",
            "EXPERT": "expert"
        }
        return mapping.get(tier_name, "medium")
    
    def _parse_test_inputs(self, raw_inputs: Any) -> List[Dict[str, Any]]:
        """Parse test inputs from various dataset formats."""
        if not raw_inputs:
            return [{"shape": [1024], "dtype": "float32", "name": "input"}]
        
        if isinstance(raw_inputs, list):
            parsed = []
            for i, inp in enumerate(raw_inputs):
                if isinstance(inp, dict):
                    parsed.append(inp)
                elif isinstance(inp, (list, tuple)):
                    parsed.append({
                        "shape": list(inp),
                        "dtype": "float32", 
                        "name": f"input_{i}"
                    })
            return parsed if parsed else [{"shape": [1024], "dtype": "float32", "name": "input"}]
        
        return [{"shape": [1024], "dtype": "float32", "name": "input"}]


class CUDATrainingDataLoader:
    """Load and prepare CUDA training data from KernelBench and AI CUDA Engineer Archive."""
    
    def __init__(
        self, 
        dataset_sources: Optional[List[str]] = None,
        data_dir: str = "./data/cuda_training",
        max_problems_per_source: int = 1000
    ):
        self.dataset_sources = dataset_sources or [
            "SakanaAI/AI-CUDA-Engineer-Archive",  # HuggingFace dataset
            "kernelbench_local",                   # Local KernelBench data
        ]
        self.data_dir = data_dir
        self.max_problems_per_source = max_problems_per_source
        self.logger = structlog.get_logger("cuda_data_loader")
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        self.logger.info(
            "CUDA data loader initialized",
            dataset_sources=self.dataset_sources,
            data_dir=data_dir,
            max_problems_per_source=max_problems_per_source
        )
    
    async def load_cuda_problems(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load CUDA training problems from various sources."""
        
        all_problems = []
        
        # Load from each configured source
        for source in self.dataset_sources:
            try:
                if source.startswith("SakanaAI/") and DATASETS_AVAILABLE:
                    problems = await self._load_from_huggingface(source)
                elif source == "kernelbench_local":
                    problems = await self._load_from_kernelbench()
                else:
                    self.logger.warning(f"Unknown dataset source: {source}")
                    continue
                
                all_problems.extend(problems)
                self.logger.info(f"Loaded {len(problems)} problems from {source}")
                
            except Exception as e:
                self.logger.error(f"Failed to load from {source}", error=str(e))
                continue
        
        # Filter and validate problems
        valid_problems = [p for p in all_problems if self._validate_problem(p)]
        
        self.logger.info(
            "Loaded and validated CUDA problems",
            total_loaded=len(all_problems),
            valid_problems=len(valid_problems)
        )
        
        # Split into train/validation (80/20)
        split_idx = int(len(valid_problems) * 0.8)
        train_problems = valid_problems[:split_idx]
        val_problems = valid_problems[split_idx:]
        
        return train_problems, val_problems
    
    async def _load_from_huggingface(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Load problems from Sakana AI CUDA Engineer Archive."""
        
        if not DATASETS_AVAILABLE:
            self.logger.error("datasets library not available for HuggingFace loading")
            return []
        
        try:
            self.logger.info(f"Loading dataset from HuggingFace: {dataset_name}")
            dataset = load_dataset(dataset_name, split="train")
            
            problems = []
            for i, item in enumerate(dataset):
                if i >= self.max_problems_per_source:
                    break
                
                problem = {
                    "id": item.get("task_id", f"hf_{i}"),
                    "pytorch_operation": item.get("torch_reference", item.get("description", "")),
                    "description": item.get("description", ""),
                    "test_inputs": self._parse_test_inputs(item.get("test_inputs", [])),
                    "expected_speedup": float(item.get("speedup_target", 2.0)),
                    "difficulty": item.get("level", "medium"),
                    "reference_implementation": item.get("cuda_kernel", ""),
                    "source": "sakana_ai_archive",
                    "category": item.get("category", "general")
                }
                problems.append(problem)
            
            self.logger.info(f"Successfully loaded {len(problems)} problems from HuggingFace")
            return problems
            
        except Exception as e:
            self.logger.error(f"Failed to load from HuggingFace dataset {dataset_name}", error=str(e))
            return []
    
    async def _load_from_kernelbench(self) -> List[Dict[str, Any]]:
        """Load problems from KernelBench dataset (local or synthetic)."""
        
        # For now, create synthetic KernelBench-style problems
        # In production, this would load from actual KernelBench data files
        
        synthetic_problems = [
            {
                "id": "matmul_basic",
                "pytorch_operation": "torch.mm(A, B)",
                "description": "Basic matrix multiplication kernel",
                "test_inputs": [
                    {"shape": [1024, 1024], "dtype": "float32", "name": "A"},
                    {"shape": [1024, 1024], "dtype": "float32", "name": "B"}
                ],
                "expected_speedup": 2.5,
                "difficulty": "easy",
                "source": "kernelbench",
                "category": "linear_algebra"
            },
            {
                "id": "softmax_2d", 
                "pytorch_operation": "torch.softmax(x, dim=1)",
                "description": "2D softmax computation kernel",
                "test_inputs": [
                    {"shape": [256, 1000], "dtype": "float32", "name": "x"}
                ],
                "expected_speedup": 1.8,
                "difficulty": "medium",
                "source": "kernelbench",
                "category": "activation"
            },
            {
                "id": "vector_add",
                "pytorch_operation": "torch.add(a, b)",
                "description": "Element-wise vector addition kernel",
                "test_inputs": [
                    {"shape": [1048576], "dtype": "float32", "name": "a"},
                    {"shape": [1048576], "dtype": "float32", "name": "b"}
                ],
                "expected_speedup": 1.5,
                "difficulty": "easy",
                "source": "kernelbench",
                "category": "elementwise"
            },
            {
                "id": "conv2d_basic",
                "pytorch_operation": "torch.conv2d(input, weight)",
                "description": "Basic 2D convolution kernel",
                "test_inputs": [
                    {"shape": [1, 64, 32, 32], "dtype": "float32", "name": "input"},
                    {"shape": [128, 64, 3, 3], "dtype": "float32", "name": "weight"}
                ],
                "expected_speedup": 3.0,
                "difficulty": "hard",
                "source": "kernelbench",
                "category": "convolution"
            },
            {
                "id": "relu_inplace",
                "pytorch_operation": "torch.relu_(x)",
                "description": "In-place ReLU activation kernel",
                "test_inputs": [
                    {"shape": [512, 1024], "dtype": "float32", "name": "x"}
                ],
                "expected_speedup": 1.3,
                "difficulty": "easy",
                "source": "kernelbench",
                "category": "activation"
            },
            {
                "id": "transpose_2d",
                "pytorch_operation": "torch.transpose(x, 0, 1)",
                "description": "2D matrix transpose kernel",
                "test_inputs": [
                    {"shape": [2048, 2048], "dtype": "float32", "name": "x"}
                ],
                "expected_speedup": 2.0,
                "difficulty": "medium",
                "source": "kernelbench",
                "category": "memory"
            },
            {
                "id": "reduce_sum",
                "pytorch_operation": "torch.sum(x, dim=1)",
                "description": "Row-wise sum reduction kernel",
                "test_inputs": [
                    {"shape": [1024, 4096], "dtype": "float32", "name": "x"}
                ],
                "expected_speedup": 2.2,
                "difficulty": "medium", 
                "source": "kernelbench",
                "category": "reduction"
            },
            {
                "id": "batch_norm_2d",
                "pytorch_operation": "torch.batch_norm(input, running_mean, running_var, weight, bias)",
                "description": "2D batch normalization kernel",
                "test_inputs": [
                    {"shape": [32, 256, 32, 32], "dtype": "float32", "name": "input"},
                    {"shape": [256], "dtype": "float32", "name": "running_mean"},
                    {"shape": [256], "dtype": "float32", "name": "running_var"},
                    {"shape": [256], "dtype": "float32", "name": "weight"},
                    {"shape": [256], "dtype": "float32", "name": "bias"}
                ],
                "expected_speedup": 2.8,
                "difficulty": "hard",
                "source": "kernelbench",
                "category": "normalization"
            }
        ]
        
        # Add more complexity and variety
        extended_problems = []
        for base_problem in synthetic_problems:
            # Add different input sizes for the same operation
            for size_variant in ["small", "medium", "large"]:
                problem = base_problem.copy()
                problem["id"] = f"{base_problem['id']}_{size_variant}"
                problem["test_inputs"] = self._scale_test_inputs(
                    base_problem["test_inputs"], size_variant
                )
                extended_problems.append(problem)
        
        return extended_problems[:self.max_problems_per_source]
    
    def _scale_test_inputs(self, original_inputs: List[Dict], size_variant: str) -> List[Dict]:
        """Scale test inputs for different size variants."""
        scale_factors = {
            "small": 0.5,
            "medium": 1.0, 
            "large": 2.0
        }
        
        factor = scale_factors.get(size_variant, 1.0)
        scaled_inputs = []
        
        for input_spec in original_inputs:
            scaled_input = input_spec.copy()
            if "shape" in scaled_input:
                # Scale dimensions (but keep reasonable sizes)
                original_shape = scaled_input["shape"]
                scaled_shape = []
                for dim in original_shape:
                    scaled_dim = max(1, int(dim * factor))
                    # Keep dimensions reasonable (not too large)
                    scaled_dim = min(scaled_dim, 4096)
                    scaled_shape.append(scaled_dim)
                scaled_input["shape"] = scaled_shape
            
            scaled_inputs.append(scaled_input)
        
        return scaled_inputs
    
    def _parse_test_inputs(self, raw_inputs: Any) -> List[Dict[str, Any]]:
        """Parse test inputs from various formats."""
        if not raw_inputs:
            return []
        
        # Handle different input formats from datasets
        parsed_inputs = []
        
        if isinstance(raw_inputs, list):
            for i, input_data in enumerate(raw_inputs):
                if isinstance(input_data, dict):
                    parsed_inputs.append(input_data)
                elif isinstance(input_data, (list, tuple)):
                    # Assume it's a shape specification
                    parsed_inputs.append({
                        "shape": list(input_data),
                        "dtype": "float32",
                        "name": f"input_{i}"
                    })
                else:
                    # Handle other formats as needed
                    self.logger.warning(f"Unknown input format: {type(input_data)}")
        
        return parsed_inputs
    
    def _validate_problem(self, problem: Dict[str, Any]) -> bool:
        """Validate that problem has required fields for training."""
        
        required_fields = ["id", "pytorch_operation", "test_inputs"]
        
        # Check required fields exist
        for field in required_fields:
            if field not in problem:
                self.logger.warning(f"Problem {problem.get('id', 'unknown')} missing field: {field}")
                return False
        
        # Validate test inputs format
        if not isinstance(problem["test_inputs"], list) or not problem["test_inputs"]:
            self.logger.warning(f"Problem {problem['id']} has invalid test_inputs")
            return False
        
        # Validate expected speedup is reasonable
        expected_speedup = problem.get("expected_speedup", 1.0)
        if not isinstance(expected_speedup, (int, float)) or expected_speedup < 0.1:
            self.logger.warning(f"Problem {problem['id']} has invalid expected_speedup: {expected_speedup}")
            return False
        
        return True
    
    def get_problem_statistics(self, problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the problem dataset."""
        if not problems:
            return {}
        
        # Count by source
        source_counts = {}
        category_counts = {}
        difficulty_counts = {}
        speedup_stats = []
        
        for problem in problems:
            source = problem.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            
            category = problem.get("category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
            
            difficulty = problem.get("difficulty", "unknown")
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
            
            speedup = problem.get("expected_speedup", 1.0)
            speedup_stats.append(speedup)
        
        stats = {
            "total_problems": len(problems),
            "sources": source_counts,
            "categories": category_counts,
            "difficulties": difficulty_counts,
            "speedup_statistics": {
                "mean": sum(speedup_stats) / len(speedup_stats) if speedup_stats else 0,
                "min": min(speedup_stats) if speedup_stats else 0,
                "max": max(speedup_stats) if speedup_stats else 0,
            }
        }
        
        return stats
    
    async def save_problems_to_disk(
        self, 
        problems: List[Dict[str, Any]], 
        filename: str = "cuda_problems.json"
    ) -> None:
        """Save problems to disk for caching."""
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(problems, f, indent=2, default=str)
            
            self.logger.info(f"Saved {len(problems)} problems to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save problems to {filepath}", error=str(e))
    
    async def load_problems_from_disk(self, filename: str = "cuda_problems.json") -> List[Dict[str, Any]]:
        """Load cached problems from disk."""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            self.logger.info(f"No cached problems found at {filepath}")
            return []
        
        try:
            with open(filepath, 'r') as f:
                problems = json.load(f)
            
            self.logger.info(f"Loaded {len(problems)} cached problems from {filepath}")
            return problems
            
        except Exception as e:
            self.logger.error(f"Failed to load problems from {filepath}", error=str(e))
            return []