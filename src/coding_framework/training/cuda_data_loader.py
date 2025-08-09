import os
import json
from typing import List, Dict, Tuple, Any, Optional
import structlog

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