"""
SakanaAI CUDA Dataset Loader with curriculum support.
Integrates with the SakanaAI/AI-CUDA-Engineer-Archive dataset.
"""

from datasets import load_dataset
from typing import Dict, List, Any, Optional, Tuple
import json
import random
from pathlib import Path
import structlog
import numpy as np


class SakanaDataLoader:
    """Loader for SakanaAI CUDA dataset with curriculum support."""
    
    def __init__(
        self,
        dataset_name: str = "SakanaAI/AI-CUDA-Engineer-Archive",
        cache_dir: Optional[str] = None,
        curriculum_enabled: bool = True
    ):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir or "./cache/datasets"
        self.curriculum_enabled = curriculum_enabled
        self.logger = structlog.get_logger("sakana_data_loader")
        
        # Create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Load and process dataset
        self._load_dataset()
        self._process_curriculum()
    
    def _load_dataset(self):
        """Load the SakanaAI dataset."""
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
        except Exception as e:
            self.logger.warning(f"Failed to load SakanaAI dataset: {e}, using synthetic data")
            self.dataset = self._create_synthetic_dataset()
    
    def _create_synthetic_dataset(self) -> Dict[str, List]:
        """Create synthetic dataset for testing when real dataset unavailable."""
        synthetic_problems = [
            # Easy problems
            {
                "problem_description": "Implement vector addition for two float arrays",
                "cuda_kernel": """
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}""",
                "difficulty": "easy",
                "shapes": [[1024], [4096], [65536]]
            },
            {
                "problem_description": "Implement element-wise multiplication",
                "cuda_kernel": """
__global__ void element_multiply(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}""",
                "difficulty": "easy",
                "shapes": [[2048], [8192]]
            },
            # Medium problems
            {
                "problem_description": "Implement parallel reduction to sum array elements",
                "cuda_kernel": """
__global__ void reduction_sum(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(output, sdata[0]);
}""",
                "difficulty": "medium",
                "shapes": [[1024], [32768]]
            },
            {
                "problem_description": "Implement matrix transpose",
                "cuda_kernel": """
__global__ void matrix_transpose(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();
    
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}""",
                "difficulty": "medium",
                "shapes": [[512, 512], [1024, 1024]]
            },
            # Hard problems
            {
                "problem_description": "Implement optimized matrix multiplication",
                "cuda_kernel": """
__global__ void matrix_multiply(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * 16 + ty;
    int col = bx * 16 + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + 15) / 16; ++tile) {
        if (row < M && tile * 16 + tx < K)
            As[ty][tx] = A[row * K + tile * 16 + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (col < N && tile * 16 + ty < K)
            Bs[ty][tx] = B[(tile * 16 + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < 16; ++k)
            sum += As[ty][k] * Bs[k][tx];
            
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
}""",
                "difficulty": "hard",
                "shapes": [[256, 256, 256], [512, 512, 512]]
            }
        ]
        
        return {"train": synthetic_problems, "test": synthetic_problems[:2]}
    
    def _process_curriculum(self):
        """Process dataset for curriculum learning."""
        if not self.curriculum_enabled:
            return
        
        # Split by difficulty
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
        
        # Check if difficulty is explicitly labeled
        if "difficulty" in example:
            return example["difficulty"]
        
        # Infer from problem characteristics
        problem_desc = example.get("problem_description", "").lower()
        kernel_code = example.get("cuda_kernel", "")
        
        # Simple heuristics for difficulty classification
        if any(keyword in problem_desc for keyword in ["vector", "element-wise", "simple", "basic", "add"]):
            return "easy"
        elif any(keyword in problem_desc for keyword in ["matrix", "reduction", "transpose", "sum"]):
            return "medium"
        elif any(keyword in problem_desc for keyword in ["convolution", "fft", "sort", "graph", "multiply"]):
            return "hard"
        
        # Check code complexity
        if kernel_code:
            if "__shared__" in kernel_code and "atomicAdd" in kernel_code:
                return "hard"
            elif "__shared__" in kernel_code or "__syncthreads()" in kernel_code:
                return "medium"
            else:
                return "easy"
        
        return "medium"  # Default to medium if unsure
    
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
            # Sample from all data
            candidates = list(self.dataset["train"]) if "train" in self.dataset else []
        
        if not candidates:
            raise ValueError(f"No problems available for difficulty: {difficulty}")
        
        example = random.choice(candidates)
        
        # Standardize problem format
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
    
    async def get_batch(
        self,
        batch_size: int,
        difficulty: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get a batch of problems."""
        batch = []
        for _ in range(batch_size):
            problem = await self.sample_problem(difficulty=difficulty)
            batch.append(problem)
        return batch
    
    def _generate_test_cases(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases for the problem."""
        
        # Extract shape information if available
        shapes = example.get("shapes", [])
        if not shapes:
            # Default test cases based on difficulty
            difficulty = self._infer_difficulty(example)
            if difficulty == "easy":
                shapes = [[1024], [4096], [65536]]
            elif difficulty == "medium":
                shapes = [[1024, 1024], [2048, 2048]]
            else:
                shapes = [[512, 512, 512], [1024, 1024, 256]]
        
        test_cases = []
        for shape in shapes:
            test_cases.append({
                "input_shapes": [shape, shape] if len(shape) <= 2 else [shape],
                "dtype": "float32",
                "grid_dims": self._calculate_grid_dims(shape),
                "block_dims": self._calculate_block_dims(shape)
            })
        
        return test_cases
    
    def _calculate_grid_dims(self, shape: List[int]) -> Tuple[int, int, int]:
        """Calculate appropriate grid dimensions for shape."""
        if len(shape) == 1:
            return (min(65535, (shape[0] + 255) // 256), 1, 1)
        elif len(shape) == 2:
            return (
                min(65535, (shape[0] + 15) // 16),
                min(65535, (shape[1] + 15) // 16),
                1
            )
        else:
            return (
                min(65535, (shape[0] + 7) // 8),
                min(65535, (shape[1] + 7) // 8),
                min(65535, (shape[2] + 7) // 8)
            )
    
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
        
        # Default targets based on difficulty
        difficulty = self._infer_difficulty(example)
        if difficulty == "easy":
            return {"speedup": 1.5, "min_speedup": 1.2}
        elif difficulty == "medium":
            return {"speedup": 2.0, "min_speedup": 1.5}
        else:
            return {"speedup": 3.0, "min_speedup": 2.0}