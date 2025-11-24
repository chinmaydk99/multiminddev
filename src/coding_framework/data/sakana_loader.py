"""
HIP Dataset Loader with curriculum support for AMD ROCm.
Provides synthetic HIP kernel examples and can integrate with external datasets.
Note: The SakanaAI dataset contains CUDA code which can be converted to HIP.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
from datasets import load_dataset


class SakanaDataLoader:
    """Loader for HIP kernel dataset with curriculum support for AMD ROCm."""

    def __init__(
        self,
        dataset_name: str = "SakanaAI/AI-CUDA-Engineer-Archive",
        cache_dir: Optional[str] = None,
        curriculum_enabled: bool = True,
        convert_cuda_to_hip: bool = True
    ):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir or "./cache/datasets"
        self.curriculum_enabled = curriculum_enabled
        self.convert_cuda_to_hip = convert_cuda_to_hip
        self.logger = structlog.get_logger("hip_data_loader")

        # Create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # Load and process dataset
        self._load_dataset()
        self._process_curriculum()

    def _load_dataset(self):
        """Load the dataset (uses HIP synthetic data by default for ROCm)."""
        try:
            self.dataset = load_dataset(
                self.dataset_name,
                cache_dir=self.cache_dir,
                trust_remote_code=False
            )
            self.logger.info(
                "Dataset loaded (will convert to HIP if needed)",
                level_1=len(self.dataset["level_1"]) if "level_1" in self.dataset else 0,
                level_2=len(self.dataset["level_2"]) if "level_2" in self.dataset else 0,
                level_3=len(self.dataset["level_3"]) if "level_3" in self.dataset else 0
            )
        except Exception as e:
            self.logger.warning(f"Failed to load external dataset: {e}, using synthetic HIP data")
            self.dataset = self._create_synthetic_dataset()

    def _cuda_to_hip(self, cuda_code: str) -> str:
        """Convert CUDA code to HIP code."""
        if not self.convert_cuda_to_hip:
            return cuda_code

        hip_code = cuda_code
        # Basic CUDA to HIP conversions
        replacements = [
            ("cuda_runtime.h", "hip/hip_runtime.h"),
            ("cudaMalloc", "hipMalloc"),
            ("cudaFree", "hipFree"),
            ("cudaMemcpy", "hipMemcpy"),
            ("cudaMemcpyHostToDevice", "hipMemcpyHostToDevice"),
            ("cudaMemcpyDeviceToHost", "hipMemcpyDeviceToHost"),
            ("cudaDeviceSynchronize", "hipDeviceSynchronize"),
            ("cudaGetLastError", "hipGetLastError"),
            ("cudaGetErrorString", "hipGetErrorString"),
            ("cudaError_t", "hipError_t"),
            ("cudaSuccess", "hipSuccess"),
            ("CUDA_CHECK", "HIP_CHECK"),
        ]

        for cuda_term, hip_term in replacements:
            hip_code = hip_code.replace(cuda_term, hip_term)

        return hip_code

    def _create_synthetic_dataset(self) -> Dict[str, List]:
        """Create synthetic HIP dataset for testing when real dataset unavailable."""
        synthetic_problems = [
            # Easy problems
            {
                "problem_description": "Implement vector addition for two float arrays",
                "hip_kernel": """
#include <hip/hip_runtime.h>

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
                "hip_kernel": """
#include <hip/hip_runtime.h>

__global__ void element_multiply(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}""",
                "difficulty": "easy",
                "shapes": [[2048], [8192]]
            },
            {
                "problem_description": "Implement scalar multiplication for a float array",
                "hip_kernel": """
#include <hip/hip_runtime.h>

__global__ void scalar_multiply(float* input, float* output, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scalar;
    }
}""",
                "difficulty": "easy",
                "shapes": [[1024], [4096]]
            },
            # Medium problems
            {
                "problem_description": "Implement parallel reduction to sum array elements using LDS",
                "hip_kernel": """
#include <hip/hip_runtime.h>

__global__ void reduction_sum(float* input, float* output, int n) {
    extern __shared__ float sdata[];  // LDS (Local Data Share) memory
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    // Parallel reduction in LDS
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
                "problem_description": "Implement matrix transpose using LDS for coalesced access",
                "hip_kernel": """
#include <hip/hip_runtime.h>

__global__ void matrix_transpose(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33];  // Padding to avoid LDS bank conflicts
    
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
            {
                "problem_description": "Implement dot product using parallel reduction",
                "hip_kernel": """
#include <hip/hip_runtime.h>

__global__ void dot_product(float* a, float* b, float* result, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes partial product
    sdata[tid] = (idx < n) ? a[idx] * b[idx] : 0;
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(result, sdata[0]);
}""",
                "difficulty": "medium",
                "shapes": [[1024], [4096], [16384]]
            },
            # Hard problems
            {
                "problem_description": "Implement optimized tiled matrix multiplication using LDS",
                "hip_kernel": """
#include <hip/hip_runtime.h>

#define TILE_SIZE 16

__global__ void matrix_multiply(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Collaborative loading into LDS
        if (row < M && tile * TILE_SIZE + tx < K)
            As[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (col < N && tile * TILE_SIZE + ty < K)
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[ty][k] * Bs[k][tx];
            
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
}""",
                "difficulty": "hard",
                "shapes": [[256, 256, 256], [512, 512, 512]]
            },
            {
                "problem_description": "Implement 1D convolution with LDS optimization",
                "hip_kernel": """
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 256
#define HALO_SIZE 3  // For kernel radius of 3

__global__ void conv1d(float* input, float* kernel_weights, float* output, 
                       int input_size, int kernel_size) {
    __shared__ float tile[BLOCK_SIZE + 2 * HALO_SIZE];
    
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    
    // Load main data
    if (gid < input_size) {
        tile[lid + HALO_SIZE] = input[gid];
    }
    
    // Load halo regions
    if (lid < HALO_SIZE) {
        int left_idx = gid - HALO_SIZE;
        tile[lid] = (left_idx >= 0) ? input[left_idx] : 0;
        
        int right_idx = gid + BLOCK_SIZE;
        if (right_idx < input_size) {
            tile[lid + BLOCK_SIZE + HALO_SIZE] = input[right_idx];
        }
    }
    
    __syncthreads();
    
    // Compute convolution
    if (gid < input_size) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            sum += tile[lid + k] * kernel_weights[k];
        }
        output[gid] = sum;
    }
}""",
                "difficulty": "hard",
                "shapes": [[4096], [16384], [65536]]
            }
        ]

        return {"train": synthetic_problems, "test": synthetic_problems[:2]}

    def _process_curriculum(self):
        """Process dataset for curriculum learning."""
        if not self.curriculum_enabled:
            return

        # Map levels to curriculum difficulties
        self.curriculum_data = {
            "easy": list(self.dataset.get("level_1", [])),
            "medium": list(self.dataset.get("level_2", [])),
            "hard": list(self.dataset.get("level_3", []))
        }

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
        kernel_code = example.get("hip_kernel", example.get("cuda_kernel", ""))

        # Simple heuristics for difficulty classification
        if any(keyword in problem_desc for keyword in ["vector", "element-wise", "simple", "basic", "add", "scalar"]):
            return "easy"
        elif any(keyword in problem_desc for keyword in ["matrix", "reduction", "transpose", "sum", "dot"]):
            return "medium"
        elif any(keyword in problem_desc for keyword in ["convolution", "fft", "sort", "graph", "multiply", "tiled"]):
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

        # Get kernel code (convert from CUDA if needed)
        kernel_code = example.get("hip_kernel", "")
        if not kernel_code:
            cuda_code = example.get("CUDA_Code", example.get("cuda_kernel", ""))
            kernel_code = self._cuda_to_hip(cuda_code)

        # Standardize problem format
        return {
            "description": f"Optimize HIP kernel for {example.get('Op_Name', 'operation')}",
            "difficulty": difficulty or self._infer_difficulty(example),
            "reference_solution": kernel_code,
            "test_cases": self._generate_test_cases(example),
            "target_performance": self._extract_target_performance(example),
            "baseline_performance": {
                "pytorch_native": example.get("PyTorch_Native_Runtime", 0),
                "pytorch_compile": example.get("PyTorch_Compile_Runtime", 0),
                "hip_speedup_native": example.get("CUDA_Speedup_Native", 1),  # Reuse CUDA metrics
                "hip_speedup_compile": example.get("CUDA_Speedup_Compile", 1)
            },
            "metadata": {
                "original_example": example,
                "dataset": self.dataset_name,
                "kernel_name": example.get("Kernel_Name", ""),
                "op_name": example.get("Op_Name", ""),
                "task_id": example.get("Task_ID", ""),
                "level_id": example.get("Level_ID", ""),
                "converted_from_cuda": "hip_kernel" not in example
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
        """Calculate appropriate block dimensions for shape (AMD wavefront-friendly)."""
        # AMD GPUs use wavefronts of 64 threads (vs NVIDIA warps of 32)
        if len(shape) == 1:
            return (256, 1, 1)  # 4 wavefronts
        elif len(shape) == 2:
            return (16, 16, 1)  # 4 wavefronts (256 threads)
        else:
            return (8, 8, 4)  # 4 wavefronts (256 threads)

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
