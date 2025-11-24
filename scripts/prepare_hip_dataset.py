#!/usr/bin/env python3
"""
Prepare HIP training dataset from HuggingFace or create synthetic data.
Converts CUDA kernels to HIP format for AMD GPU training.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any
import random
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HIPDatasetPreparer:
    """Prepare and manage HIP training datasets"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("./data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _convert_cuda_to_hip(self, cuda_code: str) -> str:
        """Convert CUDA code to HIP code"""
        hip_code = cuda_code
        
        # Header replacements
        hip_code = re.sub(r'#include\s*<cuda_runtime\.h>', '#include <hip/hip_runtime.h>', hip_code)
        hip_code = re.sub(r'#include\s*<cuda\.h>', '#include <hip/hip_runtime.h>', hip_code)
        
        # API replacements
        replacements = {
            'cudaMalloc': 'hipMalloc',
            'cudaFree': 'hipFree',
            'cudaMemcpy': 'hipMemcpy',
            'cudaMemcpyHostToDevice': 'hipMemcpyHostToDevice',
            'cudaMemcpyDeviceToHost': 'hipMemcpyDeviceToHost',
            'cudaDeviceSynchronize': 'hipDeviceSynchronize',
            'cudaGetLastError': 'hipGetLastError',
            'cudaGetErrorString': 'hipGetErrorString',
            'cudaError_t': 'hipError_t',
            'cudaSuccess': 'hipSuccess',
            'cudaMemset': 'hipMemset',
            'cudaEventCreate': 'hipEventCreate',
            'cudaEventRecord': 'hipEventRecord',
            'cudaEventSynchronize': 'hipEventSynchronize',
            'cudaEventElapsedTime': 'hipEventElapsedTime',
            'cudaEventDestroy': 'hipEventDestroy',
            'cudaEvent_t': 'hipEvent_t',
            'cudaStream_t': 'hipStream_t',
            'cudaStreamCreate': 'hipStreamCreate',
            'cudaStreamDestroy': 'hipStreamDestroy',
            'cudaStreamSynchronize': 'hipStreamSynchronize',
            # Atomic operations
            'atomicAdd': 'atomicAdd',  # Same in HIP
        }
        
        for cuda_api, hip_api in replacements.items():
            hip_code = hip_code.replace(cuda_api, hip_api)
        
        return hip_code
        
    def download_huggingface_dataset(self, num_examples: int = 1000) -> List[Dict]:
        """Download and process HuggingFace CUDA dataset, converting to HIP"""
        
        logger.info("Downloading SakanaAI/AI-CUDA-Engineer-Archive dataset...")
        
        try:
            # Load dataset
            dataset = load_dataset(
                "SakanaAI/AI-CUDA-Engineer-Archive",
                split="train"
            )
            
            processed_data = []
            
            for i, example in enumerate(dataset):
                if i >= num_examples:
                    break
                
                # Process each example
                processed = self._process_example(example)
                if processed:
                    processed_data.append(processed)
            
            logger.info(f"Processed {len(processed_data)} examples from HuggingFace")
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return []
    
    def _process_example(self, example: Dict) -> Dict:
        """Process a single dataset example and convert to HIP"""
        
        processed = {}
        
        # Extract problem/instruction
        if "problem" in example:
            processed["problem"] = example["problem"]
        elif "instruction" in example:
            processed["problem"] = example["instruction"]
        elif "task" in example:
            processed["problem"] = example["task"]
        else:
            # Try to construct from other fields
            if "kernel_name" in example and "description" in example:
                processed["problem"] = f"Implement {example['kernel_name']}: {example['description']}"
            else:
                return None
        
        # Update problem description to mention HIP
        processed["problem"] = processed["problem"].replace("CUDA", "HIP").replace("cuda", "HIP")
        
        # Extract solution/code and convert to HIP
        solution = None
        if "solution" in example:
            solution = example["solution"]
        elif "code" in example:
            solution = example["code"]
        elif "kernel_code" in example:
            solution = example["kernel_code"]
        
        if solution:
            processed["solution"] = self._convert_cuda_to_hip(solution)
        
        # Extract metadata
        if "difficulty" in example:
            processed["difficulty"] = example["difficulty"]
        else:
            # Infer difficulty from problem complexity
            processed["difficulty"] = self._infer_difficulty(processed.get("problem", ""))
        
        # Performance metrics if available
        if "performance" in example:
            processed["performance"] = example["performance"]
        
        # Test cases if available
        if "test_cases" in example:
            processed["test_cases"] = example["test_cases"]
        
        return processed
    
    def _infer_difficulty(self, problem: str) -> str:
        """Infer difficulty level from problem description"""
        
        problem_lower = problem.lower()
        
        # Keywords for different difficulty levels
        expert_keywords = ["matrix cores", "fused", "attention", "transformer", "custom gemm", "wmma"]
        advanced_keywords = ["shared memory", "lds", "tiling", "optimization", "coalescing", "bank conflict"]
        intermediate_keywords = ["reduction", "transpose", "convolution", "histogram"]
        
        if any(keyword in problem_lower for keyword in expert_keywords):
            return "expert"
        elif any(keyword in problem_lower for keyword in advanced_keywords):
            return "advanced"
        elif any(keyword in problem_lower for keyword in intermediate_keywords):
            return "intermediate"
        else:
            return "basic"
    
    def create_curriculum_dataset(self, num_examples: int = 1000) -> List[Dict]:
        """Create a curriculum-based synthetic dataset for HIP"""
        
        logger.info(f"Creating curriculum dataset with {num_examples} examples...")
        
        dataset = []
        
        # Distribution of difficulty levels
        distribution = {
            "basic": 0.3,
            "intermediate": 0.35,
            "advanced": 0.25,
            "expert": 0.1
        }
        
        for difficulty, ratio in distribution.items():
            count = int(num_examples * ratio)
            examples = self._generate_examples_for_level(difficulty, count)
            dataset.extend(examples)
        
        # Shuffle for variety
        random.shuffle(dataset)
        
        logger.info(f"Created {len(dataset)} curriculum examples")
        return dataset
    
    def _generate_examples_for_level(self, difficulty: str, count: int) -> List[Dict]:
        """Generate HIP examples for a specific difficulty level"""
        
        examples = []
        
        if difficulty == "basic":
            templates = [
                {
                    "problem": "Implement vector addition for {size} float elements using HIP",
                    "solution_template": """#include <hip/hip_runtime.h>

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        c[idx] = a[idx] + b[idx];
    }}
}}""",
                    "concepts": ["thread indexing", "boundary checking"]
                },
                {
                    "problem": "Create element-wise multiplication for {size} elements using HIP",
                    "solution_template": """#include <hip/hip_runtime.h>

__global__ void elementMul(const float* a, const float* b, float* c, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        c[idx] = a[idx] * b[idx];
    }}
}}""",
                    "concepts": ["parallel computation", "element-wise operations"]
                }
            ]
        
        elif difficulty == "intermediate":
            templates = [
                {
                    "problem": "Implement parallel reduction to sum {size} elements using HIP shared memory",
                    "solution_template": """#include <hip/hip_runtime.h>

__global__ void reduce(float* data, float* result, int n) {{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    sdata[tid] = (idx < n) ? data[idx] : 0;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] += sdata[tid + s];
        }}
        __syncthreads();
    }}
    
    if (tid == 0) atomicAdd(result, sdata[0]);
}}""",
                    "concepts": ["shared memory (LDS)", "reduction", "synchronization"]
                }
            ]
        
        elif difficulty == "advanced":
            templates = [
                {
                    "problem": "Optimize matrix multiplication for {dim}x{dim} matrices with tiling using HIP",
                    "solution_template": """#include <hip/hip_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_tiled(float* A, float* B, float* C, int N) {{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {{
        if (row < N && t * TILE_SIZE + tx < N)
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (col < N && t * TILE_SIZE + ty < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[ty][k] * Bs[k][tx];
            
        __syncthreads();
    }}
    
    if (row < N && col < N)
        C[row * N + col] = sum;
}}""",
                    "concepts": ["tiling", "shared memory (LDS)", "coalesced access", "loop unrolling"]
                }
            ]
        
        else:  # expert
            templates = [
                {
                    "problem": "Implement fused softmax attention for transformer with flash attention optimization using HIP",
                    "solution_template": """#include <hip/hip_runtime.h>

// Flash attention implementation for AMD GPUs
// Uses LDS (Local Data Share) for efficient memory access
__global__ void flash_attention(
    const float* Q, const float* K, const float* V,
    float* O, int seq_len, int head_dim
) {{
    // Advanced implementation with:
    // - Online softmax computation
    // - Tiling in LDS (Local Data Share)
    // - Recomputation to save memory
    // - Wavefront-level primitives
    
    extern __shared__ float shared_mem[];
    float* s_Q = shared_mem;
    float* s_K = shared_mem + head_dim;
    float* s_V = shared_mem + 2 * head_dim;
    
    // Implementation details depend on specific AMD architecture
    // (CDNA2/CDNA3 for MI250/MI300, RDNA3 for consumer GPUs)
}}""",
                    "concepts": ["kernel fusion", "online algorithms", "memory hierarchy", "wavefront primitives"]
                }
            ]
        
        # Generate examples from templates
        sizes = [1024, 2048, 4096, 8192, 16384, 32768]
        dims = [128, 256, 512, 1024]
        
        for _ in range(count):
            template = random.choice(templates)
            
            # Fill in parameters
            problem = template["problem"]
            if "{size}" in problem:
                problem = problem.format(size=random.choice(sizes))
            elif "{dim}" in problem:
                problem = problem.format(dim=random.choice(dims))
            
            example = {
                "problem": problem,
                "solution": template["solution_template"],
                "difficulty": difficulty,
                "concepts": template["concepts"],
                "performance_targets": self._get_performance_targets(difficulty)
            }
            
            examples.append(example)
        
        return examples
    
    def _get_performance_targets(self, difficulty: str) -> Dict:
        """Get performance targets based on difficulty"""
        
        targets = {
            "basic": {
                "bandwidth_efficiency": 0.5,
                "compute_efficiency": 0.4,
                "speedup_vs_cpu": 10
            },
            "intermediate": {
                "bandwidth_efficiency": 0.7,
                "compute_efficiency": 0.6,
                "speedup_vs_cpu": 50
            },
            "advanced": {
                "bandwidth_efficiency": 0.85,
                "compute_efficiency": 0.8,
                "speedup_vs_cpu": 100
            },
            "expert": {
                "bandwidth_efficiency": 0.95,
                "compute_efficiency": 0.9,
                "speedup_vs_cpu": 200
            }
        }
        
        return targets.get(difficulty, targets["basic"])
    
    def save_dataset(self, dataset: List[Dict], filename: str = "hip_training_data.json"):
        """Save dataset to JSON file"""
        
        output_path = self.data_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump({
                "version": "1.0",
                "target_platform": "ROCm/HIP",
                "num_examples": len(dataset),
                "problems": dataset
            }, f, indent=2)
        
        logger.info(f"Saved {len(dataset)} examples to {output_path}")
        
        # Also create separate files for SFT phases
        self._create_sft_datasets(dataset)
    
    def _create_sft_datasets(self, dataset: List[Dict]):
        """Create separate datasets for generator and optimizer SFT"""
        
        # Generator dataset - problem to initial solution
        generator_data = []
        for example in dataset:
            if "solution" in example:
                generator_data.append({
                    "input": f"Generate HIP kernel for: {example['problem']}",
                    "output": example["solution"]
                })
        
        # Optimizer dataset - unoptimized to optimized
        optimizer_data = []
        for example in dataset:
            if example.get("difficulty") in ["advanced", "expert"]:
                # Create optimization pairs
                optimizer_data.append({
                    "input": f"Optimize this HIP kernel: {example.get('solution', '')}",
                    "output": example.get("solution", "")  # In real case, would have optimized version
                })
        
        # Save SFT datasets
        with open(self.data_dir / "generator_sft_data.json", 'w') as f:
            json.dump(generator_data, f, indent=2)
        
        with open(self.data_dir / "optimizer_sft_data.json", 'w') as f:
            json.dump(optimizer_data, f, indent=2)
        
        logger.info(f"Created SFT datasets: {len(generator_data)} generator, {len(optimizer_data)} optimizer examples")


def main():
    """Main dataset preparation"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare HIP training dataset")
    parser.add_argument("--source", choices=["huggingface", "curriculum", "both"], 
                        default="both", help="Data source")
    parser.add_argument("--num-examples", type=int, default=1000, 
                        help="Number of examples to generate/download")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Output directory for datasets")
    
    args = parser.parse_args()
    
    preparer = HIPDatasetPreparer(Path(args.output_dir))
    
    dataset = []
    
    if args.source in ["huggingface", "both"]:
        # Try to download from HuggingFace and convert to HIP
        hf_data = preparer.download_huggingface_dataset(args.num_examples // 2 if args.source == "both" else args.num_examples)
        dataset.extend(hf_data)
    
    if args.source in ["curriculum", "both"]:
        # Generate curriculum data
        curr_data = preparer.create_curriculum_dataset(args.num_examples // 2 if args.source == "both" else args.num_examples)
        dataset.extend(curr_data)
    
    if dataset:
        # Save the combined dataset
        preparer.save_dataset(dataset)
        logger.info(f"✅ Dataset preparation complete! Total examples: {len(dataset)}")
    else:
        logger.error("❌ No data could be prepared")


if __name__ == "__main__":
    main()
