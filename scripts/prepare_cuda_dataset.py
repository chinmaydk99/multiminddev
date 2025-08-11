#!/usr/bin/env python3
"""
Prepare CUDA training dataset from HuggingFace or create synthetic data
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import random
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CUDADatasetPreparer:
    """Prepare and manage CUDA training datasets"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("./data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_huggingface_dataset(self, num_examples: int = 1000) -> List[Dict]:
        """Download and process HuggingFace CUDA dataset"""
        
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
        """Process a single dataset example"""
        
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
        
        # Extract solution/code
        if "solution" in example:
            processed["solution"] = example["solution"]
        elif "code" in example:
            processed["solution"] = example["code"]
        elif "kernel_code" in example:
            processed["solution"] = example["kernel_code"]
        
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
        expert_keywords = ["tensor cores", "fused", "attention", "transformer", "custom gemm"]
        advanced_keywords = ["shared memory", "tiling", "optimization", "coalescing", "bank conflict"]
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
        """Create a curriculum-based synthetic dataset"""
        
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
        """Generate examples for a specific difficulty level"""
        
        examples = []
        
        if difficulty == "basic":
            templates = [
                {
                    "problem": "Implement vector addition for {size} float elements",
                    "solution_template": """__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        c[idx] = a[idx] + b[idx];
    }}
}}""",
                    "concepts": ["thread indexing", "boundary checking"]
                },
                {
                    "problem": "Create element-wise multiplication for {size} elements",
                    "solution_template": """__global__ void elementMul(const float* a, const float* b, float* c, int n) {{
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
                    "problem": "Implement parallel reduction to sum {size} elements",
                    "solution_template": """__global__ void reduce(float* data, float* result, int n) {{
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
                    "concepts": ["shared memory", "reduction", "synchronization"]
                }
            ]
        
        elif difficulty == "advanced":
            templates = [
                {
                    "problem": "Optimize matrix multiplication for {dim}x{dim} matrices with tiling",
                    "solution_template": """__global__ void matmul_tiled(float* A, float* B, float* C, int N) {{
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
                    "concepts": ["tiling", "shared memory", "coalesced access", "loop unrolling"]
                }
            ]
        
        else:  # expert
            templates = [
                {
                    "problem": "Implement fused softmax attention for transformer with flash attention optimization",
                    "solution_template": """// Complex flash attention implementation
__global__ void flash_attention(/* parameters */) {{
    // Advanced implementation with:
    // - Online softmax computation
    // - Tiling in SRAM
    // - Recomputation to save memory
    // - Warp-level primitives
}}""",
                    "concepts": ["kernel fusion", "online algorithms", "memory hierarchy", "warp primitives"]
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
    
    def save_dataset(self, dataset: List[Dict], filename: str = "cuda_training_data.json"):
        """Save dataset to JSON file"""
        
        output_path = self.data_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump({
                "version": "1.0",
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
                    "input": f"Generate CUDA kernel for: {example['problem']}",
                    "output": example["solution"]
                })
        
        # Optimizer dataset - unoptimized to optimized
        optimizer_data = []
        for example in dataset:
            if example.get("difficulty") in ["advanced", "expert"]:
                # Create optimization pairs
                optimizer_data.append({
                    "input": f"Optimize this kernel: {example.get('solution', '')}",
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
    
    parser = argparse.ArgumentParser(description="Prepare CUDA training dataset")
    parser.add_argument("--source", choices=["huggingface", "curriculum", "both"], 
                        default="both", help="Data source")
    parser.add_argument("--num-examples", type=int, default=1000, 
                        help="Number of examples to generate/download")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Output directory for datasets")
    
    args = parser.parse_args()
    
    preparer = CUDADatasetPreparer(Path(args.output_dir))
    
    dataset = []
    
    if args.source in ["huggingface", "both"]:
        # Try to download from HuggingFace
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