"""
SFT (Supervised Fine-Tuning) data preparation pipeline for training individual agents
before multi-turn RL training. Prepares specialized datasets for each agent type.
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import structlog
from dataclasses import dataclass, field
import asyncio
import aiofiles
from datasets import Dataset, load_dataset
import torch
from transformers import PreTrainedTokenizer


@dataclass
class SFTDataItem:
    """Single SFT training example."""
    input_text: str
    output_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input": self.input_text,
            "output": self.output_text,
            "metadata": self.metadata
        }


class CUDADatasetGenerator:
    """
    Generate synthetic CUDA training data for SFT.
    Uses templates and patterns to create diverse training examples.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger()
        
        # Common CUDA operations to generate kernels for
        self.operations = [
            "element-wise addition",
            "element-wise multiplication",
            "matrix multiplication",
            "reduction (sum)",
            "reduction (max)",
            "convolution",
            "transpose",
            "softmax",
            "batch normalization",
            "layer normalization"
        ]
        
        # Data types
        self.dtypes = ["float", "double", "int", "half"]
        
        # Common tensor shapes
        self.shapes = [
            (1024,),
            (256, 256),
            (128, 512),
            (32, 1024, 768),
            (16, 128, 128, 64)
        ]
        
        # Optimization patterns
        self.optimization_patterns = {
            "shared_memory": "Use shared memory to reduce global memory access",
            "memory_coalescing": "Ensure coalesced memory access patterns",
            "vectorized_access": "Use float4/int4 for vectorized loads",
            "loop_unrolling": "Unroll loops for better instruction throughput",
            "warp_primitives": "Use warp-level primitives like __shfl_down_sync"
        }
    
    def generate_generator_examples(self, num_examples: int = 1000) -> List[SFTDataItem]:
        """Generate SFT examples for the generator agent."""
        examples = []
        
        for _ in range(num_examples):
            operation = random.choice(self.operations)
            dtype = random.choice(self.dtypes)
            shape = random.choice(self.shapes)
            
            # Create input prompt
            input_text = self._create_generator_prompt(operation, dtype, shape)
            
            # Create expected output (kernel code)
            output_text = self._create_kernel_template(operation, dtype, shape)
            
            metadata = {
                "operation": operation,
                "dtype": dtype,
                "shape": shape
            }
            
            examples.append(SFTDataItem(input_text, output_text, metadata))
        
        self.logger.info(f"Generated {len(examples)} generator SFT examples")
        return examples
    
    def generate_optimizer_examples(self, num_examples: int = 800) -> List[SFTDataItem]:
        """Generate SFT examples for the optimizer agent."""
        examples = []
        
        # First generate base kernels
        base_kernels = self.generate_generator_examples(num_examples)
        
        for base_kernel in base_kernels:
            # Choose optimization strategies
            num_opts = random.randint(1, 3)
            optimizations = random.sample(
                list(self.optimization_patterns.keys()),
                num_opts
            )
            
            # Create input (unoptimized kernel + optimization request)
            input_text = f"""Optimize this CUDA kernel:

{base_kernel.output_text}

Apply these optimizations: {', '.join(optimizations)}
Target: 2x speedup"""
            
            # Create optimized version
            output_text = self._apply_optimizations(
                base_kernel.output_text,
                optimizations
            )
            
            metadata = {
                **base_kernel.metadata,
                "optimizations": optimizations
            }
            
            examples.append(SFTDataItem(input_text, output_text, metadata))
        
        self.logger.info(f"Generated {len(examples)} optimizer SFT examples")
        return examples
    
    def _create_generator_prompt(
        self,
        operation: str,
        dtype: str,
        shape: Tuple[int, ...]
    ) -> str:
        """Create generator agent prompt."""
        return f"""Generate CUDA kernel for: {operation}
Input tensors shape: {shape}
Data type: {dtype}
Requirements:
- Include proper headers
- Use efficient thread indexing
- Include boundary checks
- Optimize for memory coalescing"""
    
    def _create_kernel_template(
        self,
        operation: str,
        dtype: str,
        shape: Tuple[int, ...]
    ) -> str:
        """Create template kernel code."""
        # Simplified template generation
        kernel_name = operation.replace(" ", "_").replace("(", "").replace(")", "")
        
        if "element-wise" in operation:
            return self._elementwise_kernel_template(kernel_name, dtype, shape)
        elif "reduction" in operation:
            return self._reduction_kernel_template(kernel_name, dtype, shape)
        elif "matrix" in operation:
            return self._matmul_kernel_template(dtype, shape)
        else:
            return self._generic_kernel_template(kernel_name, dtype, shape)
    
    def _elementwise_kernel_template(
        self,
        kernel_name: str,
        dtype: str,
        shape: Tuple[int, ...]
    ) -> str:
        """Generate element-wise kernel template."""
        return f"""#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void {kernel_name}_kernel(
    const {dtype}* __restrict__ input_a,
    const {dtype}* __restrict__ input_b,
    {dtype}* __restrict__ output,
    const int n
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {{
        output[idx] = input_a[idx] + input_b[idx];  // Modify operation as needed
    }}
}}

// Launch configuration: 
// dim3 block(256);
// dim3 grid((n + block.x - 1) / block.x);"""
    
    def _reduction_kernel_template(
        self,
        kernel_name: str,
        dtype: str,
        shape: Tuple[int, ...]
    ) -> str:
        """Generate reduction kernel template."""
        return f"""#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void {kernel_name}_kernel(
    const {dtype}* __restrict__ input,
    {dtype}* __restrict__ output,
    const int n
) {{
    extern __shared__ {dtype} sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data to shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] += sdata[tid + s];
        }}
        __syncthreads();
    }}
    
    // Write result
    if (tid == 0) {{
        output[blockIdx.x] = sdata[0];
    }}
}}"""
    
    def _matmul_kernel_template(self, dtype: str, shape: Tuple[int, ...]) -> str:
        """Generate matrix multiplication kernel template."""
        return f"""#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_kernel(
    const {dtype}* __restrict__ A,
    const {dtype}* __restrict__ B,
    {dtype}* __restrict__ C,
    const int M, const int N, const int K
) {{
    __shared__ {dtype} As[TILE_SIZE][TILE_SIZE];
    __shared__ {dtype} Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    {dtype} sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {{
        if (row < M && t * TILE_SIZE + tx < K)
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (col < N && t * TILE_SIZE + ty < K)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[ty][k] * Bs[k][tx];
            
        __syncthreads();
    }}
    
    if (row < M && col < N)
        C[row * N + col] = sum;
}}"""
    
    def _generic_kernel_template(
        self,
        kernel_name: str,
        dtype: str,
        shape: Tuple[int, ...]
    ) -> str:
        """Generate generic kernel template."""
        return f"""#include <cuda_runtime.h>

__global__ void {kernel_name}_kernel(
    const {dtype}* __restrict__ input,
    {dtype}* __restrict__ output,
    const int n
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {{
        // Process element i
        output[i] = input[i];  // Modify as needed
    }}
}}"""
    
    def _apply_optimizations(
        self,
        kernel_code: str,
        optimizations: List[str]
    ) -> str:
        """Apply optimizations to kernel code (simplified)."""
        optimized = kernel_code
        
        for opt in optimizations:
            if opt == "shared_memory" and "__shared__" not in optimized:
                # Add shared memory declaration
                optimized = optimized.replace(
                    "__global__ void",
                    "__shared__ float shared_data[256];\n\n__global__ void"
                )
            
            elif opt == "vectorized_access" and "float4" not in optimized:
                # Convert to vectorized access
                optimized = optimized.replace("float", "float4", 1)
                
            elif opt == "loop_unrolling" and "#pragma unroll" not in optimized:
                # Add unroll pragma
                optimized = optimized.replace(
                    "for (",
                    "#pragma unroll 4\n    for (",
                    1
                )
        
        return optimized


class SFTDataPipeline:
    """
    Complete SFT data preparation pipeline for all agents.
    Handles data loading, processing, and formatting for training.
    """
    
    def __init__(
        self,
        data_dir: Path = Path("data/sft"),
        use_huggingface_data: bool = True
    ):
        """
        Initialize SFT data pipeline.
        
        Args:
            data_dir: Directory to save/load SFT data
            use_huggingface_data: Whether to use HuggingFace datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.use_huggingface_data = use_huggingface_data
        self.logger = structlog.get_logger()
        
        self.generator = CUDADatasetGenerator()
    
    async def prepare_generator_data(
        self,
        num_examples: int = 10000,
        save_path: Optional[Path] = None
    ) -> Dataset:
        """
        Prepare SFT data for generator agent.
        
        Args:
            num_examples: Number of examples to generate
            save_path: Path to save the dataset
            
        Returns:
            HuggingFace Dataset object
        """
        self.logger.info("Preparing generator agent SFT data")
        
        all_examples = []
        
        # Load from HuggingFace if available
        if self.use_huggingface_data:
            try:
                # Try to load CUDA dataset from HuggingFace
                hf_data = await self._load_huggingface_cuda_data("generator")
                all_examples.extend(hf_data)
            except Exception as e:
                self.logger.warning(f"Could not load HuggingFace data: {e}")
        
        # Generate synthetic examples
        synthetic_examples = self.generator.generate_generator_examples(
            num_examples - len(all_examples)
        )
        all_examples.extend(synthetic_examples)
        
        # Convert to HuggingFace Dataset
        dataset_dict = {
            "input": [ex.input_text for ex in all_examples],
            "output": [ex.output_text for ex in all_examples],
            "metadata": [json.dumps(ex.metadata) for ex in all_examples]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Save if path provided
        if save_path:
            save_path = self.data_dir / save_path
            dataset.save_to_disk(str(save_path))
            self.logger.info(f"Saved generator dataset to {save_path}")
        
        return dataset
    
    async def prepare_optimizer_data(
        self,
        num_examples: int = 8000,
        save_path: Optional[Path] = None
    ) -> Dataset:
        """
        Prepare SFT data for optimizer agent.
        
        Args:
            num_examples: Number of examples to generate
            save_path: Path to save the dataset
            
        Returns:
            HuggingFace Dataset object
        """
        self.logger.info("Preparing optimizer agent SFT data")
        
        # Generate optimizer examples
        examples = self.generator.generate_optimizer_examples(num_examples)
        
        # Convert to HuggingFace Dataset
        dataset_dict = {
            "input": [ex.input_text for ex in examples],
            "output": [ex.output_text for ex in examples],
            "metadata": [json.dumps(ex.metadata) for ex in examples]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Save if path provided
        if save_path:
            save_path = self.data_dir / save_path
            dataset.save_to_disk(str(save_path))
            self.logger.info(f"Saved optimizer dataset to {save_path}")
        
        return dataset
    
    async def _load_huggingface_cuda_data(
        self,
        agent_type: str
    ) -> List[SFTDataItem]:
        """
        Load CUDA data from HuggingFace datasets.
        
        Args:
            agent_type: Type of agent (generator/optimizer)
            
        Returns:
            List of SFT data items
        """
        examples = []
        
        try:
            # Try to load SakanaAI CUDA dataset
            dataset = load_dataset(
                "SakanaAI/AI-CUDA-Engineer-Archive",
                split="train[:1000]"  # Load subset
            )
            
            for item in dataset:
                # Parse based on agent type
                if agent_type == "generator":
                    if "pytorch_code" in item and "cuda_kernel" in item:
                        input_text = f"Generate CUDA kernel for:\n{item['pytorch_code']}"
                        output_text = item["cuda_kernel"]
                        examples.append(
                            SFTDataItem(input_text, output_text, {"source": "sakana"})
                        )
                
                elif agent_type == "optimizer":
                    if "unoptimized_kernel" in item and "optimized_kernel" in item:
                        input_text = f"Optimize this kernel:\n{item['unoptimized_kernel']}"
                        output_text = item["optimized_kernel"]
                        examples.append(
                            SFTDataItem(input_text, output_text, {"source": "sakana"})
                        )
        
        except Exception as e:
            self.logger.warning(f"Could not load SakanaAI dataset: {e}")
        
        return examples
    
    def tokenize_for_training(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048
    ) -> Dataset:
        """
        Tokenize dataset for model training.
        
        Args:
            dataset: HuggingFace Dataset
            tokenizer: Model tokenizer
            max_length: Maximum sequence length
            
        Returns:
            Tokenized dataset
        """
        def tokenize_function(examples):
            # Combine input and output for training
            full_texts = [
                f"{inp}\n{out}" 
                for inp, out in zip(examples["input"], examples["output"])
            ]
            
            # Tokenize
            tokenized = tokenizer(
                full_texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Set labels (same as input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    async def prepare_cross_agent_data(
        self,
        num_conversations: int = 1000,
        save_path: Optional[Path] = None
    ) -> Dataset:
        """
        Prepare cross-agent compatibility training data.
        
        This creates multi-turn conversation examples for ensuring
        agents can work together effectively.
        
        Args:
            num_conversations: Number of conversations to generate
            save_path: Path to save the dataset
            
        Returns:
            HuggingFace Dataset with conversation data
        """
        self.logger.info("Preparing cross-agent compatibility data")
        
        conversations = []
        
        for _ in range(num_conversations):
            # Generate a complete conversation flow
            problem = random.choice([
                "Implement efficient matrix multiplication",
                "Create parallel reduction kernel", 
                "Optimize convolution operation",
                "Implement batch normalization"
            ])
            
            conversation = {
                "problem": problem,
                "turns": []
            }
            
            # Turn 1: Generator creates kernel
            gen_input = f"Generate CUDA kernel for: {problem}"
            gen_output = self.generator._generic_kernel_template(
                problem.replace(" ", "_"),
                "float",
                (1024, 1024)
            )
            conversation["turns"].append({
                "agent": "generator",
                "input": gen_input,
                "output": gen_output
            })
            
            # Turn 2: Tester evaluates
            test_input = "Test and profile the kernel"
            test_output = "Compilation successful. Performance: 1.0x baseline"
            conversation["turns"].append({
                "agent": "tester",
                "input": test_input,
                "output": test_output
            })
            
            # Turn 3: Optimizer improves
            opt_input = f"Optimize kernel. Current performance: 1.0x"
            opt_output = self.generator._apply_optimizations(
                gen_output,
                ["shared_memory", "memory_coalescing"]
            )
            conversation["turns"].append({
                "agent": "optimizer",
                "input": opt_input,
                "output": opt_output
            })
            
            conversations.append(conversation)
        
        # Convert to dataset format
        dataset_dict = {
            "conversation_id": list(range(num_conversations)),
            "problem": [c["problem"] for c in conversations],
            "conversation": [json.dumps(c["turns"]) for c in conversations]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Save if path provided
        if save_path:
            save_path = self.data_dir / save_path
            dataset.save_to_disk(str(save_path))
            self.logger.info(f"Saved cross-agent dataset to {save_path}")
        
        return dataset