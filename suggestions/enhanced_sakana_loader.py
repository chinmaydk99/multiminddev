"""
Enhanced SakanaAI CUDA Dataset Loader with proper level 1/2/3 support.
Handles the actual dataset structure from SakanaAI/AI-CUDA-Engineer-Archive.
"""

from datasets import load_dataset
from typing import Dict, List, Any, Optional, Tuple
import json
import random
from pathlib import Path
import structlog
import numpy as np
from dataclasses import dataclass


@dataclass
class CUDAExample:
    """Structured CUDA training example from SakanaAI dataset."""
    problem_id: str
    operation_name: str  # Op_Name from dataset
    level_id: int        # Level_ID from dataset (1, 2, 3, etc.)
    task_id: int         # Task_ID from dataset
    kernel_name: str     # Kernel_Name from dataset
    
    # Code components
    pytorch_code_module: str      # PyTorch_Code_Module
    pytorch_code_functional: str  # PyTorch_Code_Functional  
    cuda_code: str               # CUDA_Code
    
    # Performance data
    cuda_runtime: float = 0.0              # CUDA_Runtime
    pytorch_native_runtime: float = 0.0    # PyTorch_Native_Runtime
    pytorch_compile_runtime: float = 0.0   # PyTorch_Compile_Runtime
    cuda_speedup_native: float = 0.0       # CUDA_Speedup_Native
    cuda_speedup_compile: float = 0.0      # CUDA_Speedup_Compile
    
    # Correctness and debugging
    is_correct: bool = False               # Correct
    max_diff: float = 0.0                 # Max_Diff
    error_message: str = ""               # Error
    
    # Profiling data (optional)
    ncu_profile: Optional[Dict] = None     # NCU_Profile
    torch_profile: Optional[Dict] = None   # Torch_Profile
    clang_tidy: Optional[Dict] = None     # Clang_Tidy
    
    # Derived properties for curriculum
    @property
    def level_name(self) -> str:
        """Convert level_id to level name for curriculum."""
        return f"level_{self.level_id}"
    
    @property 
    def difficulty_tier(self) -> str:
        """Map level to curriculum difficulty."""
        if self.level_id == 1:
            return "easy"
        elif self.level_id == 2:
            return "medium"
        else:
            return "hard"


class SakanaDataLoader:
    """Loader for SakanaAI CUDA dataset with proper multi-level support."""
    
    def __init__(
        self,
        dataset_name: str = "SakanaAI/AI-CUDA-Engineer-Archive",
        cache_dir: Optional[str] = None,
        use_synthetic_fallback: bool = True
    ):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir or "./cache/datasets"
        self.use_synthetic_fallback = use_synthetic_fallback
        self.logger = structlog.get_logger("sakana_data_loader")
        
        # Create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Dataset state
        self.dataset = None
        self.examples_by_level = {"level_1": [], "level_2": [], "level_3": []}
        self.operation_types = set()
        self.total_examples = 0
        
        # Load and process dataset
        self._load_dataset()
        self._process_examples()
    
    def _load_dataset(self):
        """Load the SakanaAI dataset."""
        
        try:
            self.logger.info(f"Loading dataset: {self.dataset_name}")
            
            self.dataset = load_dataset(
                self.dataset_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # SakanaAI dataset has splits like level_1, level_2, level_3
            available_splits = list(self.dataset.keys())
            self.logger.info(f"Available dataset splits: {available_splits}")
            
            # Count total examples
            total_count = 0
            for split in available_splits:
                if split.startswith("level_"):
                    count = len(self.dataset[split])
                    total_count += count
                    self.logger.info(f"{split}: {count} examples")
            
            self.total_examples = total_count
            self.logger.info(f"Total examples across all levels: {total_count}")
                
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            
            if self.use_synthetic_fallback:
                self.logger.warning("Falling back to synthetic data generation")
                self._create_synthetic_dataset()
            else:
                raise
    
    def _process_examples(self):
        """Process dataset examples and organize by level."""
        
        if not self.dataset:
            return
        
        processed_count = 0
        
        # Process each level split
        for split_name in self.dataset.keys():
            if not split_name.startswith("level_"):
                continue
                
            level_data = self.dataset[split_name]
            level_examples = []
            
            self.logger.info(f"Processing {split_name} with {len(level_data)} examples")
            
            for idx, raw_example in enumerate(level_data):
                try:
                    cuda_example = self._parse_example(raw_example, idx, split_name)
                    if cuda_example:
                        level_examples.append(cuda_example)
                        self.operation_types.add(cuda_example.operation_name)
                        processed_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process {split_name}[{idx}]: {e}")
                    continue
            
            # Store processed examples
            self.examples_by_level[split_name] = level_examples
            self.logger.info(f"Processed {len(level_examples)} examples for {split_name}")
        
        self.logger.info(
            f"Dataset processing complete: {processed_count} examples, "
            f"{len(self.operation_types)} operation types"
        )
    
    def _parse_example(self, raw_example: Dict[str, Any], idx: int, split_name: str) -> Optional[CUDAExample]:
        """Parse a raw dataset example into structured format."""
        
        try:
            # Extract required fields with fallbacks
            operation_name = raw_example.get("Op_Name", "unknown_operation")
            level_id = raw_example.get("Level_ID", int(split_name.split("_")[1]) if "_" in split_name else 1)
            task_id = raw_example.get("Task_ID", idx)
            kernel_name = raw_example.get("Kernel_Name", f"kernel_{idx}")
            
            # Extract code components
            pytorch_module = raw_example.get("PyTorch_Code_Module", "")
            pytorch_functional = raw_example.get("PyTorch_Code_Functional", "")
            cuda_code = raw_example.get("CUDA_Code", "")
            
            # Skip examples without CUDA code
            if not cuda_code or cuda_code.strip() == "":
                return None
            
            # Extract performance metrics with safe defaults
            cuda_runtime = self._safe_float(raw_example.get("CUDA_Runtime"))
            pytorch_native_runtime = self._safe_float(raw_example.get("PyTorch_Native_Runtime"))
            pytorch_compile_runtime = self._safe_float(raw_example.get("PyTorch_Compile_Runtime"))
            cuda_speedup_native = self._safe_float(raw_example.get("CUDA_Speedup_Native"))
            cuda_speedup_compile = self._safe_float(raw_example.get("CUDA_Speedup_Compile"))
            
            # Extract correctness info
            is_correct = raw_example.get("Correct", False)
            max_diff = self._safe_float(raw_example.get("Max_Diff"))
            error_message = raw_example.get("Error", "") or ""
            
            # Parse profiling data (JSON strings)
            ncu_profile = self._safe_json_parse(raw_example.get("NCU_Profile"))
            torch_profile = self._safe_json_parse(raw_example.get("Torch_Profile"))
            clang_tidy = self._safe_json_parse(raw_example.get("Clang_Tidy"))
            
            return CUDAExample(
                problem_id=f"{split_name}_{task_id}_{idx}",
                operation_name=operation_name,
                level_id=level_id,
                task_id=task_id,
                kernel_name=kernel_name,
                pytorch_code_module=pytorch_module,
                pytorch_code_functional=pytorch_functional,
                cuda_code=cuda_code,
                cuda_runtime=cuda_runtime,
                pytorch_native_runtime=pytorch_native_runtime,
                pytorch_compile_runtime=pytorch_compile_runtime,
                cuda_speedup_native=cuda_speedup_native,
                cuda_speedup_compile=cuda_speedup_compile,
                is_correct=is_correct,
                max_diff=max_diff,
                error_message=error_message,
                ncu_profile=ncu_profile,
                torch_profile=torch_profile,
                clang_tidy=clang_tidy
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing example {idx}: {e}")
            return None
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float."""
        if value is None or value == "null":
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _safe_json_parse(self, value: Any) -> Optional[Dict]:
        """Safely parse JSON string."""
        if not value or value == "null":
            return None
        try:
            if isinstance(value, str):
                return json.loads(value)
            elif isinstance(value, dict):
                return value
            else:
                return None
        except (json.JSONDecodeError, TypeError):
            return None
    
    def get_examples_by_level(self, level: str, max_examples: Optional[int] = None) -> List[CUDAExample]:
        """Get examples for a specific level."""
        examples = self.examples_by_level.get(level, [])
        
        if max_examples and len(examples) > max_examples:
            # Use deterministic sampling
            random.seed(42)
            examples = random.sample(examples, max_examples)
        
        return examples
    
    def get_examples_by_operation(self, operation_name: str) -> List[CUDAExample]:
        """Get all examples for a specific operation across levels."""
        all_examples = []
        for level_examples in self.examples_by_level.values():
            for example in level_examples:
                if example.operation_name == operation_name:
                    all_examples.append(example)
        return all_examples
    
    def get_correct_examples_only(self, level: Optional[str] = None) -> List[CUDAExample]:
        """Get only functionally correct examples."""
        if level:
            examples = self.examples_by_level.get(level, [])
        else:
            examples = []
            for level_examples in self.examples_by_level.values():
                examples.extend(level_examples)
        
        return [ex for ex in examples if ex.is_correct]
    
    def get_level_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each level."""
        stats = {}
        
        for level, examples in self.examples_by_level.items():
            if not examples:
                continue
                
            correct_examples = [ex for ex in examples if ex.is_correct]
            speedups = [ex.cuda_speedup_native for ex in correct_examples if ex.cuda_speedup_native > 0]
            
            stats[level] = {
                "total_examples": len(examples),
                "correct_examples": len(correct_examples),
                "correctness_rate": len(correct_examples) / len(examples) if examples else 0,
                "avg_speedup": np.mean(speedups) if speedups else 0,
                "max_speedup": max(speedups) if speedups else 0,
                "operations": list(set(ex.operation_name for ex in examples))
            }
        
        return stats
    
    def _create_synthetic_dataset(self):
        """Create synthetic dataset when real dataset unavailable."""
        
        self.logger.warning("Creating synthetic CUDA dataset")
        
        # Synthetic operations by level
        synthetic_ops = {
            "level_1": ["vector_add", "scalar_multiply", "element_wise_ops"],
            "level_2": ["matrix_vector_multiply", "reduction_sum", "transpose"],
            "level_3": ["matrix_multiply", "convolution_2d", "fused_operations"]
        }
        
        for level, operations in synthetic_ops.items():
            level_examples = []
            level_id = int(level.split("_")[1])
            
            for op_idx, operation in enumerate(operations):
                for variant in range(5):  # 5 variants per operation
                    example = self._create_synthetic_example(
                        operation, level_id, op_idx * 5 + variant
                    )
                    level_examples.append(example)
            
            self.examples_by_level[level] = level_examples
            self.logger.info(f"Generated {len(level_examples)} synthetic examples for {level}")
    
    def _create_synthetic_example(self, operation: str, level_id: int, task_id: int) -> CUDAExample:
        """Create a synthetic CUDA example."""
        
        # Simple synthetic CUDA kernel template
        cuda_template = f"""
__global__ void {operation}_kernel(float* input, float* output, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        // Synthetic {operation} operation
        output[idx] = input[idx] * 2.0f;
    }}
}}
"""
        
        pytorch_template = f"""
def {operation}(input_tensor):
    return input_tensor * 2.0
"""
        
        return CUDAExample(
            problem_id=f"synthetic_level_{level_id}_{task_id}",
            operation_name=operation,
            level_id=level_id,
            task_id=task_id,
            kernel_name=f"{operation}_synthetic",
            pytorch_code_module=pytorch_template,
            pytorch_code_functional=pytorch_template,
            cuda_code=cuda_template,
            cuda_runtime=1.0 + random.random(),
            pytorch_native_runtime=2.0 + random.random(),
            cuda_speedup_native=1.5 + random.random(),
            is_correct=True,
            max_diff=0.001,
            error_message=""
        )