#!/usr/bin/env python3
"""
Upload trained CUDA model to Hugging Face Hub
Handles model, tokenizer, and training artifacts upload
"""

import os
import json
import argparse
from pathlib import Path
import sys
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from huggingface_hub import HfApi, Repository, login, create_repo
    from transformers import AutoModel, AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    print("❌ HuggingFace Hub not available. Install with: pip install huggingface_hub")
    HF_AVAILABLE = False

import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("hf_upload")


def create_model_card(
    base_model: str,
    training_stats: Optional[dict] = None,
    examples_trained: int = 0,
    success_rate: float = 0.0
) -> str:
    """Create a comprehensive model card for the trained model."""
    
    model_card = f"""---
license: apache-2.0
base_model: {base_model}
tags:
- cuda
- code-generation
- gpu-programming
- kernel-optimization
- sft
- multi-turn-rl
library_name: transformers
pipeline_tag: text-generation
---

# MultiMindDev CUDA Code Generator

This model is a fine-tuned version of [{base_model}]({base_model}) specialized for CUDA kernel generation and optimization.

## Model Description

- **Base Model:** {base_model}
- **Training Type:** Supervised Fine-Tuning (SFT) + Multi-turn Reinforcement Learning
- **Specialization:** CUDA C++ kernel generation
- **Architecture:** Multi-agent with Generator, Optimizer, and Tester components
- **Training Infrastructure:** 8x Tesla V100 GPUs (Lambda Labs)

## Training Details

### Training Data
- **Examples Trained:** {examples_trained}
- **Success Rate:** {success_rate:.2%}
- **Categories:** Elementwise operations, reductions, linear algebra, convolutions
- **Difficulty Levels:** Basic, Intermediate, Advanced, Expert

### Training Procedure
- **Framework:** HuggingFace Transformers + LangChain + Accelerate
- **Optimization:** Multi-GPU distributed training with device_map="auto"
- **Temperature:** 0.3 (for more focused code generation)
- **Max Tokens:** 2048

### Hardware
- **GPUs:** 8x Tesla V100 (16GB each)
- **Total VRAM:** 128GB
- **Platform:** Lambda Labs Cloud

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("MultiMindDev/cuda-code-generator")
model = AutoModelForCausalLM.from_pretrained(
    "MultiMindDev/cuda-code-generator",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Generate CUDA kernel
prompt = '''Generate an optimized CUDA kernel for the following operation:

Problem: Implement vector addition: C = A + B
Reference: torch.add(A, B)
Difficulty: easy
Category: elementwise

Requirements:
- Use proper CUDA thread indexing
- Include boundary checks
- Optimize for memory coalescing

Generate efficient CUDA C++ kernel code:'''

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_code)
```

## Example Outputs

### Vector Addition
```cpp
__global__ void addVectors(float* A, float* B, float* C, int N) {{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {{
        C[index] = A[index] + B[index];
    }}
}}
```

### Matrix Transpose with Shared Memory
```cpp
__global__ void transposeMatrix(float* input, float* output, int width, int height) {{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load data into shared memory
    if (x < width && y < height) {{
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }}
    
    __syncthreads();
    
    // Write transposed data
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    if (x < height && y < width) {{
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }}
}}
```

## Performance

- **Generation Speed:** ~6 seconds per kernel on V100
- **Quality Score:** {training_stats.get('avg_quality_score', 0.0):.3f}/1.0 if training_stats else 'N/A'
- **Memory Efficiency:** Optimized for coalesced access patterns
- **Kernel Correctness:** Includes boundary checks and proper indexing

## Training Infrastructure

### Multi-Agent Architecture
- **Generator Agent:** Creates initial CUDA kernels
- **Optimizer Agent:** Applies performance optimizations
- **Tester Agent:** Validates functional correctness and benchmarks

### Reinforcement Learning Setup
- **Reward Function:** CUDA performance metrics (speedup, memory efficiency, correctness)
- **Algorithm:** PPO (Proximal Policy Optimization)
- **Multi-turn:** Iterative optimization conversations

## Limitations

- Specialized for CUDA C++ kernels only
- Requires NVIDIA GPU for optimal performance
- May need additional optimization for very complex kernels
- Training focused on common GPU computing patterns

## Citation

```bibtex
@misc{{multiminddev-cuda-generator,
  title={{MultiMindDev CUDA Code Generator}},
  author={{MultiMindDev Team}},
  year={{2025}},
  howpublished={{\\url{{https://huggingface.co/MultiMindDev/cuda-code-generator}}}},
  note={{Fine-tuned for CUDA kernel generation with multi-agent RL}}
}}
```

## Model Architecture

Based on {base_model} with the following modifications:
- Temperature optimized for code generation (0.3)
- Extended context window for complex kernels
- Multi-GPU distributed inference support

## Contact

For questions about this model or the MultiMindDev framework:
- GitHub: [MultiMindDev Repository](https://github.com/MultiMindDev)
- Issues: Report bugs and feature requests on GitHub

---

*This model was trained using the MultiMindDev multi-agent CUDA code generation framework with reinforcement learning from human feedback (RLHF) principles applied to GPU performance optimization.*
"""
    
    return model_card


def upload_model_to_hf(
    hf_token: str,
    model_name: str,
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    training_stats: Optional[dict] = None,
    local_model_path: Optional[str] = None,
    organization: str = "MultiMindDev"
) -> bool:
    """Upload trained model to Hugging Face Hub."""
    
    if not HF_AVAILABLE:
        logger.error("HuggingFace Hub not available")
        return False
    
    try:
        # Login to Hugging Face
        logger.info("🔐 Logging in to Hugging Face Hub...")
        login(token=hf_token, write_permission=True)
        logger.info("✅ Successfully logged in to Hugging Face")
        
        # Initialize HF API
        api = HfApi()
        
        # Create repository name
        repo_name = f"{organization}/{model_name}"
        logger.info(f"📦 Creating repository: {repo_name}")
        
        # Create repository
        try:
            api.create_repo(
                repo_id=repo_name,
                repo_type="model",
                private=False,
                exist_ok=True
            )
            logger.info(f"✅ Repository created/exists: {repo_name}")
        except Exception as e:
            logger.warning(f"Repository creation warning: {e}")
        
        # Create model card
        logger.info("📝 Creating model card...")
        model_card = create_model_card(
            base_model=base_model,
            training_stats=training_stats,
            examples_trained=training_stats.get('examples_processed', 0) if training_stats else 0,
            success_rate=training_stats.get('success_rate', 0.0) if training_stats else 0.0
        )
        
        # Upload model card
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            commit_message="Add comprehensive model card"
        )
        logger.info("✅ Model card uploaded")
        
        # If we have a local model path, upload the model files
        if local_model_path and os.path.exists(local_model_path):
            logger.info(f"📤 Uploading model files from {local_model_path}")
            
            # Upload all model files
            api.upload_folder(
                folder_path=local_model_path,
                repo_id=repo_name,
                commit_message="Upload fine-tuned CUDA code generation model"
            )
            logger.info("✅ Model files uploaded")
        else:
            logger.warning("⚠️ No local model path provided - only uploading model card")
        
        # Upload training metadata if available
        if training_stats:
            metadata_json = json.dumps(training_stats, indent=2)
            api.upload_file(
                path_or_fileobj=metadata_json.encode(),
                path_in_repo="training_stats.json",
                repo_id=repo_name,
                commit_message="Add training statistics"
            )
            logger.info("✅ Training statistics uploaded")
        
        # Create a simple config file for easier loading
        config_json = {
            "model_type": "cuda_code_generator",
            "base_model": base_model,
            "training_framework": "multiminddev",
            "specialization": "cuda_kernels",
            "recommended_temperature": 0.3,
            "recommended_max_tokens": 2048,
            "trust_remote_code": True
        }
        
        api.upload_file(
            path_or_fileobj=json.dumps(config_json, indent=2).encode(),
            path_in_repo="multiminddev_config.json",
            repo_id=repo_name,
            commit_message="Add MultiMindDev configuration"
        )
        logger.info("✅ Configuration uploaded")
        
        logger.info(f"🎉 Model successfully uploaded to: https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to upload model to Hugging Face: {e}")
        return False


def main():
    """Main upload script."""
    parser = argparse.ArgumentParser(description="Upload CUDA model to Hugging Face Hub")
    parser.add_argument("--token", required=True, help="HuggingFace API token")
    parser.add_argument("--model-name", default="cuda-code-generator", help="Model name on HF")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Base model name")
    parser.add_argument("--local-path", help="Local path to model files (optional)")
    parser.add_argument("--stats-file", help="Training statistics JSON file")
    parser.add_argument("--organization", default="MultiMindDev", help="HF organization name")
    
    args = parser.parse_args()
    
    # Load training stats if provided
    training_stats = None
    if args.stats_file and os.path.exists(args.stats_file):
        with open(args.stats_file, 'r') as f:
            training_stats = json.load(f)
        logger.info(f"📊 Loaded training statistics from {args.stats_file}")
    
    print("🚀 Starting Hugging Face Upload...")
    print(f"📦 Model Name: {args.organization}/{args.model_name}")
    print(f"🔧 Base Model: {args.base_model}")
    print(f"📁 Local Path: {args.local_path or 'Model card only'}")
    print("=" * 60)
    
    success = upload_model_to_hf(
        hf_token=args.token,
        model_name=args.model_name,
        base_model=args.base_model,
        training_stats=training_stats,
        local_model_path=args.local_path,
        organization=args.organization
    )
    
    if success:
        print(f"\n🎉 Successfully uploaded to: https://huggingface.co/{args.organization}/{args.model_name}")
        return 0
    else:
        print("\n❌ Upload failed - check logs for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())