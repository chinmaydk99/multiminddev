#!/usr/bin/env python3
"""
True Supervised Fine-Tuning (SFT) for CUDA Code Generation
Uses actual gradient updates, loss computation, and SFTTrainer
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datasets import Dataset, load_dataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Core training imports
try:
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        TrainingArguments,
        BitsAndBytesConfig
    )
    from trl import SFTTrainer
    try:
        from trl import DataCollatorForCompletionOnlyLM
        COMPLETION_COLLATOR_AVAILABLE = True
    except ImportError:
        from transformers import DataCollatorForLanguageModeling
        COMPLETION_COLLATOR_AVAILABLE = False
    from peft import LoraConfig, get_peft_model, TaskType
    TRL_AVAILABLE = True
except ImportError as e:
    print(f"❌ TRL/PEFT not available: {e}")
    print("Install with: pip install trl peft bitsandbytes")
    TRL_AVAILABLE = False

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
)

logger = structlog.get_logger("true_sft")


@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    dataset_name: str = "SakanaAI/AI-CUDA-Engineer-Archive"
    output_dir: str = "./cuda-sft-model"
    max_seq_length: int = 2048
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    save_steps: int = 50
    logging_steps: int = 5
    eval_steps: int = 50
    warmup_steps: int = 100
    use_lora: bool = True
    use_4bit: bool = True
    max_samples: int = None  # Use full dataset
    use_wandb: bool = True
    wandb_project: str = "cuda-code-generation-sft"
    push_to_hub: bool = True
    hf_token: Optional[str] = None
    hub_model_id: str = "MultiMindDev/cuda-code-generator-v1"


def create_cuda_prompt_format(example: Dict[str, Any]) -> str:
    """Create a consistent prompt format for CUDA training."""
    
    # Extract fields with fallbacks
    problem = example.get('problem_description', example.get('description', 'CUDA kernel implementation'))
    torch_ref = example.get('torch_reference', example.get('torch_implementation', 'torch operation'))
    difficulty = example.get('difficulty', example.get('level', 'medium'))
    category = example.get('category', example.get('operation_type', 'general'))
    
    # Get the target CUDA kernel
    target_kernel = example.get('cuda_kernel', example.get('kernel_code', ''))
    
    if not target_kernel:
        return None  # Skip examples without target kernels
    
    # Create structured prompt
    prompt = f"""Generate an optimized CUDA kernel for the following operation:

Problem: {problem}
Reference: {torch_ref}
Difficulty: {difficulty}
Category: {category}

Requirements:
- Use proper CUDA thread indexing
- Include boundary checks
- Optimize for memory coalescing
- Add comments for key optimizations

Generate efficient CUDA C++ kernel code:

{target_kernel}"""
    
    return prompt


def prepare_sft_dataset(dataset_name: str, max_samples: Optional[int] = None) -> tuple[Dataset, Dataset]:
    """Prepare the dataset for SFT training."""
    logger.info("📊 Loading CUDA dataset for SFT", dataset=dataset_name, max_samples=max_samples)
    
    try:
        # Load the dataset - try different splits for SakanaAI dataset
        try:
            raw_dataset = load_dataset(dataset_name, split="train")
        except ValueError as e:
            if "Unknown split" in str(e):
                logger.info("🔄 'train' split not found, trying level-based splits...")
                # Try loading from level splits and combine
                level_datasets = []
                for level in ['level_1', 'level_2', 'level_3']:
                    try:
                        level_data = load_dataset(dataset_name, split=level)
                        level_datasets.append(level_data)
                        logger.info(f"✅ Loaded {len(level_data)} examples from {level}")
                    except:
                        continue
                
                if level_datasets:
                    # Combine all levels
                    from datasets import concatenate_datasets
                    raw_dataset = concatenate_datasets(level_datasets)
                    logger.info(f"✅ Combined dataset with {len(raw_dataset)} total examples")
                else:
                    raise e
            else:
                raise e
        
        logger.info(f"✅ Loaded raw dataset with {len(raw_dataset)} examples")
        
        # Process examples
        processed_examples = []
        skipped_count = 0
        
        for i, example in enumerate(raw_dataset):
            if max_samples and len(processed_examples) >= max_samples:
                break
                
            formatted_text = create_cuda_prompt_format(example)
            if formatted_text:
                processed_examples.append({"text": formatted_text})
            else:
                skipped_count += 1
        
        logger.info(f"📈 Processed {len(processed_examples)} examples, skipped {skipped_count}")
        
        # Split into train/validation (90/10)
        train_size = int(len(processed_examples) * 0.9)
        train_examples = processed_examples[:train_size]
        val_examples = processed_examples[train_size:]
        
        # Create HuggingFace Datasets
        train_dataset = Dataset.from_list(train_examples)
        val_dataset = Dataset.from_list(val_examples)
        
        logger.info(f"✅ Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        return train_dataset, val_dataset
        
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        
        # Fallback to synthetic dataset
        logger.info("🔄 Creating synthetic CUDA dataset...")
        return create_synthetic_cuda_dataset(max_samples or 100)


def create_synthetic_cuda_dataset(num_samples: int = 100) -> tuple[Dataset, Dataset]:
    """Create synthetic CUDA training data as fallback."""
    
    synthetic_examples = [
        {
            "text": """Generate an optimized CUDA kernel for the following operation:

Problem: Implement element-wise vector addition
Reference: torch.add(A, B)
Difficulty: easy
Category: elementwise

Requirements:
- Use proper CUDA thread indexing
- Include boundary checks
- Optimize for memory coalescing
- Add comments for key optimizations

Generate efficient CUDA C++ kernel code:

__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check to avoid out-of-bounds access
    if (tid < N) {
        C[tid] = A[tid] + B[tid];  // Element-wise addition
    }
}"""
        },
        {
            "text": """Generate an optimized CUDA kernel for the following operation:

Problem: Implement scalar multiplication
Reference: torch.mul(x, scalar)
Difficulty: easy
Category: elementwise

Requirements:
- Use proper CUDA thread indexing
- Include boundary checks
- Optimize for memory coalescing
- Add comments for key optimizations

Generate efficient CUDA C++ kernel code:

__global__ void scalarMultiply(float* input, float* output, float scalar, int N) {
    // Calculate global thread ID for coalesced access
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure we don't access beyond array bounds
    if (tid < N) {
        output[tid] = input[tid] * scalar;  // Scalar multiplication
    }
}"""
        },
        {
            "text": """Generate an optimized CUDA kernel for the following operation:

Problem: Implement matrix transpose with shared memory
Reference: torch.transpose(A, 0, 1)
Difficulty: medium
Category: linear_algebra

Requirements:
- Use proper CUDA thread indexing
- Include boundary checks
- Optimize for memory coalescing
- Add comments for key optimizations

Generate efficient CUDA C++ kernel code:

#define TILE_DIM 32

__global__ void matrixTranspose(float* input, float* output, int width, int height) {
    // Shared memory tile to optimize memory access
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts
    
    // Calculate input coordinates
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load data into shared memory with bounds checking
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    // Synchronize threads in the block
    __syncthreads();
    
    // Calculate output coordinates (transposed)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Write transposed data with bounds checking
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}"""
        }
    ]
    
    # Repeat examples to reach desired count
    dataset_examples = []
    for i in range(num_samples):
        example = synthetic_examples[i % len(synthetic_examples)]
        dataset_examples.append(example)
    
    # Split into train/val
    train_size = int(len(dataset_examples) * 0.9)
    train_data = dataset_examples[:train_size]
    val_data = dataset_examples[train_size:]
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    logger.info(f"✅ Created synthetic datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_dataset, val_dataset


def setup_model_and_tokenizer(config: SFTConfig):
    """Setup model and tokenizer with optional quantization and LoRA."""
    logger.info("🚀 Setting up model and tokenizer", model=config.model_name)
    
    # Configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right"  # Important for training
    )
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Configure model with optional quantization
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }
    
    if config.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    
    # Setup LoRA if requested
    if config.use_lora:
        logger.info("🔧 Configuring LoRA for efficient fine-tuning")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    logger.info("✅ Model and tokenizer ready for training")
    return model, tokenizer


def run_sft_training(config: SFTConfig) -> bool:
    """Run actual SFT training with gradient updates."""
    logger.info("🔥 Starting True SFT Training", config=config.__dict__)
    
    if not TRL_AVAILABLE:
        logger.error("❌ TRL not available - cannot run SFT training")
        return False
    
    try:
        # Setup Weights & Biases for monitoring
        if config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    name=f"cuda-sft-{config.model_name.split('/')[-1]}",
                    config=config.__dict__
                )
                logger.info("✅ Weights & Biases initialized")
            except ImportError:
                logger.warning("⚠️ Weights & Biases not available - install with: pip install wandb")
                config.use_wandb = False
        
        # Prepare dataset
        train_dataset, val_dataset = prepare_sft_dataset(config.dataset_name, config.max_samples)
        
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(config)
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_train_epochs,
            max_steps=-1,  # Use epochs instead
            warmup_steps=config.warmup_steps,
            logging_steps=config.logging_steps,
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            save_total_limit=3,
            eval_strategy="steps",
            prediction_loss_only=False,
            remove_unused_columns=False,
            push_to_hub=config.push_to_hub,
            hub_model_id=config.hub_model_id if config.push_to_hub else None,
            hub_token=config.hf_token,
            report_to="wandb" if config.use_wandb else None,
            run_name=f"cuda-sft-{config.model_name.split('/')[-1]}",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=False,  # Helps with multi-GPU
        )
        
        # Create data collator
        if COMPLETION_COLLATOR_AVAILABLE:
            # Use completion-only training if available
            response_template = "Generate efficient CUDA C++ kernel code:\n\n"
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template,
                tokenizer=tokenizer,
                mlm=False
            )
        else:
            # Fallback to standard language modeling collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
        
        # Initialize SFTTrainer with minimal parameters
        logger.info("🎯 Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Start training
        logger.info("🚀 Starting SFT training with gradient updates...")
        trainer.train()
        
        # Save the final model
        logger.info("💾 Saving trained model...")
        trainer.save_model()
        trainer.save_state()
        
        # Push to HuggingFace Hub if requested
        if config.push_to_hub and config.hf_token:
            logger.info("📤 Pushing model to HuggingFace Hub...")
            try:
                trainer.push_to_hub(
                    commit_message="Complete SFT training for CUDA code generation",
                    tags=["cuda", "code-generation", "sft", "multiminddev"]
                )
                logger.info(f"✅ Model successfully pushed to: https://huggingface.co/{config.hub_model_id}")
            except Exception as e:
                logger.error(f"❌ Failed to push to HuggingFace: {e}")
        
        # Save training metrics
        training_history = trainer.state.log_history
        metrics_file = os.path.join(config.output_dir, "training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Final training summary
        final_metrics = training_history[-1] if training_history else {}
        logger.info("🎉 SFT training completed successfully!", 
                   output_dir=config.output_dir,
                   metrics_file=metrics_file,
                   final_train_loss=final_metrics.get('train_loss', 'N/A'),
                   final_eval_loss=final_metrics.get('eval_loss', 'N/A'))
        
        # Cleanup wandb
        if config.use_wandb:
            try:
                import wandb
                wandb.finish()
            except:
                pass
        
        return True
        
    except Exception as e:
        logger.error("❌ SFT training failed", error=str(e))
        return False


def main():
    """Main SFT training entry point."""
    parser = argparse.ArgumentParser(description="True SFT Training for CUDA Code Generation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Base model")
    parser.add_argument("--dataset", default="SakanaAI/AI-CUDA-Engineer-Archive", help="Training dataset")
    parser.add_argument("--output-dir", default="./cuda-sft-model", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Max training samples (None = full dataset)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--hf-token", help="HuggingFace API token for pushing model")
    parser.add_argument("--hub-model-id", default="MultiMindDev/cuda-code-generator-v1", help="HuggingFace model ID")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases")
    parser.add_argument("--no-push", action="store_true", help="Don't push to HuggingFace Hub")
    
    args = parser.parse_args()
    
    # Get HF token from args or environment
    hf_token = args.hf_token or os.getenv('HUGGINGFACE_TOKEN')
    
    # Create config
    config = SFTConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        use_lora=not args.no_lora,
        use_4bit=not args.no_4bit,
        use_wandb=not args.no_wandb,
        push_to_hub=not args.no_push and hf_token is not None,
        hf_token=hf_token,
        hub_model_id=args.hub_model_id,
    )
    
    print("🚀 True SFT Training for CUDA Code Generation")
    print("=" * 60)
    print(f"🤖 Model: {config.model_name}")
    print(f"📊 Dataset: {config.dataset_name}")
    print(f"📁 Output: {config.output_dir}")
    print(f"🔢 Samples: {config.max_samples}")
    print(f"📦 Batch Size: {config.batch_size}")
    print(f"📈 Epochs: {config.num_train_epochs}")
    print(f"🎯 Learning Rate: {config.learning_rate}")
    print(f"🔧 LoRA: {'Enabled' if config.use_lora else 'Disabled'}")
    print(f"🗜️  4-bit: {'Enabled' if config.use_4bit else 'Disabled'}")
    print("=" * 60)
    
    success = run_sft_training(config)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())