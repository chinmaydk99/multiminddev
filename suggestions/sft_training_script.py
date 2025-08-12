#!/usr/bin/env python3
"""
SFT Training Script with QLoRA for CUDA Code Generation
Trains Generator and Optimizer agents separately using SakanaAI dataset with curriculum learning.
"""

import os
import sys
import json
import asyncio
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import structlog
from dataclasses import dataclass, field

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import data pipeline components
from coding_framework.data.sakana_loader import SakanaDataLoader
from coding_framework.data.curriculum_manager import CurriculumManager
from coding_framework.data.data_pipeline import CUDADataPipeline

# Import QLoRA/SFT components
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
        PeftModel
    )
    from datasets import Dataset
    import wandb
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ Transformers/PEFT not available. Please install: pip install transformers peft bitsandbytes")
    sys.exit(1)


@dataclass
class SFTConfig:
    """Configuration for SFT training with QLoRA."""
    
    # Model configuration
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    # Training data
    max_examples_per_tier: int = 2000
    max_seq_length: int = 2048
    
    # QLoRA configuration
    qlora_r: int = 64
    qlora_alpha: int = 128
    qlora_dropout: float = 0.1
    qlora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    
    # Curriculum settings
    start_tier: str = "level_1"  # level_1, level_2, level_3 from SakanaAI dataset
    enable_progressive_curriculum: bool = True
    tier_switch_threshold: float = 0.8  # Loss threshold to advance to next tier
    
    # Paths
    output_dir: str = "./checkpoints/sft"
    cache_dir: str = "./cache"
    
    # Monitoring
    use_wandb: bool = True
    project_name: str = "CUDA-SFT-Training"
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 50
    
    # Hardware
    num_gpus: int = 8
    bf16: bool = True
    gradient_checkpointing: bool = True


class SakanaDatasetProcessor:
    """Enhanced SakanaAI dataset processor with proper level handling."""
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.logger = structlog.get_logger("sakana_processor")
        
        # Initialize data pipeline
        self.data_pipeline = CUDADataPipeline(
            dataset_name="SakanaAI/AI-CUDA-Engineer-Archive",
            cache_dir=config.cache_dir,
            curriculum_enabled=True,
            initial_tier="easy"  # Will map to level_1
        )
        
        self.level_mapping = {
            "level_1": "easy",      # Basic operations: vector_add, scalar_multiply
            "level_2": "medium",    # Intermediate: reductions, transpose, matrix ops
            "level_3": "hard"       # Advanced: convolutions, optimized kernels
        }
    
    async def load_and_process_dataset(self) -> Dict[str, Dataset]:
        """Load SakanaAI dataset and organize by curriculum levels."""
        
        self.logger.info("Loading SakanaAI CUDA dataset...")
        
        # Get raw dataset from data pipeline
        raw_examples = []
        for level in ["level_1", "level_2", "level_3"]:
            tier = self.level_mapping[level]
            examples = await self.data_pipeline.get_training_batch(
                batch_size=self.config.max_examples_per_tier,
                use_cache=False
            )
            
            for example in examples:
                # Add level metadata
                example.metadata["dataset_level"] = level
                example.metadata["curriculum_tier"] = tier
                raw_examples.append(example)
        
        # Convert to training datasets
        datasets = {}
        for agent_type in ["generator", "optimizer"]:
            for level in ["level_1", "level_2", "level_3"]:
                level_examples = [ex for ex in raw_examples 
                                if ex.metadata.get("dataset_level") == level]
                
                if agent_type == "generator":
                    dataset = self._create_generator_dataset(level_examples)
                else:
                    dataset = self._create_optimizer_dataset(level_examples)
                
                datasets[f"{agent_type}_{level}"] = dataset
        
        self.logger.info(
            "Dataset processing complete",
            total_examples=len(raw_examples),
            datasets=list(datasets.keys())
        )
        
        return datasets
    
    def _create_generator_dataset(self, examples: List) -> Dataset:
        """Create generator training dataset from examples."""
        
        training_data = []
        
        for example in examples:
            # Create prompt for generator agent
            prompt = f"""Generate a CUDA kernel for the following problem:

Problem: {example.problem_description}

Requirements:
- Write efficient CUDA C++ code
- Use appropriate grid and block dimensions
- Include proper error checking
- Optimize for memory access patterns

CUDA Kernel:"""
            
            # Target is the reference CUDA code
            target = example.reference_solution if example.reference_solution else example.cuda_code
            
            # Create conversation format
            full_text = f"{prompt}\n{target}<|endoftext|>"
            
            training_data.append({
                "text": full_text,
                "input": prompt,
                "output": target,
                "problem_id": example.problem_id,
                "level": example.metadata.get("dataset_level", "unknown")
            })
        
        return Dataset.from_list(training_data)
    
    def _create_optimizer_dataset(self, examples: List) -> Dataset:
        """Create optimizer training dataset from examples."""
        
        training_data = []
        
        for example in examples:
            # Create initial "unoptimized" version
            initial_code = self._create_unoptimized_version(example.reference_solution or example.cuda_code)
            
            prompt = f"""Optimize the following CUDA kernel for better performance:

Problem: {example.problem_description}

Current CUDA Code:
{initial_code}

Optimization Requirements:
- Improve memory access patterns
- Use shared memory effectively
- Minimize thread divergence
- Optimize occupancy

Optimized CUDA Kernel:"""
            
            # Target is the optimized version
            target = example.reference_solution if example.reference_solution else example.cuda_code
            
            full_text = f"{prompt}\n{target}<|endoftext|>"
            
            training_data.append({
                "text": full_text,
                "input": prompt,
                "output": target,
                "problem_id": example.problem_id,
                "level": example.metadata.get("dataset_level", "unknown")
            })
        
        return Dataset.from_list(training_data)
    
    def _create_unoptimized_version(self, optimized_code: str) -> str:
        """Create a less optimized version for training data."""
        
        # Simple heuristics to "unoptimize" code
        unoptimized = optimized_code
        
        # Remove shared memory usage
        unoptimized = unoptimized.replace("__shared__", "// __shared__")
        
        # Replace coalesced memory access with simple patterns
        unoptimized = unoptimized.replace("threadIdx.x + blockIdx.x * blockDim.x", "idx")
        
        # Add some inefficiencies
        unoptimized = unoptimized.replace("gridDim.x", "1000000")  # Force bounds checking
        
        return unoptimized


class CUDAQLoRATrainer:
    """QLoRA trainer for CUDA code generation agents."""
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.logger = structlog.get_logger("qlora_trainer")
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                config=config.__dict__,
                name=f"sft-qlora-{int(time.time())}"
            )
    
    def setup_model_and_tokenizer(self, model_name: str) -> tuple:
        """Setup QLoRA model and tokenizer."""
        
        self.logger.info(f"Loading model: {model_name}")
        
        # BitsAndBytes config for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="right",
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # Prepare model for training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA config
        lora_config = LoraConfig(
            r=self.config.qlora_r,
            lora_alpha=self.config.qlora_alpha,
            target_modules=self.config.qlora_target_modules,
            lora_dropout=self.config.qlora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        self.logger.info(f"Model loaded with LoRA adapters")
        model.print_trainable_parameters()
        
        return model, tokenizer
    
    def train_agent(
        self,
        agent_type: str,
        datasets: Dict[str, Dataset],
        output_path: str
    ):
        """Train a specific agent (generator or optimizer) with curriculum."""
        
        self.logger.info(f"Training {agent_type} agent")
        
        # Setup model and tokenizer
        model, tokenizer = self.setup_model_and_tokenizer(self.config.base_model)
        
        # Progressive curriculum training
        for level in ["level_1", "level_2", "level_3"]:
            if not self.config.enable_progressive_curriculum and level != self.config.start_tier:
                continue
            
            dataset_key = f"{agent_type}_{level}"
            if dataset_key not in datasets:
                self.logger.warning(f"Dataset {dataset_key} not found, skipping")
                continue
            
            self.logger.info(f"Training on {level} (tier: {dataset_key})")
            
            # Prepare dataset
            train_dataset = self._prepare_dataset(datasets[dataset_key], tokenizer)
            
            # Create output directory for this level
            level_output_dir = Path(output_path) / level
            level_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(level_output_dir),
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                warmup_ratio=self.config.warmup_ratio,
                weight_decay=self.config.weight_decay,
                
                # Optimization settings
                bf16=self.config.bf16,
                gradient_checkpointing=self.config.gradient_checkpointing,
                dataloader_pin_memory=False,
                
                # Saving and logging
                save_steps=self.config.save_steps,
                logging_steps=self.config.logging_steps,
                save_total_limit=3,
                
                # Distributed training
                ddp_find_unused_parameters=False,
                report_to="wandb" if self.config.use_wandb else None,
                
                # Remove evaluation for now to simplify
                evaluation_strategy="no",
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )
            
            # Train
            self.logger.info(f"Starting training for {agent_type} on {level}")
            trainer.train()
            
            # Save the model for this level
            trainer.save_model()
            tokenizer.save_pretrained(level_output_dir)
            
            self.logger.info(f"Completed training for {agent_type} on {level}")
            
            # If not using progressive curriculum, only train on start tier
            if not self.config.enable_progressive_curriculum:
                break
        
        # Save final merged model
        final_output_dir = Path(output_path) / "final"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        
        self.logger.info(f"Training completed for {agent_type}")
    
    def _prepare_dataset(self, dataset: Dataset, tokenizer) -> Dataset:
        """Prepare dataset for training."""
        
        def tokenize_function(examples):
            # Tokenize the full text
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.config.max_seq_length,
                return_overflowing_tokens=False,
            )
            
            # Labels are the same as input_ids for causal LM
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset


async def main():
    """Main SFT training pipeline."""
    
    parser = argparse.ArgumentParser(description="SFT Training with QLoRA for CUDA Agents")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--agent", type=str, choices=["generator", "optimizer", "both"], 
                       default="both", help="Which agent to train")
    parser.add_argument("--start-tier", type=str, choices=["level_1", "level_2", "level_3"],
                       default="level_1", help="Starting curriculum tier")
    parser.add_argument("--max-examples", type=int, default=2000, help="Max examples per tier")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/sft", help="Output directory")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--quick-test", action="store_true", help="Quick test with minimal settings")
    
    args = parser.parse_args()
    
    # Setup logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logger = structlog.get_logger("sft_main")
    
    # Create config
    config = SFTConfig(
        start_tier=args.start_tier,
        max_examples_per_tier=args.max_examples,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb
    )
    
    # Quick test mode
    if args.quick_test:
        config.max_examples_per_tier = 100
        config.num_epochs = 1
        config.save_steps = 50
        config.eval_steps = 25
        config.logging_steps = 10
        logger.info("Running in quick test mode")
    
    logger.info("Starting SFT training pipeline", config=config.__dict__)
    
    # Initialize components
    processor = SakanaDatasetProcessor(config)
    trainer = CUDAQLoRATrainer(config)
    
    # Load and process datasets
    logger.info("Loading and processing datasets...")
    datasets = await processor.load_and_process_dataset()
    
    # Train agents
    if args.agent in ["generator", "both"]:
        logger.info("Training Generator agent...")
        trainer.train_agent("generator", datasets, f"{config.output_dir}/generator")
    
    if args.agent in ["optimizer", "both"]:
        logger.info("Training Optimizer agent...")
        trainer.train_agent("optimizer", datasets, f"{config.output_dir}/optimizer")
    
    logger.info("SFT training completed successfully!")
    
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
