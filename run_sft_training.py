#!/usr/bin/env python3
"""
Simplified SFT Training Script for CUDA Agents using QLoRA.
This script focuses only on SFT phase without any RL dependencies.
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
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import data components
from coding_framework.data.data_pipeline import CUDADataPipeline, TrainingExample
from coding_framework.data.curriculum_manager import CurriculumManager

# Import transformers and PEFT for QLoRA
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import wandb
from datasets import Dataset

logger = structlog.get_logger("sft_training")

@dataclass
class SFTConfig:
    """Configuration for SFT training only."""
    
    # Model configuration
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    
    # Training configuration
    num_examples: int = 500
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_length: int = 2048
    
    # QLoRA configuration
    qlora_r: int = 16
    qlora_alpha: int = 32
    qlora_dropout: float = 0.1
    qlora_target_modules: Optional[List[str]] = None
    
    # Paths
    output_dir: str = "./sft_checkpoints"
    
    # Data configuration
    start_difficulty: str = "easy"
    curriculum_enabled: bool = True
    
    def __post_init__(self):
        if self.qlora_target_modules is None:
            self.qlora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

class SFTTrainer:
    """Simplified SFT trainer using QLoRA."""
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.logger = structlog.get_logger("sft_trainer")
        
        # Setup wandb
        wandb.init(
            project="CUDA-SFT-Training",
            config=config.__dict__,
            name=f"sft_training_{int(time.time())}"
        )
        
        # Initialize data pipeline
        self.data_pipeline = CUDADataPipeline(
            dataset_name="SakanaAI/AI-CUDA-Engineer-Archive",
            cache_dir="./cache/datasets",
            curriculum_enabled=config.curriculum_enabled,
            initial_tier=config.start_difficulty
        )
        
        self.logger.info("SFT Trainer initialized")
    
    async def prepare_training_data(self) -> Dataset:
        """Prepare training data for SFT."""
        self.logger.info("Preparing training data...", num_examples=self.config.num_examples)
        
        # Get training batch from data pipeline
        training_batch = await self.data_pipeline.get_training_batch(
            batch_size=self.config.num_examples
        )
        
        # Convert to format expected by transformers
        texts = []
        for example in training_batch:
            # Create instruction-following format
            text = self._format_training_example(example)
            texts.append(text)
        
        dataset = Dataset.from_dict({"text": texts})
        self.logger.info("Training data prepared", num_examples=len(dataset))
        return dataset
    
    def _format_training_example(self, example: TrainingExample) -> str:
        """Format example for instruction following."""
        return f"""<|im_start|>system
You are an expert CUDA programmer. Optimize the given CUDA kernel for better performance.
<|im_end|>
<|im_start|>user

Problem: {example.problem_description}
<|im_end|>
<|im_start|>assistant
{example.reference_solution}
<|im_end|>"""
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with QLoRA."""
        self.logger.info("Loading model and tokenizer...", model=self.config.base_model)
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # BitsAndBytes config for QLoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map={"": torch.cuda.current_device()},
            trust_remote_code=False,
            torch_dtype=torch.bfloat16
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA config
        lora_config = LoraConfig(
            r=self.config.qlora_r,
            lora_alpha=self.config.qlora_alpha,
            target_modules=self.config.qlora_target_modules,
            lora_dropout=self.config.qlora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        self.logger.info("Model and tokenizer setup complete")
    
    def tokenize_function(self, examples):
        """Tokenize examples for training."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
    
    async def train_generator(self) -> str:
        """Train generator agent with QLoRA."""
        self.logger.info("🤖 Training Generator Agent with QLoRA")
        
        # Setup model and tokenizer first
        self.setup_model_and_tokenizer()
        
        # Prepare data
        dataset = await self.prepare_training_data()
        tokenized_dataset = dataset.map(
            self.tokenize_function, 
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}/generator",
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            logging_steps=10,
            save_steps=100,
            eval_strategy="no",
            save_total_limit=2,
            warmup_steps=50,
            lr_scheduler_type="cosine",
            bf16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb",
            run_name=f"generator_sft_{int(time.time())}"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train
        self.logger.info("Starting training...")
        trainer.train()
        
        # Save
        checkpoint_path = f"{self.config.output_dir}/generator"
        trainer.save_model(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        self.logger.info("✅ Generator training complete", checkpoint=checkpoint_path)
        return checkpoint_path
    
    async def train_optimizer(self) -> str:
        """Train optimizer agent with QLoRA."""
        self.logger.info("⚡ Training Optimizer Agent with QLoRA")
        
        # Fresh model for optimizer
        self.setup_model_and_tokenizer()
        
        # Similar to generator but with optimizer-focused prompts
        # For now, using same approach but saving to different path
        dataset = await self.prepare_training_data()
        tokenized_dataset = dataset.map(
            self.tokenize_function, 
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}/optimizer",
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            logging_steps=10,
            save_steps=100,
            eval_strategy="no",
            save_total_limit=2,
            warmup_steps=50,
            lr_scheduler_type="cosine",
            bf16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb",
            run_name=f"optimizer_sft_{int(time.time())}"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train
        self.logger.info("Starting optimizer training...")
        trainer.train()
        
        # Save
        checkpoint_path = f"{self.config.output_dir}/optimizer"
        trainer.save_model(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        self.logger.info("✅ Optimizer training complete", checkpoint=checkpoint_path)
        return checkpoint_path
    
    async def run_sft_training(self):
        """Run complete SFT training pipeline."""
        self.logger.info("🚀 Starting SFT Training Pipeline")
        
        # Train both agents
        generator_checkpoint = await self.train_generator()
        optimizer_checkpoint = await self.train_optimizer()
        
        self.logger.info(
            "🎉 SFT Training Complete!",
            generator_checkpoint=generator_checkpoint,
            optimizer_checkpoint=optimizer_checkpoint
        )
        
        # Save config
        config_path = f"{self.config.output_dir}/config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        return {
            "generator_checkpoint": generator_checkpoint,
            "optimizer_checkpoint": optimizer_checkpoint,
            "config": config_path
        }

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="SFT Training for CUDA Agents")
    parser.add_argument("--num-examples", type=int, default=500, help="Number of training examples")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="./sft_checkpoints", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Create config
    config = SFTConfig(
        num_examples=args.num_examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
    
    # Run training
    trainer = SFTTrainer(config)
    asyncio.run(trainer.run_sft_training())

if __name__ == "__main__":
    main()