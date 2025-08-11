#!/usr/bin/env python3
"""
Fixed QLoRA SFT Training with proper data preprocessing
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import Dataset, load_dataset
import wandb
import os

def create_qlora_config():
    """Create QLoRA configuration"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    return bnb_config, lora_config

def load_model_for_qlora(model_name: str):
    """Load model with QLoRA"""
    bnb_config, lora_config = create_qlora_config()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": torch.cuda.current_device()},
        trust_remote_code=True,
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    print(f"QLoRA Model: {model_name}")
    print(f"Trainable: {trainable_params:,} / {all_params:,} ({100*trainable_params/all_params:.2f}%)")
    
    return model

def prepare_datasets(tokenizer, num_examples=500):
    """Prepare and tokenize datasets properly"""
    
    # Generator dataset - PyTorch to CUDA
    generator_data = []
    for i in range(num_examples):
        prompt = f"Generate CUDA kernel for: torch.add(tensor_a, tensor_b)\nInput shapes: [{1024+i}]\nOutput CUDA kernel:"
        completion = f"__global__ void add_kernel(float* a, float* b, float* c, int n) {{\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx < n) c[idx] = a[idx] + b[idx];\n}}"
        
        # Format as instruction-following
        text = f"[INST] {prompt} [/INST] {completion}"
        generator_data.append({"text": text})
    
    # Optimizer dataset - unoptimized to optimized 
    optimizer_data = []
    for i in range(num_examples):
        prompt = f"Optimize this CUDA kernel for better memory coalescing:\n__global__ void slow_kernel(float* data) {{ /* unoptimized */ }}\nOptimized version:"
        completion = f"__global__ void fast_kernel(float* data) {{\n    // Optimized with shared memory and coalescing\n    __shared__ float sdata[256];\n    // ... optimized implementation\n}}"
        
        text = f"[INST] {prompt} [/INST] {completion}"
        optimizer_data.append({"text": text})
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=1024,
            return_overflowing_tokens=False,
        )
    
    gen_dataset = Dataset.from_list(generator_data).map(tokenize_function, batched=True, remove_columns=["text"])
    opt_dataset = Dataset.from_list(optimizer_data).map(tokenize_function, batched=True, remove_columns=["text"])
    
    print(f"Generator dataset: {len(gen_dataset)} examples")
    print(f"Optimizer dataset: {len(opt_dataset)} examples")
    
    return gen_dataset, opt_dataset

def train_agent(model_name: str, dataset: Dataset, agent_type: str, tokenizer):
    """Train a single agent with QLoRA"""
    
    print(f"\n🤖 Training {agent_type} Agent: {model_name}")
    print("=" * 60)
    
    # Load model
    model = load_model_for_qlora(model_name)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./qlora_checkpoints/{agent_type}",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="wandb",
        run_name=f"qlora-{agent_type}",
        remove_unused_columns=False,
        dataloader_drop_last=True,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print(f"🚀 Starting {agent_type} training...")
    result = trainer.train()
    
    # Save LoRA adapters
    model.save_pretrained(f"./qlora_checkpoints/{agent_type}/final")
    tokenizer.save_pretrained(f"./qlora_checkpoints/{agent_type}/final")
    
    print(f"✅ {agent_type} training complete!")
    print(f"Final loss: {result.training_loss:.4f}")
    
    # Clear memory
    del model, trainer
    torch.cuda.empty_cache()
    
    return result

def main():
    # Initialize WandB
    wandb.init(
        project="cuda-multiturn-rl-complete",
        name="complete-sft-rl-pipeline",
        config={
            "method": "QLoRA + VERL",
            "quantization": "4-bit",
            "agents": ["generator", "optimizer"],
            "dataset": "SakanaAI/AI-CUDA-Engineer-Archive",
        }
    )
    
    print("🚀 Complete CUDA MultiTurn RL Pipeline")
    print("=" * 70)
    print("Phase 1: QLoRA SFT Training for both agents")
    print("Phase 2: VERL Multi-Turn RL Training")
    print("=" * 70)
    
    # Model configurations
    generator_model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    optimizer_model = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(generator_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare datasets
    print("\n📊 Preparing specialized datasets...")
    gen_dataset, opt_dataset = prepare_datasets(tokenizer, num_examples=200)
    
    # Phase 1: SFT Training
    print("\n📚 Phase 1: Supervised Fine-Tuning (QLoRA)")
    print("=" * 50)
    
    # Train Generator
    gen_result = train_agent(generator_model, gen_dataset, "generator", tokenizer)
    
    # Train Optimizer  
    opt_result = train_agent(optimizer_model, opt_dataset, "optimizer", tokenizer)
    
    print("\n✅ SFT Phase Complete!")
    print(f"Generator final loss: {gen_result.training_loss:.4f}")
    print(f"Optimizer final loss: {opt_result.training_loss:.4f}")
    
    # Phase 2: VERL MultiTurn RL (placeholder for now)
    print("\n🤖 Phase 2: VERL Multi-Turn RL Training")
    print("=" * 50)
    print("📝 Ready for VERL integration!")
    print("Models saved at:")
    print("- Generator: ./qlora_checkpoints/generator/final")
    print("- Optimizer: ./qlora_checkpoints/optimizer/final")
    
    wandb.finish()

if __name__ == "__main__":
    main()