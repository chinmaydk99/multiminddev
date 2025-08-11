#!/usr/bin/env python3
"""
Complete CUDA RL Training Pipeline with QLoRA SFT and VERL-based RL.
This script orchestrates the full training process:
1. SFT with QLoRA for Generator and Optimizer agents
2. VERL-based multi-turn RL training using SFT checkpoints
"""

import os
import sys
import json
import asyncio
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import structlog
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import all our implemented components
from coding_framework.training.verl_integration import MultiAgentVERLTrainer
from coding_framework.training.multi_turn_conversation import MultiTurnConversationManager
from coding_framework.training.reward_functions.cuda_performance_reward import CUDAPerformanceReward
from coding_framework.data.data_pipeline import CUDADataPipeline, TrainingExample
from coding_framework.data.curriculum_manager import CurriculumManager
from coding_framework.cuda.compiler import CUDACompiler
from coding_framework.cuda.benchmarker import CUDABenchmarker
from coding_framework.agents.trainable_cuda_agents import (
    TrainableCUDAGeneratorAgent,
    TrainableCUDAOptimizerAgent,
    TrainableCUDATesterAgent
)

# Import libraries for SFT with QLoRA
try:
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
        TaskType,
        PeftModel
    )
    from datasets import Dataset, load_dataset
    import wandb
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers/PEFT not available - SFT phase will be skipped")


@dataclass
class CompleteTrainingConfig:
    """Configuration for complete training pipeline."""
    
    # Model configuration
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    # SFT Phase Configuration (QLoRA)
    sft_enabled: bool = True
    sft_num_examples: int = 1000
    sft_epochs: int = 2
    sft_batch_size: int = 4
    sft_gradient_accumulation_steps: int = 8
    sft_learning_rate: float = 2e-4
    sft_warmup_steps: int = 100
    sft_checkpoint_dir: str = "./checkpoints/sft"
    
    # QLoRA Configuration
    qlora_r: int = 64
    qlora_alpha: int = 128
    qlora_dropout: float = 0.1
    qlora_target_modules: list = None  # Will use default if None
    
    # RL Phase Configuration (VERL)
    rl_enabled: bool = True
    rl_algorithm: str = "grpo"  # grpo, dapo, ppo
    rl_num_episodes: int = 100
    rl_max_turns: int = 5
    rl_batch_size: int = 256
    rl_mini_batch_size: int = 32
    rl_learning_rate: float = 5e-6
    rl_kl_coef: float = 0.02
    rl_checkpoint_dir: str = "./checkpoints/rl"
    
    # VERL Distributed Configuration
    num_gpus: int = 8
    num_rollout_workers: int = 4
    num_actor_workers: int = 2
    num_critic_workers: int = 2
    
    # Curriculum Configuration
    start_difficulty: str = "easy"
    curriculum_enabled: bool = True
    
    # Reward Configuration
    target_speedup: float = 2.0
    early_termination_threshold: float = 2.0
    
    # Monitoring
    use_wandb: bool = True
    project_name: str = "CUDA-MultiTurn-RL"
    experiment_name: str = "complete_training"
    
    # Hardware
    cuda_visible_devices: str = "0,1,2,3,4,5,6,7"
    ray_object_store_memory: int = 50000000000  # 50GB
    
    # Safety & Compilation
    use_docker_sandbox: bool = True
    max_compilation_time: int = 60
    
    def __post_init__(self):
        if self.qlora_target_modules is None:
            self.qlora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                                         "gate_proj", "up_proj", "down_proj"]


class CUDATrainingPipeline:
    """Complete training pipeline orchestrator."""
    
    def __init__(self, config: CompleteTrainingConfig):
        self.config = config
        self.logger = structlog.get_logger("cuda_training_pipeline")
        
        # Set environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
        os.environ["RAY_OBJECT_STORE_MEMORY"] = str(config.ray_object_store_memory)
        os.environ["PYTHONPATH"] = f"{os.getcwd()}/src:{os.environ.get('PYTHONPATH', '')}"
        
        # Initialize data pipeline with integrated curriculum
        self.data_pipeline = CUDADataPipeline(
            dataset_name="SakanaAI/AI-CUDA-Engineer-Archive",
            cache_dir="./cache/datasets",
            curriculum_enabled=config.curriculum_enabled,
            initial_tier=config.start_difficulty
        )
        self.compiler = CUDACompiler(use_docker=config.use_docker_sandbox)
        self.benchmarker = CUDABenchmarker()
        self.reward_function = CUDAPerformanceReward(target_speedup=config.target_speedup)
        
        # Paths
        self.sft_generator_checkpoint = Path(config.sft_checkpoint_dir) / "generator"
        self.sft_optimizer_checkpoint = Path(config.sft_checkpoint_dir) / "optimizer"
        self.rl_checkpoint = Path(config.rl_checkpoint_dir)
        
        # Create directories
        self.sft_generator_checkpoint.mkdir(parents=True, exist_ok=True)
        self.sft_optimizer_checkpoint.mkdir(parents=True, exist_ok=True)
        self.rl_checkpoint.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                name=config.experiment_name,
                config=config.__dict__
            )
    
    async def run_complete_pipeline(self):
        """Run the complete training pipeline."""
        
        self.logger.info("🚀 Starting Complete CUDA RL Training Pipeline")
        self.logger.info(f"System: {torch.cuda.device_count()} GPUs available")
        
        # Phase 1: SFT with QLoRA
        if self.config.sft_enabled and TRANSFORMERS_AVAILABLE:
            self.logger.info("📚 Phase 1: SFT Warmstart Training with QLoRA")
            await self.run_sft_phase()
        else:
            self.logger.warning("⚠️ Skipping SFT phase - using base models directly")
        
        # Phase 2: VERL Multi-Turn RL Training
        if self.config.rl_enabled:
            self.logger.info("🤖 Phase 2: VERL Multi-Turn RL Training")
            await self.run_rl_phase()
        
        # Phase 3: Evaluation
        self.logger.info("📊 Phase 3: Final Evaluation")
        await self.run_evaluation()
        
        self.logger.info("✅ Training pipeline completed successfully!")
    
    async def run_sft_phase(self):
        """Run SFT training with QLoRA for both generator and optimizer agents."""
        
        # Prepare training data
        self.logger.info("Preparing SFT training data...")
        train_data = await self.prepare_sft_data()
        
        # Train Generator Agent with QLoRA
        self.logger.info("Training Generator Agent with QLoRA...")
        await self.train_agent_with_qlora(
            agent_type="generator",
            train_data=train_data,
            checkpoint_path=self.sft_generator_checkpoint
        )
        
        # Train Optimizer Agent with QLoRA
        self.logger.info("Training Optimizer Agent with QLoRA...")
        await self.train_agent_with_qlora(
            agent_type="optimizer",
            train_data=train_data,
            checkpoint_path=self.sft_optimizer_checkpoint
        )
        
        self.logger.info("✅ SFT phase completed successfully")
    
    async def prepare_sft_data(self) -> Dataset:
        """Prepare SFT training data from CUDA dataset."""
        
        examples = []
        
        # Get training batch from data pipeline
        training_batch = await self.data_pipeline.get_training_batch(
            batch_size=self.config.sft_num_examples,
            use_cache=False  # Don't cache for SFT
        )
        
        # Format for SFT training
        for item in training_batch:
            # Use the new TrainingExample methods
            generator_prompt = item.to_generator_prompt()
            
            # For optimizer, use reference solution if available
            initial_kernel = item.reference_solution or """
__global__ void kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}"""
            optimizer_prompt = item.to_optimizer_prompt(initial_kernel)
            
            examples.append({
                "generator_prompt": generator_prompt,
                "optimizer_prompt": optimizer_prompt,
                "response": item.reference_solution or initial_kernel
            })
        
        return Dataset.from_list(examples)
    
    async def train_agent_with_qlora(
        self,
        agent_type: str,
        train_data: Dataset,
        checkpoint_path: Path
    ):
        """Train a single agent with QLoRA."""
        
        # QLoRA configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.qlora_r,
            lora_alpha=self.config.qlora_alpha,
            target_modules=self.config.qlora_target_modules,
            lora_dropout=self.config.qlora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Get PEFT model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare data
        def preprocess_function(examples):
            prompt_key = f"{agent_type}_prompt"
            inputs = tokenizer(
                examples[prompt_key],
                truncation=True,
                padding="max_length",
                max_length=1024
            )
            labels = tokenizer(
                examples["response"],
                truncation=True,
                padding="max_length",
                max_length=1024
            )
            inputs["labels"] = labels["input_ids"]
            return inputs
        
        tokenized_data = train_data.map(preprocess_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(checkpoint_path),
            num_train_epochs=self.config.sft_epochs,
            per_device_train_batch_size=self.config.sft_batch_size,
            gradient_accumulation_steps=self.config.sft_gradient_accumulation_steps,
            warmup_steps=self.config.sft_warmup_steps,
            learning_rate=self.config.sft_learning_rate,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",
            save_total_limit=2,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
            group_by_length=True,
            report_to="wandb" if self.config.use_wandb else None,
            run_name=f"sft_{agent_type}"
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
        
        # Train
        self.logger.info(f"Starting QLoRA training for {agent_type} agent...")
        trainer.train()
        
        # Save final model
        trainer.save_model(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        self.logger.info(f"✅ {agent_type.capitalize()} agent SFT training completed")
    
    async def run_rl_phase(self):
        """Run VERL-based multi-turn RL training."""
        
        # Initialize agents with SFT checkpoints
        generator_agent = TrainableCUDAGeneratorAgent(
            model_name=str(self.sft_generator_checkpoint) if self.config.sft_enabled else self.config.base_model,
            use_lora=False  # Already has LoRA weights from SFT
        )
        
        optimizer_agent = TrainableCUDAOptimizerAgent(
            model_name=str(self.sft_optimizer_checkpoint) if self.config.sft_enabled else self.config.base_model,
            use_lora=False  # Already has LoRA weights from SFT
        )
        
        tester_agent = TrainableCUDATesterAgent(
            use_model=False  # Rule-based tester
        )
        
        # Initialize conversation manager
        conversation_manager = MultiTurnConversationManager(
            generator_agent=generator_agent,
            optimizer_agent=optimizer_agent,
            tester_agent=tester_agent,
            compiler=self.compiler,
            benchmarker=self.benchmarker,
            max_turns=self.config.rl_max_turns,
            early_termination_threshold=self.config.early_termination_threshold
        )
        
        # Create VERL training configuration
        verl_config = {
            "algorithm": self.config.rl_algorithm,
            "num_episodes": self.config.rl_num_episodes,
            "batch_size": self.config.rl_batch_size,
            "mini_batch_size": self.config.rl_mini_batch_size,
            "learning_rate": self.config.rl_learning_rate,
            "kl_coef": self.config.rl_kl_coef,
            "num_gpus": self.config.num_gpus,
            "num_rollout_workers": self.config.num_rollout_workers,
            "num_actor_workers": self.config.num_actor_workers,
            "num_critic_workers": self.config.num_critic_workers,
            "checkpoint_dir": str(self.rl_checkpoint),
            "use_wandb": self.config.use_wandb
        }
        
        # Initialize VERL trainer with data pipeline
        verl_trainer = MultiAgentVERLTrainer(
            config=verl_config,
            conversation_manager=conversation_manager,
            reward_function=self.reward_function,
            curriculum_manager=self.data_pipeline.curriculum_manager,
            data_loader=self.data_pipeline
        )
        
        # Run training
        self.logger.info("Starting VERL multi-turn RL training...")
        await verl_trainer.train()
        
        self.logger.info("✅ RL training phase completed")
    
    async def run_evaluation(self):
        """Run comprehensive evaluation of trained models."""
        
        self.logger.info("Running final evaluation...")
        
        # Load trained models
        generator_agent = TrainableCUDAGeneratorAgent(
            model_name=str(self.rl_checkpoint / "generator_final"),
            use_lora=False
        )
        
        optimizer_agent = TrainableCUDAOptimizerAgent(
            model_name=str(self.rl_checkpoint / "optimizer_final"),
            use_lora=False
        )
        
        # Prepare test problems using data pipeline
        test_problems = await self.data_pipeline.prepare_evaluation_set(
            num_problems=100,
            difficulty_distribution={"easy": 0.2, "medium": 0.4, "hard": 0.4}
        )
        
        results = {
            "total_problems": len(test_problems),
            "compilation_success": 0,
            "tests_passed": 0,
            "avg_speedup": 0.0,
            "avg_turns": 0.0,
            "speedups": []
        }
        
        # Evaluate each problem
        for problem in test_problems:
            # Generate initial kernel using the problem's prompt
            generator_prompt = problem.to_generator_prompt()
            generator_response = await generator_agent.generate_response(
                generator_prompt,
                max_tokens=1024
            )
            
            # Compile and test
            compilation_result = await self.compiler.compile_kernel(generator_response["text"])
            
            if compilation_result.success:
                results["compilation_success"] += 1
                
                # Benchmark using test cases
                test_inputs = []
                for test_case in problem.test_cases[:1]:  # Use first test case
                    test_inputs.extend(test_case.input_tensors or [])
                
                benchmark_result = await self.benchmarker.benchmark_kernel(
                    compilation_result.binary_path,
                    compilation_result.kernel_name,
                    test_inputs
                )
                
                if benchmark_result.functional_correct:
                    results["tests_passed"] += 1
                    results["speedups"].append(benchmark_result.speedup_vs_torch)
        
        # Calculate statistics
        results["compilation_rate"] = results["compilation_success"] / results["total_problems"]
        results["pass_rate"] = results["tests_passed"] / results["total_problems"]
        results["avg_speedup"] = sum(results["speedups"]) / len(results["speedups"]) if results["speedups"] else 0
        
        # Log results
        self.logger.info(
            "Evaluation Results",
            compilation_rate=f"{results['compilation_rate']:.2%}",
            pass_rate=f"{results['pass_rate']:.2%}",
            avg_speedup=f"{results['avg_speedup']:.2f}x"
        )
        
        # Log to wandb
        if self.config.use_wandb:
            wandb.log({
                "eval/compilation_rate": results["compilation_rate"],
                "eval/pass_rate": results["pass_rate"],
                "eval/avg_speedup": results["avg_speedup"]
            })
        
        # Save results
        with open(self.rl_checkpoint / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results


async def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="Complete CUDA RL Training Pipeline")
    
    # Add arguments
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT phase")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL phase")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of RL episodes")
    parser.add_argument("--sft-examples", type=int, default=1000, help="Number of SFT examples")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with minimal settings")
    
    args = parser.parse_args()
    
    # Create configuration
    config = CompleteTrainingConfig()
    
    # Apply command line arguments
    config.sft_enabled = not args.skip_sft
    config.rl_enabled = not args.skip_rl
    config.num_gpus = args.num_gpus
    config.rl_num_episodes = args.num_episodes
    config.sft_num_examples = args.sft_examples
    
    # Quick test mode
    if args.quick_test:
        config.sft_num_examples = 10
        config.sft_epochs = 1
        config.rl_num_episodes = 2
        config.rl_batch_size = 4
        config.num_gpus = min(2, torch.cuda.device_count())
    
    # Create and run pipeline
    pipeline = CUDATrainingPipeline(config)
    await pipeline.run_complete_pipeline()


if __name__ == "__main__":
    asyncio.run(main())