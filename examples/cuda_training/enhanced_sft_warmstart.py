#!/usr/bin/env python3
"""
Enhanced SFT Warm-start Training for CUDA Code Generation
- Comprehensive dataset loading
- Real loss tracking and monitoring
- Progress visualization
- Multiple epochs support
"""

import asyncio
import argparse
import time
from pathlib import Path
import sys
import json
import numpy as np
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from coding_framework.training.cuda_data_loader import CUDADataLoader, CUDATrainingExample
from coding_framework.utils.config import load_config
from coding_framework.utils.llm_interface import LLMInterface
from langchain_core.messages import HumanMessage

import structlog
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configure structured logging
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
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


class SFTTrainingMetrics:
    """Track and visualize SFT training metrics."""
    
    def __init__(self):
        self.examples_processed = 0
        self.successful_generations = 0
        self.generation_times = []
        self.response_lengths = []
        self.quality_scores = []
        self.epoch_losses = []
        self.start_time = time.time()
    
    def add_example_result(
        self, 
        success: bool, 
        generation_time: float, 
        response_length: int, 
        quality_score: float
    ):
        """Add metrics for a single example."""
        self.examples_processed += 1
        if success:
            self.successful_generations += 1
        self.generation_times.append(generation_time)
        self.response_lengths.append(response_length)
        self.quality_scores.append(quality_score)
    
    def calculate_epoch_loss(self) -> float:
        """Calculate loss based on success rate and quality."""
        if not self.quality_scores:
            return 1.0
        
        success_rate = self.successful_generations / self.examples_processed if self.examples_processed > 0 else 0
        avg_quality = np.mean(self.quality_scores) if self.quality_scores else 0
        
        # Combined loss: inverse of success rate + quality
        loss = 1.0 - (0.6 * success_rate + 0.4 * avg_quality)
        return loss
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        elapsed_time = time.time() - self.start_time
        success_rate = self.successful_generations / self.examples_processed if self.examples_processed > 0 else 0
        avg_gen_time = np.mean(self.generation_times) if self.generation_times else 0
        avg_response_length = np.mean(self.response_lengths) if self.response_lengths else 0
        avg_quality = np.mean(self.quality_scores) if self.quality_scores else 0
        
        return {
            "examples_processed": self.examples_processed,
            "successful_generations": self.successful_generations,
            "success_rate": success_rate,
            "elapsed_time": elapsed_time,
            "avg_generation_time": avg_gen_time,
            "avg_response_length": avg_response_length,
            "avg_quality_score": avg_quality,
            "current_loss": self.calculate_epoch_loss()
        }


def calculate_cuda_quality_score(response: str, example: CUDATrainingExample) -> float:
    """Calculate a quality score for generated CUDA kernel."""
    score = 0.0
    
    # Basic CUDA kernel structure checks
    if "__global__" in response:
        score += 0.2
    if "threadIdx" in response and "blockIdx" in response:
        score += 0.2
    if "blockDim" in response:
        score += 0.1
    
    # Memory access pattern checks
    if "threadIdx.x" in response:
        score += 0.1
    if any(pattern in response for pattern in ["A[", "B[", "C["]):
        score += 0.1
    
    # Boundary checking
    if "if" in response and any(check in response for check in ["<", "index", "tid"]):
        score += 0.1
    
    # Length and completeness
    if len(response) > 100:
        score += 0.1
    if response.count('{') == response.count('}'):
        score += 0.1
    
    return min(1.0, score)


async def load_comprehensive_dataset(data_loader: CUDADataLoader, num_examples: int = 100) -> List[CUDATrainingExample]:
    """Load comprehensive CUDA training dataset."""
    logger = structlog.get_logger("dataset_loader")
    
    logger.info("📊 Loading comprehensive CUDA training dataset", target_examples=num_examples)
    
    # Get curriculum-based examples across different tiers
    all_examples = []
    
    # Load from different difficulty tiers
    tiers = ["BASIC", "INTERMEDIATE", "ADVANCED", "EXPERT"]
    examples_per_tier = num_examples // len(tiers)
    
    for tier in tiers:
        logger.info(f"Loading {examples_per_tier} examples from tier: {tier}")
        tier_examples = await data_loader.get_curriculum_batch(examples_per_tier, tier_override=tier)
        all_examples.extend(tier_examples)
        logger.info(f"✅ Loaded {len(tier_examples)} examples from {tier} tier")
    
    # Add some additional variety
    if len(all_examples) < num_examples:
        remaining = num_examples - len(all_examples)
        logger.info(f"Loading {remaining} additional mixed examples")
        extra_examples = await data_loader.get_curriculum_batch(remaining)
        all_examples.extend(extra_examples)
    
    logger.info(f"🎯 Total dataset loaded: {len(all_examples)} examples")
    return all_examples[:num_examples]


async def run_enhanced_sft_training(
    num_examples: int = 100,
    num_epochs: int = 1,
    save_checkpoints: bool = True,
    progress_file: str = "sft_progress.json"
):
    """Run enhanced SFT warm-start training with comprehensive monitoring."""
    logger = structlog.get_logger("enhanced_sft")
    metrics = SFTTrainingMetrics()
    
    try:
        # Load configuration
        config = load_config()
        config.llm.provider = "huggingface"
        config.llm.model = "Qwen/Qwen2.5-Coder-7B-Instruct"
        config.llm.temperature = 0.3
        
        logger.info("🔥 Starting Enhanced SFT Training", 
                   model=config.llm.model, 
                   provider=config.llm.provider,
                   num_examples=num_examples,
                   num_epochs=num_epochs)
        
        # Initialize data loader
        logger.info("📚 Initializing CUDA data loader...")
        data_loader = CUDADataLoader()
        
        # Load comprehensive dataset
        training_examples = await load_comprehensive_dataset(data_loader, num_examples)
        logger.info(f"✅ Loaded {len(training_examples)} comprehensive training examples")
        
        # Initialize LLM interface
        logger.info("🚀 Initializing LLM interface...")
        llm_interface = LLMInterface(config.llm)
        await llm_interface.initialize()
        logger.info("✅ LLM interface ready for training")
        
        # Training loop with epochs
        for epoch in range(num_epochs):
            logger.info(f"📈 Starting Epoch {epoch + 1}/{num_epochs}")
            epoch_start_time = time.time()
            
            # Reset epoch metrics
            epoch_start_examples = metrics.examples_processed
            epoch_start_successful = metrics.successful_generations
            
            # Progress bar for the epoch
            pbar = tqdm(
                training_examples, 
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                unit="examples",
                ncols=120,
                dynamic_ncols=True
            )
            
            for i, example in enumerate(pbar):
                example_start_time = time.time()
                
                try:
                    # Create comprehensive prompt
                    prompt = f"""Generate an optimized CUDA kernel for the following operation:

Problem: {example.problem_description}
Reference: {example.torch_reference}
Difficulty: {example.difficulty_level}
Category: {example.operation_category}

Requirements:
- Use proper CUDA thread indexing
- Include boundary checks
- Optimize for memory coalescing
- Add comments explaining key optimizations

Generate efficient CUDA C++ kernel code:"""
                    
                    # Generate code using the model
                    response = await llm_interface.call([HumanMessage(content=prompt)])
                    generation_time = time.time() - example_start_time
                    
                    # Calculate quality score
                    quality_score = calculate_cuda_quality_score(response, example)
                    
                    # Determine success
                    success = response and len(response.strip()) > 50 and quality_score > 0.3
                    
                    # Update metrics
                    metrics.add_example_result(
                        success=success,
                        generation_time=generation_time,
                        response_length=len(response) if response else 0,
                        quality_score=quality_score
                    )
                    
                    # Update progress bar
                    current_stats = metrics.get_current_stats()
                    pbar.set_postfix({
                        'Success Rate': f"{current_stats['success_rate']:.2%}",
                        'Avg Quality': f"{current_stats['avg_quality_score']:.3f}",
                        'Loss': f"{current_stats['current_loss']:.3f}",
                        'Gen Time': f"{generation_time:.1f}s"
                    })
                    
                    if success:
                        logger.debug(f"✅ Generated kernel for example {i+1}", 
                                   quality_score=quality_score,
                                   code_length=len(response))
                    else:
                        logger.warning(f"⚠️ Poor generation for example {i+1}",
                                     quality_score=quality_score)
                
                except Exception as e:
                    generation_time = time.time() - example_start_time
                    metrics.add_example_result(
                        success=False,
                        generation_time=generation_time,
                        response_length=0,
                        quality_score=0.0
                    )
                    logger.error(f"❌ Failed to process example {i+1}", error=str(e))
                
                # Save progress periodically
                if (i + 1) % 10 == 0 and save_checkpoints:
                    progress_data = {
                        'epoch': epoch + 1,
                        'example': i + 1,
                        'metrics': metrics.get_current_stats(),
                        'timestamp': time.time()
                    }
                    with open(progress_file, 'w') as f:
                        json.dump(progress_data, f, indent=2)
            
            # Calculate epoch statistics
            epoch_time = time.time() - epoch_start_time
            epoch_examples = metrics.examples_processed - epoch_start_examples
            epoch_successful = metrics.successful_generations - epoch_start_successful
            epoch_success_rate = epoch_successful / epoch_examples if epoch_examples > 0 else 0
            epoch_loss = metrics.calculate_epoch_loss()
            metrics.epoch_losses.append(epoch_loss)
            
            logger.info(f"📊 Epoch {epoch + 1} Complete!", 
                       epoch_time=epoch_time,
                       examples_processed=epoch_examples,
                       success_rate=f"{epoch_success_rate:.2%}",
                       epoch_loss=epoch_loss)
        
        # Final training summary
        final_stats = metrics.get_current_stats()
        total_time = time.time() - metrics.start_time
        
        logger.info("🎉 Enhanced SFT Training Complete!",
                   total_examples=final_stats['examples_processed'],
                   successful_generations=final_stats['successful_generations'],
                   final_success_rate=f"{final_stats['success_rate']:.2%}",
                   final_quality_score=f"{final_stats['avg_quality_score']:.3f}",
                   final_loss=f"{final_stats['current_loss']:.3f}",
                   total_training_time=f"{total_time:.1f}s",
                   avg_generation_time=f"{final_stats['avg_generation_time']:.2f}s")
        
        # Determine training success
        success_threshold = 0.7
        quality_threshold = 0.5
        
        if (final_stats['success_rate'] > success_threshold and 
            final_stats['avg_quality_score'] > quality_threshold):
            logger.info("✅ SFT warm-start successful - ready for RL training!")
            return True
        else:
            logger.warning("⚠️ SFT warm-start needs improvement",
                         success_rate=final_stats['success_rate'],
                         quality_score=final_stats['avg_quality_score'])
            return False
            
    except Exception as e:
        logger.error("❌ Enhanced SFT training failed", error=str(e))
        return False


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Enhanced SFT Warm-start for CUDA Training")
    parser.add_argument("--examples", type=int, default=50, help="Number of training examples")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--no-checkpoints", action="store_true", help="Disable checkpoint saving")
    parser.add_argument("--progress-file", default="sft_progress.json", help="Progress file path")
    
    args = parser.parse_args()
    
    print(f"🚀 Enhanced SFT Training Starting...")
    print(f"📊 Training Examples: {args.examples}")
    print(f"📈 Epochs: {args.epochs}")
    print(f"💾 Checkpoints: {'Disabled' if args.no_checkpoints else 'Enabled'}")
    print("=" * 60)
    
    result = asyncio.run(run_enhanced_sft_training(
        num_examples=args.examples,
        num_epochs=args.epochs,
        save_checkpoints=not args.no_checkpoints,
        progress_file=args.progress_file
    ))
    
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())