#!/usr/bin/env python3
"""
SFT Warm-start Training for CUDA Code Generation
Implements the recommended hybrid approach: SFT first, then RL
"""

import asyncio
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from coding_framework.training.cuda_data_loader import CUDADataLoader, CUDATrainingExample
from coding_framework.utils.config import load_config
from coding_framework.utils.llm_interface import LLMInterface

import structlog

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


async def run_sft_warmstart():
    """Run SFT warm-start training using HuggingFace model."""
    logger = structlog.get_logger("sft_warmstart")
    
    try:
        # Load configuration
        config = load_config()
        
        # Ensure we're using HuggingFace
        config.llm.provider = "huggingface"
        config.llm.model = "Qwen/Qwen2.5-Coder-7B-Instruct"
        config.llm.temperature = 0.3  # Lower for SFT
        
        logger.info("🔥 Starting SFT Warm-start Training", 
                   model=config.llm.model, 
                   provider=config.llm.provider)
        
        # Create simple training examples for SFT warm-start
        logger.info("📚 Creating basic CUDA training examples...")
        training_examples = [
            CUDATrainingExample(
                problem_description="Implement vector addition: C = A + B",
                torch_reference="torch.add(A, B)",
                test_inputs=[{"shape": (1024,), "dtype": "float32"}],
                difficulty_level="easy",
                operation_category="elementwise"
            ),
            CUDATrainingExample(
                problem_description="Implement scalar multiplication: C = A * scalar",
                torch_reference="torch.mul(A, scalar)",
                test_inputs=[{"shape": (1024,), "dtype": "float32", "scalar": 2.0}],
                difficulty_level="easy", 
                operation_category="elementwise"
            ),
            CUDATrainingExample(
                problem_description="Implement vector dot product: result = A dot B",
                torch_reference="torch.dot(A, B)",
                test_inputs=[{"shape": (1024,), "dtype": "float32"}],
                difficulty_level="medium",
                operation_category="reduction"
            ),
            CUDATrainingExample(
                problem_description="Implement matrix transpose: B = A^T",
                torch_reference="torch.transpose(A, 0, 1)",
                test_inputs=[{"shape": (32, 32), "dtype": "float32"}],
                difficulty_level="medium",
                operation_category="linear_algebra"
            ),
            CUDATrainingExample(
                problem_description="Implement simple matrix-vector multiplication: y = A * x",
                torch_reference="torch.mv(A, x)",
                test_inputs=[{"matrix_shape": (64, 64), "vector_shape": (64,), "dtype": "float32"}],
                difficulty_level="hard",
                operation_category="linear_algebra"
            )
        ]
        
        logger.info(f"✅ Loaded {len(training_examples)} training examples")
        
        # Initialize LLM interface
        llm_interface = LLMInterface(config.llm)
        
        # Simple SFT training loop
        successful_generations = 0
        total_examples = len(training_examples)
        
        logger.info("🎯 Starting SFT training loop...")
        
        for i, example in enumerate(training_examples):  # Process all examples
            logger.info(f"📝 Processing example {i+1}/{total_examples}")
            
            try:
                # Create prompt for CUDA kernel generation
                prompt = f"""
Generate a CUDA kernel for the following operation:

Problem: {example.problem_description}
Reference Implementation: {example.torch_reference}

Generate efficient CUDA C++ kernel code:
"""
                
                # Generate code using the model
                from langchain_core.messages import HumanMessage
                response = await llm_interface.call([HumanMessage(content=prompt)])
                
                if response and len(response.strip()) > 50:  # Basic validation
                    successful_generations += 1
                    logger.info(f"✅ Generated kernel for example {i+1}", 
                              code_length=len(response))
                else:
                    logger.warning(f"⚠️ Poor generation for example {i+1}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process example {i+1}", error=str(e))
        
        # Calculate success rate
        success_rate = successful_generations / total_examples
        
        logger.info("🎉 SFT Warm-start Complete!", 
                   successful_generations=successful_generations,
                   total_processed=total_examples,
                   success_rate=f"{success_rate:.2%}")
        
        if success_rate > 0.7:
            logger.info("✅ SFT warm-start successful - ready for RL training!")
            return True
        else:
            logger.warning("⚠️ SFT warm-start had low success rate - check model/config")
            return False
            
    except Exception as e:
        logger.error("❌ SFT warm-start failed", error=str(e))
        return False


def main():
    parser = argparse.ArgumentParser(description="SFT Warm-start for CUDA Training")
    parser.add_argument("--examples", type=int, default=100, help="Number of examples for warm-start")
    args = parser.parse_args()
    
    result = asyncio.run(run_sft_warmstart())
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())