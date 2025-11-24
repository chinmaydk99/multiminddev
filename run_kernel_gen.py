#!/usr/bin/env python3
"""
End-to-end kernel generation pipeline.

This script:
1. Takes an operation + shape specification
2. Generates a HIP kernel using LLM
3. Compiles and executes in Docker sandbox
4. Measures performance vs baseline
5. Optionally refines based on feedback

Usage:
    python run_kernel_gen.py --operation rmsnorm --batch_seq 4096 --hidden 4096
    python run_kernel_gen.py --config configs/default.yaml
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.operations.base import DataType
from src.operations.normalization import RMSNormOp, LayerNormOp
from src.operations.attention import AttentionOp, FlashAttentionOp
from src.operations.gemm import GemmOp, BatchedGemmOp
from src.kernel_gen.generator import KernelGenerator
from src.kernel_gen.prompts import PromptBuilder
from src.execution.sandbox import KernelSandbox, ExecutionResult
from src.execution.compiler import HIPCompiler
from src.rewards.execution_reward import ExecutionReward, RewardConfig

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True)
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


# Operation registry
OPERATIONS = {
    "rmsnorm": RMSNormOp,
    "layernorm": LayerNormOp,
    "attention": AttentionOp,
    "flash_attention": FlashAttentionOp,
    "gemm": GemmOp,
    "batched_gemm": BatchedGemmOp,
}


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def get_operation(name: str) -> Any:
    """Get operation class by name."""
    if name not in OPERATIONS:
        raise ValueError(f"Unknown operation: {name}. Available: {list(OPERATIONS.keys())}")
    return OPERATIONS[name]()


async def run_single_generation(
    operation: Any,
    shape_values: Dict[str, int],
    dtype: DataType,
    generator: KernelGenerator,
    sandbox: KernelSandbox,
    reward_fn: ExecutionReward,
    max_refinements: int = 3,
) -> Dict[str, Any]:
    """
    Run single kernel generation with optional refinements.
    
    Returns dict with:
    - best_kernel: Best kernel code
    - best_speedup: Best achieved speedup
    - attempts: List of all attempts with results
    """
    
    logger.info(
        "Starting kernel generation",
        operation=operation.name,
        shapes=shape_values,
        dtype=dtype.value,
    )
    
    attempts = []
    best_kernel = None
    best_speedup = 0.0
    best_result = None
    
    # Generate test inputs
    inputs = operation.generate_test_inputs(shape_values, dtype, device="cuda")
    
    # Get reference outputs
    reference_outputs = operation.reference_impl(**inputs)
    
    # Measure baseline time
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        _ = operation.baseline_impl(**inputs)
    torch.cuda.synchronize()
    baseline_time_ms = (time.perf_counter() - start) / 10 * 1000
    
    logger.info(f"Baseline time: {baseline_time_ms:.3f} ms")
    
    # Initial generation
    gen_result = await generator.generate_from_operation(
        operation=operation,
        shape_values=shape_values,
        dtype=dtype,
    )
    
    if not gen_result.success:
        logger.error("Initial generation failed", error=gen_result.error_message)
        return {
            "success": False,
            "error": gen_result.error_message,
            "attempts": attempts,
        }
    
    current_code = gen_result.kernel_code
    
    # Execution + refinement loop
    for attempt_num in range(max_refinements + 1):
        logger.info(f"Attempt {attempt_num + 1}/{max_refinements + 1}")
        
        # Execute kernel
        exec_result = await sandbox.execute_kernel(
            kernel_code=current_code,
            kernel_name=f"{operation.name}_v{attempt_num}",
            input_tensors=inputs,
            reference_outputs=reference_outputs,
            shape_values=shape_values,
            baseline_time_ms=baseline_time_ms,
        )
        
        # Calculate reward
        reward, reward_components = reward_fn.calculate(exec_result)
        
        # Record attempt
        attempt_record = {
            "attempt": attempt_num + 1,
            "kernel_code": current_code,
            "compiled": exec_result.compiled,
            "correct": exec_result.numerically_correct,
            "speedup": exec_result.speedup,
            "kernel_time_ms": exec_result.kernel_time_ms,
            "reward": reward,
            "reward_components": reward_components,
        }
        attempts.append(attempt_record)
        
        logger.info(
            f"Attempt {attempt_num + 1} results",
            compiled=exec_result.compiled,
            correct=exec_result.numerically_correct,
            speedup=f"{exec_result.speedup:.2f}x",
            reward=f"{reward:.3f}",
        )
        
        # Track best
        if exec_result.numerically_correct and exec_result.speedup > best_speedup:
            best_speedup = exec_result.speedup
            best_kernel = current_code
            best_result = exec_result
        
        # Early stop if we achieved target speedup
        if exec_result.numerically_correct and exec_result.speedup >= 2.0:
            logger.info(f"Target speedup achieved! {exec_result.speedup:.2f}x")
            break
        
        # Refine if we have more attempts
        if attempt_num < max_refinements:
            logger.info("Refining kernel based on feedback...")
            
            refine_result = await generator.refine(
                operation=operation,
                shape_values=shape_values,
                dtype=dtype,
                previous_code=current_code,
                execution_result=exec_result,
            )
            
            if refine_result.success:
                current_code = refine_result.kernel_code
            else:
                logger.warning("Refinement failed", error=refine_result.error_message)
                break
    
    return {
        "success": best_kernel is not None,
        "best_kernel": best_kernel,
        "best_speedup": best_speedup,
        "best_result": best_result.to_dict() if best_result else None,
        "attempts": attempts,
        "operation": operation.name,
        "shapes": shape_values,
        "dtype": dtype.value,
        "baseline_time_ms": baseline_time_ms,
    }


async def main():
    parser = argparse.ArgumentParser(description="Generate optimized HIP kernels")
    
    # Operation selection
    parser.add_argument("--operation", "-op", type=str, default="rmsnorm",
                        choices=list(OPERATIONS.keys()),
                        help="Operation to generate kernel for")
    
    # Shape parameters (operation-specific)
    parser.add_argument("--batch_seq", type=int, default=4096,
                        help="Batch * sequence length (for normalization)")
    parser.add_argument("--hidden", type=int, default=4096,
                        help="Hidden dimension (for normalization)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size (for attention/gemm)")
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="Sequence length (for attention)")
    parser.add_argument("--num_heads", type=int, default=32,
                        help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, default=128,
                        help="Attention head dimension")
    parser.add_argument("--M", type=int, default=1,
                        help="M dimension (for GEMM)")
    parser.add_argument("--K", type=int, default=4096,
                        help="K dimension (for GEMM)")
    parser.add_argument("--N", type=int, default=4096,
                        help="N dimension (for GEMM)")
    
    # Data type
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Data type for tensors")
    
    # Generator settings
    parser.add_argument("--backend", type=str, default="openai",
                        choices=["openai", "huggingface", "vllm"],
                        help="LLM backend to use")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default depends on backend)")
    
    # Execution settings
    parser.add_argument("--use-docker", action="store_true", default=True,
                        help="Use Docker for execution")
    parser.add_argument("--no-docker", action="store_false", dest="use_docker",
                        help="Run locally (requires ROCm)")
    
    # Refinement settings
    parser.add_argument("--max-refinements", type=int, default=3,
                        help="Maximum refinement iterations")
    
    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    
    # Output
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file for results (JSON)")
    parser.add_argument("--save-kernel", type=str, default=None,
                        help="Save best kernel to file")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = load_config(args.config)
    
    # Get operation
    operation = get_operation(args.operation)
    
    # Build shape values based on operation type
    if args.operation in ["rmsnorm", "layernorm"]:
        shape_values = {
            "batch_seq": args.batch_seq,
            "hidden": args.hidden,
        }
    elif args.operation in ["attention", "flash_attention"]:
        shape_values = {
            "batch": args.batch,
            "seq_len": args.seq_len,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
        }
    elif args.operation in ["gemm"]:
        shape_values = {
            "M": args.M,
            "K": args.K,
            "N": args.N,
        }
    elif args.operation in ["batched_gemm"]:
        shape_values = {
            "batch": args.batch,
            "M": args.M,
            "K": args.K,
            "N": args.N,
        }
    else:
        raise ValueError(f"Unknown operation: {args.operation}")
    
    # Data type
    dtype_map = {
        "float16": DataType.FP16,
        "bfloat16": DataType.BF16,
        "float32": DataType.FP32,
    }
    dtype = dtype_map[args.dtype]
    
    # Initialize components
    logger.info("Initializing kernel generator...")
    
    generator = KernelGenerator(
        backend_type=args.backend,
        model_name=args.model,
    )
    
    sandbox = KernelSandbox(
        use_docker=args.use_docker,
    )
    
    reward_fn = ExecutionReward()
    
    # Run generation
    logger.info("=" * 60)
    logger.info("KERNEL GENERATION PIPELINE")
    logger.info("=" * 60)
    
    results = await run_single_generation(
        operation=operation,
        shape_values=shape_values,
        dtype=dtype,
        generator=generator,
        sandbox=sandbox,
        reward_fn=reward_fn,
        max_refinements=args.max_refinements,
    )
    
    # Print results
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    
    if results["success"]:
        logger.info(f"Best speedup: {results['best_speedup']:.2f}x")
        logger.info(f"Baseline time: {results['baseline_time_ms']:.3f} ms")
        logger.info(f"Total attempts: {len(results['attempts'])}")
        
        # Save kernel if requested
        if args.save_kernel and results["best_kernel"]:
            Path(args.save_kernel).write_text(results["best_kernel"])
            logger.info(f"Best kernel saved to: {args.save_kernel}")
    else:
        logger.error("Generation failed")
        if "error" in results:
            logger.error(f"Error: {results['error']}")
    
    # Save results if requested
    if args.output:
        # Remove non-serializable fields
        output_results = {k: v for k, v in results.items() if k != "best_result"}
        if results.get("best_result"):
            output_results["best_result"] = results["best_result"]
        
        with open(args.output, "w") as f:
            json.dump(output_results, f, indent=2, default=str)
        logger.info(f"Results saved to: {args.output}")
    
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

