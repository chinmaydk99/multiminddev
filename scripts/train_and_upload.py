#!/usr/bin/env python3
"""
Complete pipeline: Enhanced SFT Training + Hugging Face Upload
Trains the model and automatically uploads to HF Hub
"""

import asyncio
import json
import os
import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from examples.cuda_training.enhanced_sft_warmstart import run_enhanced_sft_training
from scripts.upload_to_huggingface import upload_model_to_hf

import structlog

logger = structlog.get_logger("train_and_upload")


async def complete_training_pipeline(
    num_examples: int = 50,
    num_epochs: int = 1,
    hf_token: str = None,
    model_name: str = "cuda-code-generator-v1",
    upload_to_hf: bool = True
):
    """Complete training and upload pipeline."""
    
    logger.info("🚀 Starting Complete Training Pipeline", 
               examples=num_examples, 
               epochs=num_epochs,
               upload_enabled=upload_to_hf)
    
    try:
        # Run enhanced SFT training
        logger.info("📈 Phase 1: Running Enhanced SFT Training...")
        training_success = await run_enhanced_sft_training(
            num_examples=num_examples,
            num_epochs=num_epochs,
            save_checkpoints=True,
            progress_file="sft_training_results.json"
        )
        
        if not training_success:
            logger.error("❌ Training failed - aborting upload")
            return False
        
        logger.info("✅ Training completed successfully!")
        
        # Load training results
        training_stats = None
        if os.path.exists("sft_training_results.json"):
            with open("sft_training_results.json", 'r') as f:
                training_data = json.load(f)
                training_stats = training_data.get('metrics', {})
            logger.info("📊 Loaded training statistics")
        
        # Upload to Hugging Face if requested
        if upload_to_hf and hf_token:
            logger.info("📤 Phase 2: Uploading to Hugging Face...")
            
            upload_success = upload_model_to_hf(
                hf_token=hf_token,
                model_name=model_name,
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                training_stats=training_stats,
                local_model_path=None,  # We're not saving model files locally for now
                organization="MultiMindDev"
            )
            
            if upload_success:
                logger.info("✅ Successfully uploaded to Hugging Face!")
                print(f"\n🎉 Model available at: https://huggingface.co/MultiMindDev/{model_name}")
            else:
                logger.error("❌ Upload to Hugging Face failed")
                return False
        
        logger.info("🏁 Complete pipeline finished successfully!")
        return True
        
    except Exception as e:
        logger.error("❌ Pipeline failed", error=str(e))
        return False


def main():
    """Main pipeline entry point."""
    parser = argparse.ArgumentParser(description="Complete Training + Upload Pipeline")
    parser.add_argument("--examples", type=int, default=30, help="Number of training examples")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--hf-token", help="HuggingFace API token")
    parser.add_argument("--model-name", default="cuda-code-generator-v1", help="Model name for HF")
    parser.add_argument("--no-upload", action="store_true", help="Skip HuggingFace upload")
    
    args = parser.parse_args()
    
    # Use environment variable if token not provided
    hf_token = args.hf_token or os.getenv('HUGGINGFACE_TOKEN')
    
    print("🚀 MultiMindDev Complete Training Pipeline")
    print("=" * 50)
    print(f"📊 Training Examples: {args.examples}")
    print(f"📈 Epochs: {args.epochs}")
    print(f"📦 Model Name: MultiMindDev/{args.model_name}")
    print(f"📤 HF Upload: {'Disabled' if args.no_upload else ('Enabled' if hf_token else 'No token provided')}")
    print("=" * 50)
    
    if not args.no_upload and not hf_token:
        print("⚠️  Warning: No HuggingFace token provided - skipping upload")
        print("   Set HUGGINGFACE_TOKEN environment variable or use --hf-token")
        args.no_upload = True
    
    result = asyncio.run(complete_training_pipeline(
        num_examples=args.examples,
        num_epochs=args.epochs,
        hf_token=hf_token,
        model_name=args.model_name,
        upload_to_hf=not args.no_upload
    ))
    
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())