#!/bin/bash
# Complete separated training setup for CUDA RL system
# Handles both SFT (QLoRA) and Multi-Turn RL (VERL) phases separately

set -e

echo "🚀 CUDA Multi-Agent RL Training System - Separated Training"
echo "=============================================================="

# Configuration
BASE_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
NUM_GPUS=8
SFT_OUTPUT_DIR="./checkpoints/sft"
RL_OUTPUT_DIR="./checkpoints/rl"

# Parse command line arguments
PHASE="both"  # both, sft, rl
QUICK_TEST=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --quick-test)
            QUICK_TEST=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--phase both|sft|rl] [--quick-test]"
            echo ""
            echo "Options:"
            echo "  --phase     Which phase to run (both, sft, rl)"
            echo "  --quick-test Run with minimal settings for testing"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Environment setup
echo "📋 Setting up environment..."
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_OBJECT_STORE_MEMORY=50000000000  # 50GB
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Verify GPU setup
echo "🔧 Verifying GPU setup..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Test system components
echo "🧪 Testing system components..."
python test_complete_system.py

# Phase 1: SFT Training with QLoRA
if [[ "$PHASE" == "both" || "$PHASE" == "sft" ]]; then
    echo ""
    echo "📚 Phase 1: SFT Training with QLoRA"
    echo "===================================="
    
    # SFT training arguments
    SFT_ARGS=(
        --start-tier level_1
        --max-examples 2000
        --output-dir "$SFT_OUTPUT_DIR"
        --agent both
    )
    
    if [[ "$QUICK_TEST" == "true" ]]; then
        echo "⚡ Running SFT in quick test mode"
        SFT_ARGS+=(--quick-test)
    fi
    
    echo "🏃 Starting SFT training..."
    echo "Command: python train_sft_qlora.py ${SFT_ARGS[*]}"
    
    python train_sft_qlora.py "${SFT_ARGS[@]}"
    
    echo "✅ SFT training completed!"
    echo "📁 Generator model saved to: $SFT_OUTPUT_DIR/generator/final"
    echo "📁 Optimizer model saved to: $SFT_OUTPUT_DIR/optimizer/final"
    
    # Verify SFT checkpoints exist
    if [[ ! -d "$SFT_OUTPUT_DIR/generator/final" ]]; then
        echo "❌ ERROR: Generator SFT checkpoint not found!"
        exit 1
    fi
    
    if [[ ! -d "$SFT_OUTPUT_DIR/optimizer/final" ]]; then
        echo "❌ ERROR: Optimizer SFT checkpoint not found!"
        exit 1
    fi
    
    echo "🔍 SFT checkpoints verified successfully"
fi

# Phase 2: Multi-Turn RL Training
if [[ "$PHASE" == "both" || "$PHASE" == "rl" ]]; then
    echo ""
    echo "🤖 Phase 2: Multi-Turn RL Training with VERL"
    echo "============================================="
    
    # Verify SFT checkpoints exist for RL training
    if [[ ! -d "$SFT_OUTPUT_DIR/generator/final" ]]; then
        echo "❌ ERROR: SFT generator checkpoint required for RL training"
        echo "Please run SFT training first: $0 --phase sft"
        exit 1
    fi
    
    # RL training arguments
    RL_ARGS=(
        --generator-model "$SFT_OUTPUT_DIR/generator/final"
        --optimizer-model "$SFT_OUTPUT_DIR/optimizer/final"
        --num-episodes 1000
        --max-turns 5
        --batch-size 256
        --learning-rate 5e-6
        --algorithm grpo
    )
    
    if [[ "$QUICK_TEST" == "true" ]]; then
        echo "⚡ Running RL in quick test mode"
        RL_ARGS+=(
            --test-only
            --num-episodes 10
        )
    fi
    
    echo "🏃 Starting RL training..."
    echo "Command: python train_multiturn_rl.py ${RL_ARGS[*]}"
    
    python train_multiturn_rl.py "${RL_ARGS[@]}"
    
    echo "✅ Multi-Turn RL training completed!"
    echo "📁 RL checkpoints saved to: $RL_OUTPUT_DIR"
fi

# Final summary
echo ""
echo "🎉 Training Pipeline Completed Successfully!"
echo "==========================================="
echo ""
echo "📊 Training Summary:"
echo "  - Base Model: $BASE_MODEL"
echo "  - GPUs Used: $NUM_GPUS"
echo "  - Phase Run: $PHASE"

if [[ "$PHASE" == "both" || "$PHASE" == "sft" ]]; then
    echo "  - SFT Checkpoints: $SFT_OUTPUT_DIR/"
    echo "    - Generator: $SFT_OUTPUT_DIR/generator/final"
    echo "    - Optimizer: $SFT_OUTPUT_DIR/optimizer/final"
fi

if [[ "$PHASE" == "both" || "$PHASE" == "rl" ]]; then
    echo "  - RL Checkpoints: $RL_OUTPUT_DIR/"
fi

echo ""
echo "🔄 Next Steps:"

if [[ "$PHASE" == "sft" ]]; then
    echo "  1. Run RL training: $0 --phase rl"
    echo "  2. Or run full pipeline: $0 --phase both"
elif [[ "$PHASE" == "rl" ]]; then
    echo "  1. Evaluate final models"
    echo "  2. Deploy for inference"
else
    echo "  1. Evaluate final models"
    echo "  2. Deploy for inference"
    echo "  3. Monitor performance"
fi

echo ""
echo "📚 Usage Examples:"
echo "  # Run only SFT training:"
echo "  $0 --phase sft"
echo ""
echo "  # Run only RL training (requires SFT checkpoints):"
echo "  $0 --phase rl"
echo ""
echo "  # Quick test of full pipeline:"
echo "  $0 --phase both --quick-test"
echo ""
echo "✨ Happy training!"
