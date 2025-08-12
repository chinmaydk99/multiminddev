#!/usr/bin/env python3
"""
Test script to validate SakanaAI dataset integration and curriculum system.
Run this before starting training to ensure everything works.
"""

import os
import sys
import asyncio
from pathlib import Path
import structlog

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from coding_framework.data.sakana_loader import SakanaDataLoader
from coding_framework.data.curriculum_manager import SakanaCurriculumManager


async def test_sakana_dataset_loading():
    """Test SakanaAI dataset loading with different scenarios."""
    
    print("🔍 Testing SakanaAI Dataset Loading")
    print("=" * 50)
    
    # Test 1: Load dataset normally
    print("\n1. Testing normal dataset loading...")
    try:
        loader = SakanaDataLoader(
            dataset_name="SakanaAI/AI-CUDA-Engineer-Archive",
            cache_dir="./cache/test_datasets",
            use_synthetic_fallback=True
        )
        
        print(f"✅ Dataset loaded successfully")
        print(f"   Total examples across levels: {loader.total_examples}")
        
        # Check each level
        for level in ["level_1", "level_2", "level_3"]:
            examples = loader.get_examples_by_level(level, max_examples=10)
            print(f"   {level}: {len(examples)} examples (showing max 10)")
            
            if examples:
                example = examples[0]
                print(f"      Sample: {example.operation_name}")
                print(f"      Correct: {example.is_correct}")
                print(f"      Speedup: {example.cuda_speedup_native:.2f}x")
        
        print("\n📊 Dataset Statistics:")
        stats = loader.get_level_statistics()
        for level, level_stats in stats.items():
            print(f"   {level}:")
            print(f"      Total: {level_stats['total_examples']}")
            print(f"      Correct: {level_stats['correct_examples']} ({level_stats['correctness_rate']:.1%})")
            print(f"      Avg Speedup: {level_stats['avg_speedup']:.2f}x")
            print(f"      Operations: {len(level_stats['operations'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        print("🔄 This will fall back to synthetic data during training")
        return False


async def test_curriculum_manager():
    """Test curriculum manager with SakanaAI levels."""
    
    print("\n🎯 Testing Curriculum Manager")
    print("=" * 50)
    
    try:
        # Initialize curriculum manager
        curriculum = SakanaCurriculumManager(
            initial_level="level_1",
            advancement_window=10  # Small window for testing
        )
        
        print(f"✅ Curriculum manager initialized")
        print(f"   Current level: {curriculum.get_current_level()}")
        print(f"   Available levels: {curriculum.level_order}")
        
        # Test level configurations
        for level in curriculum.level_order:
            config = curriculum.get_level_config(level)
            print(f"\n   {level} configuration:")
            print(f"      Difficulty: {config.difficulty_tier}")
            print(f"      Min compilation rate: {config.min_compilation_rate}")
            print(f"      Target speedup: {config.target_speedup}x")
            print(f"      Min episodes: {config.min_episodes}")
        
        # Simulate training episodes
        print(f"\n🏃 Simulating training episodes...")
        
        # Mock conversation result
        class MockConversationResult:
            def __init__(self, success=True, speedup=1.5, reward=0.7):
                self.conversation_success = success
                self.current_performance = {"speedup": speedup}
                self.final_reward = reward
                self.turns = [MockTurn()]
        
        class MockTurn:
            def __init__(self):
                self.compilation_success = True
        
        # Record some episode results
        for i in range(15):
            result = MockConversationResult(
                success=True,
                speedup=1.2 + i * 0.1,  # Gradually improving
                reward=0.5 + i * 0.03
            )
            curriculum.record_episode_result(result)
        
        # Check advancement
        should_advance = curriculum.should_advance_level()
        print(f"   Should advance from level_1: {should_advance}")
        
        # Get summary
        summary = curriculum.get_level_summary()
        print(f"\n📈 Level 1 Summary:")
        print(f"   Episodes: {summary['episode_count']}")
        print(f"   Recent metrics: {summary['recent_metrics']}")
        print(f"   Meets advancement: {summary['meets_advancement']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Curriculum manager test failed: {e}")
        return False


async def test_training_integration():
    """Test integration between dataset and curriculum."""
    
    print("\n🔗 Testing Dataset-Curriculum Integration")
    print("=" * 50)
    
    try:
        # Load dataset
        loader = SakanaDataLoader(use_synthetic_fallback=True)
        
        # Initialize curriculum
        curriculum = SakanaCurriculumManager()
        
        # Get examples for current curriculum level
        current_level = curriculum.get_current_level()
        examples = loader.get_examples_by_level(current_level, max_examples=5)
        
        print(f"✅ Integration test passed")
        print(f"   Current curriculum level: {current_level}")
        print(f"   Examples available: {len(examples)}")
        
        if examples:
            print(f"   Sample operations:")
            for i, example in enumerate(examples[:3]):
                print(f"      {i+1}. {example.operation_name} (Level {example.level_id})")
        
        # Test curriculum progression simulation
        print(f"\n🔄 Testing curriculum progression...")
        
        for level in curriculum.level_order:
            level_examples = loader.get_examples_by_level(level, max_examples=1)
            level_config = curriculum.get_level_config(level)
            
            print(f"   {level} ({level_config.difficulty_tier}): "
                  f"{len(loader.examples_by_level.get(level, []))} total examples")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


async def test_data_pipeline_components():
    """Test data pipeline components."""
    
    print("\n🔧 Testing Data Pipeline Components")
    print("=" * 50)
    
    try:
        # Test imports
        from coding_framework.data.data_pipeline import CUDADataPipeline
        from coding_framework.training.reward_functions.cuda_performance_reward import CUDAPerformanceReward
        
        print("✅ Core imports successful")
        
        # Test data pipeline initialization
        pipeline = CUDADataPipeline(
            dataset_name="SakanaAI/AI-CUDA-Engineer-Archive",
            curriculum_enabled=True,
            initial_tier="easy"
        )
        
        print("✅ Data pipeline initialized")
        
        # Test reward function
        reward_fn = CUDAPerformanceReward(target_speedup=2.0)
        print("✅ Reward function initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Data pipeline test failed: {e}")
        return False


async def main():
    """Run all tests."""
    
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
    
    print("🧪 SakanaAI Dataset Integration Test Suite")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Dataset Loading", test_sakana_dataset_loading),
        ("Curriculum Manager", test_curriculum_manager), 
        ("Training Integration", test_training_integration),
        ("Data Pipeline Components", test_data_pipeline_components)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n🎯 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System ready for training.")
        print("\n🚀 Next steps:")
        print("   1. Run SFT training: python train_sft_qlora.py")
        print("   2. Run RL training: python train_multiturn_rl.py")
        print("   3. Or use the complete pipeline: ./separated_training_setup.sh")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("   The system may still work with synthetic data fallbacks.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
