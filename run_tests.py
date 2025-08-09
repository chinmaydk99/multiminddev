#!/usr/bin/env python3
"""
Unified test runner for the CUDA Multi-Agent System.

Replaces scattered test scripts with a single, organized test runner.
"""

import sys
import asyncio
import subprocess
from pathlib import Path
import argparse

# Add source to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔧 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 {description} - EXCEPTION: {e}")
        return False


async def run_async_tests():
    """Run async validation tests."""
    print("\n🧪 Running async validation tests...")
    
    try:
        from tests.validation.test_system_validation import (
            test_agent_initialization, 
            test_basic_workflow,
            test_import_validation
        )
        
        # Run import test
        test_import_validation()
        print("✅ Import validation - PASSED")
        
        # Run async tests
        await test_agent_initialization()
        print("✅ Agent initialization - PASSED")
        
        await test_basic_workflow()
        print("✅ Basic workflow - PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Async tests - FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run CUDA Multi-Agent System tests")
    parser.add_argument("--type", choices=["unit", "integration", "performance", "all"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--cuda", action="store_true", help="Include CUDA-dependent tests")
    parser.add_argument("--remote", action="store_true", help="Run tests on remote Lambda Labs")
    
    args = parser.parse_args()
    
    print("CUDA Multi-Agent System Test Runner")
    print("=" * 50)
    
    test_results = []
    
    # Run import and basic validation
    print("\n📦 Running system validation...")
    async_success = asyncio.run(run_async_tests())
    test_results.append(("System Validation", async_success))
    
    # Run pytest if available
    if args.type in ["unit", "all"]:
        success = run_command(
            [sys.executable, "-m", "pytest", "tests/unit/", "-v"],
            "Unit Tests"
        )
        test_results.append(("Unit Tests", success))
    
    if args.type in ["integration", "all"]:
        success = run_command(
            [sys.executable, "-m", "pytest", "tests/integration/", "-v"],
            "Integration Tests"
        )
        test_results.append(("Integration Tests", success))
    
    if args.cuda and args.type in ["performance", "all"]:
        success = run_command(
            [sys.executable, "-m", "pytest", "tests/performance/", "-v", "--tb=short"],
            "Performance Tests (CUDA Required)"
        )
        test_results.append(("Performance Tests", success))
    
    # Remote testing
    if args.remote:
        print("\n🌐 Remote testing not implemented in clean version")
        print("   Use Lambda Labs scripts manually if needed")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    success_rate = passed / total if total > 0 else 0
    print(f"\nOverall: {passed}/{total} tests passed ({success_rate:.1%})")
    
    if success_rate >= 0.8:
        print("\n🎉 Tests completed successfully!")
        return 0
    else:
        print("\n⚠️ Some tests failed - check output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())