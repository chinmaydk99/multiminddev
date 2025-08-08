#!/usr/bin/env python3
"""
Quick test script to verify our fixes work.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from coding_framework.agents import CodeGeneratorAgent
from coding_framework.training.reward_functions import CorrectnessReward, StyleReward, EfficiencyReward
from coding_framework.utils.config import Config
from coding_framework.utils.llm_interface import LLMInterface


async def test_code_extraction():
    """Test code extraction from LLM response."""
    print("🧪 Testing code extraction...")
    
    # Mock LLM response with markdown formatting
    mock_response = """Here's a solution to reverse a string:

```python
def reverse_string(s):
    return s[::-1]
```

This function uses Python's slice notation to reverse the string efficiently."""
    
    config = Config()
    llm_interface = LLMInterface(config.llm)
    await llm_interface.initialize()
    
    agent = CodeGeneratorAgent(
        config=config.agents.generator,
        llm_interface=llm_interface,
        agent_id="test_generator"
    )
    
    # Test code extraction
    code_info = agent._extract_code_from_response(mock_response)
    print(f"✅ Extracted code: {repr(code_info['code'])}")
    print(f"✅ Explanation: {repr(code_info['explanation'])}")
    
    # Test syntax validation
    try:
        import ast
        ast.parse(code_info['code'])
        print("✅ Code has valid syntax")
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
    
    return code_info['code']


async def test_reward_functions():
    """Test reward function execution."""
    print("\n🧪 Testing reward functions...")
    
    # Simple test code
    test_code = """def reverse_string(s):
    return s[::-1]"""
    
    test_problem = "Write a function that takes a string and returns it reversed."
    test_cases = [
        {"input": "hello", "expected_output": "olleh"},
        {"input": "world", "expected_output": "dlrow"},
    ]
    
    # Test EfficiencyReward
    print("Testing EfficiencyReward...")
    efficiency_reward = EfficiencyReward(weight=1.0)
    try:
        reward = await efficiency_reward.calculate_reward(
            test_problem, test_code, test_cases
        )
        print(f"✅ EfficiencyReward: {reward}")
    except Exception as e:
        print(f"❌ EfficiencyReward failed: {e}")
    
    # Test CorrectnessReward  
    print("Testing CorrectnessReward...")
    correctness_reward = CorrectnessReward(weight=1.0)
    try:
        reward = await correctness_reward.calculate_reward(
            test_problem, test_code, test_cases
        )
        print(f"✅ CorrectnessReward: {reward}")
    except Exception as e:
        print(f"❌ CorrectnessReward failed: {e}")


async def main():
    """Run all tests."""
    print("🚀 Testing VERL Training Fixes")
    print("=" * 50)
    
    try:
        await test_code_extraction()
        await test_reward_functions()
        print("\n✅ All tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())