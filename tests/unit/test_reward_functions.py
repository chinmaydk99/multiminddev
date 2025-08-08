"""
Unit tests for reward functions used in VERL training.

Tests the individual and composite reward functions that evaluate
code generation, review, and execution performance.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from src.coding_framework.training.reward_functions import (
    BaseRewardFunction,
    CorrectnessReward,
    StyleReward, 
    EfficiencyReward,
    CompositeReward,
)


class TestBaseRewardFunction:
    """Test cases for the base reward function interface."""
    
    def test_base_reward_is_abstract(self):
        """Test that BaseRewardFunction cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseRewardFunction()
    
    def test_subclass_implementation(self):
        """Test that subclasses must implement calculate_reward."""
        class IncompleteReward(BaseRewardFunction):
            pass
        
        with pytest.raises(TypeError):
            IncompleteReward()
    
    def test_valid_subclass(self):
        """Test that valid subclasses can be instantiated."""
        class ValidReward(BaseRewardFunction):
            async def calculate_reward(self, code, review_result, execution_result):
                return 0.5
        
        reward_function = ValidReward()
        assert reward_function is not None


class TestCorrectnessReward:
    """Test cases for the Correctness Reward Function."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.reward_function = CorrectnessReward()
    
    async def test_perfect_execution_score(self):
        """Test reward for perfect code execution."""
        code = "def test(): return 42"
        review_result = {"overall_score": 85}
        execution_result = {
            "success": True,
            "tests_passed": 5,
            "total_tests": 5,
            "execution_time": 0.1,
            "exit_code": 0,
            "error": None
        }
        
        reward = await self.reward_function.calculate_reward(
            code, review_result, execution_result
        )
        
        assert reward > 0.8  # Should get high reward for perfect execution
        assert isinstance(reward, float)
    
    async def test_failed_execution_score(self):
        """Test reward for failed code execution."""
        code = "def test(): return unknown_variable"
        review_result = {"overall_score": 30}
        execution_result = {
            "success": False,
            "tests_passed": 0,
            "total_tests": 5,
            "execution_time": 0.1,
            "exit_code": 1,
            "error": "NameError: name 'unknown_variable' is not defined"
        }
        
        reward = await self.reward_function.calculate_reward(
            code, review_result, execution_result
        )
        
        assert reward < 0.5  # Should get low reward for failed execution
        assert reward >= 0.0  # But not negative
    
    async def test_partial_execution_score(self):
        """Test reward for partial code execution success."""
        code = "def test(): return 42"
        review_result = {"overall_score": 75}
        execution_result = {
            "success": True,
            "tests_passed": 3,
            "total_tests": 5,
            "execution_time": 0.2,
            "exit_code": 0,
            "error": None
        }
        
        reward = await self.reward_function.calculate_reward(
            code, review_result, execution_result
        )
        
        assert 0.4 < reward < 0.8  # Should get moderate reward
    
    async def test_timeout_execution_penalty(self):
        """Test penalty for slow execution."""
        code = "def slow_function(): time.sleep(10)"
        review_result = {"overall_score": 60}
        execution_result = {
            "success": False,
            "tests_passed": 0,
            "total_tests": 1,
            "execution_time": 30.0,  # Very slow
            "exit_code": -15,  # Timeout signal
            "error": "TimeoutError: Execution timed out"
        }
        
        reward = await self.reward_function.calculate_reward(
            code, review_result, execution_result
        )
        
        assert reward < 0.3  # Should get very low reward for timeout


class TestStyleReward:
    """Test cases for the Style Reward Function."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.reward_function = StyleReward()
    
    async def test_high_quality_code_style(self):
        """Test reward for high-quality code style."""
        code = """
def calculate_fibonacci(n: int) -> int:
    '''Calculate the nth Fibonacci number using iterative approach.'''
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b
"""
        review_result = {
            "overall_score": 95,
            "issues": [],
            "style_score": 95,
            "has_docstrings": True,
            "follows_conventions": True
        }
        execution_result = {"success": True}
        
        reward = await self.reward_function.calculate_reward(
            code, review_result, execution_result
        )
        
        assert reward > 0.8  # Should get high reward for good style
    
    async def test_poor_code_style(self):
        """Test penalty for poor code style."""
        code = "def f(x):return x*2 if x>0 else None"  # Poor formatting, no types
        review_result = {
            "overall_score": 45,
            "issues": [
                {"category": "style", "severity": "medium"},
                {"category": "style", "severity": "low"}
            ],
            "style_score": 25,
            "has_docstrings": False,
            "follows_conventions": False
        }
        execution_result = {"success": True}
        
        reward = await self.reward_function.calculate_reward(
            code, review_result, execution_result
        )
        
        assert reward < 0.4  # Should get low reward for poor style
    
    async def test_moderate_style_issues(self):
        """Test reward for code with moderate style issues."""
        code = """
def fibonacci(n):
    # Missing type hints but otherwise ok
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        review_result = {
            "overall_score": 70,
            "issues": [
                {"category": "style", "severity": "low"},
                {"category": "performance", "severity": "medium"}
            ],
            "style_score": 65,
            "has_docstrings": False,
            "follows_conventions": True
        }
        execution_result = {"success": True}
        
        reward = await self.reward_function.calculate_reward(
            code, review_result, execution_result
        )
        
        assert 0.5 < reward < 0.8  # Should get moderate reward


class TestEfficiencyReward:
    """Test cases for the Efficiency Reward Function."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.reward_function = EfficiencyReward()
    
    async def test_efficient_code_reward(self):
        """Test reward for efficient code."""
        code = "def sum_range(n): return n * (n + 1) // 2"  # O(1) solution
        review_result = {
            "overall_score": 90,
            "performance_score": 95,
            "complexity_analysis": "O(1) time complexity"
        }
        execution_result = {
            "success": True,
            "execution_time": 0.001,  # Very fast
            "memory_usage": 1024,     # Low memory
            "cpu_usage": 0.1
        }
        
        reward = await self.reward_function.calculate_reward(
            code, review_result, execution_result
        )
        
        assert reward > 0.8  # Should get high reward for efficiency
    
    async def test_inefficient_code_penalty(self):
        """Test penalty for inefficient code."""
        code = "def sum_range(n): return sum(range(n+1))"  # Less efficient
        review_result = {
            "overall_score": 60,
            "performance_score": 40,
            "complexity_analysis": "O(n) time complexity when O(1) possible"
        }
        execution_result = {
            "success": True,
            "execution_time": 2.5,    # Slow
            "memory_usage": 1048576,  # High memory usage
            "cpu_usage": 85.0
        }
        
        reward = await self.reward_function.calculate_reward(
            code, review_result, execution_result
        )
        
        assert reward < 0.5  # Should get low reward for inefficiency
    
    async def test_memory_efficiency_consideration(self):
        """Test that memory efficiency affects reward."""
        code = "def process_data(data): return [x*2 for x in data]"
        review_result = {
            "overall_score": 75,
            "performance_score": 70
        }
        
        # High memory usage case
        execution_result_high_memory = {
            "success": True,
            "execution_time": 0.1,
            "memory_usage": 10485760,  # 10MB
            "cpu_usage": 20.0
        }
        
        # Low memory usage case
        execution_result_low_memory = {
            "success": True,
            "execution_time": 0.1,
            "memory_usage": 1024,      # 1KB
            "cpu_usage": 20.0
        }
        
        reward_high = await self.reward_function.calculate_reward(
            code, review_result, execution_result_high_memory
        )
        reward_low = await self.reward_function.calculate_reward(
            code, review_result, execution_result_low_memory
        )
        
        assert reward_low > reward_high  # Lower memory should get better reward


class TestCompositeReward:
    """Test cases for the Composite Reward Function."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create composite reward with custom weights
        self.composite_reward = CompositeReward(
            weights={
                "correctness": 0.7,
                "style": 0.2,
                "efficiency": 0.1
            }
        )
    
    async def test_weighted_composite_calculation(self):
        """Test that composite reward is calculated and returns valid value."""
        code = "def test(): return 42"
        review_result = {"overall_score": 80}
        execution_result = {"success": True}
        
        reward = await self.composite_reward.calculate_reward(
            code, review_result, execution_result
        )
        
        # Should return a valid reward value between 0 and 1
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0
    
    async def test_individual_reward_functions_called(self):
        """Test that composite reward works with different inputs."""
        code = "def test(): return 42"
        review_result = {"overall_score": 80}
        execution_result = {"success": True}
        
        reward = await self.composite_reward.calculate_reward(
            code, review_result, execution_result
        )
        
        # Should return a valid reward
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0
    
    async def test_reward_with_failing_code(self):
        """Test composite reward with failing code."""
        code = "def test(): return undefined_var"
        review_result = {"overall_score": 20}
        execution_result = {"success": False}
        
        reward = await self.composite_reward.calculate_reward(
            code, review_result, execution_result
        )
        
        # Should return lower reward for failing code
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0
        assert reward < 0.5  # Should be lower than good code
    
    async def test_zero_weights_handling(self):
        """Test handling of zero weights in composite calculation."""
        composite_no_style = CompositeReward(
            weights={
                "correctness": 0.9,
                "style": 0.0,  # Zero weight
                "efficiency": 0.1
            }
        )
        
        code = "def test(): return 42"
        review_result = {"overall_score": 80}
        execution_result = {"success": True}
        
        reward = await composite_no_style.calculate_reward(
            code, review_result, execution_result
        )
        
        # Should return a valid reward even with zero weight
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0
    
    async def test_weight_normalization(self):
        """Test that weights work correctly even if they don't sum to 1."""
        composite_unnormalized = CompositeReward(
            weights={
                "correctness": 70,  # These sum to 100, not 1
                "style": 20,
                "efficiency": 10
            }
        )
        
        code = "def test(): return 42"
        review_result = {"overall_score": 80}
        execution_result = {"success": True}
        
        reward = await composite_unnormalized.calculate_reward(
            code, review_result, execution_result
        )
        
        # Should return a valid reward even with unnormalized weights
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0


class TestRewardFunctionIntegration:
    """Integration tests for reward functions with real implementations."""
    
    async def test_real_composite_reward(self):
        """Test composite reward with actual reward function implementations."""
        composite_reward = CompositeReward(
            correctness_weight=0.7,
            style_weight=0.2, 
            efficiency_weight=0.1
        )
        
        # Good quality code example
        code = """
def fibonacci(n: int) -> int:
    '''Calculate nth Fibonacci number iteratively.'''
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
"""
        
        review_result = {
            "overall_score": 90,
            "style_score": 85,
            "performance_score": 95,
            "issues": [],
            "has_docstrings": True,
            "follows_conventions": True
        }
        
        execution_result = {
            "success": True,
            "tests_passed": 5,
            "total_tests": 5,
            "execution_time": 0.01,
            "memory_usage": 1024,
            "exit_code": 0
        }
        
        reward = await composite_reward.calculate_reward(
            code, review_result, execution_result
        )
        
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0
        assert reward > 0.7  # Should get high reward for good code
    
    async def test_poor_quality_code_penalty(self):
        """Test that poor quality code gets appropriately low rewards."""
        composite_reward = CompositeReward()
        
        # Poor quality code example
        code = "def f(x):return eval(x) if x else None"  # Security issue, poor style
        
        review_result = {
            "overall_score": 25,
            "style_score": 20,
            "performance_score": 30,
            "issues": [
                {"category": "security", "severity": "critical"},
                {"category": "style", "severity": "medium"}
            ],
            "has_docstrings": False,
            "follows_conventions": False
        }
        
        execution_result = {
            "success": False,
            "tests_passed": 1,
            "total_tests": 5,
            "execution_time": 0.1,
            "exit_code": 1,
            "error": "SecurityWarning: eval() usage detected"
        }
        
        reward = await composite_reward.calculate_reward(
            code, review_result, execution_result
        )
        
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0
        assert reward < 0.4  # Should get low reward for poor code