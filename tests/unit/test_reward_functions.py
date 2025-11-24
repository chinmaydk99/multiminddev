"""
Unit tests for HIP performance reward functions.

Tests the reward function that evaluates HIP kernel generation and optimization
performance on AMD ROCm GPUs.
"""

import pytest
from unittest.mock import MagicMock

from src.coding_framework.training.reward_functions import (
    BaseRewardFunction,
    HIPPerformanceReward,
    RewardComponents,
)


class TestRewardComponents:
    """Test cases for RewardComponents dataclass."""
    
    def test_default_components(self):
        """Test default reward component values."""
        components = RewardComponents()
        
        assert components.compilation_success == 0.0
        assert components.functional_correctness == 0.0
        assert components.performance_score == 0.0
        assert components.efficiency_score == 0.0
        assert components.code_quality_score == 0.0
        assert components.improvement_bonus == 0.0
    
    def test_custom_components(self):
        """Test creating components with custom values."""
        components = RewardComponents(
            compilation_success=1.0,
            functional_correctness=0.8,
            performance_score=0.9,
            efficiency_score=0.7,
            code_quality_score=0.6
        )
        
        assert components.compilation_success == 1.0
        assert components.functional_correctness == 0.8
        assert components.performance_score == 0.9
    
    def test_total_score_calculation(self):
        """Test weighted total score calculation."""
        components = RewardComponents(
            compilation_success=1.0,
            functional_correctness=1.0,
            performance_score=1.0,
            efficiency_score=1.0,
            code_quality_score=1.0
        )
        
        weights = {
            "compilation": 0.3,
            "correctness": 0.3,
            "performance": 0.25,
            "efficiency": 0.1,
            "code_quality": 0.05
        }
        
        total = components.total_score(weights)
        
        # All components are 1.0, so total should be 1.0
        assert abs(total - 1.0) < 0.001
    
    def test_partial_score_calculation(self):
        """Test total score with partial component values."""
        components = RewardComponents(
            compilation_success=1.0,
            functional_correctness=0.5,
            performance_score=0.0,
            efficiency_score=0.0,
            code_quality_score=0.0
        )
        
        weights = {
            "compilation": 0.3,
            "correctness": 0.3,
            "performance": 0.25,
            "efficiency": 0.1,
            "code_quality": 0.05
        }
        
        total = components.total_score(weights)
        
        # 0.3 * 1.0 + 0.3 * 0.5 = 0.45
        expected = 0.3 + 0.15
        assert abs(total - expected) < 0.001


class TestHIPPerformanceReward:
    """Test cases for HIPPerformanceReward function."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.reward_function = HIPPerformanceReward(
            target_speedup=2.0,
            max_speedup_reward=4.0
        )
    
    def test_initialization(self):
        """Test reward function initialization."""
        assert self.reward_function.target_speedup == 2.0
        assert self.reward_function.max_speedup_reward == 4.0
        assert "compilation" in self.reward_function.reward_weights
        assert "correctness" in self.reward_function.reward_weights
    
    def test_custom_weights(self):
        """Test reward function with custom weights."""
        custom_weights = {
            "compilation": 0.4,
            "correctness": 0.4,
            "performance": 0.1,
            "efficiency": 0.05,
            "code_quality": 0.05
        }
        
        reward_fn = HIPPerformanceReward(reward_weights=custom_weights)
        
        assert reward_fn.reward_weights["compilation"] == 0.4
        assert reward_fn.reward_weights["performance"] == 0.1
    
    @pytest.mark.asyncio
    async def test_calculate_reward_successful_compilation(self):
        """Test reward calculation for successfully compiled kernel."""
        problem = {
            "description": "Vector addition kernel",
            "difficulty": "easy"
        }
        
        generated_code = """
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""
        
        # Mock successful compilation
        compilation_result = MagicMock()
        compilation_result.success = True
        compilation_result.compilation_warnings = []
        compilation_result.register_pressure = 32
        compilation_result.shared_memory_usage = 0
        
        # Mock successful benchmark
        benchmark_result = MagicMock()
        benchmark_result.success = True
        benchmark_result.functional_correct = True
        benchmark_result.speedup_vs_torch = 1.5
        benchmark_result.memory_bandwidth_gb_s = 500.0
        benchmark_result.compute_efficiency = 0.7
        benchmark_result.occupancy = 0.8
        benchmark_result.execution_time_ms = 0.1
        benchmark_result.numerical_accuracy = 0.9999
        
        context = {"previous_performance": 1.0}
        
        total_reward, components = await self.reward_function.calculate_reward(
            problem, generated_code, compilation_result, benchmark_result, context
        )
        
        assert isinstance(total_reward, float)
        assert 0.0 <= total_reward <= 1.0
        assert components.compilation_success > 0
    
    @pytest.mark.asyncio
    async def test_calculate_reward_failed_compilation(self):
        """Test reward calculation for failed compilation."""
        problem = {"description": "Test kernel"}
        generated_code = "invalid code {{"
        
        # Mock failed compilation
        compilation_result = MagicMock()
        compilation_result.success = False
        compilation_result.compilation_warnings = ["syntax error"]
        compilation_result.register_pressure = 0
        compilation_result.shared_memory_usage = 0
        
        # Mock no benchmark (compilation failed)
        benchmark_result = MagicMock()
        benchmark_result.success = False
        benchmark_result.functional_correct = False
        benchmark_result.speedup_vs_torch = 0.0
        
        context = {}
        
        total_reward, components = await self.reward_function.calculate_reward(
            problem, generated_code, compilation_result, benchmark_result, context
        )
        
        assert isinstance(total_reward, float)
        assert 0.0 <= total_reward <= 1.0
        # Failed compilation should result in low reward
        assert components.compilation_success < 0.5
    
    @pytest.mark.asyncio
    async def test_calculate_reward_high_performance(self):
        """Test reward calculation for high-performance kernel."""
        problem = {
            "description": "Matrix multiplication",
            "baseline_performance": {"speedup": 1.0}
        }
        
        generated_code = """
__global__ void matmul_tiled(float* A, float* B, float* C, int N) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    // Tiled implementation
}
"""
        
        # Mock successful compilation
        compilation_result = MagicMock()
        compilation_result.success = True
        compilation_result.compilation_warnings = []
        compilation_result.register_pressure = 48
        compilation_result.shared_memory_usage = 2048
        
        # Mock high-performance benchmark
        benchmark_result = MagicMock()
        benchmark_result.success = True
        benchmark_result.functional_correct = True
        benchmark_result.speedup_vs_torch = 3.0  # 3x speedup
        benchmark_result.memory_bandwidth_gb_s = 800.0
        benchmark_result.compute_efficiency = 0.85
        benchmark_result.occupancy = 0.9
        benchmark_result.execution_time_ms = 0.05
        benchmark_result.numerical_accuracy = 0.99999
        
        context = {"previous_performance": 1.5}
        
        total_reward, components = await self.reward_function.calculate_reward(
            problem, generated_code, compilation_result, benchmark_result, context
        )
        
        assert isinstance(total_reward, float)
        assert 0.0 <= total_reward <= 1.0
        # High performance should result in high reward
        assert total_reward > 0.5


class TestBaseRewardFunction:
    """Test cases for BaseRewardFunction abstract class."""
    
    def test_base_reward_is_abstract(self):
        """Test that BaseRewardFunction cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseRewardFunction()
    
    def test_valid_subclass(self):
        """Test that valid subclasses can be instantiated."""
        class ValidReward(BaseRewardFunction):
            async def calculate_reward(self, *args, **kwargs):
                return 0.5
        
        reward_function = ValidReward()
        assert reward_function is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
