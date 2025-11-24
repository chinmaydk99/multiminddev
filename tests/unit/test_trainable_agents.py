"""
Unit tests for trainable agent implementations.
Tests the multi-agent RL architecture with GRPO/DAPO support for HIP/ROCm.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.coding_framework.agents.trainable_agent import GenerationOutput, TrainableAgent
from src.coding_framework.agents.trainable_hip_agents import (
    TrainableHIPGeneratorAgent,
    TrainableHIPOptimizerAgent,
    TrainableHIPTesterAgent,
)
from src.coding_framework.training.multi_turn_conversation import (
    AgentType,
    ConversationTurn,
    HIPConversationState,
    MultiTurnConversationManager,
    TurnStatus,
)


class TestTrainableAgent:
    """Test base trainable agent functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock HuggingFace model."""
        mock = MagicMock()
        mock.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
        mock.train = MagicMock()
        mock.eval = MagicMock()
        mock.generate = MagicMock(return_value=MagicMock(
            sequences=torch.tensor([[1, 2, 3, 4, 5]]),
            scores=[torch.randn(1, 100) for _ in range(5)]
        ))
        return mock
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        mock = MagicMock()
        mock.pad_token_id = 0
        mock.eos_token_id = 1
        mock.return_value = MagicMock(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 1]])
        )
        mock.decode = MagicMock(return_value="Generated text")
        return mock
    
    @patch('src.coding_framework.agents.trainable_agent.AutoModelForCausalLM')
    @patch('src.coding_framework.agents.trainable_agent.AutoTokenizer')
    def test_agent_initialization(self, mock_tokenizer_class, mock_model_class, mock_model, mock_tokenizer):
        """Test trainable agent initialization."""
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        agent = TrainableAgent(
            agent_id="test_agent",
            agent_type="generator",
            model_name="test-model",
            device="cpu"
        )
        
        assert agent.agent_id == "test_agent"
        assert agent.agent_type == "generator"
        assert agent.model is not None
        assert agent.tokenizer is not None


class TestHIPAgents:
    """Test specialized HIP agents for AMD ROCm."""
    
    @pytest.mark.asyncio
    @patch('src.coding_framework.agents.trainable_agent.AutoModelForCausalLM')
    @patch('src.coding_framework.agents.trainable_agent.AutoTokenizer')
    async def test_generator_agent(self, mock_tokenizer_class, mock_model_class):
        """Test HIP generator agent."""
        mock_model_class.from_pretrained.return_value = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        
        agent = TrainableHIPGeneratorAgent(
            agent_id="hip_gen",
            model_name="test-model"
        )
        
        # Mock the generation
        with patch.object(agent, 'generate_with_log_probs') as mock_gen:
            mock_gen.return_value = GenerationOutput(
                text="__global__ void kernel() {}",
                token_ids=torch.tensor([1, 2, 3]),
                log_probs=torch.tensor([-0.1, -0.2, -0.3]),
                attention_mask=torch.ones(3)
            )
            
            result = await agent.generate_hip_kernel(
                "Create a simple HIP kernel",
                tensor_info={"shape": (1024,), "dtype": "float"}
            )
            
            assert "kernel_code" in result
            assert "__global__" in result["kernel_code"]
            assert result["is_valid_syntax"] is not None
    
    @pytest.mark.asyncio
    @patch('src.coding_framework.agents.trainable_agent.AutoModelForCausalLM')
    @patch('src.coding_framework.agents.trainable_agent.AutoTokenizer')
    async def test_optimizer_agent(self, mock_tokenizer_class, mock_model_class):
        """Test HIP optimizer agent."""
        mock_model_class.from_pretrained.return_value = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        
        agent = TrainableHIPOptimizerAgent(
            agent_id="hip_opt",
            model_name="test-model"
        )
        
        kernel_code = "__global__ void simple_kernel() {}"
        
        with patch.object(agent, 'generate_with_log_probs') as mock_gen:
            mock_gen.return_value = GenerationOutput(
                text="__global__ void optimized_kernel() { __shared__ float cache[256]; }",
                token_ids=torch.tensor([1, 2, 3]),
                log_probs=torch.tensor([-0.1, -0.2, -0.3]),
                attention_mask=torch.ones(3)
            )
            
            result = await agent.optimize_kernel(
                kernel_code,
                optimization_targets=["lds_memory"]
            )
            
            assert "optimized_code" in result
            assert "applied_optimizations" in result
            assert "optimization_score" in result
    
    @pytest.mark.asyncio
    async def test_tester_agent_rule_based(self):
        """Test rule-based HIP tester agent."""
        agent = TrainableHIPTesterAgent(
            agent_id="hip_tester",
            use_trained_model=False
        )
        
        kernel_code = """
        __global__ void test_kernel(float* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = data[idx] * 2.0f;
            }
        }
        """
        
        # Mock hipcc compilation
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="",
                stderr=""
            )
            
            result = await agent.test_kernel(kernel_code)
            
            assert "compilation" in result
            assert "performance" in result
            assert "test_report" in result


class TestHIPConversationState:
    """Test HIP conversation state management."""
    
    def test_conversation_state_creation(self):
        """Test creating a HIP conversation state."""
        state = HIPConversationState(
            conversation_id="test_001",
            problem_description="Optimize matrix multiplication",
            difficulty_tier="medium"
        )
        
        assert state.conversation_id == "test_001"
        assert state.problem_description == "Optimize matrix multiplication"
        assert state.difficulty_tier == "medium"
        assert len(state.turns) == 0
        assert state.final_reward == 0.0
    
    def test_conversation_turn_creation(self):
        """Test creating conversation turns."""
        turn = ConversationTurn(
            turn_number=0,
            agent_type=AgentType.GENERATOR,
            prompt="Generate kernel",
            response="__global__ void kernel() {}",
            status=TurnStatus.SUCCESS
        )
        
        assert turn.turn_number == 0
        assert turn.agent_type == AgentType.GENERATOR
        assert turn.status == TurnStatus.SUCCESS
    
    def test_agent_types(self):
        """Test all agent types are defined."""
        assert AgentType.GENERATOR.value == "generator"
        assert AgentType.OPTIMIZER.value == "optimizer"
        assert AgentType.TESTER.value == "tester"
    
    def test_turn_statuses(self):
        """Test all turn statuses are defined."""
        assert TurnStatus.SUCCESS.value == "success"
        assert TurnStatus.FAILURE.value == "failure"
        assert TurnStatus.PARTIAL.value == "partial"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
