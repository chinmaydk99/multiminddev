"""
Unit tests for trainable agent implementations.
Tests the multi-agent RL architecture with GRPO/DAPO support.
"""

import pytest
import torch
import asyncio
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

from src.coding_framework.agents.trainable_agent import TrainableAgent, GenerationOutput
from src.coding_framework.agents.trainable_cuda_agents import (
    TrainableCUDAGeneratorAgent,
    TrainableCUDAOptimizerAgent,
    TrainableCUDATesterAgent
)
from src.coding_framework.training.multi_turn_conversation import (
    MultiTurnConversationManager,
    TurnLevelRewardDistributor,
    CUDAConversationState,
    ConversationTurn,
    AgentRole
)
from src.coding_framework.training.sft_data_preparation import (
    SFTDataPipeline,
    CUDADatasetGenerator,
    SFTDataItem
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
    def test_agent_initialization(self, mock_tokenizer_class, mock_model_class):
        """Test trainable agent initialization."""
        mock_model_class.from_pretrained.return_value = self.mock_model()
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer()
        
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
        assert agent.optimizer is not None
    
    @pytest.mark.asyncio
    @patch('src.coding_framework.agents.trainable_agent.AutoModelForCausalLM')
    @patch('src.coding_framework.agents.trainable_agent.AutoTokenizer')
    async def test_generate_with_log_probs(self, mock_tokenizer_class, mock_model_class):
        """Test generation with log probability tracking."""
        mock_model_class.from_pretrained.return_value = self.mock_model()
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer()
        
        agent = TrainableAgent(
            agent_id="test_agent",
            agent_type="generator",
            model_name="test-model",
            device="cpu"
        )
        
        output = await agent.generate_with_log_probs(
            prompt="Test prompt",
            max_new_tokens=10
        )
        
        assert isinstance(output, GenerationOutput)
        assert output.text == "Generated text"
        assert output.token_ids is not None
        assert output.log_probs is not None
    
    def test_update_parameters(self):
        """Test parameter update with PPO."""
        with patch('src.coding_framework.agents.trainable_agent.AutoModelForCausalLM') as mock_model_class, \
             patch('src.coding_framework.agents.trainable_agent.AutoTokenizer') as mock_tokenizer_class:
            
            mock_model_class.from_pretrained.return_value = self.mock_model()
            mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer()
            
            agent = TrainableAgent(
                agent_id="test_agent",
                agent_type="generator",
                model_name="test-model",
                device="cpu"
            )
            
            # Test parameter update
            rewards = torch.tensor([0.5, 0.8, 0.3])
            log_probs = torch.tensor([-0.5, -0.2, -0.7])
            
            metrics = agent.update_parameters(rewards, log_probs)
            
            assert "policy_loss" in metrics
            assert "entropy" in metrics
            assert "total_loss" in metrics


class TestCUDAAgents:
    """Test specialized CUDA agents."""
    
    @pytest.mark.asyncio
    @patch('src.coding_framework.agents.trainable_agent.AutoModelForCausalLM')
    @patch('src.coding_framework.agents.trainable_agent.AutoTokenizer')
    async def test_generator_agent(self, mock_tokenizer_class, mock_model_class):
        """Test CUDA generator agent."""
        mock_model_class.from_pretrained.return_value = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        
        agent = TrainableCUDAGeneratorAgent(
            agent_id="cuda_gen",
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
            
            result = await agent.generate_cuda_kernel(
                "Create a simple CUDA kernel",
                tensor_info={"shape": (1024,), "dtype": "float"}
            )
            
            assert "kernel_code" in result
            assert "__global__" in result["kernel_code"]
            assert result["is_valid_syntax"] is not None
    
    @pytest.mark.asyncio
    @patch('src.coding_framework.agents.trainable_agent.AutoModelForCausalLM')
    @patch('src.coding_framework.agents.trainable_agent.AutoTokenizer')
    async def test_optimizer_agent(self, mock_tokenizer_class, mock_model_class):
        """Test CUDA optimizer agent."""
        mock_model_class.from_pretrained.return_value = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        
        agent = TrainableCUDAOptimizerAgent(
            agent_id="cuda_opt",
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
                optimization_targets=["shared_memory"]
            )
            
            assert "optimized_code" in result
            assert "applied_optimizations" in result
            assert "optimization_score" in result
    
    @pytest.mark.asyncio
    async def test_tester_agent_rule_based(self):
        """Test rule-based CUDA tester agent."""
        agent = TrainableCUDATesterAgent(
            agent_id="cuda_tester",
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
        
        # Mock nvcc compilation
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


class TestMultiTurnConversation:
    """Test multi-turn conversation management."""
    
    @pytest.fixture
    def conversation_state(self):
        """Create a sample conversation state."""
        state = CUDAConversationState(
            problem="Optimize matrix multiplication",
            problem_id="test_001",
            difficulty="medium"
        )
        return state
    
    def test_conversation_turn_creation(self, conversation_state):
        """Test adding turns to conversation."""
        turn = conversation_state.add_turn(
            agent_type=AgentRole.GENERATOR,
            input_text="Generate kernel",
            output_text="__global__ void kernel() {}",
            immediate_reward=0.5
        )
        
        assert turn.turn_id == 0
        assert turn.agent_type == AgentRole.GENERATOR
        assert turn.immediate_reward == 0.5
        assert conversation_state.num_turns == 1
    
    def test_performance_tracking(self, conversation_state):
        """Test performance history tracking."""
        conversation_state.update_kernel("kernel_v1", 1.0)
        conversation_state.update_kernel("kernel_v2", 1.5)
        conversation_state.update_kernel("kernel_v3", 2.0)
        
        assert conversation_state.current_performance == 2.0
        assert conversation_state.best_performance == 2.0
        assert conversation_state.best_kernel == "kernel_v3"
        assert len(conversation_state.performance_history) == 3
    
    def test_early_termination(self, conversation_state):
        """Test early termination conditions."""
        # Test max turns
        for i in range(5):
            conversation_state.add_turn(
                AgentRole.GENERATOR,
                f"input_{i}",
                f"output_{i}"
            )
        assert conversation_state.should_terminate_early() == True
        
        # Test performance threshold
        state2 = CUDAConversationState(
            problem="test",
            problem_id="test_002"
        )
        state2.update_kernel("kernel", 2.5)  # > 2.0 threshold
        assert state2.should_terminate_early() == True
    
    def test_reward_distribution(self):
        """Test turn-level reward distribution."""
        distributor = TurnLevelRewardDistributor(
            discount_factor=0.9,
            immediate_weight=0.3,
            final_weight=0.7
        )
        
        state = CUDAConversationState(
            problem="test",
            problem_id="test_003"
        )
        
        # Add some turns
        state.add_turn(AgentRole.GENERATOR, "gen", "kernel", immediate_reward=0.3)
        state.add_turn(AgentRole.TESTER, "test", "results", immediate_reward=0.0)
        state.add_turn(AgentRole.OPTIMIZER, "opt", "optimized", immediate_reward=0.5)
        state.final_reward = 1.0
        
        rewards = distributor.distribute_rewards(state)
        
        assert len(rewards) == 3
        assert rewards[1] == 0.0  # Tester gets no reward
        assert rewards[0] > 0  # Generator gets discounted reward
        assert rewards[2] > 0  # Optimizer gets reward


class TestSFTDataPreparation:
    """Test SFT data preparation pipeline."""
    
    def test_cuda_dataset_generator(self):
        """Test synthetic CUDA dataset generation."""
        generator = CUDADatasetGenerator()
        
        # Test generator examples
        gen_examples = generator.generate_generator_examples(100)
        assert len(gen_examples) == 100
        assert all(isinstance(ex, SFTDataItem) for ex in gen_examples)
        assert all("Generate CUDA kernel" in ex.input_text for ex in gen_examples)
        
        # Test optimizer examples
        opt_examples = generator.generate_optimizer_examples(50)
        assert len(opt_examples) == 50
        assert all("Optimize" in ex.input_text for ex in opt_examples)
    
    @pytest.mark.asyncio
    async def test_sft_pipeline(self):
        """Test complete SFT data pipeline."""
        pipeline = SFTDataPipeline(use_huggingface_data=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline.data_dir = Path(tmpdir)
            
            # Test generator data preparation
            dataset = await pipeline.prepare_generator_data(
                num_examples=10,
                save_path=Path("generator_data")
            )
            
            assert len(dataset) == 10
            assert "input" in dataset.column_names
            assert "output" in dataset.column_names
            
            # Check saved file exists
            saved_path = pipeline.data_dir / "generator_data"
            assert saved_path.exists()


class TestGRPOConfiguration:
    """Test GRPO/DAPO configuration handling."""
    
    def test_grpo_config(self):
        """Test GRPO-specific configuration."""
        from src.coding_framework.training.multi_turn_rl_trainer import MultiTurnRLConfig
        
        config = MultiTurnRLConfig(algorithm="grpo")
        
        assert config.algorithm == "grpo"
        assert config.grpo_group_size == 16
        assert config.grpo_kl_coef == 0.0
        assert config.grpo_clip_ratio_low == 0.2
        assert config.grpo_clip_ratio_high == 0.28
    
    def test_dapo_config(self):
        """Test DAPO-specific configuration."""
        from src.coding_framework.training.multi_turn_rl_trainer import MultiTurnRLConfig
        
        config = MultiTurnRLConfig(algorithm="dapo")
        
        assert config.algorithm == "dapo"
        assert config.dapo_use_kl_in_reward == False
        assert config.dapo_loss_agg_mode == "token-mean"
        assert config.dapo_overlong_penalty_factor == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])