"""
Integration tests for LangGraph workflow orchestration.

Tests the complete workflow coordination between agents including
state management, routing logic, and error handling.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from src.coding_framework.orchestration.workflows import (
    CodingWorkflow,
    WorkflowState,
)
from src.coding_framework.orchestration.supervisor import CodingSupervisor


class TestCodingWorkflow:
    """Integration tests for the CodingWorkflow class."""
    
    @pytest.fixture
    def workflow_config(self):
        """Create workflow configuration for testing."""
        return {
            "max_iterations": 3,
            "human_in_loop": False,
            "target_review_score": 80.0,
            "min_execution_score": 50.0,
            "human_feedback_threshold": 30.0,
            "generation": {"temperature": 0.7},
            "review": {"temperature": 0.3},
            "execution": {"timeout": 30},
        }
    
    @pytest.fixture
    def mock_agents(self, mock_code_generator, mock_code_reviewer, mock_code_executor):
        """Create mock agents dictionary."""
        return {
            "generator": mock_code_generator,
            "reviewer": mock_code_reviewer,
            "executor": mock_code_executor,
        }
    
    def test_workflow_initialization(self, mock_agents, workflow_config):
        """Test workflow initialization."""
        workflow = CodingWorkflow(mock_agents, workflow_config)
        
        assert workflow.agents == mock_agents
        assert workflow.config == workflow_config
        assert workflow.workflow is not None
    
    async def test_code_generation_step(self, mock_agents, workflow_config):
        """Test the code generation workflow step."""
        workflow = CodingWorkflow(mock_agents, workflow_config)
        
        # Mock successful generation
        mock_agents["generator"].process_request.return_value = AsyncMock()
        mock_agents["generator"].process_request.return_value.success = True
        mock_agents["generator"].process_request.return_value.content = "def test(): return 42"
        mock_agents["generator"].process_request.return_value.metadata = {"language": "python"}
        
        # Create initial state
        initial_state: WorkflowState = {
            "problem": "Write a test function",
            "context": {"language": "python"},
            "generated_code": None,
            "code_review": None,
            "execution_result": None,
            "generator_metadata": None,
            "reviewer_metadata": None,
            "executor_metadata": None,
            "current_step": "start",
            "next_agent": None,
            "iteration_count": 0,
            "max_iterations": 3,
            "review_score": None,
            "execution_success": None,
            "final_solution": None,
            "errors": [],
            "warnings": [],
            "requires_human_input": False,
            "human_feedback": None,
        }
        
        # Execute generation step
        result_state = await workflow._generate_code(initial_state)
        
        assert result_state["generated_code"] == "def test(): return 42"
        assert result_state["generator_metadata"]["language"] == "python"
        assert result_state["current_step"] == "generation_complete"
        assert mock_agents["generator"].process_request.called
    
    async def test_code_review_step(self, mock_agents, workflow_config):
        """Test the code review workflow step."""
        workflow = CodingWorkflow(mock_agents, workflow_config)
        
        # Mock successful review
        mock_agents["reviewer"].process_request.return_value = AsyncMock()
        mock_agents["reviewer"].process_request.return_value.success = True
        mock_agents["reviewer"].process_request.return_value.content = "Code looks good (Score: 85/100)"
        mock_agents["reviewer"].process_request.return_value.metadata = {"overall_score": 85.0}
        
        # Create state with generated code
        state_with_code: WorkflowState = {
            "problem": "Write a test function",
            "context": {"language": "python"},
            "generated_code": "def test(): return 42",
            "code_review": None,
            "execution_result": None,
            "generator_metadata": {"language": "python"},
            "reviewer_metadata": None,
            "executor_metadata": None,
            "current_step": "generation_complete",
            "next_agent": None,
            "iteration_count": 0,
            "max_iterations": 3,
            "review_score": None,
            "execution_success": None,
            "final_solution": None,
            "errors": [],
            "warnings": [],
            "requires_human_input": False,
            "human_feedback": None,
        }
        
        # Execute review step
        result_state = await workflow._review_code(state_with_code)
        
        assert "Code looks good" in result_state["code_review"]
        assert result_state["review_score"] == 85.0
        assert result_state["current_step"] == "review_complete"
        assert mock_agents["reviewer"].process_request.called
    
    async def test_code_execution_step(self, mock_agents, workflow_config):
        """Test the code execution workflow step."""
        workflow = CodingWorkflow(mock_agents, workflow_config)
        
        # Mock successful execution
        mock_agents["executor"].process_request.return_value = AsyncMock()
        mock_agents["executor"].process_request.return_value.success = True
        mock_agents["executor"].process_request.return_value.content = "Execution successful: 42"
        mock_agents["executor"].process_request.return_value.metadata = {"success": True}
        
        # Create state with generated code
        state_with_code: WorkflowState = {
            "problem": "Write a test function", 
            "context": {"language": "python"},
            "generated_code": "def test(): return 42",
            "code_review": "Code looks good",
            "execution_result": None,
            "generator_metadata": {"language": "python"},
            "reviewer_metadata": {"overall_score": 85.0},
            "executor_metadata": None,
            "current_step": "review_complete",
            "next_agent": None,
            "iteration_count": 0,
            "max_iterations": 3,
            "review_score": 85.0,
            "execution_success": None,
            "final_solution": None,
            "errors": [],
            "warnings": [],
            "requires_human_input": False,
            "human_feedback": None,
        }
        
        # Execute execution step
        result_state = await workflow._execute_code(state_with_code)
        
        assert "Execution successful" in result_state["execution_result"]
        assert result_state["execution_success"] is True
        assert result_state["current_step"] == "execution_complete"
        assert mock_agents["executor"].process_request.called
    
    async def test_supervisor_routing_logic(self, mock_agents, workflow_config):
        """Test workflow supervisor routing decisions."""
        workflow = CodingWorkflow(mock_agents, workflow_config)
        
        # Test initial routing
        initial_state: WorkflowState = {
            "problem": "Test problem",
            "context": {},
            "generated_code": None,
            "code_review": None,
            "execution_result": None,
            "generator_metadata": None,
            "reviewer_metadata": None,
            "executor_metadata": None,
            "current_step": "start",
            "next_agent": None,
            "iteration_count": 0,
            "max_iterations": 3,
            "review_score": None,
            "execution_success": None,
            "final_solution": None,
            "errors": [],
            "warnings": [],
            "requires_human_input": False,
            "human_feedback": None,
        }
        
        result_state = await workflow._supervise_workflow(initial_state)
        assert result_state["next_agent"] == "generator"
        
        # Test routing after generation
        after_generation: WorkflowState = initial_state.copy()
        after_generation["current_step"] = "generation_complete"
        after_generation["generated_code"] = "def test(): return 42"
        
        result_state = await workflow._supervise_workflow(after_generation)
        assert result_state["next_agent"] == "reviewer"
        
        # Test routing after review
        after_review: WorkflowState = after_generation.copy()
        after_review["current_step"] = "review_complete"
        after_review["code_review"] = "Good code"
        after_review["review_score"] = 85.0
        
        result_state = await workflow._supervise_workflow(after_review)
        assert result_state["next_agent"] == "executor"
    
    def test_should_execute_code_logic(self, mock_agents, workflow_config):
        """Test logic for determining if code should be executed."""
        workflow = CodingWorkflow(mock_agents, workflow_config)
        
        # Test normal case - should execute
        normal_state: WorkflowState = {
            "problem": "Test",
            "context": {},
            "generated_code": "def test(): return 42",
            "code_review": None,
            "execution_result": None,
            "generator_metadata": None,
            "reviewer_metadata": {"overall_score": 80.0},
            "executor_metadata": None,
            "current_step": "review_complete",
            "next_agent": None,
            "iteration_count": 0,
            "max_iterations": 3,
            "review_score": 80.0,
            "execution_success": None,
            "final_solution": None,
            "errors": [],
            "warnings": [],
            "requires_human_input": False,
            "human_feedback": None,
        }
        
        assert workflow._should_execute_code(normal_state) is True
        
        # Test security concerns - should not execute
        security_state = normal_state.copy()
        security_state["reviewer_metadata"] = {
            "overall_score": 80.0,
            "security_concerns": ["Potential code injection vulnerability"]
        }
        
        assert workflow._should_execute_code(security_state) is False
        
        # Test low score - should not execute
        low_score_state = normal_state.copy()
        low_score_state["review_score"] = 30.0
        low_score_state["reviewer_metadata"] = {"overall_score": 30.0}
        
        assert workflow._should_execute_code(low_score_state) is False
    
    def test_needs_iteration_logic(self, mock_agents, workflow_config):
        """Test logic for determining if another iteration is needed."""
        workflow = CodingWorkflow(mock_agents, workflow_config)
        
        # Test successful execution with good score - no iteration needed
        success_state: WorkflowState = {
            "problem": "Test",
            "context": {},
            "generated_code": "def test(): return 42",
            "code_review": "Good code",
            "execution_result": "Success: 42",
            "generator_metadata": None,
            "reviewer_metadata": None,
            "executor_metadata": None,
            "current_step": "execution_complete",
            "next_agent": None,
            "iteration_count": 1,
            "max_iterations": 3,
            "review_score": 85.0,
            "execution_success": True,
            "final_solution": None,
            "errors": [],
            "warnings": [],
            "requires_human_input": False,
            "human_feedback": None,
        }
        
        assert workflow._needs_iteration(success_state) is False
        
        # Test failed execution - iteration needed
        failed_state = success_state.copy()
        failed_state["execution_success"] = False
        failed_state["iteration_count"] = 1
        
        assert workflow._needs_iteration(failed_state) is True
        
        # Test failed execution at max iterations - no iteration
        failed_at_max = failed_state.copy()
        failed_at_max["iteration_count"] = 2  # max_iterations - 1
        
        assert workflow._needs_iteration(failed_at_max) is False
    
    async def test_complete_workflow_execution(self, mock_agents, workflow_config):
        """Test complete workflow execution from start to finish."""
        workflow = CodingWorkflow(mock_agents, workflow_config)
        
        # Mock all agent responses
        mock_agents["generator"].process_request.return_value = AsyncMock()
        mock_agents["generator"].process_request.return_value.success = True
        mock_agents["generator"].process_request.return_value.content = "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
        mock_agents["generator"].process_request.return_value.metadata = {"language": "python"}
        
        mock_agents["reviewer"].process_request.return_value = AsyncMock()
        mock_agents["reviewer"].process_request.return_value.success = True
        mock_agents["reviewer"].process_request.return_value.content = "Good implementation (Score: 85/100)"
        mock_agents["reviewer"].process_request.return_value.metadata = {"overall_score": 85.0}
        
        mock_agents["executor"].process_request.return_value = AsyncMock()
        mock_agents["executor"].process_request.return_value.success = True
        mock_agents["executor"].process_request.return_value.content = "Execution successful: 55"
        mock_agents["executor"].process_request.return_value.metadata = {"success": True}
        
        # Run complete workflow
        result = await workflow.run(
            "Write a Fibonacci function",
            {"language": "python", "max_iterations": 2}
        )
        
        # Verify final result
        assert result["success"] is True
        assert "fibonacci" in result["code"]
        assert result["review"] == "Good implementation (Score: 85/100)"
        assert result["execution"] == "Execution successful: 55"
        assert result["review_score"] == 85.0
        assert result["execution_success"] is True
        assert result["iterations"] >= 0


class TestCodingSupervisor:
    """Integration tests for the CodingSupervisor class."""
    
    async def test_supervisor_initialization(self, test_config):
        """Test supervisor initialization process."""
        supervisor = CodingSupervisor(test_config)
        
        assert supervisor.config == test_config
        assert supervisor.is_initialized is False
        assert len(supervisor.agents) == 0
    
    async def test_supervisor_agent_initialization(self, test_config):
        """Test supervisor agent initialization."""
        supervisor = CodingSupervisor(test_config)
        
        # Mock LLM interface initialization
        with patch.object(supervisor, '_initialize_llm_interface') as mock_llm_init:
            mock_llm_init.return_value = None
            supervisor.llm_interface = Mock()
            supervisor.llm_interface.initialize = AsyncMock()
            
            # Mock agent initialization
            with patch.object(supervisor, '_verify_agent_health') as mock_health:
                mock_health.return_value = None
                
                await supervisor._initialize_agents()
                
                assert len(supervisor.agents) == 3
                assert "generator" in supervisor.agents
                assert "reviewer" in supervisor.agents
                assert "executor" in supervisor.agents
    
    async def test_health_check_integration(self, test_config):
        """Test supervisor health check integration."""
        supervisor = CodingSupervisor(test_config)
        
        # Mock initialized state
        supervisor.is_initialized = True
        supervisor.agents = {
            "generator": Mock(),
            "reviewer": Mock(),
            "executor": Mock(),
        }
        supervisor.llm_interface = Mock()
        
        # Mock agent health checks
        for agent in supervisor.agents.values():
            agent.health_check = AsyncMock(return_value={"status": "healthy"})
        
        supervisor.llm_interface.health_check = AsyncMock(
            return_value={"status": "healthy"}
        )
        
        health_result = await supervisor.health_check()
        
        assert health_result["system"]["status"] == "healthy"
        assert "generator" in health_result
        assert "reviewer" in health_result
        assert "executor" in health_result
        assert "llm_interface" in health_result
    
    async def test_problem_solving_integration(self, test_config):
        """Test complete problem solving integration."""
        supervisor = CodingSupervisor(test_config)
        
        # Mock initialization
        supervisor.is_initialized = True
        supervisor.workflow = Mock()
        supervisor.workflow.run = AsyncMock(return_value={
            "success": True,
            "code": "def solution(): return 42",
            "review": "Good code (Score: 85/100)",
            "execution": "Output: 42",
            "review_score": 85.0,
            "execution_success": True,
            "iterations": 1,
            "errors": [],
            "metadata": {},
        })
        
        result = await supervisor.solve_problem(
            "Write a function that returns 42",
            context={"language": "python"}
        )
        
        assert result["success"] is True
        assert "def solution" in result["code"]
        assert result["review_score"] == 85.0
        assert "execution_time" in result
        assert "timestamp" in result
        
        # Verify workflow was called
        supervisor.workflow.run.assert_called_once_with(
            "Write a function that returns 42",
            {"language": "python"}
        )
    
    async def test_individual_agent_operations(self, test_config):
        """Test individual agent operations through supervisor."""
        supervisor = CodingSupervisor(test_config)
        supervisor.is_initialized = True
        
        # Mock agents
        mock_agent = Mock()
        mock_agent.process_request = AsyncMock(return_value=Mock(
            success=True,
            content="Mock response",
            metadata={},
            dict=lambda: {"success": True, "content": "Mock response"}
        ))
        
        supervisor.agents = {
            "generator": mock_agent,
            "reviewer": mock_agent,
            "executor": mock_agent,
        }
        
        # Test code generation
        gen_result = await supervisor.generate_code("Test problem")
        assert gen_result["success"] is True
        assert gen_result["content"] == "Mock response"
        
        # Test code review
        review_result = await supervisor.review_code("def test(): pass")
        assert review_result["success"] is True
        assert review_result["content"] == "Mock response"
        
        # Test code execution
        exec_result = await supervisor.execute_code("print('hello')")
        assert exec_result["success"] is True
        assert exec_result["content"] == "Mock response"
    
    def test_performance_metrics_tracking(self, test_config):
        """Test performance metrics tracking."""
        supervisor = CodingSupervisor(test_config)
        
        # Test initial metrics
        assert supervisor.performance_metrics == {}
        
        # Mock workflow result
        mock_result = {
            "success": True,
            "review_score": 85.0,
            "iterations": 2,
        }
        
        # Update metrics
        supervisor._update_performance_metrics(mock_result, 15.5)
        
        metrics = supervisor.get_performance_metrics()
        assert metrics["total_problems_solved"] == 1
        assert metrics["successful_solutions"] == 1
        assert metrics["avg_execution_time"] == 15.5
        assert metrics["avg_review_score"] == 85.0
        assert metrics["avg_iterations"] == 2
        
        # Test metrics reset
        supervisor.reset_performance_metrics()
        assert supervisor.performance_metrics == {}


# Error handling integration tests
class TestErrorHandlingIntegration:
    """Test error handling across workflow components."""
    
    async def test_agent_failure_handling(self, mock_agents, workflow_config):
        """Test workflow handling of agent failures."""
        workflow = CodingWorkflow(mock_agents, workflow_config)
        
        # Mock generator failure
        mock_agents["generator"].process_request.side_effect = Exception("Generator failed")
        
        # Create initial state
        initial_state: WorkflowState = {
            "problem": "Test problem",
            "context": {},
            "generated_code": None,
            "code_review": None,
            "execution_result": None,
            "generator_metadata": None,
            "reviewer_metadata": None,
            "executor_metadata": None,
            "current_step": "start",
            "next_agent": None,
            "iteration_count": 0,
            "max_iterations": 3,
            "review_score": None,
            "execution_success": None,
            "final_solution": None,
            "errors": [],
            "warnings": [],
            "requires_human_input": False,
            "human_feedback": None,
        }
        
        # Execute generation step with failure
        result_state = await workflow._generate_code(initial_state)
        
        assert len(result_state["errors"]) > 0
        assert "Generator failed" in result_state["errors"][0]
        assert result_state["generated_code"] is None
    
    async def test_workflow_error_recovery(self, mock_agents, workflow_config):
        """Test workflow error recovery mechanisms."""
        workflow = CodingWorkflow(mock_agents, workflow_config)
        
        # Mock partial failure - generation succeeds, review fails
        mock_agents["generator"].process_request.return_value = AsyncMock()
        mock_agents["generator"].process_request.return_value.success = True
        mock_agents["generator"].process_request.return_value.content = "def test(): return 42"
        mock_agents["generator"].process_request.return_value.metadata = {}
        
        mock_agents["reviewer"].process_request.side_effect = Exception("Review failed")
        mock_agents["executor"].process_request.return_value = AsyncMock()
        mock_agents["executor"].process_request.return_value.success = True
        mock_agents["executor"].process_request.return_value.content = "Execution: 42"
        mock_agents["executor"].process_request.return_value.metadata = {"success": True}
        
        # Run workflow with partial failure
        result = await workflow.run("Test problem", {"language": "python"})
        
        # Should still return a result with errors recorded
        assert result["success"] is False  # Overall failure due to review error
        assert "error" in result or len(result.get("errors", [])) > 0
    
    async def test_supervisor_error_handling(self, test_config):
        """Test supervisor error handling and graceful degradation."""
        supervisor = CodingSupervisor(test_config)
        
        # Test uninitialized supervisor
        result = await supervisor.solve_problem("Test problem")
        # Should handle initialization automatically or return appropriate error
        assert "success" in result