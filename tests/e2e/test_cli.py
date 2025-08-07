"""
End-to-end tests for the CLI interface.

Tests the complete user interaction flow through the command-line interface
including problem solving, code review, and execution workflows.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from click.testing import CliRunner

from src.coding_framework.cli import main


class TestCLIEndToEnd:
    """End-to-end tests for CLI functionality."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.runner = CliRunner()
        self.temp_dir = None
    
    def teardown_method(self):
        """Clean up after each test."""
        if self.temp_dir:
            self.temp_dir.cleanup()
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory with test configuration."""
        self.temp_dir = tempfile.TemporaryDirectory()
        config_dir = Path(self.temp_dir.name)
        
        # Create minimal test config
        config_content = """
version: "test"
environment: "testing"
debug: true

llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
  api_key: "test-key"
  temperature: 0.7

agents:
  generator:
    temperature: 0.7
  reviewer:
    temperature: 0.3
  executor:
    execution_timeout: 30

workflow:
  max_iterations: 2
  human_in_loop: false
"""
        
        config_path = config_dir / "test_config.yaml"
        config_path.write_text(config_content)
        
        return config_dir
    
    def test_cli_main_help(self):
        """Test CLI main help command."""
        result = self.runner.invoke(main, ["--help"])
        
        assert result.exit_code == 0
        assert "Multi-Agent Coding Framework" in result.output
        assert "solve" in result.output
        assert "health" in result.output
    
    def test_cli_solve_help(self):
        """Test CLI solve command help."""
        result = self.runner.invoke(main, ["solve", "--help"])
        
        assert result.exit_code == 0
        assert "Solve a coding problem" in result.output
        assert "--language" in result.output
        assert "--include-tests" in result.output
    
    @patch('src.coding_framework.cli.CodingSupervisor')
    def test_solve_command_basic(self, mock_supervisor_class, temp_config_dir):
        """Test basic solve command functionality."""
        # Mock supervisor
        mock_supervisor = AsyncMock()
        mock_supervisor.initialize = AsyncMock()
        mock_supervisor.solve_problem = AsyncMock(return_value={
            "success": True,
            "code": "def reverse_string(s):\n    return s[::-1]",
            "review": "Good implementation (Score: 90/100)",
            "execution": "Test passed: 'hello' -> 'olleh'",
            "review_score": 90.0,
            "execution_success": True,
            "metrics": {"execution_time": 12.5}
        })
        mock_supervisor_class.return_value = mock_supervisor
        
        config_path = temp_config_dir / "test_config.yaml"
        
        result = self.runner.invoke(main, [
            "--config", str(config_path),
            "solve", 
            "Write a function to reverse a string",
            "--language", "python"
        ])
        
        assert result.exit_code == 0
        assert "def reverse_string" in result.output
        assert "Good implementation" in result.output
        mock_supervisor.solve_problem.assert_called_once()
    
    @patch('src.coding_framework.cli.CodingSupervisor')
    def test_solve_command_with_options(self, mock_supervisor_class, temp_config_dir):
        """Test solve command with various options."""
        mock_supervisor = AsyncMock()
        mock_supervisor.initialize = AsyncMock()
        mock_supervisor.solve_problem = AsyncMock(return_value={
            "success": True,
            "code": "# Implementation with tests\ndef fibonacci(n):\n    pass",
            "review": "Needs implementation",
            "execution": "Tests failed",
            "metrics": {}
        })
        mock_supervisor_class.return_value = mock_supervisor
        
        config_path = temp_config_dir / "test_config.yaml"
        
        result = self.runner.invoke(main, [
            "--config", str(config_path),
            "--verbose",
            "solve",
            "Implement fibonacci sequence", 
            "--language", "python",
            "--style", "clean",
            "--include-tests",
            "--focus-areas", "performance,correctness"
        ])
        
        assert result.exit_code == 0
        
        # Verify supervisor was called with correct context
        call_args = mock_supervisor.solve_problem.call_args
        assert call_args[0][0] == "Implement fibonacci sequence"
        context = call_args[1]["context"]
        assert context["language"] == "python"
        assert context["style"] == "clean"
        assert context["include_tests"] is True
        assert context["focus_areas"] == ["performance", "correctness"]
    
    def test_solve_command_missing_problem(self):
        """Test solve command with missing problem argument."""
        result = self.runner.invoke(main, ["solve"])
        
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "Error" in result.output
    
    @patch('src.coding_framework.cli.CodingSupervisor')
    def test_health_command(self, mock_supervisor_class, temp_config_dir):
        """Test health check command."""
        mock_supervisor = AsyncMock()
        mock_supervisor.initialize = AsyncMock()
        mock_supervisor.health_check = AsyncMock(return_value={
            "system": {"status": "healthy"},
            "generator": {"status": "healthy", "response_time": 0.1},
            "reviewer": {"status": "healthy", "response_time": 0.2}, 
            "executor": {"status": "unhealthy", "error": "Docker not available"},
            "llm_interface": {"status": "healthy", "response_time": 0.5}
        })
        mock_supervisor_class.return_value = mock_supervisor
        
        config_path = temp_config_dir / "test_config.yaml"
        
        result = self.runner.invoke(main, [
            "--config", str(config_path),
            "health"
        ])
        
        assert result.exit_code == 0
        assert "Agent Health Status" in result.output
        assert "generator" in result.output.lower()
        assert "reviewer" in result.output.lower()
        assert "executor" in result.output.lower()
    
    @patch('src.coding_framework.cli.CodingSupervisor')
    def test_health_command_specific_agent(self, mock_supervisor_class, temp_config_dir):
        """Test health check for specific agent."""
        mock_supervisor = AsyncMock()
        mock_supervisor.initialize = AsyncMock()
        mock_supervisor.check_agent_health = AsyncMock(return_value={
            "status": "healthy",
            "agent_type": "code_generator",
            "response_time": 0.15
        })
        mock_supervisor_class.return_value = mock_supervisor
        
        config_path = temp_config_dir / "test_config.yaml"
        
        result = self.runner.invoke(main, [
            "--config", str(config_path),
            "health",
            "--agent", "generator"
        ])
        
        assert result.exit_code == 0
        mock_supervisor.check_agent_health.assert_called_once_with("generator")
    
    @patch('src.coding_framework.cli.CodingSupervisor')
    def test_review_command(self, mock_supervisor_class, temp_config_dir):
        """Test code review command."""
        # Create temporary code file
        code_file = temp_config_dir / "test_code.py"
        code_file.write_text("""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""")
        
        mock_supervisor = AsyncMock()
        mock_supervisor.initialize = AsyncMock()
        mock_supervisor.review_code = AsyncMock(return_value={
            "success": True,
            "content": "Good recursive implementation (Score: 80/100)",
            "metadata": {"issues_found": 1, "overall_score": 80}
        })
        mock_supervisor_class.return_value = mock_supervisor
        
        config_path = temp_config_dir / "test_config.yaml"
        
        result = self.runner.invoke(main, [
            "--config", str(config_path),
            "review",
            str(code_file),
            "--focus", "performance,correctness",
            "--severity", "high"
        ])
        
        assert result.exit_code == 0
        assert "Good recursive implementation" in result.output
        mock_supervisor.review_code.assert_called_once()
    
    def test_review_command_nonexistent_file(self, temp_config_dir):
        """Test review command with non-existent file."""
        config_path = temp_config_dir / "test_config.yaml"
        
        result = self.runner.invoke(main, [
            "--config", str(config_path),
            "review",
            "nonexistent_file.py"
        ])
        
        assert result.exit_code != 0
    
    @patch('src.coding_framework.cli.CodingSupervisor')
    def test_execute_command(self, mock_supervisor_class, temp_config_dir):
        """Test code execution command."""
        # Create temporary code file
        code_file = temp_config_dir / "hello.py"
        code_file.write_text("print('Hello, World!')")
        
        mock_supervisor = AsyncMock()
        mock_supervisor.initialize = AsyncMock()
        mock_supervisor.execute_code = AsyncMock(return_value={
            "success": True,
            "content": "✅ SUCCESS\\nOutput: Hello, World!\\nExecution time: 0.125s",
            "metadata": {"success": True, "execution_time": 0.125}
        })
        mock_supervisor_class.return_value = mock_supervisor
        
        config_path = temp_config_dir / "test_config.yaml"
        
        result = self.runner.invoke(main, [
            "--config", str(config_path),
            "execute",
            str(code_file),
            "--language", "python",
            "--timeout", "30"
        ])
        
        assert result.exit_code == 0
        assert "SUCCESS" in result.output
        assert "Hello, World!" in result.output
        mock_supervisor.execute_code.assert_called_once()
    
    @patch('src.coding_framework.cli.CodingSupervisor')
    def test_train_command(self, mock_supervisor_class, temp_config_dir):
        """Test training command."""
        mock_supervisor = AsyncMock()
        mock_supervisor.initialize = AsyncMock()
        mock_supervisor.train_agents = AsyncMock(return_value={
            "success": True,
            "algorithm": "ppo",
            "episodes": 50,
            "metrics": {
                "final_reward": 0.85,
                "training_time": 1800,
                "convergence_episode": 35
            }
        })
        mock_supervisor_class.return_value = mock_supervisor
        
        config_path = temp_config_dir / "test_config.yaml"
        
        result = self.runner.invoke(main, [
            "--config", str(config_path),
            "train",
            "--algorithm", "ppo", 
            "--episodes", "50",
            "--wandb-project", "test-training"
        ])
        
        assert result.exit_code == 0
        assert "Training Results" in result.output
        mock_supervisor.train_agents.assert_called_once_with(
            algorithm="ppo",
            episodes=50
        )
    
    def test_invalid_config_file(self):
        """Test CLI with invalid configuration file."""
        result = self.runner.invoke(main, [
            "--config", "nonexistent_config.yaml",
            "solve", 
            "test problem"
        ])
        
        assert result.exit_code != 0
        assert "Configuration file not found" in result.output
    
    @patch('src.coding_framework.cli.CodingSupervisor')
    def test_solve_with_output_file(self, mock_supervisor_class, temp_config_dir):
        """Test solve command with output file."""
        mock_supervisor = AsyncMock()
        mock_supervisor.initialize = AsyncMock()
        mock_supervisor.solve_problem = AsyncMock(return_value={
            "success": True,
            "problem": "Write a hello function", 
            "code": "def hello(): return 'Hello!'",
            "review": "Simple but correct",
            "execution": "Output: Hello!",
            "metrics": {"execution_time": 5.2}
        })
        mock_supervisor_class.return_value = mock_supervisor
        
        config_path = temp_config_dir / "test_config.yaml"
        output_file = temp_config_dir / "results.json"
        
        result = self.runner.invoke(main, [
            "--config", str(config_path),
            "solve",
            "Write a hello function",
            "--output", str(output_file)
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        
        # Verify output file contents
        with open(output_file) as f:
            output_data = json.load(f)
        
        assert output_data["problem"] == "Write a hello function"
        assert output_data["solution"] == "def hello(): return 'Hello!'"
        assert output_data["review"] == "Simple but correct"
    
    @patch('src.coding_framework.cli.CodingSupervisor')
    def test_error_handling_agent_failure(self, mock_supervisor_class, temp_config_dir):
        """Test CLI error handling when agent fails."""
        mock_supervisor = AsyncMock()
        mock_supervisor.initialize = AsyncMock()
        mock_supervisor.solve_problem = AsyncMock(return_value={
            "success": False,
            "error": "LLM API connection failed"
        })
        mock_supervisor_class.return_value = mock_supervisor
        
        config_path = temp_config_dir / "test_config.yaml"
        
        result = self.runner.invoke(main, [
            "--config", str(config_path),
            "solve",
            "test problem"
        ])
        
        # CLI should handle the error gracefully
        assert "LLM API connection failed" in result.output or "Error" in result.output
    
    @patch('src.coding_framework.cli.CodingSupervisor')
    def test_timeout_handling(self, mock_supervisor_class, temp_config_dir):
        """Test CLI timeout handling."""
        import asyncio
        
        async def timeout_side_effect(*args, **kwargs):
            await asyncio.sleep(2)  # Simulate long operation
            return {"success": True}
        
        mock_supervisor = AsyncMock()
        mock_supervisor.initialize = AsyncMock()
        mock_supervisor.solve_problem = AsyncMock(side_effect=timeout_side_effect)
        mock_supervisor_class.return_value = mock_supervisor
        
        config_path = temp_config_dir / "test_config.yaml"
        
        result = self.runner.invoke(main, [
            "--config", str(config_path),
            "solve",
            "test problem",
            "--timeout", "1"  # 1 second timeout
        ])
        
        # Should handle timeout gracefully
        assert "timed out" in result.output or result.exit_code != 0
    
    def test_verbose_mode(self, temp_config_dir):
        """Test verbose mode output."""
        config_path = temp_config_dir / "test_config.yaml"
        
        result = self.runner.invoke(main, [
            "--config", str(config_path),
            "--verbose",
            "--help"
        ])
        
        assert result.exit_code == 0
        # Should include the welcome panel in verbose mode
        assert "Multi-Agent Coding Framework" in result.output


class TestCLIIntegration:
    """Integration tests for CLI with real components."""
    
    def setup_method(self):
        """Set up for integration tests."""
        self.runner = CliRunner()
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.getenv("INTEGRATION_TESTS"),
        reason="Integration tests disabled"
    )
    def test_real_solve_command(self, temp_config_dir):
        """Test solve command with real components (requires API keys)."""
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("No LLM API keys available for integration test")
        
        config_path = temp_config_dir / "test_config.yaml"
        
        result = self.runner.invoke(main, [
            "--config", str(config_path),
            "solve",
            "Write a function to add two numbers",
            "--language", "python",
            "--timeout", "60"
        ], catch_exceptions=False)
        
        assert result.exit_code == 0
        assert "def" in result.output  # Should contain a function definition
    
    @pytest.mark.docker
    @pytest.mark.skipif(
        not os.getenv("DOCKER_TESTS"),
        reason="Docker tests disabled"
    )
    def test_real_execute_command(self, temp_config_dir):
        """Test execute command with real Docker execution."""
        # Create simple test file
        test_file = temp_config_dir / "simple.py"
        test_file.write_text("print('Integration test successful')")
        
        config_path = temp_config_dir / "test_config.yaml"
        
        result = self.runner.invoke(main, [
            "--config", str(config_path),
            "execute",
            str(test_file),
            "--language", "python"
        ])
        
        assert result.exit_code == 0
        assert "Integration test successful" in result.output


# Performance and stress tests
class TestCLIPerformance:
    """Performance tests for CLI operations."""
    
    def setup_method(self):
        """Set up for performance tests."""
        self.runner = CliRunner()
    
    @pytest.mark.slow
    def test_cli_startup_time(self, temp_config_dir):
        """Test CLI startup time."""
        import time
        
        config_path = temp_config_dir / "test_config.yaml"
        
        start_time = time.time()
        result = self.runner.invoke(main, [
            "--config", str(config_path),
            "--help"
        ])
        end_time = time.time()
        
        startup_time = end_time - start_time
        
        assert result.exit_code == 0
        assert startup_time < 2.0  # Should start in under 2 seconds
    
    @pytest.mark.slow
    @patch('src.coding_framework.cli.CodingSupervisor')
    def test_concurrent_commands(self, mock_supervisor_class, temp_config_dir):
        """Test handling of concurrent CLI commands."""
        import threading
        import time
        
        mock_supervisor = AsyncMock()
        mock_supervisor.initialize = AsyncMock()
        mock_supervisor.solve_problem = AsyncMock(side_effect=lambda *args, **kwargs: {
            "success": True,
            "code": f"def solution_{threading.current_thread().ident}(): pass",
            "review": "Good",
            "execution": "Success"
        })
        mock_supervisor_class.return_value = mock_supervisor
        
        config_path = temp_config_dir / "test_config.yaml"
        
        def run_solve_command(problem_id):
            return self.runner.invoke(main, [
                "--config", str(config_path),
                "solve",
                f"Problem {problem_id}"
            ])
        
        # Run multiple commands concurrently
        threads = []
        results = []
        
        for i in range(3):
            thread = threading.Thread(
                target=lambda i=i: results.append(run_solve_command(i))
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All should succeed
        for result in results:
            assert result.exit_code == 0