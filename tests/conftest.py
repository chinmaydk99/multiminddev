"""
Pytest configuration and shared fixtures for the test suite.

This module provides common fixtures, test configuration, and utilities
used across all test modules in the framework.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, AsyncGenerator
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
from pydantic import BaseModel

from src.coding_framework.utils.config import Config, LLMConfig, AgentConfig, WorkflowConfig
from src.coding_framework.utils.llm_interface import LLMInterface
from src.coding_framework.agents import (
    CodeGeneratorAgent,
    CodeReviewerAgent, 
    CodeExecutorAgent,
)
from src.coding_framework.orchestration import CodingSupervisor


# Configure pytest-asyncio
pytest_asyncio.asyncio_mode = "auto"


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    test_env = {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "DEBUG": "true",
        "DEV_MODE": "true",
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    return test_env


@pytest.fixture
def test_config() -> Config:
    """Create a test configuration."""
    return Config(
        version="test",
        environment="testing",
        debug=True,
        llm=LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test-key",
            temperature=0.7,
            max_tokens=1000,
            timeout=30,
        ),
        workflow=WorkflowConfig(
            max_iterations=2,
            human_in_loop=False,
            target_review_score=70.0,
            min_execution_score=50.0,
        )
    )


@pytest.fixture
def mock_llm_interface():
    """Create a mock LLM interface for testing."""
    mock_interface = AsyncMock(spec=LLMInterface)
    
    # Mock successful responses
    mock_interface.call.return_value = "Mock LLM response for testing"
    mock_interface.call_with_retry.return_value = "Mock LLM response with retry"
    mock_interface.health_check.return_value = {
        "status": "healthy",
        "provider": "mock",
        "response_time": 0.1,
    }
    mock_interface.is_initialized = True
    
    return mock_interface


@pytest.fixture
async def mock_code_generator(test_config, mock_llm_interface):
    """Create a mock code generator agent."""
    agent = CodeGeneratorAgent(
        config=test_config.agents.generator,
        llm_interface=mock_llm_interface,
        agent_id="test-generator",
    )
    return agent


@pytest.fixture
async def mock_code_reviewer(test_config, mock_llm_interface):
    """Create a mock code reviewer agent."""
    agent = CodeReviewerAgent(
        config=test_config.agents.reviewer,
        llm_interface=mock_llm_interface,
        agent_id="test-reviewer",
    )
    return agent


@pytest.fixture
async def mock_code_executor(test_config, mock_llm_interface):
    """Create a mock code executor agent."""
    agent = CodeExecutorAgent(
        config=test_config.agents.executor,
        llm_interface=mock_llm_interface,
        agent_id="test-executor",
    )
    # Disable Docker for tests by default
    agent.docker_available = False
    return agent


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return """
def fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def main():
    print(fibonacci(10))

if __name__ == "__main__":
    main()
"""


@pytest.fixture
def sample_problem():
    """Sample coding problem for testing."""
    return "Write a function to calculate the nth Fibonacci number"


@pytest.fixture
def sample_code_review():
    """Sample code review result."""
    return {
        "overall_score": 75.0,
        "issues": [
            {
                "description": "Recursive implementation may be inefficient for large n",
                "severity": "medium",
                "line_number": 4,
                "category": "performance",
            }
        ],
        "suggestions": [
            "Consider using iterative approach or memoization for better performance"
        ],
        "strengths": [
            "Clear function documentation",
            "Proper edge case handling",
        ],
    }


@pytest.fixture
def sample_execution_result():
    """Sample code execution result."""
    return {
        "success": True,
        "output": "55\n",
        "error": "",
        "exit_code": 0,
        "execution_time": 0.125,
    }


@pytest.fixture
async def mock_supervisor(test_config):
    """Create a mock supervisor for testing."""
    supervisor = Mock(spec=CodingSupervisor)
    supervisor.config = test_config
    supervisor.is_initialized = True
    
    # Mock methods
    supervisor.initialize = AsyncMock()
    supervisor.solve_problem = AsyncMock(return_value={
        "success": True,
        "code": "def test(): return 42",
        "review": "Code looks good",
        "execution": "Output: 42",
        "iterations": 1,
    })
    supervisor.health_check = AsyncMock(return_value={
        "system": {"status": "healthy"},
        "generator": {"status": "healthy"},
        "reviewer": {"status": "healthy"},
        "executor": {"status": "healthy"},
    })
    
    return supervisor


@pytest.fixture
def docker_available():
    """Check if Docker is available for testing."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


class MockResponse:
    """Mock response class for testing."""
    
    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.metadata = metadata or {}


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI response."""
    return MockResponse(
        content="# Mock Python function\ndef mock_function():\n    return 'Hello, World!'"
    )


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic response."""
    return MockResponse(
        content="Here's a Python function:\n\n```python\ndef example():\n    return 42\n```"
    )


# Pytest markers for test categorization
pytest_markers = [
    "unit: Unit tests for individual components",
    "integration: Integration tests for component interactions",
    "e2e: End-to-end tests for complete workflows", 
    "slow: Tests that take longer to run",
    "docker: Tests that require Docker",
    "llm: Tests that require LLM API access",
]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    for marker in pytest_markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add markers based on test name
        if "docker" in item.name.lower():
            item.add_marker(pytest.mark.docker)
        if "llm" in item.name.lower() or "openai" in item.name.lower() or "anthropic" in item.name.lower():
            item.add_marker(pytest.mark.llm)
        if "slow" in item.name.lower() or "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.slow)


# Helper functions for tests
def assert_valid_python_code(code: str) -> bool:
    """Assert that a string contains valid Python code."""
    try:
        compile(code, '<test>', 'exec')
        return True
    except SyntaxError:
        return False


def assert_contains_function(code: str, function_name: str) -> bool:
    """Assert that code contains a function with the given name."""
    return f"def {function_name}" in code


def create_test_problem(
    description: str,
    language: str = "python",
    difficulty: str = "easy"
) -> Dict[str, Any]:
    """Create a test problem dictionary."""
    return {
        "description": description,
        "language": language,
        "difficulty": difficulty,
        "expected_functions": [],
        "test_cases": [],
    }


# Skip conditions
skip_if_no_docker = pytest.mark.skipif(
    not pytest.importorskip("docker", reason="Docker not available"),
    reason="Docker not available"
)

skip_if_no_api_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"),
    reason="No LLM API keys available"
)

skip_if_slow = pytest.mark.skipif(
    not os.getenv("RUN_SLOW_TESTS"),
    reason="Slow tests disabled (set RUN_SLOW_TESTS=1 to enable)"
)