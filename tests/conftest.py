"""
Pytest configuration and shared fixtures for the test suite.

This module provides common fixtures, test configuration, and utilities
used across all test modules in the multi-turn RL framework.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock

import pytest
import pytest_asyncio

from src.coding_framework.utils.config import Config, AgentConfig, WorkflowConfig


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
def test_config() -> Config:
    """Create a test configuration."""
    return Config(
        version="test",
        environment="testing",
        debug=True,
    )


@pytest.fixture
def mock_trainable_cuda_generator():
    """Create a mock trainable CUDA generator for testing (without loading actual model)."""
    # Create a minimal mock that doesn't load heavy ML dependencies
    mock_agent = Mock()
    mock_agent.agent_id = "test-cuda-generator"
    mock_agent.agent_type = "generator"
    mock_agent.model_name = "test-model"
    return mock_agent


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names and paths."""
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
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
skip_if_slow = pytest.mark.skipif(
    not os.getenv("RUN_SLOW_TESTS"),
    reason="Slow tests disabled (set RUN_SLOW_TESTS=1 to enable)"
)

skip_if_no_cuda = pytest.mark.skipif(
    not os.getenv("CUDA_AVAILABLE"),
    reason="CUDA not available for testing"
)