"""
Unit tests for agent implementations.

Tests the core functionality of each specialized agent including
code generation, review, and execution capabilities.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from src.coding_framework.agents import (
    BaseAgent,
    CodeGeneratorAgent, 
    CodeReviewerAgent,
    CodeExecutorAgent,
)
from src.coding_framework.agents.base_agent import AgentResponse


class TestBaseAgent:
    """Test cases for the BaseAgent abstract class."""
    
    def test_agent_initialization(self, test_config, mock_llm_interface):
        """Test base agent initialization."""
        # Create a concrete implementation for testing
        class TestAgent(BaseAgent):
            @property
            def agent_type(self):
                return "test_agent"
            
            @property
            def system_prompt(self):
                return "Test system prompt"
            
            async def process_request(self, request, context=None, **kwargs):
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=True,
                    content="test response",
                    execution_time=0.1,
                )
        
        agent = TestAgent(
            config=test_config.agents.generator,
            llm_interface=mock_llm_interface,
            agent_id="test-001",
        )
        
        assert agent.agent_type == "test_agent"
        assert agent.agent_id == "test-001"
        assert agent.system_prompt == "Test system prompt"
        assert agent.state == {}
        assert agent.conversation_history == []
    
    def test_state_management(self, test_config, mock_llm_interface):
        """Test agent state management."""
        class TestAgent(BaseAgent):
            @property
            def agent_type(self):
                return "test_agent"
            
            @property
            def system_prompt(self):
                return "Test system prompt"
            
            async def process_request(self, request, context=None, **kwargs):
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=True,
                    content="test response",
                    execution_time=0.1,
                )
        
        agent = TestAgent(
            config=test_config.agents.generator,
            llm_interface=mock_llm_interface,
        )
        
        # Test state updates
        agent.update_state("test_key", "test_value")
        assert agent.get_state("test_key") == "test_value"
        assert agent.get_state("missing_key", "default") == "default"
    
    async def test_health_check(self, test_config, mock_llm_interface):
        """Test agent health check functionality."""
        class TestAgent(BaseAgent):
            @property
            def agent_type(self):
                return "test_agent"
            
            @property
            def system_prompt(self):
                return "Test system prompt"
            
            async def process_request(self, request, context=None, **kwargs):
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=True,
                    content="test response",
                    execution_time=0.1,
                )
        
        agent = TestAgent(
            config=test_config.agents.generator,
            llm_interface=mock_llm_interface,
        )
        
        health_status = await agent.health_check()
        
        assert health_status["status"] == "healthy"
        assert health_status["agent_type"] == "test_agent"
        assert "response_time" in health_status
        assert health_status["llm_responsive"] is True


class TestCodeGeneratorAgent:
    """Test cases for the Code Generator Agent."""
    
    async def test_agent_initialization(self, test_config, mock_llm_interface):
        """Test code generator agent initialization."""
        agent = CodeGeneratorAgent(
            config=test_config.agents.generator,
            llm_interface=mock_llm_interface,
            agent_id="generator-001",
        )
        
        assert agent.agent_type == "code_generator"
        assert agent.agent_id == "generator-001"
        assert "Senior Software Engineer" in agent.system_prompt
    
    async def test_code_generation(self, mock_code_generator, sample_problem):
        """Test basic code generation functionality."""
        # Mock LLM response with Python code
        mock_code_generator.llm_interface.call.return_value = """
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```
"""
        
        result = await mock_code_generator.process_request(
            sample_problem,
            context={"language": "python"}
        )
        
        assert result.success is True
        assert "def fibonacci" in result.content
        assert result.metadata["language"] == "python"
        assert "detected_functions" in result.metadata
    
    async def test_language_detection(self, mock_code_generator):
        """Test programming language detection and requirements."""
        test_cases = [
            ("python", "Follow PEP 8"),
            ("javascript", "modern ES6+"),
            ("java", "Oracle conventions"),
            ("cpp", "modern C++17"),
        ]
        
        for language, expected_requirement in test_cases:
            requirements = mock_code_generator._get_language_requirements(language)
            assert expected_requirement.lower() in requirements.lower()
    
    async def test_code_optimization(self, mock_code_generator):
        """Test code optimization functionality."""
        sample_code = "def slow_func(n):\n    return sum(range(n))"
        
        mock_code_generator.llm_interface.call.return_value = """
Optimized version using mathematical formula:

```python
def fast_func(n):
    return n * (n - 1) // 2
```
"""
        
        result = await mock_code_generator.optimize_code(
            sample_code,
            optimization_focus="performance"
        )
        
        assert result.success is True
        assert "fast_func" in result.content
        assert result.metadata["focus"] == "performance"
    
    def test_function_detection(self, mock_code_generator):
        """Test function detection in generated code."""
        test_cases = [
            ("python", "def test_func():", ["test_func"]),
            ("javascript", "function myFunc() {}", ["myFunc"]),
            ("java", "public void testMethod() {", ["testMethod"]),
        ]
        
        for language, code, expected_functions in test_cases:
            detected = mock_code_generator._detect_functions(code, language)
            for func in expected_functions:
                assert func in detected
    
    def test_complexity_estimation(self, mock_code_generator):
        """Test code complexity estimation."""
        simple_code = "def hello():\n    return 'Hello World'"
        complex_code = """
def complex_function(data):
    for item in data:
        if item > 0:
            for sub_item in item:
                try:
                    while sub_item.has_next():
                        yield process(sub_item)
                except Exception as e:
                    handle_error(e)
"""
        
        simple_complexity = mock_code_generator._estimate_complexity(simple_code)
        complex_complexity = mock_code_generator._estimate_complexity(complex_code)
        
        assert simple_complexity == "simple"
        assert complex_complexity in ["moderate", "complex"]


class TestCodeReviewerAgent:
    """Test cases for the Code Reviewer Agent."""
    
    async def test_agent_initialization(self, test_config, mock_llm_interface):
        """Test code reviewer agent initialization."""
        agent = CodeReviewerAgent(
            config=test_config.agents.reviewer,
            llm_interface=mock_llm_interface,
            agent_id="reviewer-001",
        )
        
        assert agent.agent_type == "code_reviewer"
        assert agent.agent_id == "reviewer-001"
        assert "Senior Code Reviewer" in agent.system_prompt
    
    async def test_code_review(self, mock_code_reviewer, sample_python_code):
        """Test basic code review functionality."""
        # Mock LLM response with review
        mock_code_reviewer.llm_interface.call.return_value = """
**Overall Assessment:** Good implementation with minor improvements needed (Score: 75/100)

**Issues Found:**
- Inefficient recursive implementation for large values
- Missing input validation

**Suggestions:**
- Use memoization or iterative approach
- Add parameter validation

**Strengths:**
- Clear documentation
- Proper edge case handling

**Security Concerns:**
- No security issues found

**Performance Notes:**
- Exponential time complexity could be improved
"""
        
        result = await mock_code_reviewer.process_request(
            f"Review this code:\n```python\n{sample_python_code}\n```"
        )
        
        assert result.success is True
        assert "Overall Assessment" in result.content
        assert result.metadata["overall_score"] > 0
        assert "issues_found" in result.metadata
    
    async def test_security_focused_review(self, mock_code_reviewer):
        """Test security-focused code review."""
        dangerous_code = """
import subprocess
import os

def execute_command(user_input):
    os.system(user_input)  # Dangerous!
    return subprocess.call(user_input, shell=True)
"""
        
        mock_code_reviewer.llm_interface.call.return_value = """
**Security Review Results:**

**Critical Issues:**
- Direct execution of user input via os.system() - Command injection vulnerability
- subprocess.call with shell=True - Another command injection risk

**Recommendations:**
- Use parameterized commands
- Validate and sanitize all inputs
- Avoid shell=True parameter
"""
        
        result = await mock_code_reviewer.security_focused_review(dangerous_code)
        
        assert result.success is True
        assert "injection" in result.content.lower()
        assert result.metadata["review_type"] == "security"
    
    def test_issue_categorization(self, mock_code_reviewer):
        """Test issue categorization logic."""
        test_cases = [
            ("SQL injection vulnerability found", "security"),
            ("Function is too slow for large inputs", "performance"), 
            ("Variable name should be more descriptive", "style"),
            ("Logic error in condition check", "correctness"),
            ("Complex nested structure hard to maintain", "maintainability"),
        ]
        
        for description, expected_category in test_cases:
            category = mock_code_reviewer._categorize_issue(description)
            assert category == expected_category
    
    def test_language_detection(self, mock_code_reviewer):
        """Test programming language detection."""
        test_cases = [
            ("def test(): import os", "python"),
            ("function test() { const x = 1; }", "javascript"),
            ("public class Test { private int x; }", "java"),
            ("#include <iostream>\nstd::cout << 'hello';", "cpp"),
        ]
        
        for code, expected_lang in test_cases:
            detected = mock_code_reviewer._detect_language(code)
            assert detected == expected_lang


class TestCodeExecutorAgent:
    """Test cases for the Code Executor Agent."""
    
    async def test_agent_initialization(self, test_config, mock_llm_interface):
        """Test code executor agent initialization."""
        agent = CodeExecutorAgent(
            config=test_config.agents.executor,
            llm_interface=mock_llm_interface,
            agent_id="executor-001",
        )
        
        assert agent.agent_type == "code_executor"
        assert agent.agent_id == "executor-001"
        assert "Code Execution Specialist" in agent.system_prompt
    
    async def test_local_python_execution(self, mock_code_executor):
        """Test local Python code execution."""
        simple_code = "print('Hello, World!')"
        
        # Mock subprocess execution
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"Hello, World!\n", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            result = await mock_code_executor._execute_python_locally(simple_code, 30)
            
            assert result.success is True
            assert "Hello, World!" in result.output
            assert result.exit_code == 0
    
    def test_docker_image_selection(self, mock_code_executor):
        """Test Docker image selection for different languages."""
        test_cases = [
            ("python", "python:3.9-slim"),
            ("javascript", "node:18-slim"),
            ("java", "openjdk:17-slim"),
            ("go", "golang:1.19-alpine"),
        ]
        
        for language, expected_image in test_cases:
            image = mock_code_executor._get_docker_image(language)
            assert image == expected_image
    
    def test_execution_command_generation(self, mock_code_executor):
        """Test execution command generation."""
        test_cases = [
            ("python", "test.py", "python test.py"),
            ("javascript", "test.js", "node test.js"),
            ("java", "Test.java", "javac Test.java && java Test"),
            ("go", "test.go", "go run test.go"),
        ]
        
        for language, filename, expected_cmd in test_cases:
            command = mock_code_executor._get_execution_command(language, filename)
            assert expected_cmd in command
    
    async def test_code_execution_analysis(self, mock_code_executor):
        """Test code execution analysis."""
        from src.coding_framework.agents.code_executor import ExecutionResult
        
        sample_result = ExecutionResult(
            success=True,
            output="42\n",
            error="",
            exit_code=0,
            execution_time=0.125,
        )
        
        mock_code_executor.llm_interface.call.return_value = """
**Code Analysis:**
The code executed successfully and produced the expected output.

**Performance:** Execution time of 0.125 seconds is reasonable.

**Correctness:** Output matches expected result.

**Recommendations:** Code appears to be working correctly.
"""
        
        analysis = await mock_code_executor._analyze_execution_results(
            "print(42)",
            sample_result,
            "python"
        )
        
        assert "successfully" in analysis
        assert "0.125" in analysis
    
    @pytest.mark.docker
    async def test_docker_execution(self, mock_code_executor, docker_available):
        """Test Docker-based code execution (requires Docker)."""
        if not docker_available:
            pytest.skip("Docker not available")
        
        # Enable Docker for this test
        mock_code_executor.docker_available = True
        
        simple_code = "print('Hello from Docker')"
        
        with patch.object(mock_code_executor, 'docker_client') as mock_docker:
            mock_container = Mock()
            mock_docker.containers.run.return_value = b"Hello from Docker\n"
            
            result = await mock_code_executor._execute_in_docker(
                simple_code, "python", 30, []
            )
            
            assert result.success is True
            mock_docker.containers.run.assert_called_once()


# Integration tests for agent interactions
class TestAgentIntegrations:
    """Integration tests for agent interactions."""
    
    async def test_generator_to_reviewer_flow(
        self, 
        mock_code_generator, 
        mock_code_reviewer, 
        sample_problem
    ):
        """Test flow from code generation to review."""
        # Generate code
        mock_code_generator.llm_interface.call.return_value = """
```python
def solution(n):
    return n * 2
```
"""
        
        generation_result = await mock_code_generator.process_request(sample_problem)
        
        # Review generated code
        mock_code_reviewer.llm_interface.call.return_value = """
**Overall Assessment:** Simple solution (Score: 85/100)

**Strengths:**
- Clean and readable implementation

**Suggestions:**
- Consider edge cases
"""
        
        review_result = await mock_code_reviewer.process_request(
            f"Review: {generation_result.content}"
        )
        
        assert generation_result.success is True
        assert review_result.success is True
        assert "def solution" in generation_result.content
        assert "85/100" in review_result.content
    
    async def test_full_agent_pipeline(
        self,
        mock_code_generator,
        mock_code_reviewer, 
        mock_code_executor,
        sample_problem,
    ):
        """Test complete pipeline: generate -> review -> execute."""
        # Step 1: Generate code
        mock_code_generator.llm_interface.call.return_value = """
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(5))
```
"""
        
        gen_result = await mock_code_generator.process_request(sample_problem)
        
        # Step 2: Review code
        mock_code_reviewer.llm_interface.call.return_value = """
**Overall Assessment:** Functional but inefficient (Score: 65/100)

**Issues Found:**
- Exponential time complexity

**Suggestions:**
- Use memoization for better performance
"""
        
        review_result = await mock_code_reviewer.process_request(gen_result.content)
        
        # Step 3: Execute code
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"5\n", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            exec_result = await mock_code_executor.process_request(gen_result.content)
        
        # Verify pipeline results
        assert gen_result.success is True
        assert "fibonacci" in gen_result.content
        
        assert review_result.success is True
        assert "65/100" in review_result.content
        
        assert exec_result.success is True
        # Execution result should contain analysis