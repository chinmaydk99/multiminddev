from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import structlog
from pydantic import BaseModel, Field

from ...agents.base_agent import TrainableAgent


class EvaluationConfig(BaseModel):
    """Configuration for code generation evaluation."""
    
    dataset_path: str = Field(description="Path to evaluation dataset")
    results_dir: str = Field(default="./evaluation_results", description="Directory to save results")
    max_samples: int = Field(default=100, description="Maximum samples for pass@k evaluation")
    execution_timeout: int = Field(default=10, description="Code execution timeout in seconds")
    enable_parallel: bool = Field(default=True, description="Enable parallel evaluation")
    max_workers: int = Field(default=4, description="Maximum parallel workers")
    save_individual_results: bool = Field(default=True, description="Save individual problem results")


class EvaluationResult(BaseModel):
    """Result from benchmark evaluation."""
    
    benchmark: str
    total_problems: int
    solved_problems: int
    pass_at_1: float
    pass_at_10: float = 0.0
    pass_at_100: float = 0.0
    success: bool
    evaluation_time: float
    problem_results: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseEvaluator(ABC):
    """Abstract base class for code generation benchmark evaluators."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = structlog.get_logger(
            component=f"{self.__class__.__name__.lower()}"
        )
        
    @abstractmethod
    async def evaluate_agent(self, agent: TrainableAgent) -> EvaluationResult:
        """
        Evaluate an agent on the benchmark dataset.
        
        Args:
            agent: The agent to evaluate
            
        Returns:
            Evaluation results with metrics and detailed scores
        """
        pass
        
    @abstractmethod
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load the benchmark dataset.
        
        Returns:
            List of problems in standard format
        """
        pass
        
    def calculate_pass_at_k(
        self, 
        problem_results: List[Dict[str, Any]], 
        k: int
    ) -> float:
        """
        Calculate pass@k metric across all problems.
        
        Args:
            problem_results: List of per-problem results
            k: Number of samples to consider
            
        Returns:
            Pass@k score between 0.0 and 1.0
        """
        import math
        
        total_problems = len(problem_results)
        if total_problems == 0:
            return 0.0
            
        pass_at_k_sum = 0.0
        
        for result in problem_results:
            solutions_passed = result.get("solutions_passed", 0)
            solutions_generated = result.get("solutions_generated", 0)
            
            if solutions_generated >= k:
                # Probability that at least one of k solutions passes
                if solutions_passed >= solutions_generated:
                    pass_at_k_prob = 1.0
                else:
                    try:
                        pass_at_k_prob = 1.0 - (
                            math.comb(solutions_generated - solutions_passed, k) / 
                            math.comb(solutions_generated, k)
                        )
                    except (ValueError, ZeroDivisionError):
                        pass_at_k_prob = 0.0
            else:
                # Fewer solutions than k, use what we have
                pass_at_k_prob = 1.0 if solutions_passed > 0 else 0.0
                
            pass_at_k_sum += pass_at_k_prob
            
        return pass_at_k_sum / total_problems
        
    async def execute_code_safely(
        self, 
        code: str, 
        test_cases: str,
        problem_id: str,
        timeout: int = None
    ) -> Dict[str, Any]:
        """
        Execute code safely with test cases.
        
        Args:
            code: The code to execute
            test_cases: Test cases to run
            problem_id: Problem identifier for logging
            timeout: Execution timeout in seconds
            
        Returns:
            Execution results with success status and output
        """
        import subprocess
        import tempfile
        import time
        from pathlib import Path
        
        timeout = timeout or self.config.execution_timeout
        
        execution_result = {
            "success": False,
            "output": "",
            "error": "",
            "execution_time": 0.0,
            "timeout": False
        }
        
        try:
            start_time = time.time()
            
            # Create temporary file for code execution
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.py', 
                delete=False
            ) as f:
                # Combine code with test cases
                full_code = f"{code}\n\n{test_cases}"
                f.write(full_code)
                temp_file = f.name
                
            try:
                # Execute code in subprocess
                result = subprocess.run(
                    ["python", temp_file],
                    timeout=timeout,
                    capture_output=True,
                    text=True
                )
                
                execution_time = time.time() - start_time
                execution_result["execution_time"] = execution_time
                execution_result["output"] = result.stdout
                execution_result["error"] = result.stderr
                
                # Check if execution was successful (no errors and proper exit code)
                execution_result["success"] = (
                    result.returncode == 0 and 
                    "AssertionError" not in result.stderr and
                    "Error" not in result.stderr
                )
                
            except subprocess.TimeoutExpired:
                execution_result["timeout"] = True
                execution_result["error"] = f"Execution timeout after {timeout} seconds"
                
            finally:
                # Clean up temporary file
                try:
                    Path(temp_file).unlink(missing_ok=True)
                except Exception:
                    pass  # Ignore cleanup errors
                    
        except Exception as e:
            execution_result["error"] = f"Execution setup failed: {str(e)}"
            self.logger.warning(
                f"Code execution failed for {problem_id}",
                error=str(e)
            )
            
        return execution_result
        
    async def generate_solutions(
        self,
        agent: TrainableAgent,
        problem: Dict[str, Any],
        num_solutions: int = 1
    ) -> List[str]:
        """
        Generate multiple solutions for a problem using the agent.
        
        Args:
            agent: The agent to use for generation
            problem: Problem specification
            num_solutions: Number of solutions to generate
            
        Returns:
            List of generated code solutions
        """
        solutions = []
        
        prompt = problem.get("prompt", "") or problem.get("problem", "")
        if not prompt:
            self.logger.warning("No prompt found in problem", problem_keys=list(problem.keys()))
            return solutions
            
        for i in range(num_solutions):
            try:
                response = await agent.process_request(
                    prompt,
                    context={
                        "evaluation_mode": True,
                        "benchmark": self.__class__.__name__.replace("Evaluator", "").lower(),
                        "problem_id": problem.get("task_id", problem.get("id", f"unknown_{i}")),
                        "sample_idx": i,
                        "temperature": 0.8 if num_solutions > 1 else 0.2  # Higher temp for diversity
                    }
                )
                
                if response.success and response.content:
                    solutions.append(response.content)
                else:
                    # Add a placeholder for failed generation
                    solutions.append(f"# Generation failed: {response.error or 'Unknown error'}")
                    
            except Exception as e:
                self.logger.warning(f"Solution generation failed: {e}")
                solutions.append(f"# Generation exception: {str(e)}")
                
        return solutions
        
    def extract_code_solution(self, response: str) -> str:
        """
        Extract clean code solution from agent response.
        
        Args:
            response: Raw response from agent
            
        Returns:
            Cleaned code solution
        """
        import re
        
        # Try to extract code from markdown blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
            
        code_blocks = re.findall(r'```\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
            
        # If no code blocks found, return the whole response
        return response.strip()
        
    async def save_results(
        self, 
        results: EvaluationResult,
        agent_id: str
    ) -> str:
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results to save
            agent_id: Identifier of the evaluated agent
            
        Returns:
            Path to saved results file
        """
        import json
        import time
        from pathlib import Path
        
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        filename = f"{results.benchmark}_{agent_id}_{timestamp}.json"
        results_file = results_dir / filename
        
        # Convert results to dict and save
        results_data = {
            "agent_id": agent_id,
            "timestamp": timestamp,
            "config": self.config.dict(),
            "results": results.dict()
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
            
        self.logger.info(
            f"Evaluation results saved",
            benchmark=results.benchmark,
            agent_id=agent_id,
            path=str(results_file)
        )
        
        return str(results_file)