import json
import asyncio
import time
from typing import Dict, Any, List
from pathlib import Path

from .base_evaluator import BaseEvaluator, EvaluationConfig, EvaluationResult
from ...agents.base_agent import BaseAgent


class HumanEvalConfig(EvaluationConfig):
    """Configuration specific to HumanEval benchmark."""
    
    dataset_path: str = "./data/benchmarks/humaneval.jsonl"
    max_samples: int = 100  # For pass@k evaluation
    temperature: float = 0.8  # For diverse solution generation
    

class HumanEvalEvaluator(BaseEvaluator):
    """
    Evaluator for HumanEval benchmark.
    
    HumanEval consists of 164 programming problems with function signatures,
    docstrings, and test cases. Evaluates using pass@k metrics.
    """
    
    def __init__(self, config: HumanEvalConfig = None):
        self.config = config or HumanEvalConfig()
        super().__init__(self.config)
        self.problems = []
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load HumanEval problems from dataset file."""
        
        dataset_path = Path(self.config.dataset_path)
        if not dataset_path.exists():
            # Try to download or create sample data
            self.logger.warning(
                f"HumanEval dataset not found at {dataset_path}, creating sample data"
            )
            return self._create_sample_problems()
            
        problems = []
        
        try:
            with open(dataset_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        problem = json.loads(line.strip())
                        problems.append(problem)
                    except json.JSONDecodeError as e:
                        self.logger.warning(
                            f"Skipping invalid JSON on line {line_num}: {e}"
                        )
                        continue
                        
            self.logger.info(f"Loaded {len(problems)} HumanEval problems")
            return problems
            
        except Exception as e:
            self.logger.error(f"Failed to load HumanEval dataset: {e}")
            return self._create_sample_problems()
            
    def _create_sample_problems(self) -> List[Dict[str, Any]]:
        """Create sample HumanEval-style problems for testing."""
        
        sample_problems = [
            {
                "task_id": "HumanEval/0",
                "prompt": 'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
                "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
                "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n"
            },
            {
                "task_id": "HumanEval/1", 
                "prompt": 'def separate_paren_groups(paren_string: str) -> List[str]:\n    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups(\'( ) (( )) (( )( ))\')\n    [\'()\', \'(())\', \'(()())\']\n    """\n',
                "canonical_solution": "    result = []\n    current_string = []\n    current_depth = 0\n\n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n\n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string = []\n\n    return result\n",
                "test": "def check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == [\n        '(()())', '((()))', '()', '((())()())'\n    ]\n    assert candidate('() (()) ((())) (((())))') == [\n        '()', '(())', '((()))', '(((())))'\n    ]\n    assert candidate('(()(())((())))') == [\n        '(()(())((())))'\n    ]\n    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\n"
            }
        ]
        
        self.logger.info(f"Created {len(sample_problems)} sample HumanEval problems")
        return sample_problems
        
    async def evaluate_agent(self, agent: BaseAgent) -> EvaluationResult:
        """Evaluate agent on HumanEval benchmark."""
        
        start_time = time.time()
        
        # Load problems
        problems = self.load_dataset()
        if not problems:
            return EvaluationResult(
                benchmark="humaneval",
                total_problems=0,
                solved_problems=0,
                pass_at_1=0.0,
                success=False,
                evaluation_time=0.0,
                error="No problems loaded"
            )
            
        self.logger.info(
            f"Starting HumanEval evaluation",
            agent_id=agent.agent_id,
            total_problems=len(problems)
        )
        
        # Process problems
        problem_results = []
        solved_count = 0
        
        # Determine number of solutions to generate per problem
        solutions_per_problem = min(self.config.max_samples, 10)  # Reasonable limit
        
        for i, problem in enumerate(problems):
            problem_id = problem.get("task_id", f"HumanEval/{i}")
            
            try:
                # Generate multiple solutions for pass@k evaluation
                solutions = await self.generate_solutions(
                    agent, problem, solutions_per_problem
                )
                
                # Test each solution
                execution_results = []
                for sol_idx, solution in enumerate(solutions):
                    # Extract clean code
                    clean_code = self.extract_code_solution(solution)
                    
                    # Execute with test cases
                    exec_result = await self.execute_code_safely(
                        clean_code,
                        problem.get("test", ""),
                        f"{problem_id}_solution_{sol_idx}"
                    )
                    
                    execution_results.append(exec_result["success"])
                    
                # Calculate problem-level results
                solutions_passed = sum(execution_results)
                problem_pass_at_1 = 1.0 if solutions_passed > 0 else 0.0
                
                problem_result = {
                    "task_id": problem_id,
                    "solutions_generated": len(solutions),
                    "solutions_passed": solutions_passed,
                    "pass_at_1": problem_pass_at_1,
                    "execution_results": execution_results,
                    "solutions": solutions if self.config.save_individual_results else []
                }
                
                problem_results.append(problem_result)
                
                if problem_pass_at_1 > 0:
                    solved_count += 1
                    
                # Log progress
                if (i + 1) % 10 == 0 or i == len(problems) - 1:
                    current_pass_rate = solved_count / (i + 1)
                    self.logger.info(
                        f"HumanEval progress: {i+1}/{len(problems)}",
                        pass_at_1=current_pass_rate,
                        solved=solved_count
                    )
                    
            except Exception as e:
                self.logger.error(f"Error evaluating {problem_id}: {e}")
                problem_result = {
                    "task_id": problem_id,
                    "error": str(e),
                    "pass_at_1": 0.0,
                    "solutions_generated": 0,
                    "solutions_passed": 0
                }
                problem_results.append(problem_result)
                
        # Calculate final metrics
        evaluation_time = time.time() - start_time
        pass_at_1 = solved_count / len(problems) if problems else 0.0
        pass_at_10 = self.calculate_pass_at_k(problem_results, 10)
        pass_at_100 = self.calculate_pass_at_k(problem_results, min(solutions_per_problem, 100))
        
        result = EvaluationResult(
            benchmark="humaneval",
            total_problems=len(problems),
            solved_problems=solved_count,
            pass_at_1=pass_at_1,
            pass_at_10=pass_at_10,
            pass_at_100=pass_at_100,
            success=True,
            evaluation_time=evaluation_time,
            problem_results=problem_results,
            metadata={
                "solutions_per_problem": solutions_per_problem,
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "config": self.config.dict()
            }
        )
        
        self.logger.info(
            "HumanEval evaluation completed",
            pass_at_1=pass_at_1,
            pass_at_10=pass_at_10,
            solved_problems=solved_count,
            total_problems=len(problems),
            evaluation_time=evaluation_time
        )
        
        # Save results
        if self.config.save_individual_results:
            await self.save_results(result, agent.agent_id)
            
        return result
        
    def extract_code_solution(self, response: str) -> str:
        """Extract code solution from agent response for HumanEval."""
        
        # HumanEval expects function definitions
        import re
        
        # Clean the response
        clean_response = super().extract_code_solution(response)
        
        # Ensure we have necessary imports
        if "List[" in clean_response and "from typing import" not in clean_response:
            clean_response = "from typing import List\n\n" + clean_response
            
        # Make sure function is properly defined
        if not clean_response.strip().startswith("def "):
            # Try to find function definition in the response
            func_match = re.search(r'(def \w+.*?(?=\ndef|\Z))', clean_response, re.DOTALL)
            if func_match:
                clean_response = func_match.group(1)
            else:
                self.logger.warning("No function definition found in response")
                
        return clean_response