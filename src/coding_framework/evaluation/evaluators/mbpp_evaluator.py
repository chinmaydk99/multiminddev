import json
import time
from typing import Dict, Any, List
from pathlib import Path

from .base_evaluator import BaseEvaluator, EvaluationConfig, EvaluationResult
from ...agents.base_agent import BaseAgent


class MBPPConfig(EvaluationConfig):
    """Configuration specific to MBPP benchmark."""
    
    dataset_path: str = "./data/benchmarks/mbpp.jsonl"
    use_three_shot: bool = True  # Use 3-shot prompting as per MBPP standard
    max_samples: int = 50  # MBPP typically uses fewer samples than HumanEval
    

class MBPPEvaluator(BaseEvaluator):
    """
    Evaluator for MBPP (Mostly Basic Python Problems) benchmark.
    
    MBPP consists of 1000 entry-level Python programming problems.
    Uses 3-shot prompting with expert programmer context.
    """
    
    def __init__(self, config: MBPPConfig = None):
        self.config = config or MBPPConfig()
        super().__init__(self.config)
        
        # 3-shot examples for prompting
        self.few_shot_examples = [
            {
                "task_id": "MBPP/example_1",
                "text": "Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].",
                "code": """def min_cost(cost, m, n): 
    tc = [[0 for x in range(n+1)] for x in range(m+1)] 
    tc[0][0] = cost[0][0] 
    for i in range(1, m+1): 
        tc[i][0] = tc[i-1][0] + cost[i][0] 
    for j in range(1, n+1): 
        tc[0][j] = tc[0][j-1] + cost[0][j] 
    for i in range(1, m+1): 
        for j in range(1, n+1): 
            tc[i][j] = min(tc[i-1][j], tc[i][j-1]) + cost[i][j] 
    return tc[m][n]"""
            },
            {
                "task_id": "MBPP/example_2", 
                "text": "Write a function to find all words which are at least 4 characters long in a string.",
                "code": """def find_char_long(text):
    return [word for word in text.split() if len(word) >= 4]"""
            },
            {
                "task_id": "MBPP/example_3",
                "text": "Write a function to find the n largest integers from a given list of numbers, returned in descending order.",
                "code": """def heap_queue_largest(nums, n):
    import heapq
    return heapq.nlargest(n, nums)"""
            }
        ]
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load MBPP problems from dataset file."""
        
        dataset_path = Path(self.config.dataset_path)
        if not dataset_path.exists():
            self.logger.warning(
                f"MBPP dataset not found at {dataset_path}, creating sample data"
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
                        
            self.logger.info(f"Loaded {len(problems)} MBPP problems")
            return problems
            
        except Exception as e:
            self.logger.error(f"Failed to load MBPP dataset: {e}")
            return self._create_sample_problems()
            
    def _create_sample_problems(self) -> List[Dict[str, Any]]:
        """Create sample MBPP-style problems for testing."""
        
        sample_problems = [
            {
                "task_id": "MBPP/1",
                "text": "Write a function to find the similar elements from the given two tuple lists.",
                "code": "def similar_elements(test_tup1, test_tup2):\n    res = tuple(set(test_tup1) & set(test_tup2))\n    return (res) ",
                "test_list": [
                    "assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))",
                    "assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))", 
                    "assert set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))"
                ]
            },
            {
                "task_id": "MBPP/2",
                "text": "Write a function to identify non-prime numbers.",
                "code": "def is_not_prime(n):\n    result = False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            result = True\n    return result",
                "test_list": [
                    "assert is_not_prime(2) == False",
                    "assert is_not_prime(10) == True",
                    "assert is_not_prime(35) == True"
                ]
            },
            {
                "task_id": "MBPP/3", 
                "text": "Write a function to find the largest integers from a given list of numbers.",
                "code": "def heap_queue_largest(nums,n):\n    import heapq\n    largest_nums = heapq.nlargest(n, nums)\n    return largest_nums",
                "test_list": [
                    "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]",
                    "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]",
                    "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]"
                ]
            }
        ]
        
        self.logger.info(f"Created {len(sample_problems)} sample MBPP problems")
        return sample_problems
        
    def create_few_shot_prompt(self, problem_text: str) -> str:
        """Create 3-shot prompt for MBPP as per standard evaluation."""
        
        if not self.config.use_three_shot:
            return f"Please solve the following problem:\n{problem_text}"
            
        prompt = "You are an expert Python programmer. Here are some examples:\n\n"
        
        # Add few-shot examples
        for i, example in enumerate(self.few_shot_examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Problem: {example['text']}\n"
            prompt += f"Solution:\n{example['code']}\n\n"
            
        # Add the actual problem
        prompt += f"Now solve this problem:\n{problem_text}\n\nSolution:"
        
        return prompt
        
    async def evaluate_agent(self, agent: BaseAgent) -> EvaluationResult:
        """Evaluate agent on MBPP benchmark."""
        
        start_time = time.time()
        
        # Load problems
        problems = self.load_dataset()
        if not problems:
            return EvaluationResult(
                benchmark="mbpp",
                total_problems=0,
                solved_problems=0,
                pass_at_1=0.0,
                success=False,
                evaluation_time=0.0,
                error="No problems loaded"
            )
            
        self.logger.info(
            f"Starting MBPP evaluation",
            agent_id=agent.agent_id,
            total_problems=len(problems),
            use_three_shot=self.config.use_three_shot
        )
        
        # Process problems
        problem_results = []
        solved_count = 0
        
        # MBPP typically uses fewer samples than HumanEval
        solutions_per_problem = min(self.config.max_samples, 5)
        
        for i, problem in enumerate(problems):
            problem_id = problem.get("task_id", f"MBPP/{i}")
            problem_text = problem.get("text", "")
            
            if not problem_text:
                self.logger.warning(f"No problem text for {problem_id}")
                continue
                
            try:
                # Create few-shot prompt
                few_shot_prompt = self.create_few_shot_prompt(problem_text)
                
                # Generate solutions using the few-shot prompt
                solutions = []
                for sol_idx in range(solutions_per_problem):
                    try:
                        response = await agent.process_request(
                            few_shot_prompt,
                            context={
                                "evaluation_mode": True,
                                "benchmark": "mbpp",
                                "problem_id": problem_id,
                                "sample_idx": sol_idx,
                                "temperature": 0.6 if solutions_per_problem > 1 else 0.2,
                                "use_few_shot": self.config.use_three_shot
                            }
                        )
                        
                        if response.success and response.content:
                            solutions.append(response.content)
                        else:
                            solutions.append(f"# Generation failed: {response.error or 'Unknown error'}")
                            
                    except Exception as e:
                        self.logger.warning(f"Solution generation failed for {problem_id}: {e}")
                        solutions.append(f"# Generation exception: {str(e)}")
                        
                # Test each solution
                execution_results = []
                test_cases = problem.get("test_list", [])
                
                for sol_idx, solution in enumerate(solutions):
                    # Extract clean code
                    clean_code = self.extract_code_solution(solution)
                    
                    # Create test script for MBPP format
                    test_script = self._create_mbpp_test_script(clean_code, test_cases)
                    
                    # Execute with test cases
                    exec_result = await self.execute_code_safely(
                        clean_code,
                        test_script,
                        f"{problem_id}_solution_{sol_idx}"
                    )
                    
                    execution_results.append(exec_result["success"])
                    
                # Calculate problem-level results
                solutions_passed = sum(execution_results)
                problem_pass_at_1 = 1.0 if solutions_passed > 0 else 0.0
                
                problem_result = {
                    "task_id": problem_id,
                    "text": problem_text,
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
                        f"MBPP progress: {i+1}/{len(problems)}",
                        pass_at_1=current_pass_rate,
                        solved=solved_count
                    )
                    
            except Exception as e:
                self.logger.error(f"Error evaluating {problem_id}: {e}")
                problem_result = {
                    "task_id": problem_id,
                    "text": problem_text,
                    "error": str(e),
                    "pass_at_1": 0.0,
                    "solutions_generated": 0,
                    "solutions_passed": 0
                }
                problem_results.append(problem_result)
                
        # Calculate final metrics
        evaluation_time = time.time() - start_time
        pass_at_1 = solved_count / len(problems) if problems else 0.0
        pass_at_10 = self.calculate_pass_at_k(problem_results, min(10, solutions_per_problem))
        
        result = EvaluationResult(
            benchmark="mbpp",
            total_problems=len(problems),
            solved_problems=solved_count,
            pass_at_1=pass_at_1,
            pass_at_10=pass_at_10,
            success=True,
            evaluation_time=evaluation_time,
            problem_results=problem_results,
            metadata={
                "solutions_per_problem": solutions_per_problem,
                "use_three_shot": self.config.use_three_shot,
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "config": self.config.dict()
            }
        )
        
        self.logger.info(
            "MBPP evaluation completed",
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
        
    def _create_mbpp_test_script(self, code: str, test_cases: List[str]) -> str:
        """Create test script for MBPP format."""
        
        test_script = ""
        
        # Add the test cases
        for test_case in test_cases:
            test_script += f"{test_case}\n"
            
        # Add a simple execution check
        test_script += "\nprint('All tests passed!')\n"
        
        return test_script
        
    def extract_code_solution(self, response: str) -> str:
        """Extract code solution from agent response for MBPP."""
        
        # MBPP expects function definitions
        import re
        
        # Clean the response first
        clean_response = super().extract_code_solution(response)
        
        # MBPP problems often need import statements
        # Check if we need to add common imports
        if any(lib in clean_response for lib in ['heapq', 'math', 're', 'itertools']):
            imports_to_add = []
            if 'heapq.' in clean_response and 'import heapq' not in clean_response:
                imports_to_add.append('import heapq')
            if 'math.' in clean_response and 'import math' not in clean_response:
                imports_to_add.append('import math')
            if 're.' in clean_response and 'import re' not in clean_response:
                imports_to_add.append('import re')
                
            if imports_to_add:
                clean_response = '\n'.join(imports_to_add) + '\n\n' + clean_response
                
        return clean_response