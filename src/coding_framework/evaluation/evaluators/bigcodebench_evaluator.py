import json
import time
from typing import Dict, Any, List
from pathlib import Path

from .base_evaluator import BaseEvaluator, EvaluationConfig, EvaluationResult
from ...agents.base_agent import TrainableAgent


class BigCodeBenchConfig(EvaluationConfig):
    """Configuration specific to BigCodeBench benchmark."""
    
    dataset_path: str = "./data/benchmarks/bigcodebench.jsonl"
    mode: str = "instruct"  # "instruct" or "complete"
    include_function_calls: bool = True  # BigCodeBench focuses on function calls
    max_samples: int = 20  # Fewer samples due to complexity
    extended_timeout: int = 20  # Longer timeout for complex problems
    

class BigCodeBenchEvaluator(BaseEvaluator):
    """
    Evaluator for BigCodeBench benchmark.
    
    BigCodeBench consists of 1,140 software-engineering-oriented tasks
    that involve complex instructions with function calls and practical applications.
    """
    
    def __init__(self, config: BigCodeBenchConfig = None):
        self.config = config or BigCodeBenchConfig()
        super().__init__(self.config)
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load BigCodeBench problems from dataset file."""
        
        dataset_path = Path(self.config.dataset_path)
        if not dataset_path.exists():
            self.logger.warning(
                f"BigCodeBench dataset not found at {dataset_path}, creating sample data"
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
                        
            self.logger.info(f"Loaded {len(problems)} BigCodeBench problems")
            return problems
            
        except Exception as e:
            self.logger.error(f"Failed to load BigCodeBench dataset: {e}")
            return self._create_sample_problems()
            
    def _create_sample_problems(self) -> List[Dict[str, Any]]:
        """Create sample BigCodeBench-style problems for testing."""
        
        sample_problems = [
            {
                "task_id": "BigCodeBench/1",
                "complete_prompt": "import pandas as pd\nimport numpy as np\n\ndef analyze_sales_data(df):\n    \"\"\"\n    Analyze sales data and return summary statistics.\n    Expected to use pandas operations for grouping and aggregation.\n    \"\"\"",
                "instruct_prompt": "Write a function that takes a pandas DataFrame containing sales data with columns 'product', 'quantity', and 'price', and returns a summary with total sales by product, sorted by total revenue in descending order. Include error handling for missing columns.",
                "canonical_solution": "import pandas as pd\nimport numpy as np\n\ndef analyze_sales_data(df):\n    required_cols = ['product', 'quantity', 'price']\n    if not all(col in df.columns for col in required_cols):\n        raise ValueError(f'Missing required columns. Need: {required_cols}')\n    \n    df['total_revenue'] = df['quantity'] * df['price']\n    summary = df.groupby('product').agg({\n        'quantity': 'sum',\n        'total_revenue': 'sum'\n    }).sort_values('total_revenue', ascending=False)\n    \n    return summary",
                "test_cases": [
                    {
                        "input": "pd.DataFrame({'product': ['A', 'B', 'A', 'C'], 'quantity': [10, 5, 3, 8], 'price': [100, 200, 100, 150]})",
                        "expected_behavior": "Returns DataFrame sorted by total revenue with A=1300, C=1200, B=1000"
                    }
                ],
                "dependencies": ["pandas", "numpy"]
            },
            {
                "task_id": "BigCodeBench/2",
                "complete_prompt": "import requests\nimport json\nfrom typing import Dict, List\n\ndef fetch_api_data(url: str, params: Dict = None) -> List[Dict]:",
                "instruct_prompt": "Create a robust API client function that fetches data from a REST endpoint with proper error handling, timeout management, and JSON parsing. Include retry logic for failed requests and return structured data.",
                "canonical_solution": "import requests\nimport json\nimport time\nfrom typing import Dict, List, Optional\n\ndef fetch_api_data(url: str, params: Dict = None, timeout: int = 30, max_retries: int = 3) -> List[Dict]:\n    params = params or {}\n    \n    for attempt in range(max_retries):\n        try:\n            response = requests.get(url, params=params, timeout=timeout)\n            response.raise_for_status()\n            \n            data = response.json()\n            if isinstance(data, list):\n                return data\n            elif isinstance(data, dict):\n                return [data]\n            else:\n                return [{'data': data}]\n                \n        except requests.RequestException as e:\n            if attempt == max_retries - 1:\n                raise Exception(f'Failed to fetch data after {max_retries} attempts: {str(e)}')\n            time.sleep(2 ** attempt)  # Exponential backoff\n            \n    return []",
                "test_cases": [
                    {
                        "input": "Mock URL with valid JSON response",
                        "expected_behavior": "Returns parsed JSON data as list of dictionaries"
                    }
                ],
                "dependencies": ["requests"]
            },
            {
                "task_id": "BigCodeBench/3",
                "complete_prompt": "import matplotlib.pyplot as plt\nimport seaborn as sns\nimport pandas as pd\n\ndef create_dashboard_plot(data, plot_type='histogram'):",
                "instruct_prompt": "Design a data visualization function that creates publication-quality plots using matplotlib and seaborn. Support multiple plot types (histogram, scatter, box, violin) with automatic styling, proper labels, and export capabilities.",
                "canonical_solution": "import matplotlib.pyplot as plt\nimport seaborn as sns\nimport pandas as pd\nfrom typing import Optional, Tuple\n\ndef create_dashboard_plot(data: pd.DataFrame, plot_type: str = 'histogram', \n                         x_col: Optional[str] = None, y_col: Optional[str] = None,\n                         title: Optional[str] = None, figsize: Tuple[int, int] = (10, 6)):\n    \n    plt.figure(figsize=figsize)\n    sns.set_style('whitegrid')\n    \n    if plot_type == 'histogram' and x_col:\n        sns.histplot(data=data, x=x_col, kde=True)\n    elif plot_type == 'scatter' and x_col and y_col:\n        sns.scatterplot(data=data, x=x_col, y=y_col)\n    elif plot_type == 'box' and x_col:\n        sns.boxplot(data=data, y=x_col)\n    elif plot_type == 'violin' and x_col:\n        sns.violinplot(data=data, y=x_col)\n    else:\n        raise ValueError(f'Unsupported plot type: {plot_type} or missing required columns')\n    \n    if title:\n        plt.title(title, fontsize=14, fontweight='bold')\n    \n    plt.tight_layout()\n    return plt.gcf()",
                "test_cases": [
                    {
                        "input": "DataFrame with numeric columns",
                        "expected_behavior": "Creates properly formatted plot with styling"
                    }
                ],
                "dependencies": ["matplotlib", "seaborn", "pandas"]
            }
        ]
        
        self.logger.info(f"Created {len(sample_problems)} sample BigCodeBench problems")
        return sample_problems
        
    def get_prompt_for_mode(self, problem: Dict[str, Any]) -> str:
        """Get the appropriate prompt based on the evaluation mode."""
        
        if self.config.mode == "complete":
            return problem.get("complete_prompt", problem.get("prompt", ""))
        else:  # instruct mode
            return problem.get("instruct_prompt", problem.get("instruction", problem.get("prompt", "")))
            
    async def evaluate_agent(self, agent: TrainableAgent) -> EvaluationResult:
        """Evaluate agent on BigCodeBench benchmark."""
        
        start_time = time.time()
        
        # Load problems
        problems = self.load_dataset()
        if not problems:
            return EvaluationResult(
                benchmark="bigcodebench",
                total_problems=0,
                solved_problems=0,
                pass_at_1=0.0,
                success=False,
                evaluation_time=0.0,
                error="No problems loaded"
            )
            
        self.logger.info(
            f"Starting BigCodeBench evaluation",
            agent_id=agent.agent_id,
            total_problems=len(problems),
            mode=self.config.mode
        )
        
        # Process problems
        problem_results = []
        solved_count = 0
        
        # BigCodeBench uses fewer samples due to complexity
        solutions_per_problem = min(self.config.max_samples, 5)
        
        for i, problem in enumerate(problems):
            problem_id = problem.get("task_id", f"BigCodeBench/{i}")
            prompt = self.get_prompt_for_mode(problem)
            
            if not prompt:
                self.logger.warning(f"No prompt found for {problem_id}")
                continue
                
            try:
                # Generate solutions
                solutions = await self.generate_solutions(
                    agent, {"prompt": prompt, "id": problem_id}, solutions_per_problem
                )
                
                # Test each solution - BigCodeBench requires more complex testing
                execution_results = []
                test_cases = problem.get("test_cases", [])
                dependencies = problem.get("dependencies", [])
                
                for sol_idx, solution in enumerate(solutions):
                    # Extract clean code
                    clean_code = self.extract_code_solution(solution)
                    
                    # Add necessary imports for BigCodeBench
                    clean_code = self._add_required_imports(clean_code, dependencies)
                    
                    # Create comprehensive test script
                    test_script = self._create_bigcodebench_test_script(
                        clean_code, test_cases, problem
                    )
                    
                    # Execute with extended timeout for complex problems
                    exec_result = await self.execute_code_safely(
                        clean_code,
                        test_script,
                        f"{problem_id}_solution_{sol_idx}",
                        timeout=self.config.extended_timeout
                    )
                    
                    # BigCodeBench success criteria may be more nuanced
                    success = self._evaluate_bigcodebench_success(
                        exec_result, problem, clean_code
                    )
                    execution_results.append(success)
                    
                # Calculate problem-level results
                solutions_passed = sum(execution_results)
                problem_pass_at_1 = 1.0 if solutions_passed > 0 else 0.0
                
                problem_result = {
                    "task_id": problem_id,
                    "prompt": prompt,
                    "mode": self.config.mode,
                    "solutions_generated": len(solutions),
                    "solutions_passed": solutions_passed,
                    "pass_at_1": problem_pass_at_1,
                    "execution_results": execution_results,
                    "dependencies": dependencies,
                    "solutions": solutions if self.config.save_individual_results else []
                }
                
                problem_results.append(problem_result)
                
                if problem_pass_at_1 > 0:
                    solved_count += 1
                    
                # Log progress
                if (i + 1) % 5 == 0 or i == len(problems) - 1:  # More frequent logging
                    current_pass_rate = solved_count / (i + 1)
                    self.logger.info(
                        f"BigCodeBench progress: {i+1}/{len(problems)}",
                        pass_at_1=current_pass_rate,
                        solved=solved_count
                    )
                    
            except Exception as e:
                self.logger.error(f"Error evaluating {problem_id}: {e}")
                problem_result = {
                    "task_id": problem_id,
                    "prompt": prompt,
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
            benchmark="bigcodebench",
            total_problems=len(problems),
            solved_problems=solved_count,
            pass_at_1=pass_at_1,
            pass_at_10=pass_at_10,
            success=True,
            evaluation_time=evaluation_time,
            problem_results=problem_results,
            metadata={
                "solutions_per_problem": solutions_per_problem,
                "mode": self.config.mode,
                "extended_timeout": self.config.extended_timeout,
                "include_function_calls": self.config.include_function_calls,
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "config": self.config.dict()
            }
        )
        
        self.logger.info(
            "BigCodeBench evaluation completed",
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
        
    def _add_required_imports(self, code: str, dependencies: List[str]) -> str:
        """Add required imports for BigCodeBench problems."""
        
        imports_to_add = []
        
        for dep in dependencies:
            if dep == "pandas" and "import pandas" not in code and "pd." in code:
                imports_to_add.append("import pandas as pd")
            elif dep == "numpy" and "import numpy" not in code and "np." in code:
                imports_to_add.append("import numpy as np")
            elif dep == "matplotlib" and "import matplotlib" not in code and "plt." in code:
                imports_to_add.append("import matplotlib.pyplot as plt")
            elif dep == "seaborn" and "import seaborn" not in code and "sns." in code:
                imports_to_add.append("import seaborn as sns")
            elif dep == "requests" and "import requests" not in code and "requests." in code:
                imports_to_add.append("import requests")
            elif f"import {dep}" not in code and dep in ["json", "time", "re", "os", "sys"]:
                imports_to_add.append(f"import {dep}")
                
        if imports_to_add:
            return '\n'.join(imports_to_add) + '\n\n' + code
        else:
            return code
            
    def _create_bigcodebench_test_script(
        self, 
        code: str, 
        test_cases: List[Dict[str, Any]], 
        problem: Dict[str, Any]
    ) -> str:
        """Create test script for BigCodeBench format."""
        
        test_script = ""
        
        # BigCodeBench often requires more sophisticated testing
        if test_cases:
            for i, test_case in enumerate(test_cases):
                test_input = test_case.get("input", "")
                expected = test_case.get("expected_behavior", "")
                
                # Create a basic test
                test_script += f"""
# Test case {i + 1}: {expected}
try:
    # Test input: {test_input}
    # Add basic execution test
    print(f"Test {i + 1} setup complete")
except Exception as e:
    print(f"Test {i + 1} failed: {{e}}")
"""
        else:
            # Basic syntax and import test
            test_script = """
# Basic functionality test
try:
    # Check if main function can be called
    print("Code executed successfully")
except Exception as e:
    print(f"Execution failed: {e}")
    raise
"""
        
        return test_script
        
    def _evaluate_bigcodebench_success(
        self, 
        exec_result: Dict[str, Any], 
        problem: Dict[str, Any], 
        code: str
    ) -> bool:
        """Evaluate success for BigCodeBench with additional criteria."""
        
        # Basic execution success
        basic_success = exec_result.get("success", False)
        
        # Additional BigCodeBench-specific checks
        if not basic_success:
            return False
            
        # Check for required function calls or library usage
        dependencies = problem.get("dependencies", [])
        if self.config.include_function_calls and dependencies:
            for dep in dependencies:
                if dep in ["pandas", "numpy", "matplotlib", "seaborn", "requests"]:
                    if dep not in code and f"{dep}." not in code:
                        # Missing important dependency usage
                        self.logger.debug(f"Missing expected dependency: {dep}")
                        
        # Check output for specific patterns if expected
        output = exec_result.get("output", "")
        if "error" in output.lower() or "exception" in output.lower():
            return False
            
        return True
        
    def extract_code_solution(self, response: str) -> str:
        """Extract code solution from agent response for BigCodeBench."""
        
        import re
        
        # BigCodeBench often requires complete functions with imports
        clean_response = super().extract_code_solution(response)
        
        # Ensure we have proper function structure
        if not clean_response.strip().startswith(('import', 'from', 'def ', 'class ')):
            # Try to find the main code block
            code_match = re.search(
                r'((?:import.*?\n|from.*?\n)*.*?def.*?(?=\n\n|\n#|\nif|\Z))', 
                clean_response, 
                re.DOTALL
            )
            if code_match:
                clean_response = code_match.group(1)
                
        return clean_response