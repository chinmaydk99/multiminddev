import asyncio
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import structlog

from .evaluators import (
    HumanEvalEvaluator, 
    MBPPEvaluator, 
    BigCodeBenchEvaluator,
    HumanEvalConfig,
    MBPPConfig, 
    BigCodeBenchConfig
)
from ..agents.trainable_agent import TrainableAgent


class BenchmarkManagerConfig:
    """Configuration for benchmark manager."""
    
    def __init__(
        self,
        results_dir: str = "./evaluation_results",
        parallel_execution: bool = True,
        max_concurrent_evaluations: int = 3,
        save_detailed_results: bool = True,
        timeout_per_benchmark: int = 3600  # 1 hour per benchmark
    ):
        self.results_dir = results_dir
        self.parallel_execution = parallel_execution
        self.max_concurrent_evaluations = max_concurrent_evaluations
        self.save_detailed_results = save_detailed_results
        self.timeout_per_benchmark = timeout_per_benchmark


class BenchmarkManager:
    """
    Manages multiple code generation benchmarks and evaluation.
    
    Coordinates evaluation across HumanEval, MBPP, BigCodeBench and other
    benchmarks while providing comprehensive reporting and VERL integration.
    """
    
    def __init__(self, config: BenchmarkManagerConfig = None):
        self.config = config or BenchmarkManagerConfig()
        self.logger = structlog.get_logger(component="benchmark_manager")
        
        # Initialize evaluators with their specific configs
        self.evaluators = {
            "humaneval": HumanEvalEvaluator(HumanEvalConfig(
                results_dir=self.config.results_dir,
                save_individual_results=self.config.save_detailed_results
            )),
            "mbpp": MBPPEvaluator(MBPPConfig(
                results_dir=self.config.results_dir,
                save_individual_results=self.config.save_detailed_results
            )),
            "bigcodebench": BigCodeBenchEvaluator(BigCodeBenchConfig(
                results_dir=self.config.results_dir,
                save_individual_results=self.config.save_detailed_results
            ))
        }
        
        # Results storage
        self.evaluation_history: List[Dict[str, Any]] = []
        
    async def run_comprehensive_evaluation(
        self,
        agent: TrainableAgent,
        benchmarks: List[str] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run evaluation across multiple benchmarks.
        
        Args:
            agent: The agent to evaluate
            benchmarks: List of benchmark names to run (default: all)
            save_results: Whether to save results to disk
            
        Returns:
            Comprehensive evaluation results
        """
        
        benchmarks = benchmarks or list(self.evaluators.keys())
        start_time = time.time()
        
        self.logger.info(
            "Starting comprehensive evaluation",
            benchmarks=benchmarks,
            agent_id=agent.agent_id,
            agent_type=agent.agent_type
        )
        
        results = {
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type,
            "benchmarks": {},
            "aggregate": {},
            "metadata": {
                "start_time": start_time,
                "benchmarks_requested": benchmarks,
                "config": self.config.__dict__
            }
        }
        
        # Run evaluations (parallel or sequential based on config)
        if self.config.parallel_execution and len(benchmarks) > 1:
            benchmark_results = await self._run_parallel_evaluations(agent, benchmarks)
        else:
            benchmark_results = await self._run_sequential_evaluations(agent, benchmarks)
            
        # Process results
        for benchmark_name, result in benchmark_results.items():
            if result:
                results["benchmarks"][benchmark_name] = result.dict() if hasattr(result, 'dict') else result
                
                # Log individual benchmark results
                if isinstance(result, dict):
                    pass_at_1 = result.get("pass_at_1", 0.0)
                else:
                    pass_at_1 = getattr(result, 'pass_at_1', 0.0)
                    
                self.logger.info(
                    f"{benchmark_name} evaluation completed",
                    pass_at_1=pass_at_1,
                    agent_id=agent.agent_id
                )
            else:
                results["benchmarks"][benchmark_name] = {
                    "error": "Evaluation failed",
                    "success": False
                }
                
        # Calculate aggregate metrics
        results["aggregate"] = self._calculate_aggregate_metrics(results["benchmarks"])
        
        # Record timing
        total_time = time.time() - start_time
        results["metadata"]["total_evaluation_time"] = total_time
        results["metadata"]["end_time"] = time.time()
        
        # Save results if requested
        if save_results:
            results_file = await self._save_comprehensive_results(results)
            results["metadata"]["results_file"] = results_file
            
        # Add to history
        self.evaluation_history.append({
            "timestamp": start_time,
            "agent_id": agent.agent_id,
            "benchmarks": benchmarks,
            "aggregate_score": results["aggregate"].get("weighted_pass_at_1", 0.0),
            "total_time": total_time
        })
        
        self.logger.info(
            "Comprehensive evaluation completed",
            total_time=total_time,
            aggregate_score=results["aggregate"].get("weighted_pass_at_1", 0.0),
            benchmarks_completed=len([b for b in results["benchmarks"].values() if b.get("success", False)])
        )
        
        return results
        
    async def _run_parallel_evaluations(
        self, 
        agent: TrainableAgent, 
        benchmarks: List[str]
    ) -> Dict[str, Any]:
        """Run evaluations in parallel with concurrency limits."""
        
        semaphore = asyncio.Semaphore(self.config.max_concurrent_evaluations)
        
        async def run_single_evaluation(benchmark_name: str):
            async with semaphore:
                try:
                    if benchmark_name not in self.evaluators:
                        self.logger.warning(f"Unknown benchmark: {benchmark_name}")
                        return None
                        
                    evaluator = self.evaluators[benchmark_name]
                    
                    # Run with timeout
                    result = await asyncio.wait_for(
                        evaluator.evaluate_agent(agent),
                        timeout=self.config.timeout_per_benchmark
                    )
                    
                    return result
                    
                except asyncio.TimeoutError:
                    self.logger.error(f"{benchmark_name} evaluation timed out")
                    return {"error": "Evaluation timed out", "success": False}
                except Exception as e:
                    self.logger.error(f"{benchmark_name} evaluation failed: {e}")
                    return {"error": str(e), "success": False}
                    
        # Create tasks for all benchmarks
        tasks = {
            benchmark: asyncio.create_task(run_single_evaluation(benchmark))
            for benchmark in benchmarks
        }
        
        # Wait for all tasks to complete
        results = {}
        for benchmark, task in tasks.items():
            try:
                results[benchmark] = await task
            except Exception as e:
                self.logger.error(f"Task for {benchmark} failed: {e}")
                results[benchmark] = {"error": str(e), "success": False}
                
        return results
        
    async def _run_sequential_evaluations(
        self, 
        agent: TrainableAgent, 
        benchmarks: List[str]
    ) -> Dict[str, Any]:
        """Run evaluations sequentially."""
        
        results = {}
        
        for benchmark_name in benchmarks:
            try:
                if benchmark_name not in self.evaluators:
                    self.logger.warning(f"Unknown benchmark: {benchmark_name}")
                    results[benchmark_name] = {"error": "Unknown benchmark", "success": False}
                    continue
                    
                evaluator = self.evaluators[benchmark_name]
                
                self.logger.info(f"Starting {benchmark_name} evaluation")
                
                result = await asyncio.wait_for(
                    evaluator.evaluate_agent(agent),
                    timeout=self.config.timeout_per_benchmark
                )
                
                results[benchmark_name] = result
                
            except asyncio.TimeoutError:
                self.logger.error(f"{benchmark_name} evaluation timed out")
                results[benchmark_name] = {"error": "Evaluation timed out", "success": False}
            except Exception as e:
                self.logger.error(f"{benchmark_name} evaluation failed: {e}")
                results[benchmark_name] = {"error": str(e), "success": False}
                
        return results
        
    def _calculate_aggregate_metrics(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate performance metrics across benchmarks."""
        
        aggregate = {
            "total_problems": 0,
            "total_solved": 0,
            "weighted_pass_at_1": 0.0,
            "benchmark_scores": {},
            "success_rate": 0.0
        }
        
        # Weights for different benchmarks (can be configured)
        benchmark_weights = {
            "humaneval": 0.4,    # Most standard benchmark
            "mbpp": 0.3,         # Entry-level problems  
            "bigcodebench": 0.3  # Complex real-world tasks
        }
        
        successful_benchmarks = 0
        total_weight = 0
        weighted_score = 0
        
        for benchmark_name, result in benchmark_results.items():
            if not isinstance(result, dict) or not result.get("success", False):
                continue
                
            successful_benchmarks += 1
            pass_at_1 = result.get("pass_at_1", 0.0)
            weight = benchmark_weights.get(benchmark_name, 0.1)
            
            weighted_score += pass_at_1 * weight
            total_weight += weight
            
            # Store individual benchmark score
            aggregate["benchmark_scores"][benchmark_name] = {
                "pass_at_1": pass_at_1,
                "pass_at_10": result.get("pass_at_10", 0.0),
                "total_problems": result.get("total_problems", 0),
                "solved_problems": result.get("solved_problems", 0),
                "weight": weight
            }
            
            # Accumulate totals
            aggregate["total_problems"] += result.get("total_problems", 0)
            aggregate["total_solved"] += result.get("solved_problems", 0)
            
        # Calculate final metrics
        if total_weight > 0:
            aggregate["weighted_pass_at_1"] = weighted_score / total_weight
            
        aggregate["success_rate"] = successful_benchmarks / max(len(benchmark_results), 1)
        
        # Add overall pass rate
        if aggregate["total_problems"] > 0:
            aggregate["overall_pass_rate"] = aggregate["total_solved"] / aggregate["total_problems"]
        else:
            aggregate["overall_pass_rate"] = 0.0
            
        return aggregate
        
    async def _save_comprehensive_results(self, results: Dict[str, Any]) -> str:
        """Save comprehensive evaluation results."""
        
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        agent_id = results["agent_id"]
        filename = f"comprehensive_evaluation_{agent_id}_{timestamp}.json"
        results_file = results_dir / filename
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        self.logger.info(
            "Comprehensive evaluation results saved",
            path=str(results_file),
            agent_id=agent_id
        )
        
        return str(results_file)
        
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get history of all evaluations run by this manager."""
        return self.evaluation_history.copy()
        
    def get_benchmark_comparison(
        self, 
        agent_ids: List[str] = None,
        benchmark: str = None
    ) -> Dict[str, Any]:
        """
        Get comparison of agents across benchmarks.
        
        Args:
            agent_ids: List of agent IDs to compare (default: all from history)
            benchmark: Specific benchmark to compare (default: all)
            
        Returns:
            Comparison data structure
        """
        
        if not self.evaluation_history:
            return {"error": "No evaluation history available"}
            
        # Filter history based on criteria
        filtered_history = self.evaluation_history
        if agent_ids:
            filtered_history = [h for h in filtered_history if h["agent_id"] in agent_ids]
            
        # Create comparison structure
        comparison = {
            "agents": {},
            "benchmark_rankings": {},
            "summary": {
                "total_evaluations": len(filtered_history),
                "unique_agents": len(set(h["agent_id"] for h in filtered_history)),
                "benchmarks_covered": set()
            }
        }
        
        # Process each evaluation in history
        for eval_record in filtered_history:
            agent_id = eval_record["agent_id"]
            if agent_id not in comparison["agents"]:
                comparison["agents"][agent_id] = {
                    "evaluations": [],
                    "best_score": 0.0,
                    "average_score": 0.0
                }
                
            comparison["agents"][agent_id]["evaluations"].append(eval_record)
            comparison["agents"][agent_id]["best_score"] = max(
                comparison["agents"][agent_id]["best_score"],
                eval_record.get("aggregate_score", 0.0)
            )
            
            comparison["summary"]["benchmarks_covered"].update(eval_record.get("benchmarks", []))
            
        # Calculate averages
        for agent_id, agent_data in comparison["agents"].items():
            scores = [e.get("aggregate_score", 0.0) for e in agent_data["evaluations"]]
            agent_data["average_score"] = sum(scores) / len(scores) if scores else 0.0
            
        comparison["summary"]["benchmarks_covered"] = list(comparison["summary"]["benchmarks_covered"])
        
        return comparison
        
    async def quick_evaluation(
        self,
        agent: TrainableAgent,
        benchmark: str = "humaneval",
        num_problems: int = 20
    ) -> Dict[str, Any]:
        """
        Run a quick evaluation on a subset of problems.
        
        Args:
            agent: Agent to evaluate
            benchmark: Benchmark to use
            num_problems: Number of problems to test
            
        Returns:
            Quick evaluation results
        """
        
        if benchmark not in self.evaluators:
            return {"error": f"Unknown benchmark: {benchmark}"}
            
        self.logger.info(
            f"Starting quick evaluation",
            benchmark=benchmark,
            num_problems=num_problems,
            agent_id=agent.agent_id
        )
        
        evaluator = self.evaluators[benchmark]
        
        # Load limited dataset
        original_problems = evaluator.load_dataset()
        if len(original_problems) > num_problems:
            # Take a representative sample
            import random
            problems_sample = random.sample(original_problems, num_problems)
        else:
            problems_sample = original_problems
            
        # Temporarily modify evaluator dataset
        original_dataset_method = evaluator.load_dataset
        evaluator.load_dataset = lambda: problems_sample
        
        try:
            result = await evaluator.evaluate_agent(agent)
            
            # Add quick eval metadata
            if hasattr(result, 'metadata'):
                result.metadata["quick_evaluation"] = True
                result.metadata["sampled_problems"] = num_problems
            elif isinstance(result, dict):
                result["metadata"] = result.get("metadata", {})
                result["metadata"]["quick_evaluation"] = True
                result["metadata"]["sampled_problems"] = num_problems
                
            return result
            
        finally:
            # Restore original dataset method
            evaluator.load_dataset = original_dataset_method