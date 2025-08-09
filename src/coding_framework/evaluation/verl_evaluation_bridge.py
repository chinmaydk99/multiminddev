"""
VERL Evaluation Bridge - Integrates our evaluation framework with VERL's training loop
for continuous evaluation during distributed training.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
import structlog

from .benchmark_manager import BenchmarkManager, BenchmarkManagerConfig
from ..agents.trainable_agent import TrainableAgent


class VERLEvaluationConfig:
    """Configuration for VERL evaluation integration."""
    
    def __init__(
        self,
        evaluation_frequency: int = 50,  # Evaluate every N training steps
        benchmarks: List[str] = None,
        quick_eval_problems: int = 20,  # Quick evaluation problem count
        full_eval_frequency: int = 500,  # Full evaluation every N steps
        enable_continuous_eval: bool = True,
        save_intermediate_results: bool = True,
        evaluation_timeout: int = 1800,  # 30 minutes for evaluation
        min_improvement_threshold: float = 0.01,  # Minimum improvement to save model
        early_stopping_patience: int = 5,  # Stop if no improvement for N evaluations
        results_dir: str = "./verl_evaluation_results"
    ):
        self.evaluation_frequency = evaluation_frequency
        self.benchmarks = benchmarks or ["humaneval", "mbpp", "bigcodebench"]
        self.quick_eval_problems = quick_eval_problems
        self.full_eval_frequency = full_eval_frequency
        self.enable_continuous_eval = enable_continuous_eval
        self.save_intermediate_results = save_intermediate_results
        self.evaluation_timeout = evaluation_timeout
        self.min_improvement_threshold = min_improvement_threshold
        self.early_stopping_patience = early_stopping_patience
        self.results_dir = results_dir


class VERLEvaluationBridge:
    """
    Bridge between VERL distributed training and our evaluation framework.
    
    Provides continuous evaluation during training, early stopping based on
    benchmark performance, and comprehensive result tracking.
    """
    
    def __init__(self, config: VERLEvaluationConfig = None):
        self.config = config or VERLEvaluationConfig()
        self.logger = structlog.get_logger(component="verl_evaluation_bridge")
        
        # Initialize benchmark manager
        benchmark_config = BenchmarkManagerConfig(
            results_dir=self.config.results_dir,
            parallel_execution=True,
            save_detailed_results=self.config.save_intermediate_results
        )
        self.benchmark_manager = BenchmarkManager(benchmark_config)
        
        # Evaluation state tracking
        self.evaluation_history: List[Dict[str, Any]] = []
        self.best_score = 0.0
        self.best_model_path: Optional[str] = None
        self.evaluations_without_improvement = 0
        self.training_step = 0
        
        # Callbacks
        self.evaluation_callbacks: List[Callable] = []
        
    async def should_evaluate(self, step: int) -> bool:
        """Determine if evaluation should be run at this training step."""
        
        if not self.config.enable_continuous_eval:
            return False
            
        # Quick evaluation frequency
        if step % self.config.evaluation_frequency == 0:
            return True
            
        # Full evaluation frequency
        if step % self.config.full_eval_frequency == 0:
            return True
            
        return False
        
    async def evaluate_during_training(
        self,
        agent: TrainableAgent,
        step: int,
        training_metrics: Optional[Dict[str, Any]] = None,
        full_evaluation: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate agent during VERL training.
        
        Args:
            agent: The agent being trained
            step: Current training step
            training_metrics: Current training metrics from VERL
            full_evaluation: Whether to run full or quick evaluation
            
        Returns:
            Evaluation results with recommendations
        """
        
        self.training_step = step
        
        self.logger.info(
            f"Running {'full' if full_evaluation else 'quick'} evaluation",
            step=step,
            agent_id=agent.agent_id
        )
        
        evaluation_start = time.time()
        
        try:
            if full_evaluation or step % self.config.full_eval_frequency == 0:
                # Full evaluation on all benchmarks
                eval_results = await asyncio.wait_for(
                    self.benchmark_manager.run_comprehensive_evaluation(
                        agent=agent,
                        benchmarks=self.config.benchmarks,
                        save_results=self.config.save_intermediate_results
                    ),
                    timeout=self.config.evaluation_timeout
                )
                evaluation_type = "full"
            else:
                # Quick evaluation on primary benchmark (HumanEval)
                eval_results = await asyncio.wait_for(
                    self.benchmark_manager.quick_evaluation(
                        agent=agent,
                        benchmark="humaneval",
                        num_problems=self.config.quick_eval_problems
                    ),
                    timeout=self.config.evaluation_timeout // 2
                )
                evaluation_type = "quick"
                
            evaluation_time = time.time() - evaluation_start
            
            # Process evaluation results
            evaluation_record = {
                "step": step,
                "type": evaluation_type,
                "results": eval_results,
                "evaluation_time": evaluation_time,
                "training_metrics": training_metrics or {},
                "timestamp": time.time()
            }
            
            # Calculate current score
            if evaluation_type == "full" and "aggregate" in eval_results:
                current_score = eval_results["aggregate"].get("weighted_pass_at_1", 0.0)
            elif evaluation_type == "quick":
                current_score = eval_results.get("pass_at_1", 0.0)
            else:
                current_score = 0.0
                
            evaluation_record["score"] = current_score
            
            # Check for improvement
            improvement = current_score - self.best_score
            is_improvement = improvement > self.config.min_improvement_threshold
            
            if is_improvement:
                self.best_score = current_score
                self.evaluations_without_improvement = 0
                evaluation_record["is_best"] = True
                
                self.logger.info(
                    "New best model found",
                    step=step,
                    score=current_score,
                    improvement=improvement
                )
            else:
                self.evaluations_without_improvement += 1
                evaluation_record["is_best"] = False
                
            # Add to history
            self.evaluation_history.append(evaluation_record)
            
            # Run callbacks
            await self._run_evaluation_callbacks(evaluation_record)
            
            # Generate recommendations
            recommendations = self._generate_training_recommendations(evaluation_record)
            evaluation_record["recommendations"] = recommendations
            
            self.logger.info(
                f"Evaluation completed",
                type=evaluation_type,
                step=step,
                score=current_score,
                is_improvement=is_improvement,
                evaluation_time=evaluation_time
            )
            
            return evaluation_record
            
        except asyncio.TimeoutError:
            self.logger.error(f"Evaluation timed out at step {step}")
            return {
                "step": step,
                "type": evaluation_type,
                "error": "Evaluation timeout",
                "timestamp": time.time()
            }
        except Exception as e:
            self.logger.error(f"Evaluation failed at step {step}: {e}")
            return {
                "step": step,
                "type": evaluation_type,
                "error": str(e),
                "timestamp": time.time()
            }
            
    def should_early_stop(self) -> bool:
        """Determine if training should be stopped early based on evaluation results."""
        
        if not self.config.enable_continuous_eval:
            return False
            
        # Check if we've exceeded patience without improvement
        should_stop = (
            self.evaluations_without_improvement >= self.config.early_stopping_patience
        )
        
        if should_stop:
            self.logger.info(
                "Early stopping triggered",
                evaluations_without_improvement=self.evaluations_without_improvement,
                patience=self.config.early_stopping_patience,
                best_score=self.best_score
            )
            
        return should_stop
        
    def _generate_training_recommendations(
        self, 
        evaluation_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate training recommendations based on evaluation results."""
        
        recommendations = {
            "continue_training": True,
            "adjust_learning_rate": False,
            "save_checkpoint": False,
            "early_stop": False,
            "messages": []
        }
        
        current_score = evaluation_record.get("score", 0.0)
        step = evaluation_record.get("step", 0)
        
        # Recommend checkpoint saving for improvements
        if evaluation_record.get("is_best", False):
            recommendations["save_checkpoint"] = True
            recommendations["messages"].append(f"New best score: {current_score:.3f}")
            
        # Recommend early stopping if no improvement
        if self.should_early_stop():
            recommendations["early_stop"] = True
            recommendations["continue_training"] = False
            recommendations["messages"].append(
                f"Consider early stopping - no improvement for {self.evaluations_without_improvement} evaluations"
            )
            
        # Learning rate adjustment recommendations
        if len(self.evaluation_history) >= 3:
            recent_scores = [r.get("score", 0.0) for r in self.evaluation_history[-3:]]
            if all(s <= recent_scores[0] + 0.001 for s in recent_scores[1:]):
                recommendations["adjust_learning_rate"] = True
                recommendations["messages"].append("Consider reducing learning rate - plateau detected")
                
        # Performance-based recommendations
        if current_score < 0.1:
            recommendations["messages"].append("Very low performance - check reward function and training data")
        elif current_score > 0.5:
            recommendations["messages"].append("Good performance - continue current approach")
        elif 0.1 <= current_score <= 0.5:
            recommendations["messages"].append("Moderate performance - consider hyperparameter tuning")
            
        return recommendations
        
    async def _run_evaluation_callbacks(self, evaluation_record: Dict[str, Any]) -> None:
        """Run registered evaluation callbacks."""
        
        for callback in self.evaluation_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(evaluation_record)
                else:
                    callback(evaluation_record)
            except Exception as e:
                self.logger.warning(f"Evaluation callback failed: {e}")
                
    def register_callback(self, callback: Callable) -> None:
        """Register a callback to be called after each evaluation."""
        self.evaluation_callbacks.append(callback)
        
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations performed."""
        
        if not self.evaluation_history:
            return {"message": "No evaluations performed yet"}
            
        scores = [r.get("score", 0.0) for r in self.evaluation_history if "score" in r]
        evaluation_times = [r.get("evaluation_time", 0.0) for r in self.evaluation_history]
        
        summary = {
            "total_evaluations": len(self.evaluation_history),
            "best_score": self.best_score,
            "current_score": scores[-1] if scores else 0.0,
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "evaluations_without_improvement": self.evaluations_without_improvement,
            "average_evaluation_time": sum(evaluation_times) / len(evaluation_times) if evaluation_times else 0.0,
            "total_evaluation_time": sum(evaluation_times),
            "evaluation_frequency": self.config.evaluation_frequency,
            "early_stopping_enabled": self.config.enable_continuous_eval
        }
        
        return summary
        
    def get_training_curve(self) -> Dict[str, List[Any]]:
        """Get training curve data for visualization."""
        
        return {
            "steps": [r.get("step", 0) for r in self.evaluation_history],
            "scores": [r.get("score", 0.0) for r in self.evaluation_history],
            "evaluation_types": [r.get("type", "unknown") for r in self.evaluation_history],
            "timestamps": [r.get("timestamp", 0) for r in self.evaluation_history],
            "is_best": [r.get("is_best", False) for r in self.evaluation_history]
        }
        
    async def final_evaluation(self, agent: TrainableAgent) -> Dict[str, Any]:
        """Run final comprehensive evaluation at the end of training."""
        
        self.logger.info("Running final comprehensive evaluation", agent_id=agent.agent_id)
        
        try:
            final_results = await self.benchmark_manager.run_comprehensive_evaluation(
                agent=agent,
                benchmarks=self.config.benchmarks,
                save_results=True
            )
            
            # Add final evaluation metadata
            final_results["metadata"]["evaluation_type"] = "final"
            final_results["metadata"]["training_step"] = self.training_step
            final_results["metadata"]["best_training_score"] = self.best_score
            final_results["metadata"]["evaluation_summary"] = self.get_evaluation_summary()
            
            self.logger.info(
                "Final evaluation completed",
                final_score=final_results["aggregate"].get("weighted_pass_at_1", 0.0),
                total_evaluations=len(self.evaluation_history)
            )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Final evaluation failed: {e}")
            return {
                "error": str(e),
                "metadata": {
                    "evaluation_type": "final",
                    "training_step": self.training_step,
                    "evaluation_summary": self.get_evaluation_summary()
                }
            }
            
    def reset(self) -> None:
        """Reset evaluation state for new training run."""
        
        self.evaluation_history.clear()
        self.best_score = 0.0
        self.best_model_path = None
        self.evaluations_without_improvement = 0
        self.training_step = 0
        
        self.logger.info("VERL evaluation bridge reset for new training run")