#!/usr/bin/env python3
"""
Analyze and visualize training results from both SFT and Multi-Turn RL phases.

This script processes the checkpoints and training artifacts to provide
comprehensive analysis of the multi-agent system performance.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class TrainingResultsAnalyzer:
    """Analyzes and reports on training results."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.logger = structlog.get_logger()
        
    def analyze_sft_results(self) -> Dict[str, Any]:
        """Analyze SFT training results."""
        sft_dir = self.checkpoint_dir / "sft_test"
        
        results = {
            "generator_agents": 0,
            "optimizer_agents": 0,
            "total_checkpoints": 0
        }
        
        if not sft_dir.exists():
            self.logger.warning("SFT checkpoint directory not found")
            return results
        
        # Count generator checkpoints
        generator_dir = sft_dir / "generator"
        if generator_dir.exists():
            generator_checkpoints = list(generator_dir.glob("*.json"))
            results["generator_agents"] = len(generator_checkpoints)
            results["total_checkpoints"] += len(generator_checkpoints)
        
        # Count optimizer checkpoints
        optimizer_dir = sft_dir / "optimizer"
        if optimizer_dir.exists():
            optimizer_checkpoints = list(optimizer_dir.glob("*.json"))
            results["optimizer_agents"] = len(optimizer_checkpoints)
            results["total_checkpoints"] += len(optimizer_checkpoints)
        
        return results
    
    def analyze_multiturn_rl_results(self) -> Dict[str, Any]:
        """Analyze Multi-Turn RL training results."""
        rl_dir = self.checkpoint_dir / "multiturn_rl_test"
        
        results = {
            "episodes_completed": 0,
            "final_avg_reward": 0.0,
            "final_avg_performance_improvement": 0.0,
            "final_success_rate": 0.0,
            "reward_progression": [],
            "improvement_progression": [],
            "success_rate_progression": [],
            "learning_trend": "unknown",
            "target_achievement": False
        }
        
        if not rl_dir.exists():
            self.logger.warning("Multi-Turn RL checkpoint directory not found")
            return results
        
        # Find the latest checkpoint
        checkpoints = sorted(rl_dir.glob("checkpoint_episode_*.json"))
        if not checkpoints:
            self.logger.warning("No Multi-Turn RL checkpoints found")
            return results
        
        latest_checkpoint = checkpoints[-1]
        
        try:
            with open(latest_checkpoint, 'r') as f:
                checkpoint_data = json.load(f)
            
            metrics = checkpoint_data.get("metrics", {})
            config = checkpoint_data.get("config", {})
            
            # Extract metrics
            episode_rewards = metrics.get("episode_rewards", [])
            performance_improvements = metrics.get("performance_improvements", [])
            compilation_success_rates = metrics.get("compilation_success_rates", [])
            
            # Calculate final statistics
            results["episodes_completed"] = len(episode_rewards)
            results["final_avg_reward"] = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
            results["final_avg_performance_improvement"] = sum(performance_improvements) / len(performance_improvements) if performance_improvements else 0
            results["final_success_rate"] = sum(compilation_success_rates) / len(compilation_success_rates) if compilation_success_rates else 0
            
            # Calculate progressions (moving averages)
            results["reward_progression"] = self._calculate_moving_average(episode_rewards, window=5)
            results["improvement_progression"] = self._calculate_moving_average(performance_improvements, window=5)
            results["success_rate_progression"] = self._calculate_moving_average(compilation_success_rates, window=5)
            
            # Determine learning trend
            if len(results["reward_progression"]) >= 2:
                if results["reward_progression"][-1] > results["reward_progression"][0]:
                    results["learning_trend"] = "improving"
                elif results["reward_progression"][-1] < results["reward_progression"][0]:
                    results["learning_trend"] = "declining"
                else:
                    results["learning_trend"] = "stable"
            
            # Check target achievement
            target_speedup = config.get("target_speedup", 2.0)
            max_improvement = max(performance_improvements) if performance_improvements else 0
            results["target_achievement"] = max_improvement >= target_speedup
            results["target_speedup"] = target_speedup
            results["max_improvement_achieved"] = max_improvement
            
        except Exception as e:
            self.logger.error(f"Failed to analyze Multi-Turn RL results: {e}")
        
        return results
    
    def _calculate_moving_average(self, data: List[float], window: int = 5) -> List[float]:
        """Calculate moving average of data."""
        if len(data) < window:
            return data
        
        moving_avg = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            window_data = data[start_idx:i + 1]
            moving_avg.append(sum(window_data) / len(window_data))
        
        return moving_avg
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        self.logger.info("🔍 Analyzing training results...")
        
        # Analyze both phases
        sft_results = self.analyze_sft_results()
        rl_results = self.analyze_multiturn_rl_results()
        
        # Generate overall assessment
        overall_assessment = self._assess_overall_performance(sft_results, rl_results)
        
        report = {
            "sft_phase": sft_results,
            "multiturn_rl_phase": rl_results,
            "overall_assessment": overall_assessment,
            "timestamp": "2025-08-09 19:18:00"
        }
        
        return report
    
    def _assess_overall_performance(self, sft_results: Dict[str, Any], rl_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system performance."""
        assessment = {
            "sft_success": sft_results["total_checkpoints"] > 0,
            "rl_success": rl_results["episodes_completed"] > 0,
            "learning_observed": rl_results["learning_trend"] == "improving",
            "target_achieved": rl_results["target_achievement"],
            "system_ready": False,
            "recommendations": []
        }
        
        # Determine if system is ready for production
        if (assessment["sft_success"] and 
            assessment["rl_success"] and 
            assessment["learning_observed"] and
            rl_results["final_success_rate"] > 0.7):
            assessment["system_ready"] = True
        
        # Generate recommendations
        if not assessment["learning_observed"]:
            assessment["recommendations"].append("Increase training episodes or adjust learning rate")
        
        if rl_results["final_success_rate"] < 0.8:
            assessment["recommendations"].append("Improve compilation success rate through better error handling")
        
        if not assessment["target_achieved"]:
            assessment["recommendations"].append(f"Continue training to achieve {rl_results.get('target_speedup', 2.0)}x speedup target")
        
        if not assessment["recommendations"]:
            assessment["recommendations"].append("System performing well - ready for production testing")
        
        return assessment
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted training report."""
        print("\n" + "="*80)
        print("🎯 MULTI-AGENT MULTI-TURN RL TRAINING RESULTS")
        print("="*80)
        
        # SFT Phase Results
        sft = report["sft_phase"]
        print(f"\n📊 SFT PHASE RESULTS:")
        print(f"  ✅ Generator Agents Trained: {sft['generator_agents']}")
        print(f"  ✅ Optimizer Agents Trained: {sft['optimizer_agents']}")
        print(f"  📁 Total Checkpoints Saved: {sft['total_checkpoints']}")
        
        # Multi-Turn RL Phase Results
        rl = report["multiturn_rl_phase"]
        print(f"\n🚀 MULTI-TURN RL PHASE RESULTS:")
        print(f"  🎮 Episodes Completed: {rl['episodes_completed']}")
        print(f"  🏆 Final Average Reward: {rl['final_avg_reward']:.3f}")
        print(f"  ⚡ Final Avg Performance Improvement: {rl['final_avg_performance_improvement']:.3f}x")
        print(f"  ✅ Final Compilation Success Rate: {rl['final_success_rate']:.1%}")
        print(f"  📈 Learning Trend: {rl['learning_trend']}")
        print(f"  🎯 Target Achievement: {'✅ YES' if rl['target_achievement'] else '❌ NO'}")
        
        if rl['target_achievement']:
            print(f"     Max Speedup Achieved: {rl['max_improvement_achieved']:.2f}x (Target: {rl['target_speedup']:.1f}x)")
        
        # Overall Assessment
        assessment = report["overall_assessment"]
        print(f"\n🔍 OVERALL ASSESSMENT:")
        print(f"  SFT Success: {'✅' if assessment['sft_success'] else '❌'}")
        print(f"  RL Success: {'✅' if assessment['rl_success'] else '❌'}")
        print(f"  Learning Observed: {'✅' if assessment['learning_observed'] else '❌'}")
        print(f"  System Ready: {'✅ YES' if assessment['system_ready'] else '❌ NO'}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(assessment['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*80)
        print("🎉 TRAINING ANALYSIS COMPLETE!")
        print("="*80 + "\n")


def main():
    """Main analysis function."""
    logger.info("🔍 Starting training results analysis")
    
    # Create analyzer
    analyzer = TrainingResultsAnalyzer("checkpoints")
    
    # Generate and print report
    report = analyzer.generate_report()
    analyzer.print_report(report)
    
    # Save report to file
    report_path = Path("training_results_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📄 Detailed report saved to: {report_path}")


if __name__ == "__main__":
    main()