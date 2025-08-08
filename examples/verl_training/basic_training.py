"""
Basic VERL training example for the multi-agent coding framework.

This example demonstrates how to train the Code Generator Agent using
VERL PPO with basic reward functions on simple coding problems.
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# Add src to path to import the framework
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

from coding_framework.orchestration import CodingSupervisor
from coding_framework.utils import load_config


async def train_coding_agents():
    """Complete example of VERL training loop."""

    print("VERL + LangGraph Multi-Agent Coding Framework")
    print("=" * 60)
    print("Starting basic PPO training example")
    print("")

    try:
        # 1. Load configuration optimized for training
        config_path = Path(__file__).parent / "configs" / "ppo_basic.yaml"
        if not config_path.exists():
            print(f"WARNING: Configuration file not found: {config_path}")
            print("Using default configuration...")
            config_path = "config/default.yaml"  # Fallback to default

        print(f"Loading configuration from: {config_path}")
        config = load_config(str(config_path))

        # 2. Initialize supervisor and training components
        print("Initializing CodingSupervisor...")
        supervisor = CodingSupervisor(config)
        await supervisor.initialize()

        print("Supervisor initialized successfully")
        print(f"   - Agents: {list(supervisor.agents.keys())}")
        print("")

        # 3. Run training for specified episodes (default 10 for demo)
        episodes = 10  # Small number for demo
        algorithm = "ppo"

        print(f"Starting VERL {algorithm.upper()} training...")
        print(f"   - Episodes: {episodes}")
        print(f"   - Algorithm: {algorithm}")
        print(f"   - Data path: {config.training.data_path}")
        print("")

        # Execute training
        training_results = await supervisor.train_agents(algorithm=algorithm, episodes=episodes)

        # 4. Display training results
        print("Training Results:")
        print("=" * 40)

        if training_results.get("success", False):
            print("SUCCESS: Training completed successfully!")

            metrics = training_results.get("metrics", {})
            print(f"   • Final reward: {metrics.get('final_reward', 'N/A'):.3f}")
            print(f"   • Best reward: {metrics.get('best_reward', 'N/A'):.3f}")
            print(f"   • Episodes completed: {metrics.get('episodes_completed', 'N/A')}")
            print(f"   • Training time: {training_results.get('training_time', 'N/A'):.1f}s")

            if "convergence_episode" in metrics:
                print(f"   • Convergence episode: {metrics['convergence_episode']}")

        else:
            print("ERROR: Training failed!")
            error = training_results.get("error", "Unknown error")
            print(f"   Error: {error}")
            return False

        print("")

        # 5. Test trained agent on validation problems
        print("Testing Testing trained agent on validation problem...")

        test_problem = "Write a function to reverse a string"
        print(f"   Problem: {test_problem}")

        validation_results = await supervisor.solve_problem(
            test_problem, context={"validation": True, "include_tests": True}
        )

        if validation_results.get("success", False):
            print("SUCCESS: Validation test passed!")

            code = validation_results.get("code", "")
            if code:
                print("   Generated code:")
                print("   " + "─" * 40)
                for i, line in enumerate(code.split("\n")[:10], 1):  # Show first 10 lines
                    print(f"   {i:2d}│ {line}")
                if len(code.split("\n")) > 10:
                    print("   ..│ (truncated)")
                print("   " + "─" * 40)

            review = validation_results.get("review", {})
            if isinstance(review, dict):
                print(f"   Review score: {review.get('score', 'N/A')}")

            execution = validation_results.get("execution", {})
            if isinstance(execution, dict):
                print(f"   Execution: {'SUCCESS: Success' if execution.get('success') else 'ERROR: Failed'}")

        else:
            print("ERROR: Validation test failed!")
            error = validation_results.get("error", "Unknown error")
            print(f"   Error: {error}")

        print("")
        print("Testing Training Summary:")
        print("=" * 40)

        # Get final performance metrics
        performance_metrics = supervisor.get_performance_metrics()
        training_metrics = performance_metrics.get("training_metrics", {})

        print(f"   • Total training sessions: {training_metrics.get('total_training_sessions', 0)}")
        print(
            f"   • Successful sessions: {training_metrics.get('successful_training_sessions', 0)}"
        )
        print(f"   • Average final reward: {training_metrics.get('avg_final_reward', 0.0):.3f}")
        print(f"   • Best final reward: {training_metrics.get('best_final_reward', 0.0):.3f}")

        return True

    except Exception as e:
        print(f"ERROR: Training example failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Clean up
        if "supervisor" in locals():
            try:
                await supervisor.shutdown()
                print("Supervisor shutdown completed")
            except Exception as e:
                print(f"Testing  Shutdown warning: {e}")


async def quick_demo():
    """Quick demo with minimal training."""
    print("Testing Quick VERL Training Demo (5 episodes)")
    print("=" * 50)

    try:
        # Use default config for demo
        from coding_framework.utils.config import Config

        config = Config()  # Default configuration

        supervisor = CodingSupervisor(config)
        await supervisor.initialize()

        # Very short training for demo
        training_results = await supervisor.train_agents(algorithm="ppo", episodes=5)

        if training_results.get("success"):
            print("SUCCESS: Demo training completed!")
            metrics = training_results.get("metrics", {})
            print(f"   Final reward: {metrics.get('final_reward', 0.0):.3f}")
        else:
            print("ERROR: Demo training failed!")
            print(f"   Error: {training_results.get('error', 'Unknown')}")

        await supervisor.shutdown()

    except Exception as e:
        print(f"ERROR: Demo failed: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="VERL Training Example")
    parser.add_argument("--demo", action="store_true", help="Run quick demo (5 episodes)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of training episodes")

    args = parser.parse_args()

    if args.demo:
        asyncio.run(quick_demo())
    else:
        asyncio.run(train_coding_agents())


if __name__ == "__main__":
    main()
