"""
Supervisor for multi-agent coordination and workflow management.

This module provides the main CodingSupervisor class that orchestrates
the three specialized agents using LangGraph workflows.
"""

import asyncio
import time
from typing import Any, Optional

import structlog

from ..agents import BaseAgent, CodeExecutorAgent, CodeGeneratorAgent, CodeReviewerAgent
from ..utils.config import Config
from ..utils.llm_interface import LLMInterface
from .workflows import CodingWorkflow


class CodingSupervisor:
    """
    Main supervisor for the multi-agent coding system.

    Coordinates code generation, review, and execution agents using
    LangGraph workflows with intelligent routing and error handling.
    """

    def __init__(self, config: Config):
        """
        Initialize the coding supervisor.

        Args:
            config: System configuration
        """
        self.config = config
        self.logger = structlog.get_logger(component="supervisor")

        # Agent instances
        self.agents: dict[str, BaseAgent] = {}
        self.llm_interface: Optional[LLMInterface] = None
        self.workflow: Optional[CodingWorkflow] = None

        # System state
        self.is_initialized = False
        self.performance_metrics = {}

        self.logger.info("Supervisor initialized", config_version=config.version)

    async def initialize(self) -> None:
        """
        Initialize all system components.

        Sets up LLM interfaces, initializes agents, and creates workflows.
        """
        if self.is_initialized:
            self.logger.warning("Supervisor already initialized")
            return

        try:
            self.logger.info("Initializing supervisor components")

            # Initialize LLM interface
            await self._initialize_llm_interface()

            # Initialize agents
            await self._initialize_agents()

            # Initialize workflow
            await self._initialize_workflow()

            self.is_initialized = True
            self.logger.info("Supervisor initialization completed")

        except Exception as e:
            self.logger.error("Supervisor initialization failed", error=str(e))
            raise

    async def _initialize_llm_interface(self) -> None:
        """Initialize the LLM interface."""
        self.llm_interface = LLMInterface(self.config.llm)
        await self.llm_interface.initialize()

        self.logger.info("LLM interface initialized", provider=self.config.llm.provider)

    async def _initialize_agents(self) -> None:
        """Initialize all specialized agents."""
        try:
            # Initialize Code Generator Agent
            self.agents["generator"] = CodeGeneratorAgent(
                config=self.config.agents.generator,
                llm_interface=self.llm_interface,
                agent_id="generator_001",
            )

            # Initialize Code Reviewer Agent
            self.agents["reviewer"] = CodeReviewerAgent(
                config=self.config.agents.reviewer,
                llm_interface=self.llm_interface,
                agent_id="reviewer_001",
            )

            # Initialize Code Executor Agent
            self.agents["executor"] = CodeExecutorAgent(
                config=self.config.agents.executor,
                llm_interface=self.llm_interface,
                agent_id="executor_001",
            )

            self.logger.info("All agents initialized", agent_count=len(self.agents))

            # Perform agent health checks
            await self._verify_agent_health()

        except Exception as e:
            self.logger.error("Agent initialization failed", error=str(e))
            raise

    async def _initialize_workflow(self) -> None:
        """Initialize the LangGraph workflow."""
        workflow_config = {
            "max_iterations": self.config.workflow.max_iterations,
            "human_in_loop": self.config.workflow.human_in_loop,
            "min_execution_score": self.config.workflow.min_execution_score,
            "target_review_score": self.config.workflow.target_review_score,
            "human_feedback_threshold": self.config.workflow.human_feedback_threshold,
            "generation": self.config.agents.generator.dict(),
            "review": self.config.agents.reviewer.dict(),
            "execution": self.config.agents.executor.dict(),
        }

        self.workflow = CodingWorkflow(self.agents, workflow_config)
        self.logger.info("Workflow initialized")

    async def _verify_agent_health(self) -> None:
        """Verify that all agents are healthy and responsive."""
        health_checks = []

        for agent_name, agent in self.agents.items():
            health_checks.append(self._check_single_agent_health(agent_name, agent))

        results = await asyncio.gather(*health_checks, return_exceptions=True)

        for agent_name, result in zip(self.agents.keys(), results):
            if isinstance(result, Exception):
                self.logger.error("Agent health check failed", agent=agent_name, error=str(result))
                raise result
            else:
                self.logger.info("Agent health verified", agent=agent_name, status=result["status"])

    async def _check_single_agent_health(
        self,
        agent_name: str,
        agent: BaseAgent,
    ) -> dict[str, Any]:
        """Check health of a single agent."""
        try:
            health_status = await agent.health_check()
            return health_status
        except Exception as e:
            self.logger.error("Agent health check failed", agent=agent_name, error=str(e))
            raise

    async def solve_problem(
        self,
        problem: str,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Solve a coding problem using the multi-agent system.

        Args:
            problem: Problem description
            context: Additional context and parameters

        Returns:
            Complete solution with code, review, and execution results
        """
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()
        context = context or {}

        self.logger.info(
            "Starting problem solving",
            problem_length=len(problem),
            context_keys=list(context.keys()),
        )

        try:
            # Run the workflow
            result = await self.workflow.run(problem, context)

            execution_time = time.time() - start_time

            # Update performance metrics
            self._update_performance_metrics(result, execution_time)

            # Add timing information
            result["execution_time"] = execution_time
            result["timestamp"] = time.time()

            self.logger.info(
                "Problem solving completed",
                success=result["success"],
                execution_time=execution_time,
                iterations=result.get("iterations", 0),
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time

            self.logger.error("Problem solving failed", error=str(e), execution_time=execution_time)

            return {
                "success": False,
                "error": str(e),
                "problem": problem,
                "execution_time": execution_time,
                "timestamp": time.time(),
            }

    async def review_code(
        self,
        code: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Review code using the reviewer agent.

        Args:
            code: Code to review
            context: Additional context
            **kwargs: Review parameters

        Returns:
            Code review results
        """
        if not self.is_initialized:
            await self.initialize()

        self.logger.info("Starting code review", code_length=len(code))

        try:
            reviewer = self.agents["reviewer"]
            result = await reviewer.process_request(code, context, **kwargs)

            self.logger.info("Code review completed", success=result.success)
            return result.dict()

        except Exception as e:
            self.logger.error("Code review failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def execute_code(
        self,
        code: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute code using the executor agent.

        Args:
            code: Code to execute
            context: Additional context
            **kwargs: Execution parameters

        Returns:
            Code execution results
        """
        if not self.is_initialized:
            await self.initialize()

        self.logger.info("Starting code execution", code_length=len(code))

        try:
            executor = self.agents["executor"]
            result = await executor.process_request(code, context, **kwargs)

            self.logger.info("Code execution completed", success=result.success)
            return result.dict()

        except Exception as e:
            self.logger.error("Code execution failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def generate_code(
        self,
        problem: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate code using the generator agent.

        Args:
            problem: Problem description
            context: Additional context
            **kwargs: Generation parameters

        Returns:
            Code generation results
        """
        if not self.is_initialized:
            await self.initialize()

        self.logger.info("Starting code generation", problem_length=len(problem))

        try:
            generator = self.agents["generator"]
            result = await generator.process_request(problem, context, **kwargs)

            self.logger.info("Code generation completed", success=result.success)
            return result.dict()

        except Exception as e:
            self.logger.error("Code generation failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def health_check(self) -> dict[str, Any]:
        """
        Perform comprehensive system health check.

        Returns:
            Health status for all components
        """
        if not self.is_initialized:
            return {"status": "not_initialized"}

        self.logger.info("Performing system health check")

        try:
            health_results = {}

            # Check each agent
            for agent_name, agent in self.agents.items():
                health_results[agent_name] = await agent.health_check()

            # Check LLM interface
            health_results["llm_interface"] = await self.llm_interface.health_check()

            # System overall status
            all_healthy = all(
                result.get("status") == "healthy" for result in health_results.values()
            )

            health_results["system"] = {
                "status": "healthy" if all_healthy else "unhealthy",
                "components": len(health_results),
                "initialized": self.is_initialized,
                "performance_metrics": self.performance_metrics,
            }

            self.logger.info(
                "Health check completed", overall_status=health_results["system"]["status"]
            )

            return health_results

        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }

    async def check_agent_health(self, agent_name: str) -> dict[str, Any]:
        """
        Check health of a specific agent.

        Args:
            agent_name: Name of the agent to check

        Returns:
            Agent health status
        """
        if not self.is_initialized:
            await self.initialize()

        if agent_name not in self.agents:
            return {
                "status": "not_found",
                "error": f"Agent '{agent_name}' not found",
            }

        try:
            agent = self.agents[agent_name]
            health_status = await agent.health_check()

            self.logger.info(
                "Agent health checked", agent=agent_name, status=health_status["status"]
            )

            return health_status

        except Exception as e:
            self.logger.error("Agent health check failed", agent=agent_name, error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }

    async def train_agents(
        self,
        algorithm: str = "ppo",
        episodes: int = 100,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Train agents using VERL reinforcement learning.

        Args:
            algorithm: RL algorithm to use
            episodes: Number of training episodes
            **kwargs: Additional training parameters

        Returns:
            Training results and metrics
        """
        if not self.is_initialized:
            await self.initialize()

        self.logger.info("Starting VERL agent training", algorithm=algorithm, episodes=episodes)

        try:
            # Import training components
            from ..training import CompositeReward, TrainingDataLoader, VERLPPOTrainer

            # 1. Initialize training components
            self.logger.info("Initializing VERL training components")

            trainer = VERLPPOTrainer(
                self.config.training, verl_config=getattr(self.config.training, "verl", None)
            )

            data_loader = TrainingDataLoader(self.config.training.data_path, validation_split=0.2)

            # Set up reward function with weights from config
            reward_weights = getattr(
                self.config.training,
                "reward_weights",
                {"correctness": 0.7, "style": 0.2, "efficiency": 0.1},
            )

            reward_function = CompositeReward(
                weights=reward_weights,
                correctness_kwargs={"timeout": 10.0, "safe_execution": True},
                style_kwargs={"use_reviewer_agent": True, "use_ruff": True},
                efficiency_kwargs={"max_execution_time": 5.0, "max_memory_mb": 100.0},
            )

            # 2. Load and validate training data
            self.logger.info("Loading training data")
            train_data, val_data = await data_loader.load_problems()

            self.logger.info(
                "Training data loaded",
                training_problems=len(train_data),
                validation_problems=len(val_data),
            )

            # 3. Switch Code Generator to training mode
            generator_agent = self.agents["generator"]
            original_training_mode = generator_agent.get_state("training_mode", False)
            generator_agent.update_state("training_mode", True)

            # Configure training context with other agents for reward calculation
            training_context = {
                "executor_agent": self.agents.get("executor"),
                "reviewer_agent": self.agents.get("reviewer"),
                "supervisor": self,
            }

            self.logger.info("Starting VERL PPO training loop")

            # 4. Execute VERL training loop
            training_results = await trainer.train_agent(
                agent=generator_agent,
                training_data=train_data,
                validation_data=val_data,
                episodes=episodes,
                reward_function=reward_function,
                context=training_context,
                **kwargs,
            )

            # 5. Switch back to inference mode and update performance metrics
            generator_agent.update_state("training_mode", original_training_mode)
            self._update_training_metrics(training_results)

            self.logger.info(
                "VERL training completed",
                success=training_results.success,
                final_reward=training_results.metrics.get("final_reward", 0.0),
                episodes_completed=training_results.metrics.get("episodes_completed", 0),
            )

            # Convert TrainingResults to dict for return
            return training_results.dict()

        except Exception as e:
            # Ensure agent is back in original mode on error
            if "generator_agent" in locals():
                generator_agent.update_state(
                    "training_mode", locals().get("original_training_mode", False)
                )

            error_msg = f"VERL training failed: {str(e)}"
            self.logger.error(error_msg, error=str(e))

            return {
                "success": False,
                "error": error_msg,
                "algorithm": algorithm,
                "episodes": 0,
                "metrics": {},
                "training_time": 0.0,
                "timestamp": time.time(),
            }

    def _update_performance_metrics(
        self,
        result: dict[str, Any],
        execution_time: float,
    ) -> None:
        """Update system performance metrics."""
        # Update counters
        self.performance_metrics["total_problems_solved"] = (
            self.performance_metrics.get("total_problems_solved", 0) + 1
        )

        if result.get("success", False):
            self.performance_metrics["successful_solutions"] = (
                self.performance_metrics.get("successful_solutions", 0) + 1
            )

        # Update timing metrics
        times = self.performance_metrics.get("execution_times", [])
        times.append(execution_time)
        if len(times) > 100:  # Keep only last 100 times
            times = times[-100:]
        self.performance_metrics["execution_times"] = times
        self.performance_metrics["avg_execution_time"] = sum(times) / len(times)

        # Update quality metrics
        if "review_score" in result and result["review_score"]:
            scores = self.performance_metrics.get("review_scores", [])
            scores.append(result["review_score"])
            if len(scores) > 100:
                scores = scores[-100:]
            self.performance_metrics["review_scores"] = scores
            self.performance_metrics["avg_review_score"] = sum(scores) / len(scores)

        # Update iteration metrics
        if "iterations" in result:
            iterations = self.performance_metrics.get("iteration_counts", [])
            iterations.append(result["iterations"])
            if len(iterations) > 100:
                iterations = iterations[-100:]
            self.performance_metrics["iteration_counts"] = iterations
            self.performance_metrics["avg_iterations"] = sum(iterations) / len(iterations)

    def _update_training_metrics(self, training_results: Any) -> None:
        """
        Update system metrics with training results.

        Args:
            training_results: Training results from VERL trainer
        """
        from ..training import TrainingResults

        # Handle both dict and TrainingResults objects
        if isinstance(training_results, TrainingResults):
            results_dict = training_results.dict()
        else:
            results_dict = training_results

        # Initialize training metrics if not exists
        if "training_metrics" not in self.performance_metrics:
            self.performance_metrics["training_metrics"] = {}

        training_metrics = self.performance_metrics["training_metrics"]

        # Update training session counter
        training_metrics["total_training_sessions"] = (
            training_metrics.get("total_training_sessions", 0) + 1
        )

        if results_dict.get("success", False):
            training_metrics["successful_training_sessions"] = (
                training_metrics.get("successful_training_sessions", 0) + 1
            )

        # Update training metrics
        metrics = results_dict.get("metrics", {})

        # Track final rewards
        if "final_reward" in metrics:
            final_rewards = training_metrics.get("final_rewards", [])
            final_rewards.append(metrics["final_reward"])
            if len(final_rewards) > 50:  # Keep last 50 training sessions
                final_rewards = final_rewards[-25:]
            training_metrics["final_rewards"] = final_rewards
            training_metrics["avg_final_reward"] = sum(final_rewards) / len(final_rewards)
            training_metrics["best_final_reward"] = max(final_rewards)

        # Track training times
        training_time = results_dict.get("training_time", 0.0)
        if training_time > 0:
            training_times = training_metrics.get("training_times", [])
            training_times.append(training_time)
            if len(training_times) > 50:
                training_times = training_times[-25:]
            training_metrics["training_times"] = training_times
            training_metrics["avg_training_time"] = sum(training_times) / len(training_times)

        # Track episodes completed
        episodes_completed = metrics.get("episodes_completed", 0)
        if episodes_completed > 0:
            episodes_list = training_metrics.get("episodes_completed", [])
            episodes_list.append(episodes_completed)
            if len(episodes_list) > 50:
                episodes_list = episodes_list[-25:]
            training_metrics["episodes_completed"] = episodes_list
            training_metrics["avg_episodes_completed"] = sum(episodes_list) / len(episodes_list)

        self.logger.info(
            "Training metrics updated",
            total_sessions=training_metrics.get("total_training_sessions", 0),
            successful_sessions=training_metrics.get("successful_training_sessions", 0),
            final_reward=metrics.get("final_reward", 0.0),
        )

    def get_performance_metrics(self) -> dict[str, Any]:
        """
        Get current performance metrics.

        Returns:
            Performance metrics dictionary
        """
        return self.performance_metrics.copy()

    def reset_performance_metrics(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics = {}
        self.logger.info("Performance metrics reset")

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the supervisor and all components.
        """
        self.logger.info("Shutting down supervisor")

        try:
            # Shutdown agents
            for agent_name, agent in self.agents.items():
                if hasattr(agent, "shutdown"):
                    await agent.shutdown()
                self.logger.info("Agent shutdown", agent=agent_name)

            # Shutdown LLM interface
            if self.llm_interface and hasattr(self.llm_interface, "shutdown"):
                await self.llm_interface.shutdown()

            self.is_initialized = False
            self.logger.info("Supervisor shutdown completed")

        except Exception as e:
            self.logger.error("Shutdown failed", error=str(e))
            raise

    def __repr__(self) -> str:
        """String representation of the supervisor."""
        return (
            f"CodingSupervisor("
            f"agents={len(self.agents)}, "
            f"initialized={self.is_initialized}, "
            f"solved={self.performance_metrics.get('total_problems_solved', 0)})"
        )
