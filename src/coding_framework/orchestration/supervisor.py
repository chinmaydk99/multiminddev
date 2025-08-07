"""
Supervisor for multi-agent coordination and workflow management.

This module provides the main CodingSupervisor class that orchestrates
the three specialized agents using LangGraph workflows.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel

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
        self.agents: Dict[str, BaseAgent] = {}
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
        
        self.logger.info("LLM interface initialized",
                        provider=self.config.llm.provider)
    
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
            
            self.logger.info("All agents initialized", 
                           agent_count=len(self.agents))
            
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
                self.logger.error("Agent health check failed",
                                agent=agent_name, error=str(result))
                raise result
            else:
                self.logger.info("Agent health verified",
                               agent=agent_name, status=result["status"])
    
    async def _check_single_agent_health(
        self,
        agent_name: str,
        agent: BaseAgent,
    ) -> Dict[str, Any]:
        """Check health of a single agent."""
        try:
            health_status = await agent.health_check()
            return health_status
        except Exception as e:
            self.logger.error("Agent health check failed",
                            agent=agent_name, error=str(e))
            raise
    
    async def solve_problem(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
        
        self.logger.info("Starting problem solving",
                        problem_length=len(problem),
                        context_keys=list(context.keys()))
        
        try:
            # Run the workflow
            result = await self.workflow.run(problem, context)
            
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self._update_performance_metrics(result, execution_time)
            
            # Add timing information
            result["execution_time"] = execution_time
            result["timestamp"] = time.time()
            
            self.logger.info("Problem solving completed",
                           success=result["success"],
                           execution_time=execution_time,
                           iterations=result.get("iterations", 0))
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            self.logger.error("Problem solving failed",
                            error=str(e),
                            execution_time=execution_time)
            
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
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
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
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
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
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
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
    
    async def health_check(self) -> Dict[str, Any]:
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
                result.get("status") == "healthy" 
                for result in health_results.values()
            )
            
            health_results["system"] = {
                "status": "healthy" if all_healthy else "unhealthy",
                "components": len(health_results),
                "initialized": self.is_initialized,
                "performance_metrics": self.performance_metrics,
            }
            
            self.logger.info("Health check completed", 
                           overall_status=health_results["system"]["status"])
            
            return health_results
            
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }
    
    async def check_agent_health(self, agent_name: str) -> Dict[str, Any]:
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
            
            self.logger.info("Agent health checked",
                           agent=agent_name,
                           status=health_status["status"])
            
            return health_status
            
        except Exception as e:
            self.logger.error("Agent health check failed",
                            agent=agent_name, error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }
    
    async def train_agents(
        self,
        algorithm: str = "ppo",
        episodes: int = 100,
        **kwargs,
    ) -> Dict[str, Any]:
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
        
        self.logger.info("Starting agent training",
                        algorithm=algorithm,
                        episodes=episodes)
        
        try:
            # This is a placeholder for VERL training integration
            # In a full implementation, this would:
            # 1. Initialize VERL training pipeline
            # 2. Set up reward functions
            # 3. Configure training data
            # 4. Run RL training loop
            # 5. Update agent models
            
            # For now, return a mock training result
            training_results = {
                "success": True,
                "algorithm": algorithm,
                "episodes": episodes,
                "metrics": {
                    "final_reward": 0.85,
                    "training_time": 3600,
                    "convergence_episode": 75,
                    "best_score": 0.92,
                },
                "message": "VERL training not yet implemented - this is a placeholder",
            }
            
            self.logger.info("Training completed (placeholder)",
                           success=training_results["success"])
            
            return training_results
            
        except Exception as e:
            self.logger.error("Training failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "algorithm": algorithm,
            }
    
    def _update_performance_metrics(
        self,
        result: Dict[str, Any],
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
    
    def get_performance_metrics(self) -> Dict[str, Any]:
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
                if hasattr(agent, 'shutdown'):
                    await agent.shutdown()
                self.logger.info("Agent shutdown", agent=agent_name)
            
            # Shutdown LLM interface
            if self.llm_interface and hasattr(self.llm_interface, 'shutdown'):
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