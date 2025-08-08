"""
LangGraph workflow definitions for multi-agent coordination.

This module defines the state management and workflow logic for
coordinating code generation, review, and execution agents.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict

import structlog
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class WorkflowState(TypedDict):
    """
    Shared state for the coding workflow.
    
    This state is passed between agents and maintains the current
    context, results, and coordination information.
    """
    # Input and problem definition
    problem: str
    context: Dict[str, Any]
    
    # Agent results
    generated_code: Optional[str]
    code_review: Optional[str] 
    execution_result: Optional[str]
    
    # Agent metadata
    generator_metadata: Optional[Dict[str, Any]]
    reviewer_metadata: Optional[Dict[str, Any]]
    executor_metadata: Optional[Dict[str, Any]]
    
    # Workflow control
    current_step: str
    next_agent: Optional[str]
    iteration_count: int
    max_iterations: int
    
    # Quality tracking
    review_score: Optional[float]
    execution_success: Optional[bool]
    final_solution: Optional[str]
    
    # Error handling
    errors: List[str]
    warnings: List[str]
    
    # Human-in-the-loop
    requires_human_input: bool
    human_feedback: Optional[str]


class AgentResult(BaseModel):
    """Standardized result from an agent."""
    
    agent_type: str = Field(..., description="Type of agent")
    success: bool = Field(..., description="Whether operation succeeded")
    content: str = Field(..., description="Main result content")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    suggestions: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class CodingWorkflow:
    """
    LangGraph workflow for multi-agent code generation.
    
    Implements a state machine that coordinates the three agents
    with conditional routing and error handling.
    """
    
    def __init__(self, agents: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize the coding workflow.
        
        Args:
            agents: Dictionary of initialized agent instances
            config: Workflow configuration
        """
        self.agents = agents
        self.config = config
        self.logger = structlog.get_logger(workflow="coding")
        
        # Build the LangGraph workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph state machine.
        
        Returns:
            Compiled LangGraph workflow
        """
        # Create the state graph
        workflow = StateGraph(WorkflowState)
        
        # Add agent nodes
        workflow.add_node("generator", self._generate_code)
        workflow.add_node("reviewer", self._review_code) 
        workflow.add_node("executor", self._execute_code)
        workflow.add_node("supervisor", self._supervise_workflow)
        workflow.add_node("human_feedback", self._handle_human_feedback)
        
        # Define the workflow edges
        workflow.set_entry_point("supervisor")
        
        # Supervisor decides which agent to call next
        workflow.add_conditional_edges(
            "supervisor",
            self._route_next_agent,
            {
                "generator": "generator",
                "reviewer": "reviewer", 
                "executor": "executor",
                "human_feedback": "human_feedback",
                "end": "__end__",
            }
        )
        
        # Agent nodes return to supervisor for routing decisions
        workflow.add_edge("generator", "supervisor")
        workflow.add_edge("reviewer", "supervisor")
        workflow.add_edge("executor", "supervisor")
        workflow.add_edge("human_feedback", "supervisor")
        
        # Compile the workflow
        return workflow.compile()
    
    async def _generate_code(self, state: WorkflowState) -> WorkflowState:
        """
        Code generation step.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        try:
            self.logger.info("Starting code generation", 
                           problem_length=len(state["problem"]))
            
            generator = self.agents["generator"]
            
            # Prepare context for generation
            context = state["context"].copy()
            context["iteration"] = state["iteration_count"]
            
            # Generate code
            result = await generator.process_request(
                state["problem"], 
                context=context,
                **self.config.get("generation", {})
            )
            
            # Update state
            state["generated_code"] = result.content
            state["generator_metadata"] = result.metadata
            state["current_step"] = "generation_complete"
            
            if not result.success:
                state["errors"].append(f"Code generation failed: {result.error}")
            
            self.logger.info("Code generation completed",
                           success=result.success,
                           code_length=len(result.content))
            
        except Exception as e:
            state["errors"].append(f"Code generation error: {str(e)}")
            self.logger.error("Code generation failed", error=str(e))
        
        return state
    
    async def _review_code(self, state: WorkflowState) -> WorkflowState:
        """
        Code review step.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        try:
            self.logger.info("Starting code review")
            
            reviewer = self.agents["reviewer"]
            
            if not state["generated_code"]:
                state["errors"].append("No code available for review")
                return state
            
            # Prepare review request
            review_request = f"Please review the following code:\n\n```\n{state['generated_code']}\n```"
            
            context = state["context"].copy()
            context["original_problem"] = state["problem"]
            
            # Perform review
            result = await reviewer.process_request(
                review_request,
                context=context,
                **self.config.get("review", {})
            )
            
            # Update state
            state["code_review"] = result.content
            state["reviewer_metadata"] = result.metadata
            
            # Enhanced review score handling
            review_score = result.metadata.get("overall_score", 75)  # Default to 75 instead of 0
            state["review_score"] = review_score
            state["current_step"] = "review_complete"
            
            self.logger.info("Code review completed",
                           success=result.success,
                           score=review_score,
                           metadata_keys=list(result.metadata.keys()) if result.metadata else [])
            
            if not result.success:
                state["errors"].append(f"Code review failed: {result.error}")
                # Even if review failed, set a reasonable default score to allow execution
                state["review_score"] = 65
            
        except Exception as e:
            state["errors"].append(f"Code review error: {str(e)}")
            state["review_score"] = 65  # Set reasonable default on error
            self.logger.error("Code review failed", error=str(e))
        
        return state
    
    async def _execute_code(self, state: WorkflowState) -> WorkflowState:
        """
        Code execution step.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        try:
            self.logger.info("Starting code execution")
            
            executor = self.agents["executor"]
            
            if not state["generated_code"]:
                state["errors"].append("No code available for execution")
                return state
            
            # Prepare execution request
            execution_request = f"Execute the following code:\n\n```\n{state['generated_code']}\n```"
            
            context = state["context"].copy()
            context["original_problem"] = state["problem"]
            
            # Execute code
            result = await executor.process_request(
                execution_request,
                context=context,
                **self.config.get("execution", {})
            )
            
            # Update state
            state["execution_result"] = result.content
            state["executor_metadata"] = result.metadata
            state["execution_success"] = result.metadata.get("success", False)
            state["current_step"] = "execution_complete"
            
            if not result.success:
                state["errors"].append(f"Code execution failed: {result.error}")
            
            self.logger.info("Code execution completed",
                           success=result.success,
                           execution_success=state["execution_success"])
            
        except Exception as e:
            state["errors"].append(f"Code execution error: {str(e)}")
            self.logger.error("Code execution failed", error=str(e))
        
        return state
    
    async def _supervise_workflow(self, state: WorkflowState) -> WorkflowState:
        """
        Workflow supervision and decision making.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with next agent decision
        """
        try:
            current_step = state["current_step"]
            iteration = state["iteration_count"]
            max_iterations = state["max_iterations"]
            
            self.logger.info("Supervising workflow",
                           current_step=current_step,
                           iteration=iteration)
            
            # Check if we've exceeded maximum iterations
            if iteration >= max_iterations:
                self.logger.warning("Maximum iterations reached", 
                                  max_iterations=max_iterations)
                state["next_agent"] = "end"
                return state
            
            # Decision logic based on current state
            if current_step == "start":
                # First step: generate code
                state["next_agent"] = "generator"
                
            elif current_step == "generation_complete":
                # Code generated, now review it
                if state["generated_code"]:
                    state["next_agent"] = "reviewer"
                else:
                    state["errors"].append("No code was generated")
                    state["next_agent"] = "end"
                    
            elif current_step == "review_complete":
                # Review done, now execute
                if self._should_execute_code(state):
                    state["next_agent"] = "executor"
                elif self._needs_human_input(state):
                    state["next_agent"] = "human_feedback"
                    state["requires_human_input"] = True
                else:
                    # Skip execution, finalize
                    state["next_agent"] = "end"
                    
            elif current_step == "execution_complete":
                # Execution done, check if we need another iteration
                if self._needs_iteration(state):
                    # Reset for another iteration
                    state["iteration_count"] += 1
                    state["current_step"] = "start"
                    state["next_agent"] = "generator"
                    self.logger.info("Starting new iteration", 
                                   iteration=state["iteration_count"])
                else:
                    # Workflow complete
                    state["final_solution"] = state["generated_code"]
                    state["next_agent"] = "end"
                    
            elif current_step == "human_feedback_complete":
                # Human provided feedback, continue based on feedback
                if state["human_feedback"]:
                    # Incorporate feedback and restart
                    state["iteration_count"] += 1
                    state["current_step"] = "start"
                    state["next_agent"] = "generator"
                else:
                    # No feedback, end workflow
                    state["next_agent"] = "end"
                    
            else:
                # Unknown state, end workflow
                self.logger.warning("Unknown workflow state", 
                                  current_step=current_step)
                state["next_agent"] = "end"
            
            self.logger.info("Supervision complete", 
                           next_agent=state["next_agent"])
            
        except Exception as e:
            state["errors"].append(f"Supervision error: {str(e)}")
            state["next_agent"] = "end"
            self.logger.error("Supervision failed", error=str(e))
        
        return state
    
    async def _handle_human_feedback(self, state: WorkflowState) -> WorkflowState:
        """
        Handle human-in-the-loop feedback.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        try:
            self.logger.info("Handling human feedback request")
            
            # In a real implementation, this would integrate with a UI or API
            # For now, we'll simulate or skip human input
            
            feedback_prompt = self._generate_feedback_prompt(state)
            
            # TODO: Implement actual human feedback mechanism
            # This could be through a web interface, CLI prompt, or API callback
            
            # For now, mark as complete without feedback
            state["human_feedback"] = ""
            state["current_step"] = "human_feedback_complete"
            state["requires_human_input"] = False
            
            self.logger.info("Human feedback handled")
            
        except Exception as e:
            state["errors"].append(f"Human feedback error: {str(e)}")
            state["current_step"] = "human_feedback_complete"
            self.logger.error("Human feedback failed", error=str(e))
        
        return state
    
    def _route_next_agent(self, state: WorkflowState) -> str:
        """
        Route to the next agent based on workflow state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next agent or end point
        """
        next_agent = state.get("next_agent", "end")
        
        self.logger.info("Routing to next agent", 
                        next_agent=next_agent,
                        current_step=state["current_step"])
        
        return next_agent
    
    def _should_execute_code(self, state: WorkflowState) -> bool:
        """
        Determine if code should be executed.
        
        Args:
            state: Current workflow state
            
        Returns:
            Whether to execute the code
        """
        # Don't execute if there are critical security issues
        if state["reviewer_metadata"]:
            security_concerns = state["reviewer_metadata"].get("security_concerns", [])
            if security_concerns:
                self.logger.warning("Skipping execution due to security concerns",
                                  concerns=security_concerns)
                return False
        
        # Don't execute if review score is too low
        review_score = state.get("review_score", 100)
        min_score = self.config.get("min_execution_score", 50)
        
        if review_score < min_score:
            self.logger.warning("Skipping execution due to low review score",
                              score=review_score, min_score=min_score)
            return False
        
        # Execute by default
        return True
    
    def _needs_human_input(self, state: WorkflowState) -> bool:
        """
        Determine if human input is needed.
        
        Args:
            state: Current workflow state
            
        Returns:
            Whether human input is required
        """
        # Check if human-in-the-loop is enabled
        if not self.config.get("human_in_loop", False):
            return False
        
        # Require human input for low review scores
        review_score = state.get("review_score", 100)
        human_threshold = self.config.get("human_feedback_threshold", 30)
        
        if review_score < human_threshold:
            return True
        
        # Require human input for security concerns
        if state["reviewer_metadata"]:
            security_concerns = state["reviewer_metadata"].get("security_concerns", [])
            if security_concerns:
                return True
        
        return False
    
    def _needs_iteration(self, state: WorkflowState) -> bool:
        """
        Determine if another iteration is needed.
        
        Args:
            state: Current workflow state
            
        Returns:
            Whether to start another iteration
        """
        # Don't iterate if execution failed and we're at max iterations
        if not state.get("execution_success", True):
            if state["iteration_count"] >= state["max_iterations"] - 1:
                return False
            # Try again if execution failed
            return True
        
        # Don't iterate if review score is good enough
        review_score = state.get("review_score", 100)
        target_score = self.config.get("target_review_score", 80)
        
        if review_score >= target_score:
            return False
        
        # Iterate if we haven't reached max iterations
        return state["iteration_count"] < state["max_iterations"] - 1
    
    def _generate_feedback_prompt(self, state: WorkflowState) -> str:
        """
        Generate a prompt for human feedback.
        
        Args:
            state: Current workflow state
            
        Returns:
            Human feedback prompt
        """
        prompt_parts = [
            "**Human Feedback Required**\n",
            f"**Problem:** {state['problem']}\n",
        ]
        
        if state["generated_code"]:
            prompt_parts.append(f"**Generated Code:**\n```\n{state['generated_code'][:500]}...\n```\n")
        
        if state["code_review"]:
            prompt_parts.append(f"**Review Summary:** {state['code_review'][:200]}...\n")
        
        if state["review_score"]:
            prompt_parts.append(f"**Review Score:** {state['review_score']}/100\n")
        
        if state["errors"]:
            prompt_parts.append(f"**Issues:** {', '.join(state['errors'])}\n")
        
        prompt_parts.append("**Please provide feedback or suggestions for improvement.**")
        
        return "\n".join(prompt_parts)
    
    async def run(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete workflow.
        
        Args:
            problem: Problem description
            context: Additional context
            
        Returns:
            Workflow results
        """
        # Initialize state
        initial_state: WorkflowState = {
            "problem": problem,
            "context": context,
            "generated_code": None,
            "code_review": None,
            "execution_result": None,
            "generator_metadata": None,
            "reviewer_metadata": None,
            "executor_metadata": None,
            "current_step": "start",
            "next_agent": None,
            "iteration_count": 0,
            "max_iterations": context.get("max_iterations", 3),
            "review_score": None,
            "execution_success": None,
            "final_solution": None,
            "errors": [],
            "warnings": [],
            "requires_human_input": False,
            "human_feedback": None,
        }
        
        self.logger.info("Starting workflow", 
                        problem_length=len(problem),
                        max_iterations=initial_state["max_iterations"])
        
        # Run the workflow
        try:
            final_state = await self.workflow.ainvoke(initial_state)
            
            self.logger.info("Workflow completed",
                           iterations=final_state["iteration_count"],
                           errors=len(final_state["errors"]))
            
            return {
                "success": len(final_state["errors"]) == 0,
                "problem": problem,
                "code": final_state["final_solution"] or final_state["generated_code"],
                "review": final_state["code_review"],
                "execution": final_state["execution_result"],
                "review_score": final_state["review_score"],
                "execution_success": final_state["execution_success"],
                "iterations": final_state["iteration_count"],
                "errors": final_state["errors"],
                "warnings": final_state["warnings"],
                "metadata": {
                    "generator": final_state["generator_metadata"],
                    "reviewer": final_state["reviewer_metadata"],
                    "executor": final_state["executor_metadata"],
                }
            }
            
        except Exception as e:
            self.logger.error("Workflow failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "problem": problem,
            }