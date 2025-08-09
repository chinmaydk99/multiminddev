from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
import time
import structlog

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    # Mock add_messages for typing
    def add_messages(x): 
        return x

from ..agents.cuda_generator import CUDAGeneratorAgent
from ..agents.cuda_optimizer import CUDAOptimizerAgent
from ..agents.cuda_tester import CUDATesterAgent
from ..cuda.compiler import CUDACompiler
from ..cuda.benchmarker import CUDABenchmarker


class CUDAWorkflowState(TypedDict):
    """Extended workflow state for CUDA kernel generation."""
    
    # Input specification
    pytorch_operation: str
    target_performance: float  # Target speedup ratio
    test_inputs: List[Dict[str, Any]]
    
    # Multi-turn conversation history
    generated_kernels: List[str]  # All kernel variants generated
    optimization_history: List[str]  # Optimization suggestions applied
    performance_history: List[float]  # Performance across iterations
    
    # Current best solution
    best_kernel: Optional[str]
    best_performance: float
    best_iteration: int
    
    # Conversation control
    optimization_turn: int
    max_optimization_turns: int
    
    # Agent responses and metadata
    messages: Annotated[list, add_messages]
    current_agent: str
    workflow_status: str  # "running", "completed", "failed"
    
    # Performance tracking
    start_time: float
    total_execution_time: float


class CUDAKernelWorkflow:
    """LangGraph workflow for multi-turn CUDA kernel optimization."""
    
    def __init__(
        self, 
        cuda_generator: CUDAGeneratorAgent,
        cuda_optimizer: CUDAOptimizerAgent, 
        cuda_tester: CUDATesterAgent,
        config: Optional[Dict[str, Any]] = None
    ):
        self.cuda_generator = cuda_generator
        self.cuda_optimizer = cuda_optimizer
        self.cuda_tester = cuda_tester
        self.config = config or {}
        
        # Initialize CUDA execution environment
        self.compiler = CUDACompiler()
        self.benchmarker = CUDABenchmarker()
        
        self.logger = structlog.get_logger("cuda_workflow")
        
        if not LANGGRAPH_AVAILABLE:
            self.logger.warning("LangGraph not available - using fallback workflow")
            self.workflow = None
        else:
            # Build multi-turn workflow graph
            self.workflow = self._build_workflow_graph()
        
        self.logger.info("CUDA workflow initialized", config=config)
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build LangGraph state machine for CUDA optimization."""
        if not LANGGRAPH_AVAILABLE:
            raise RuntimeError("LangGraph not available")
        
        workflow = StateGraph(CUDAWorkflowState)
        
        # Add workflow nodes
        workflow.add_node("generate_initial", self._generate_initial_kernel)
        workflow.add_node("test_kernel", self._test_and_profile_kernel)
        workflow.add_node("optimize_kernel", self._optimize_kernel_performance)
        # Note: evaluate_improvement node removed for simplicity
        workflow.add_node("finalize_result", self._finalize_best_kernel)
        
        # Add workflow edges  
        workflow.add_edge("generate_initial", "test_kernel")
        workflow.add_conditional_edges(
            "test_kernel",
            self._should_continue_optimizing,
            {
                "optimize": "optimize_kernel",
                "finalize": "finalize_result"
            }
        )
        workflow.add_edge("optimize_kernel", "test_kernel")
        # Note: evaluate_improvement edge removed
        workflow.add_edge("finalize_result", END)
        
        # Set entry point
        workflow.set_entry_point("generate_initial")
        
        return workflow.compile()
    
    async def run_workflow(
        self,
        pytorch_operation: str,
        test_inputs: List[Dict[str, Any]],
        target_performance: float = 2.0,
        max_optimization_turns: int = 5
    ) -> Dict[str, Any]:
        """
        Run complete CUDA kernel optimization workflow.
        
        Args:
            pytorch_operation: Description of PyTorch operation to convert
            test_inputs: Test input specifications
            target_performance: Target speedup ratio
            max_optimization_turns: Maximum optimization iterations
            
        Returns:
            Workflow results with best kernel and performance metrics
        """
        start_time = time.time()
        
        # Initialize workflow state
        initial_state: CUDAWorkflowState = {
            "pytorch_operation": pytorch_operation,
            "target_performance": target_performance,
            "test_inputs": test_inputs,
            "generated_kernels": [],
            "optimization_history": [],
            "performance_history": [],
            "best_kernel": None,
            "best_performance": 0.0,
            "best_iteration": -1,
            "optimization_turn": 0,
            "max_optimization_turns": max_optimization_turns,
            "messages": [],
            "current_agent": "cuda_generator",
            "workflow_status": "running",
            "start_time": start_time,
            "total_execution_time": 0.0
        }
        
        try:
            if self.workflow and LANGGRAPH_AVAILABLE:
                # Use LangGraph workflow
                result = await self.workflow.ainvoke(initial_state)
            else:
                # Use fallback sequential workflow
                result = await self._run_fallback_workflow(initial_state)
            
            result["total_execution_time"] = time.time() - start_time
            result["workflow_status"] = "completed"
            
            self.logger.info(
                "CUDA workflow completed successfully",
                execution_time=result["total_execution_time"],
                best_performance=result.get("best_performance", 0.0),
                optimization_turns=result.get("optimization_turn", 0)
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Workflow failed: {str(e)}"
            
            self.logger.error(
                "CUDA workflow failed",
                error=error_msg,
                execution_time=execution_time
            )
            
            return {
                "workflow_status": "failed",
                "error": error_msg,
                "total_execution_time": execution_time,
                "best_kernel": None,
                "best_performance": 0.0
            }
    
    async def _run_fallback_workflow(self, state: CUDAWorkflowState) -> Dict[str, Any]:
        """Fallback workflow implementation without LangGraph."""
        
        # Step 1: Generate initial kernel
        state = await self._generate_initial_kernel(state)
        if not state["generated_kernels"]:
            raise RuntimeError("Failed to generate initial kernel")
        
        # Step 2: Test and optimize in loop
        while state["optimization_turn"] < state["max_optimization_turns"]:
            # Test current kernel
            state = await self._test_and_profile_kernel(state)
            
            # Check if we should continue optimizing
            if self._should_continue_optimizing(state) == "finalize":
                break
            
            # Optimize kernel
            state = await self._optimize_kernel_performance(state)
            state["optimization_turn"] += 1
        
        # Step 3: Finalize results
        state = await self._finalize_best_kernel(state)
        
        return dict(state)
    
    async def _generate_initial_kernel(self, state: CUDAWorkflowState) -> CUDAWorkflowState:
        """Generate initial CUDA kernel from PyTorch operation."""
        
        self.logger.info("Generating initial CUDA kernel", operation=state["pytorch_operation"])
        
        response = await self.cuda_generator.process_request(
            state["pytorch_operation"],
            context={
                "test_inputs": state["test_inputs"],
                "target_performance": state["target_performance"]
            }
        )
        
        if response.success and response.content:
            state["generated_kernels"].append(response.content)
            state["best_kernel"] = response.content
            state["messages"].append({
                "role": "assistant",
                "content": response.content,
                "agent": "cuda_generator"
            })
            
            self.logger.info("Initial kernel generated successfully")
        else:
            error_msg = response.error or "Failed to generate initial kernel"
            self.logger.error("Initial kernel generation failed", error=error_msg)
            raise RuntimeError(error_msg)
        
        return state
    
    async def _test_and_profile_kernel(self, state: CUDAWorkflowState) -> CUDAWorkflowState:
        """Test kernel functionality and measure performance."""
        
        current_kernel = state["generated_kernels"][-1]
        
        self.logger.info(
            "Testing and profiling kernel",
            kernel_length=len(current_kernel),
            turn=state["optimization_turn"]
        )
        
        # Get analysis from tester agent
        response = await self.cuda_tester.process_request(
            current_kernel,
            context={
                "test_inputs": state["test_inputs"],
                "pytorch_operation": state["pytorch_operation"]
            }
        )
        
        if response.success:
            # Extract performance score from metadata
            performance = response.metadata.get("estimated_speedup", 1.0)
            state["performance_history"].append(performance)
            
            # Update best if improved
            if performance > state["best_performance"]:
                state["best_kernel"] = current_kernel
                state["best_performance"] = performance
                state["best_iteration"] = state["optimization_turn"]
            
            state["messages"].append({
                "role": "assistant", 
                "content": response.content,
                "agent": "cuda_tester",
                "metadata": response.metadata
            })
            
            self.logger.info(
                "Kernel testing completed",
                performance=performance,
                is_best=performance > state["best_performance"]
            )
        else:
            self.logger.warning("Kernel testing failed", error=response.error)
            # Add minimal performance entry to avoid breaking workflow
            state["performance_history"].append(0.0)
        
        return state
    
    async def _optimize_kernel_performance(self, state: CUDAWorkflowState) -> CUDAWorkflowState:
        """Generate optimized kernel variant."""
        
        current_kernel = state["generated_kernels"][-1]
        performance_feedback = self._generate_performance_feedback(state)
        
        self.logger.info(
            "Optimizing kernel performance",
            turn=state["optimization_turn"],
            current_performance=state["performance_history"][-1] if state["performance_history"] else 0.0
        )
        
        response = await self.cuda_optimizer.process_request(
            current_kernel,
            context={
                "performance_feedback": performance_feedback,
                "target_performance": state["target_performance"],
                "optimization_history": state["optimization_history"]
            }
        )
        
        if response.success and response.content:
            state["generated_kernels"].append(response.content)
            optimization_applied = response.metadata.get("optimizations_applied", ["general_optimization"])
            state["optimization_history"].extend(optimization_applied)
            
            state["messages"].append({
                "role": "assistant",
                "content": response.content,
                "agent": "cuda_optimizer",
                "metadata": response.metadata
            })
            
            self.logger.info(
                "Kernel optimization completed",
                optimizations_applied=optimization_applied
            )
        else:
            self.logger.warning("Kernel optimization failed", error=response.error)
            # Keep current kernel if optimization fails
            state["generated_kernels"].append(current_kernel)
        
        return state
    
    def _should_continue_optimizing(self, state: CUDAWorkflowState) -> str:
        """Decide whether to continue optimization or finalize."""
        
        # Stop if max turns reached
        if state["optimization_turn"] >= state["max_optimization_turns"]:
            self.logger.info("Max optimization turns reached", turns=state["optimization_turn"])
            return "finalize"
            
        # Stop if target performance achieved  
        if state["best_performance"] >= state["target_performance"]:
            self.logger.info(
                "Target performance achieved",
                best_performance=state["best_performance"],
                target=state["target_performance"]
            )
            return "finalize"
            
        # Stop if no improvement in last 2 iterations
        if len(state["performance_history"]) >= 3:
            recent_performance = state["performance_history"][-3:]
            if max(recent_performance) <= state["best_performance"]:
                self.logger.info("No recent improvement, finalizing")
                return "finalize"
        
        return "optimize"
    
    async def _finalize_best_kernel(self, state: CUDAWorkflowState) -> CUDAWorkflowState:
        """Finalize workflow with best kernel results."""
        
        self.logger.info(
            "Finalizing workflow results",
            best_performance=state["best_performance"],
            best_iteration=state["best_iteration"],
            total_turns=state["optimization_turn"]
        )
        
        # Ensure we have a best kernel
        if state["best_kernel"] is None and state["generated_kernels"]:
            state["best_kernel"] = state["generated_kernels"][-1]
        
        state["workflow_status"] = "completed"
        state["current_agent"] = "finalized"
        
        return state
    
    def _generate_performance_feedback(self, state: CUDAWorkflowState) -> str:
        """Generate performance feedback for optimization."""
        if not state["performance_history"]:
            return "No performance data available yet."
        
        current_perf = state["performance_history"][-1]
        target_perf = state["target_performance"]
        
        if current_perf >= target_perf:
            return f"Performance target achieved: {current_perf:.2f}x >= {target_perf}x"
        else:
            gap = target_perf - current_perf
            return f"Performance gap: {gap:.2f}x below target. Current: {current_perf:.2f}x, Target: {target_perf}x"
    
    async def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status and metrics."""
        return {
            "compiler_info": self.compiler.get_compilation_info(),
            "agents_healthy": {
                "generator": (await self.cuda_generator.health_check())["status"] == "healthy",
                "optimizer": (await self.cuda_optimizer.health_check())["status"] == "healthy", 
                "tester": (await self.cuda_tester.health_check())["status"] == "healthy"
            },
            "workflow_ready": LANGGRAPH_AVAILABLE,
            "config": self.config
        }