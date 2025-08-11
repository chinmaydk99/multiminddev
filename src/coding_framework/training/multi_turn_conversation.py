import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog
import torch


class AgentType(Enum):
    GENERATOR = "generator"
    OPTIMIZER = "optimizer"
    TESTER = "tester"

class TurnStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"

@dataclass
class ConversationTurn:
    """Single turn in a multi-agent conversation."""
    turn_number: int
    agent_type: AgentType
    prompt: str
    response: str
    status: TurnStatus

    # Performance metrics for this turn
    execution_time: float = 0.0
    compilation_success: bool = False
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    # Context and metadata
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # RL training data
    log_probs: Optional[torch.Tensor] = None
    token_ids: Optional[List[int]] = None
    attention_mask: Optional[torch.Tensor] = None

@dataclass
class CUDAConversationState:
    """Complete state of a multi-turn CUDA optimization conversation."""
    conversation_id: str
    problem_description: str
    difficulty_tier: str

    # Conversation history
    turns: List[ConversationTurn] = field(default_factory=list)

    # Current state
    current_kernel_code: str = ""
    current_performance: Dict[str, float] = field(default_factory=dict)
    target_performance: Dict[str, float] = field(default_factory=dict)

    # Conversation outcomes
    final_reward: float = 0.0
    conversation_success: bool = False
    termination_reason: str = ""

    # Metadata
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_turns: int = 0

class MultiTurnConversationManager:
    """Manages multi-turn conversations between CUDA optimization agents."""

    def __init__(
        self,
        generator_agent,
        optimizer_agent,
        tester_agent,
        compiler,
        benchmarker,
        max_turns: int = 5,
        early_termination_threshold: float = 2.0,  # 2x speedup
        performance_patience: int = 2  # Turns to wait for improvement
    ):
        self.generator_agent = generator_agent
        self.optimizer_agent = optimizer_agent
        self.tester_agent = tester_agent
        self.compiler = compiler
        self.benchmarker = benchmarker

        self.max_turns = max_turns
        self.early_termination_threshold = early_termination_threshold
        self.performance_patience = performance_patience

        self.logger = structlog.get_logger()

    async def run_conversation(
        self,
        problem: Dict[str, Any],
        conversation_id: str
    ) -> CUDAConversationState:
        """Run complete multi-turn conversation for CUDA optimization."""

        # Initialize conversation state
        conversation = CUDAConversationState(
            conversation_id=conversation_id,
            problem_description=problem["description"],
            difficulty_tier=problem.get("difficulty", "medium"),
            target_performance=problem.get("target_performance", {"speedup": 2.0})
        )

        self.logger.info(
            "Starting multi-turn conversation",
            conversation_id=conversation_id,
            problem=problem["description"][:100] + "...",
            max_turns=self.max_turns
        )

        try:
            # Phase 1: Initial code generation
            await self._generation_phase(conversation, problem)

            # Phase 2: Iterative optimization
            await self._optimization_phase(conversation, problem)

            # Phase 3: Final evaluation
            await self._final_evaluation(conversation)

        except Exception as e:
            self.logger.error(
                "Conversation failed with exception",
                conversation_id=conversation_id,
                error=str(e)
            )
            conversation.termination_reason = f"Exception: {str(e)}"

        conversation.end_time = time.time()
        conversation.total_turns = len(conversation.turns)

        self.logger.info(
            "Conversation completed",
            conversation_id=conversation_id,
            total_turns=conversation.total_turns,
            success=conversation.conversation_success,
            final_reward=conversation.final_reward,
            termination_reason=conversation.termination_reason
        )

        return conversation

    async def _generation_phase(
        self,
        conversation: CUDAConversationState,
        problem: Dict[str, Any]
    ):
        """Phase 1: Generate initial CUDA kernel."""

        # Prepare prompt for generator
        generator_prompt = self._create_generator_prompt(problem)

        # Generate initial kernel
        generator_response = await self.generator_agent.generate_response(
            generator_prompt,
            max_tokens=1024,
            temperature=0.7
        )

        # Record turn
        turn = ConversationTurn(
            turn_number=0,
            agent_type=AgentType.GENERATOR,
            prompt=generator_prompt,
            response=generator_response["text"],
            status=TurnStatus.SUCCESS,  # Will be updated after testing
            log_probs=generator_response.get("log_probs"),
            token_ids=generator_response.get("token_ids")
        )

        # Extract kernel code from response
        kernel_code = self._extract_kernel_code(generator_response["text"])
        conversation.current_kernel_code = kernel_code

        # Test initial generation
        await self._test_kernel(conversation, turn, kernel_code)

        conversation.turns.append(turn)

    async def _optimization_phase(
        self,
        conversation: CUDAConversationState,
        problem: Dict[str, Any]
    ):
        """Phase 2: Iterative optimization through agent collaboration."""

        turns_without_improvement = 0
        best_performance = 0.0

        for turn_number in range(1, self.max_turns):

            # Check early termination conditions
            current_speedup = conversation.current_performance.get("speedup", 0.0)

            if current_speedup >= self.early_termination_threshold:
                conversation.termination_reason = f"Target speedup achieved: {current_speedup:.2f}x"
                conversation.conversation_success = True
                break

            if turns_without_improvement >= self.performance_patience:
                conversation.termination_reason = f"No improvement for {self.performance_patience} turns"
                break

            # Determine next agent based on current state
            if conversation.turns[-1].compilation_success:
                # If code compiles, use optimizer to improve performance
                next_agent = self.optimizer_agent
                next_agent_type = AgentType.OPTIMIZER
                prompt = self._create_optimizer_prompt(conversation, problem)
            else:
                # If code doesn't compile, use generator to fix it
                next_agent = self.generator_agent
                next_agent_type = AgentType.GENERATOR
                prompt = self._create_fix_prompt(conversation, problem)

            # Generate response
            response = await next_agent.generate_response(
                prompt,
                max_tokens=1024,
                temperature=0.5  # Slightly more conservative in optimization
            )

            # Create turn record
            turn = ConversationTurn(
                turn_number=turn_number,
                agent_type=next_agent_type,
                prompt=prompt,
                response=response["text"],
                status=TurnStatus.SUCCESS,
                log_probs=response.get("log_probs"),
                token_ids=response.get("token_ids")
            )

            # Extract and test new kernel code
            new_kernel_code = self._extract_kernel_code(response["text"])
            if new_kernel_code and new_kernel_code != conversation.current_kernel_code:
                conversation.current_kernel_code = new_kernel_code
                await self._test_kernel(conversation, turn, new_kernel_code)

                # Check for performance improvement
                new_performance = conversation.current_performance.get("speedup", 0.0)
                if new_performance > best_performance:
                    best_performance = new_performance
                    turns_without_improvement = 0
                else:
                    turns_without_improvement += 1
            else:
                # No new code generated
                turn.status = TurnStatus.FAILURE
                turns_without_improvement += 1

            conversation.turns.append(turn)

        # If we exit the loop without success, mark as incomplete
        if not conversation.conversation_success:
            if conversation.termination_reason == "":
                conversation.termination_reason = "Maximum turns reached"

    async def _test_kernel(
        self,
        conversation: CUDAConversationState,
        turn: ConversationTurn,
        kernel_code: str
    ):
        """Test kernel compilation and performance."""

        turn_start_time = time.time()

        try:
            # Compile kernel
            compilation_result = await self.compiler.compile_kernel(
                kernel_code,
                kernel_name=f"kernel_{conversation.conversation_id}_{turn.turn_number}"
            )

            turn.compilation_success = compilation_result.success
            turn.context["compilation_result"] = compilation_result

            if compilation_result.success:
                # Benchmark kernel
                test_cases = self._generate_test_cases(conversation.difficulty_tier)
                benchmark_result = await self.benchmarker.benchmark_kernel(
                    compilation_result.binary_path,
                    compilation_result.kernel_name,
                    test_inputs=[],  # Will be generated from test_cases
                    test_cases=test_cases
                )

                turn.context["benchmark_result"] = benchmark_result

                if benchmark_result.success and benchmark_result.functional_correct:
                    # Update conversation performance
                    conversation.current_performance = {
                        "speedup": benchmark_result.speedup_vs_torch,
                        "memory_bandwidth": benchmark_result.memory_bandwidth_gb_s,
                        "execution_time": benchmark_result.execution_time_ms,
                        "accuracy": benchmark_result.numerical_accuracy
                    }

                    turn.performance_metrics = conversation.current_performance.copy()
                    turn.status = TurnStatus.SUCCESS
                else:
                    turn.status = TurnStatus.FAILURE
                    turn.context["failure_reason"] = benchmark_result.error_message
            else:
                turn.status = TurnStatus.FAILURE
                turn.context["failure_reason"] = compilation_result.stderr

        except Exception as e:
            turn.status = TurnStatus.FAILURE
            turn.context["failure_reason"] = str(e)
            self.logger.error(
                "Kernel testing failed",
                conversation_id=conversation.conversation_id,
                turn_number=turn.turn_number,
                error=str(e)
            )

        turn.execution_time = time.time() - turn_start_time

    async def _final_evaluation(self, conversation: CUDAConversationState):
        """Final evaluation and reward calculation."""

        # Calculate final reward based on conversation outcome
        final_reward = 0.0

        # Base reward for compilation success
        if conversation.turns and conversation.turns[-1].compilation_success:
            final_reward += 0.3

        # Performance reward
        current_speedup = conversation.current_performance.get("speedup", 0.0)
        target_speedup = conversation.target_performance.get("speedup", 2.0)

        if current_speedup > 0:
            performance_ratio = min(current_speedup / target_speedup, 2.0)
            final_reward += 0.5 * performance_ratio

        # Efficiency reward (fewer turns is better)
        if conversation.conversation_success:
            efficiency_bonus = max(0, (self.max_turns - len(conversation.turns)) / self.max_turns)
            final_reward += 0.2 * efficiency_bonus

        conversation.final_reward = final_reward

    def _create_generator_prompt(self, problem: Dict[str, Any]) -> str:
        """Create prompt for initial code generation."""
        return f"""
Generate a CUDA kernel to solve the following problem:

Problem: {problem["description"]}

Requirements:
- Write efficient CUDA C++ code
- Include proper error checking
- Use appropriate grid and block dimensions
- Optimize for the given problem size

Please provide a complete CUDA kernel implementation.
"""

    def _create_optimizer_prompt(
        self,
        conversation: CUDAConversationState,
        problem: Dict[str, Any]
    ) -> str:
        """Create prompt for optimization phase."""

        current_perf = conversation.current_performance
        last_turn = conversation.turns[-1]

        return f"""
Optimize the following CUDA kernel for better performance:

Original Problem: {problem["description"]}

Current Kernel:
{conversation.current_kernel_code}

Current Performance:
- Speedup: {current_perf.get("speedup", 0.0):.2f}x
- Memory Bandwidth: {current_perf.get("memory_bandwidth", 0.0):.2f} GB/s
- Execution Time: {current_perf.get("execution_time", 0.0):.2f} ms

Target Performance:
- Speedup: {conversation.target_performance.get("speedup", 2.0):.2f}x

Please provide an optimized version that improves performance. Consider:
- Memory access patterns
- Shared memory usage
- Thread divergence
- Occupancy optimization
"""

    def _create_fix_prompt(
        self,
        conversation: CUDAConversationState,
        problem: Dict[str, Any]
    ) -> str:
        """Create prompt for fixing compilation errors."""

        last_turn = conversation.turns[-1]
        compilation_result = last_turn.context.get("compilation_result")

        error_message = ""
        if compilation_result:
            error_message = compilation_result.stderr

        return f"""
Fix the compilation errors in the following CUDA kernel:

Original Problem: {problem["description"]}

Current Kernel (with errors):
{conversation.current_kernel_code}

Compilation Error:
{error_message}

Please provide a corrected version that compiles successfully.
"""

    def _extract_kernel_code(self, response: str) -> str:
        """Extract CUDA kernel code from agent response."""

        # Look for code blocks
        # First try to find code in markdown code blocks
        code_blocks = re.findall(r'```(?:cuda|cpp|c)?\n(.*?)\n```', response, re.DOTALL)

        if code_blocks:
            return code_blocks[0].strip()

        # If no code blocks, look for __global__ keyword
        global_match = re.search(r'(__global__.*?(?=\n\n|\n__|\\n#|\\Z))', response, re.DOTALL)
        if global_match:
            return global_match.group(1).strip()

        # As fallback, return the entire response
        return response.strip()

    def _generate_test_cases(self, difficulty_tier: str) -> List[Dict[str, Any]]:
        """Generate test cases based on difficulty tier."""

        if difficulty_tier == "easy":
            return [
                {
                    "input_shapes": [[1024], [1024]],
                    "dtype": torch.float32,
                    "grid_dims": (4, 1, 1),
                    "block_dims": (256, 1, 1)
                }
            ]
        elif difficulty_tier == "medium":
            return [
                {
                    "input_shapes": [[4096], [4096]],
                    "dtype": torch.float32,
                    "grid_dims": (16, 1, 1),
                    "block_dims": (256, 1, 1)
                },
                {
                    "input_shapes": [[1024, 1024], [1024, 1024]],
                    "dtype": torch.float32,
                    "grid_dims": (32, 32, 1),
                    "block_dims": (16, 16, 1)
                }
            ]
        else:  # hard
            return [
                {
                    "input_shapes": [[8192], [8192]],
                    "dtype": torch.float32,
                    "grid_dims": (32, 1, 1),
                    "block_dims": (256, 1, 1)
                },
                {
                    "input_shapes": [[2048, 2048], [2048, 2048]],
                    "dtype": torch.float32,
                    "grid_dims": (64, 64, 1),
                    "block_dims": (16, 16, 1)
                }
            ]
