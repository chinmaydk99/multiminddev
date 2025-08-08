"""
Base Agent class providing common functionality for all coding agents.

This module defines the abstract base class that all specialized agents inherit from,
providing shared functionality for LLM interaction, logging, and state management.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import structlog
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..utils.config import AgentConfig
from ..utils.llm_interface import LLMInterface


class AgentResponse(BaseModel):
    """Standard response format for all agents."""

    agent_type: str = Field(..., description="Type of agent that generated the response")
    success: bool = Field(..., description="Whether the operation was successful")
    content: str = Field(..., description="Main response content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    execution_time: float = Field(..., description="Time taken to generate response")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")
    error: Optional[str] = Field(None, description="Error message if operation failed")


class BaseAgent(ABC):
    """
    Abstract base class for all coding agents.

    Provides common functionality including LLM interaction, logging,
    configuration management, and standardized response handling.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_interface: LLMInterface,
        agent_id: Optional[str] = None,
    ):
        """
        Initialize the base agent.

        Args:
            config: Agent configuration settings
            llm_interface: Interface for LLM communication
            agent_id: Unique identifier for this agent instance
        """
        self.config = config
        self.llm_interface = llm_interface
        self.agent_id = agent_id or f"{self.agent_type}_{int(time.time())}"

        # Set up structured logging
        self.logger = structlog.get_logger(
            agent_type=self.agent_type,
            agent_id=self.agent_id,
        )

        # Initialize agent state
        self.state: dict[str, Any] = {}
        self.conversation_history: list[BaseMessage] = []
        self.performance_metrics: dict[str, float] = {}

        self.logger.info("Agent initialized", config=config.dict())

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the agent type identifier."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    @abstractmethod
    async def process_request(
        self,
        request: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> AgentResponse:
        """
        Process a request and return a standardized response.

        Args:
            request: The input request to process
            context: Additional context information
            **kwargs: Additional arguments specific to the agent

        Returns:
            Standardized agent response
        """
        pass

    async def _call_llm(
        self,
        messages: list[BaseMessage],
        **kwargs,
    ) -> str:
        """
        Make an LLM call with error handling and logging.

        Args:
            messages: Messages to send to the LLM
            **kwargs: Additional arguments for the LLM call

        Returns:
            LLM response content

        Raises:
            Exception: If LLM call fails after retries
        """
        start_time = time.time()

        try:
            # Extract temperature and max_tokens to avoid duplicate parameter issues
            temperature = kwargs.pop("temperature", self.config.temperature)
            max_tokens = kwargs.pop("max_tokens", self.config.max_tokens)

            response = await self.llm_interface.call(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            execution_time = time.time() - start_time

            self.logger.info(
                "LLM call successful",
                execution_time=execution_time,
                response_length=len(response),
            )

            # Update performance metrics
            self.performance_metrics["last_llm_call_time"] = execution_time
            self.performance_metrics["total_llm_calls"] = (
                self.performance_metrics.get("total_llm_calls", 0) + 1
            )

            return response

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(
                "LLM call failed",
                error=str(e),
                execution_time=execution_time,
            )
            raise

    def _build_messages(
        self,
        request: str,
        context: Optional[dict[str, Any]] = None,
        include_history: bool = True,
    ) -> list[BaseMessage]:
        """
        Build message list for LLM call.

        Args:
            request: The user request
            context: Additional context to include
            include_history: Whether to include conversation history

        Returns:
            List of messages for LLM
        """
        messages = [SystemMessage(content=self.system_prompt)]

        # Add conversation history if requested
        if include_history and self.conversation_history:
            messages.extend(self.conversation_history[-10:])  # Last 10 messages

        # Add context if provided
        if context:
            context_str = self._format_context(context)
            if context_str:
                messages.append(SystemMessage(content=f"Context: {context_str}"))

        # Add the current request
        messages.append(HumanMessage(content=request))

        return messages

    def _format_context(self, context: dict[str, Any]) -> str:
        """
        Format context dictionary into a readable string.

        Args:
            context: Context dictionary to format

        Returns:
            Formatted context string
        """
        if not context:
            return ""

        formatted_parts = []

        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                formatted_parts.append(f"{key}: {value}")
            elif isinstance(value, (list, dict)):
                formatted_parts.append(f"{key}: {str(value)[:500]}...")  # Truncate long values

        return " | ".join(formatted_parts)

    def update_state(self, key: str, value: Any) -> None:
        """
        Update agent state.

        Args:
            key: State key to update
            value: New value for the state key
        """
        self.state[key] = value
        self.logger.debug("State updated", key=key, value_type=type(value).__name__)

    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get value from agent state.

        Args:
            key: State key to retrieve
            default: Default value if key not found

        Returns:
            State value or default
        """
        return self.state.get(key, default)

    def add_to_history(self, message: BaseMessage) -> None:
        """
        Add message to conversation history.

        Args:
            message: Message to add to history
        """
        self.conversation_history.append(message)

        # Keep history size manageable
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-30:]

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        self.logger.info("Conversation history cleared")

    def get_performance_metrics(self) -> dict[str, float]:
        """
        Get current performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        return self.performance_metrics.copy()

    async def health_check(self) -> dict[str, Any]:
        """
        Perform agent health check.

        Returns:
            Health status information
        """
        start_time = time.time()

        try:
            # Test LLM connection
            test_messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Say 'OK' if you can hear me."),
            ]

            await self._call_llm(test_messages, max_tokens=10)

            health_status = {
                "status": "healthy",
                "agent_type": self.agent_type,
                "agent_id": self.agent_id,
                "llm_responsive": True,
                "response_time": time.time() - start_time,
                "state_keys": list(self.state.keys()),
                "history_length": len(self.conversation_history),
                "performance_metrics": self.performance_metrics,
            }

            self.logger.info("Health check passed", **health_status)
            return health_status

        except Exception as e:
            health_status = {
                "status": "unhealthy",
                "agent_type": self.agent_type,
                "agent_id": self.agent_id,
                "llm_responsive": False,
                "error": str(e),
                "response_time": time.time() - start_time,
            }

            self.logger.error("Health check failed", **health_status)
            return health_status

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(id={self.agent_id}, type={self.agent_type})"
