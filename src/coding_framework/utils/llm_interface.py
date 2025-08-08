"""
LLM Interface abstraction for multiple providers.

This module provides a unified interface for different LLM providers
including OpenAI, Anthropic, and local models.
"""

import asyncio
import time
from typing import Any, Optional

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

try:
    from langchain_huggingface import HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

from .config import LLMConfig


class LLMResponse(BaseModel):
    """Response from LLM provider."""

    content: str
    metadata: dict[str, Any]
    tokens_used: Optional[int] = None
    response_time: float


class LLMInterface:
    """
    Unified interface for LLM providers.

    Supports OpenAI, Anthropic, and local model providers with
    consistent API and error handling.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM interface.

        Args:
            config: LLM configuration
        """
        self.config = config
        self.logger = structlog.get_logger(
            component="llm_interface",
            provider=config.provider,
        )

        self.client = None
        self.is_initialized = False

        # Rate limiting
        self.last_request_time = 0.0
        self.min_request_interval = 0.1  # Minimum time between requests

    async def initialize(self) -> None:
        """Initialize the LLM client."""
        if self.is_initialized:
            return

        try:
            if self.config.provider == "openai":
                self.client = ChatOpenAI(
                    model=self.config.model,
                    openai_api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                )

            elif self.config.provider == "anthropic":
                self.client = ChatAnthropic(
                    model=self.config.model,
                    anthropic_api_key=self.config.api_key,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                )

            elif self.config.provider == "huggingface":
                if not HUGGINGFACE_AVAILABLE:
                    raise ImportError(
                        "HuggingFace dependencies not installed. "
                        "Install with: pip install transformers langchain-huggingface torch"
                    )

                self.client = await self._initialize_huggingface_model()

            elif self.config.provider == "local":
                # For local models, use OpenAI-compatible API
                self.client = ChatOpenAI(
                    model=self.config.model,
                    base_url=self.config.base_url or "http://localhost:8000",
                    api_key="dummy",  # Local models often don't need real API keys
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                )

            else:
                raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

            # Test connection
            await self._test_connection()

            self.is_initialized = True
            self.logger.info("LLM interface initialized successfully")

        except Exception as e:
            self.logger.error("LLM interface initialization failed", error=str(e))
            raise

    async def _test_connection(self) -> None:
        """Test connection to LLM provider."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            test_messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Say 'OK' if you can hear me."),
            ]

            response = await self.client.ainvoke(test_messages)
            self.logger.info("Connection test successful", response_length=len(response.content))

        except Exception as e:
            self.logger.error("Connection test failed", error=str(e))
            raise

    async def _initialize_huggingface_model(self):
        """Initialize HuggingFace model for local inference."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.logger.info("Loading HuggingFace model", model=self.config.model)

            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info("Using device", device=device)

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model,
                trust_remote_code=True,
            )

            # Add padding token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self.config.model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )

            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                device=0 if device == "cuda" else -1,
                return_full_text=False,
            )

            # Wrap in LangChain
            hf_llm = HuggingFacePipeline(pipeline=pipe)

            # Store model info for cleanup
            self.hf_model = model
            self.hf_tokenizer = tokenizer
            self.hf_pipeline = pipe

            self.logger.info("HuggingFace model loaded successfully")
            return hf_llm

        except Exception as e:
            self.logger.error("Failed to initialize HuggingFace model", error=str(e))
            raise

    async def _call_huggingface_model(
        self, messages: list[BaseMessage], kwargs: dict[str, Any]
    ) -> str:
        """Make a call to HuggingFace model with proper formatting."""
        try:
            # Format messages into a single prompt
            formatted_prompt = self._format_messages_for_hf(messages)

            # Run in thread pool to avoid blocking
            import asyncio

            loop = asyncio.get_event_loop()

            def sync_generate():
                # Use the pipeline directly for better control
                result = self.hf_pipeline(
                    formatted_prompt,
                    max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                    temperature=kwargs.get("temperature", self.config.temperature),
                    do_sample=True,
                    pad_token_id=self.hf_tokenizer.eos_token_id,
                )
                return result[0]["generated_text"] if result else ""

            response = await loop.run_in_executor(None, sync_generate)

            return response.strip()

        except Exception as e:
            self.logger.error("HuggingFace model call failed", error=str(e))
            raise

    def _format_messages_for_hf(self, messages: list[BaseMessage]) -> str:
        """Format LangChain messages for HuggingFace models."""
        formatted_parts = []

        for message in messages:
            content = message.content

            if hasattr(message, "type"):
                msg_type = message.type
            else:
                msg_type = message.__class__.__name__.lower().replace("message", "")

            if msg_type == "system":
                formatted_parts.append(f"System: {content}")
            elif msg_type == "human":
                formatted_parts.append(f"Human: {content}")
            elif msg_type == "ai":
                formatted_parts.append(f"Assistant: {content}")
            else:
                formatted_parts.append(content)

        # Add assistant prompt at the end
        formatted_parts.append("Assistant:")

        return "\n\n".join(formatted_parts)

    async def call(
        self,
        messages: list[BaseMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Make a call to the LLM.

        Args:
            messages: List of messages to send
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional provider-specific parameters

        Returns:
            LLM response content

        Raises:
            Exception: If LLM call fails
        """
        if not self.is_initialized:
            await self.initialize()

        # Rate limiting
        await self._rate_limit()

        start_time = time.time()

        try:
            # Prepare parameters
            call_kwargs = {}

            if temperature is not None:
                if hasattr(self.client, "temperature"):
                    self.client.temperature = temperature
                else:
                    call_kwargs["temperature"] = temperature

            if max_tokens is not None:
                if hasattr(self.client, "max_tokens"):
                    self.client.max_tokens = max_tokens
                else:
                    call_kwargs["max_tokens"] = max_tokens

            # Add any provider-specific parameters
            call_kwargs.update(kwargs)

            self.logger.info(
                "Making LLM call",
                message_count=len(messages),
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
            )

            # Make the call
            if self.config.provider == "huggingface":
                # HuggingFace models need special handling
                response = await self._call_huggingface_model(messages, call_kwargs)
                content = response
            else:
                if call_kwargs:
                    response = await self.client.ainvoke(messages, **call_kwargs)
                else:
                    response = await self.client.ainvoke(messages)

                # Extract content
                content = response.content if hasattr(response, "content") else str(response)

            response_time = time.time() - start_time

            # Log successful call
            self.logger.info(
                "LLM call successful", response_time=response_time, response_length=len(content)
            )

            return content

        except Exception as e:
            response_time = time.time() - start_time

            self.logger.error("LLM call failed", error=str(e), response_time=response_time)
            raise

    async def call_with_retry(
        self,
        messages: list[BaseMessage],
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        **kwargs,
    ) -> str:
        """
        Make LLM call with automatic retry on failure.

        Args:
            messages: Messages to send
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
            **kwargs: Additional call parameters

        Returns:
            LLM response content

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await self.call(messages, **kwargs)

            except Exception as e:
                last_exception = e

                if attempt < max_retries:
                    wait_time = backoff_factor**attempt
                    self.logger.warning(
                        "LLM call failed, retrying",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        wait_time=wait_time,
                        error=str(e),
                    )

                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(
                        "All retry attempts failed", attempts=max_retries + 1, final_error=str(e)
                    )

        # If we get here, all retries failed
        raise last_exception

    async def batch_call(
        self,
        message_batches: list[list[BaseMessage]],
        concurrency: int = 5,
        **kwargs,
    ) -> list[str]:
        """
        Make multiple LLM calls in parallel with concurrency control.

        Args:
            message_batches: List of message lists to process
            concurrency: Maximum concurrent requests
            **kwargs: Additional call parameters

        Returns:
            List of response contents in same order as input
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def make_single_call(messages: list[BaseMessage]) -> str:
            async with semaphore:
                return await self.call_with_retry(messages, **kwargs)

        self.logger.info(
            "Starting batch LLM calls", batch_size=len(message_batches), concurrency=concurrency
        )

        try:
            start_time = time.time()

            # Execute all calls
            results = await asyncio.gather(
                *[make_single_call(messages) for messages in message_batches],
                return_exceptions=True,
            )

            total_time = time.time() - start_time

            # Check for exceptions
            successful_results = []
            failed_count = 0

            for result in results:
                if isinstance(result, Exception):
                    failed_count += 1
                    successful_results.append("")  # Placeholder for failed calls
                    self.logger.error("Batch call failed", error=str(result))
                else:
                    successful_results.append(result)

            self.logger.info(
                "Batch LLM calls completed",
                total_time=total_time,
                successful=len(results) - failed_count,
                failed=failed_count,
            )

            return successful_results

        except Exception as e:
            self.logger.error("Batch LLM calls failed", error=str(e))
            raise

    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            await asyncio.sleep(wait_time)

        self.last_request_time = time.time()

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check of the LLM interface.

        Returns:
            Health status information
        """
        if not self.is_initialized:
            return {
                "status": "not_initialized",
                "provider": self.config.provider,
                "model": self.config.model,
            }

        try:
            start_time = time.time()

            # Make a simple test call
            from langchain_core.messages import HumanMessage, SystemMessage

            test_messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Say 'OK' if you can hear me."),
            ]

            response = await self.call(test_messages, max_tokens=10)
            response_time = time.time() - start_time

            return {
                "status": "healthy",
                "provider": self.config.provider,
                "model": self.config.model,
                "response_time": response_time,
                "response_length": len(response),
                "initialized": self.is_initialized,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.config.provider,
                "model": self.config.model,
                "error": str(e),
                "initialized": self.is_initialized,
            }

    def get_provider_info(self) -> dict[str, Any]:
        """
        Get information about the current LLM provider.

        Returns:
            Provider information
        """
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
            "initialized": self.is_initialized,
        }

    async def shutdown(self) -> None:
        """Shutdown the LLM interface."""
        if hasattr(self.client, "close"):
            await self.client.close()

        self.is_initialized = False
        self.logger.info("LLM interface shutdown")

    def __repr__(self) -> str:
        """String representation of LLM interface."""
        return (
            f"LLMInterface("
            f"provider={self.config.provider}, "
            f"model={self.config.model}, "
            f"initialized={self.is_initialized})"
        )
