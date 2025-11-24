"""
LLM-based kernel generator.

Supports multiple backends:
- OpenAI API (and compatible APIs like Together, Anyscale)
- HuggingFace Transformers (local)
- vLLM (local high-performance)
"""

import asyncio
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import structlog

from .prompts import KernelPrompt, PromptBuilder, SYSTEM_PROMPT

logger = structlog.get_logger()


@dataclass
class GenerationResult:
    """Result from kernel generation."""
    
    success: bool = False
    kernel_code: str = ""
    
    # Extraction info
    raw_response: str = ""
    extracted_from_block: bool = False
    
    # Metadata
    model_name: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generation_time_s: float = 0.0
    
    # Error info
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "kernel_code": self.kernel_code,
            "raw_response": self.raw_response,
            "model_name": self.model_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "generation_time_s": self.generation_time_s,
            "error_message": self.error_message,
        }


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Generate completion from the LLM."""
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI API backend (also works with compatible APIs)."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.model = model
        
        # Import here to avoid dependency if not used
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
    
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Generate using OpenAI API."""
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }


class HuggingFaceBackend(LLMBackend):
    """HuggingFace Transformers backend for local models."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        device: str = "auto",
        torch_dtype: str = "bfloat16",
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy load model."""
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype_map.get(self.torch_dtype, torch.bfloat16),
                device_map=self.device,
            )
    
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Generate using local HuggingFace model."""
        
        self._load_model()
        
        # Format prompt for instruction-tuned model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Use chat template if available
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt_text = self._tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            prompt_text = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._model.device)
        prompt_tokens = inputs.input_ids.shape[1]
        
        # Generate
        outputs = await asyncio.to_thread(
            self._model.generate,
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        
        # Decode
        generated_ids = outputs[0][prompt_tokens:]
        content = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return {
            "content": content,
            "model": self.model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(generated_ids),
        }


class VLLMBackend(LLMBackend):
    """vLLM backend for high-performance local inference."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        tensor_parallel_size: int = 1,
    ):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self._llm = None
    
    def _load_model(self):
        """Lazy load vLLM model."""
        if self._llm is None:
            from vllm import LLM
            self._llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                trust_remote_code=True,
            )
    
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Generate using vLLM."""
        
        self._load_model()
        
        from vllm import SamplingParams
        
        # Format prompt
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        outputs = await asyncio.to_thread(
            self._llm.generate,
            [prompt],
            sampling_params,
        )
        
        output = outputs[0]
        content = output.outputs[0].text
        
        return {
            "content": content,
            "model": self.model_name,
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids),
        }


class KernelGenerator:
    """
    Main kernel generator class.
    
    Uses an LLM backend to generate HIP kernels from operation specifications.
    """
    
    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        backend_type: str = "openai",
        model_name: Optional[str] = None,
        **backend_kwargs,
    ):
        """
        Initialize kernel generator.
        
        Args:
            backend: Pre-configured LLM backend
            backend_type: "openai", "huggingface", or "vllm"
            model_name: Model to use (backend-specific)
            **backend_kwargs: Additional arguments for backend
        """
        
        if backend is not None:
            self.backend = backend
        else:
            self.backend = self._create_backend(backend_type, model_name, **backend_kwargs)
        
        self.prompt_builder = PromptBuilder()
        self.logger = logger.bind(component="KernelGenerator")
    
    def _create_backend(
        self, 
        backend_type: str, 
        model_name: Optional[str],
        **kwargs
    ) -> LLMBackend:
        """Create LLM backend based on type."""
        
        if backend_type == "openai":
            return OpenAIBackend(
                model=model_name or "gpt-4-turbo-preview",
                **kwargs
            )
        elif backend_type == "huggingface":
            return HuggingFaceBackend(
                model_name=model_name or "Qwen/Qwen2.5-Coder-7B-Instruct",
                **kwargs
            )
        elif backend_type == "vllm":
            return VLLMBackend(
                model_name=model_name or "Qwen/Qwen2.5-Coder-7B-Instruct",
                **kwargs
            )
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
    
    async def generate(
        self,
        prompt: KernelPrompt,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """
        Generate a kernel from a prompt.
        
        Args:
            prompt: Structured kernel prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            GenerationResult with kernel code
        """
        
        import time
        start_time = time.time()
        
        result = GenerationResult()
        
        try:
            # Generate from LLM
            response = await self.backend.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt.to_user_prompt(),
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            result.raw_response = response["content"]
            result.model_name = response.get("model", "")
            result.prompt_tokens = response.get("prompt_tokens", 0)
            result.completion_tokens = response.get("completion_tokens", 0)
            
            # Extract kernel code
            kernel_code = self._extract_kernel_code(response["content"])
            
            if kernel_code:
                result.success = True
                result.kernel_code = kernel_code
                result.extracted_from_block = True
            else:
                # Fallback: use entire response if it looks like code
                if "__global__" in response["content"]:
                    result.success = True
                    result.kernel_code = response["content"]
                    result.extracted_from_block = False
                else:
                    result.success = False
                    result.error_message = "Could not extract kernel code from response"
            
            result.generation_time_s = time.time() - start_time
            
            self.logger.info(
                "Kernel generated",
                success=result.success,
                tokens=result.completion_tokens,
                time_s=f"{result.generation_time_s:.2f}",
            )
        
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.generation_time_s = time.time() - start_time
            
            self.logger.error(
                "Generation failed",
                error=str(e),
            )
        
        return result
    
    async def generate_from_operation(
        self,
        operation: Any,  # OperationSpec
        shape_values: Dict[str, int],
        dtype: Any = None,  # DataType
        **kwargs,
    ) -> GenerationResult:
        """
        Generate kernel directly from operation specification.
        
        Convenience method that builds prompt automatically.
        """
        from src.operations.base import DataType
        
        dtype = dtype or DataType.FP16
        
        prompt = self.prompt_builder.build_prompt(
            operation=operation,
            shape_values=shape_values,
            dtype=dtype,
        )
        
        return await self.generate(prompt, **kwargs)
    
    async def refine(
        self,
        operation: Any,  # OperationSpec
        shape_values: Dict[str, int],
        dtype: Any,  # DataType
        previous_code: str,
        execution_result: Any,  # ExecutionResult
        **kwargs,
    ) -> GenerationResult:
        """
        Refine a previous kernel based on execution feedback.
        
        This is the key RL loop: generate → execute → get feedback → refine.
        """
        
        prompt = self.prompt_builder.build_refinement_prompt(
            operation=operation,
            shape_values=shape_values,
            dtype=dtype,
            previous_code=previous_code,
            execution_result=execution_result,
        )
        
        return await self.generate(prompt, **kwargs)
    
    def _extract_kernel_code(self, response: str) -> Optional[str]:
        """Extract kernel code from LLM response."""
        
        # Try to find code in ```cpp or ``` blocks
        patterns = [
            r'```cpp\n(.*?)```',
            r'```c\+\+\n(.*?)```',
            r'```cuda\n(.*?)```',
            r'```hip\n(.*?)```',
            r'```\n(.*?)```',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                # Verify it looks like kernel code
                if "__global__" in match:
                    return match.strip()
        
        # If no code blocks, try to find kernel directly
        # Look for __global__ function
        kernel_pattern = r'(__global__\s+\w+\s+\w+\s*\([^)]*\)\s*\{.*?\})'
        matches = re.findall(kernel_pattern, response, re.DOTALL)
        if matches:
            # Return the full response up to and including the kernel
            # This captures any necessary includes
            kernel_start = response.find("__global__")
            if kernel_start > 0:
                # Look for #include before the kernel
                include_start = response.rfind("#include", 0, kernel_start)
                if include_start >= 0:
                    return response[include_start:].strip()
            return matches[0]
        
        return None

