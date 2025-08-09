"""
Trainable Agent base class that owns HuggingFace model parameters and supports
reinforcement learning training through VERL.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from dataclasses import dataclass
import structlog
from pathlib import Path
import json

from .base_agent import BaseAgent


@dataclass
class GenerationOutput:
    """Output from model generation including log probabilities."""
    text: str
    token_ids: torch.Tensor
    log_probs: torch.Tensor
    attention_mask: torch.Tensor
    value_estimates: Optional[torch.Tensor] = None


class TrainableAgent(BaseAgent):
    """
    Base class for agents with owned, trainable HuggingFace model parameters.
    
    This agent:
    - Loads models from HuggingFace Hub
    - Owns model parameters that can be updated through RL
    - Tracks log probabilities for PPO training
    - Supports both SFT and RL training modes
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-5,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_length: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize trainable agent with HuggingFace model.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (generator, optimizer, tester)
            model_name: HuggingFace model name or path
            device: Device to load model on
            learning_rate: Learning rate for optimizer
            load_in_8bit: Whether to load model in 8-bit precision
            load_in_4bit: Whether to load model in 4-bit precision
            max_length: Maximum sequence length
            temperature: Generation temperature
        """
        # Initialize base agent without LLM interface
        super().__init__(
            agent_id=agent_id,
            agent_type=agent_type,
            llm_interface=None,  # We own the model directly
            **kwargs
        )
        
        self.model_name = model_name
        self.device = device
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.temperature = temperature
        
        # Load model and tokenizer from HuggingFace
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.optimizer: Optional[AdamW] = None
        
        # Training state
        self.is_training_mode = False
        self.gradient_accumulation_steps = 1
        self.current_accumulation_step = 0
        
        # Generation config
        self.generation_config = GenerationConfig(
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            pad_token_id=None,  # Will be set from tokenizer
            eos_token_id=None,  # Will be set from tokenizer
        )
        
        # Load model
        self._load_model(load_in_8bit, load_in_4bit)
        
        self.logger.info(
            "TrainableAgent initialized",
            agent_id=agent_id,
            agent_type=agent_type,
            model_name=model_name,
            device=device,
            trainable_params=self.count_trainable_parameters()
        )
    
    def _load_model(self, load_in_8bit: bool = False, load_in_4bit: bool = False) -> None:
        """Load model and tokenizer from HuggingFace."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"  # For batch generation
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Update generation config with tokenizer info
            self.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.generation_config.eos_token_id = self.tokenizer.eos_token_id
            
            # Load model with appropriate precision
            load_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            if load_in_8bit:
                load_kwargs["load_in_8bit"] = True
            elif load_in_4bit:
                load_kwargs["load_in_4bit"] = True
                load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            else:
                load_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            
            # Move to device if not using device_map
            if "device_map" not in load_kwargs or load_kwargs["device_map"] is None:
                self.model = self.model.to(self.device)
            
            # Set up optimizer for trainable parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if trainable_params:
                self.optimizer = AdamW(trainable_params, lr=self.learning_rate)
            
            self.logger.info(
                "Model loaded successfully",
                model_name=self.model_name,
                num_parameters=sum(p.numel() for p in self.model.parameters()),
                trainable_parameters=sum(p.numel() for p in trainable_params)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def count_trainable_parameters(self) -> int:
        """Count number of trainable parameters in the model."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def set_training_mode(self, mode: bool = True) -> None:
        """Set agent to training or evaluation mode."""
        self.is_training_mode = mode
        if self.model:
            self.model.train(mode)
    
    async def generate_with_log_probs(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: Optional[float] = None,
        return_full_output: bool = True
    ) -> GenerationOutput:
        """
        Generate response while tracking log probabilities for RL training.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Override default temperature
            return_full_output: Whether to return full generation details
            
        Returns:
            GenerationOutput with text, token_ids, and log_probs
        """
        if temperature is None:
            temperature = self.temperature
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - max_new_tokens,
            padding=True
        ).to(self.device)
        
        # Generate with model
        with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
            if self.is_training_mode:
                # During training, we need to track gradients
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_attentions=False,
                )
            else:
                # During inference, no gradient tracking needed
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.95,
                        return_dict_in_generate=True,
                        output_scores=True,
                        output_attentions=False,
                    )
        
        # Extract generated tokens (excluding prompt)
        generated_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
        
        # Calculate log probabilities if we have scores
        log_probs = None
        if outputs.scores:
            # Stack scores and apply log_softmax
            scores = torch.stack(outputs.scores, dim=1)
            log_probs = F.log_softmax(scores, dim=-1)
            
            # Gather log probs for generated tokens
            log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=generated_ids.unsqueeze(-1)
            ).squeeze(-1)
        
        # Decode generated text
        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return GenerationOutput(
            text=generated_text,
            token_ids=generated_ids,
            log_probs=log_probs,
            attention_mask=torch.ones_like(generated_ids)
        )
    
    async def process(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a prompt and generate response.
        
        This overrides the base agent's process method to use our owned model.
        """
        try:
            # Add agent-specific context to prompt
            formatted_prompt = self._format_prompt(prompt, context)
            
            # Generate response with our model
            generation_output = await self.generate_with_log_probs(
                formatted_prompt,
                max_new_tokens=kwargs.get("max_new_tokens", 512),
                temperature=kwargs.get("temperature", self.temperature)
            )
            
            # Track in conversation history
            self.conversation_history.append({
                "role": "user",
                "content": prompt
            })
            self.conversation_history.append({
                "role": "assistant", 
                "content": generation_output.text
            })
            
            # Return structured output
            return {
                "response": generation_output.text,
                "token_ids": generation_output.token_ids.tolist() if generation_output.token_ids is not None else None,
                "log_probs": generation_output.log_probs.tolist() if generation_output.log_probs is not None else None,
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "model": self.model_name,
                "context": context
            }
            
        except Exception as e:
            self.logger.error(f"Error in trainable agent processing: {e}")
            return {
                "error": str(e),
                "agent_id": self.agent_id,
                "agent_type": self.agent_type
            }
    
    def _format_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Format prompt with agent-specific instructions."""
        # Base prompt formatting - can be overridden by subclasses
        agent_instruction = f"You are a {self.agent_type} agent. "
        
        if self.agent_type == "generator":
            agent_instruction += "Generate high-quality CUDA kernel code based on the given requirements."
        elif self.agent_type == "optimizer":
            agent_instruction += "Optimize the given CUDA kernel for better performance."
        elif self.agent_type == "tester":
            agent_instruction += "Test and validate the given CUDA kernel code."
        
        formatted = f"{agent_instruction}\n\n{prompt}"
        
        # Add context if provided
        if context:
            context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            formatted = f"{formatted}\n\nContext:\n{context_str}"
        
        return formatted
    
    def update_parameters(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        advantages: Optional[torch.Tensor] = None,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01
    ) -> Dict[str, float]:
        """
        Update model parameters using PPO algorithm.
        
        Args:
            rewards: Tensor of rewards for each token/action
            log_probs: Log probabilities from generation
            advantages: Advantage estimates (if None, uses rewards)
            clip_ratio: PPO clipping ratio
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            
        Returns:
            Dictionary of training metrics
        """
        if self.optimizer is None:
            self.logger.warning("No optimizer available - model may be frozen")
            return {}
        
        # Use advantages if provided, otherwise use rewards
        if advantages is None:
            advantages = rewards
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate PPO loss
        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Add entropy bonus for exploration
        entropy = -(log_probs * torch.exp(log_probs)).mean()
        
        # Total loss
        total_loss = policy_loss - entropy_coef * entropy
        
        # Backward pass with gradient accumulation
        total_loss = total_loss / self.gradient_accumulation_steps
        total_loss.backward()
        
        self.current_accumulation_step += 1
        
        # Update parameters if we've accumulated enough gradients
        if self.current_accumulation_step >= self.gradient_accumulation_steps:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.current_accumulation_step = 0
        
        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_advantage": advantages.mean().item()
        }
    
    async def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save model checkpoint."""
        path = Path(checkpoint_path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.model:
            self.model.save_pretrained(path / "model")
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(path / "tokenizer")
        
        # Save agent metadata
        metadata = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "model_name": self.model_name,
            "learning_rate": self.learning_rate,
            "max_length": self.max_length,
            "temperature": self.temperature
        }
        
        with open(path / "agent_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    async def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        path = Path(checkpoint_path)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            path / "model",
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            path / "tokenizer",
            trust_remote_code=True
        )
        
        # Set up optimizer for loaded model
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if trainable_params:
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate)
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    async def sft_train_step(
        self,
        input_text: str,
        target_text: str,
        learning_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Perform supervised fine-tuning step.
        
        Args:
            input_text: Input prompt
            target_text: Target response for SFT
            learning_rate: Override default learning rate
            
        Returns:
            Training metrics
        """
        if self.optimizer is None:
            self.logger.warning("No optimizer available for SFT")
            return {}
        
        # Update learning rate if specified
        if learning_rate is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        # Prepare input and target
        full_text = f"{input_text}\n{target_text}"
        encoding = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        ).to(self.device)
        
        # Forward pass
        self.model.train()
        outputs = self.model(**encoding, labels=encoding.input_ids)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return {
            "sft_loss": loss.item(),
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }