"""
Configuration management for the Multi-Agent Coding Framework.

This module provides configuration classes and loading utilities
using Pydantic for validation and type safety.
"""

import os
from pathlib import Path
from typing import Optional, Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""

    provider: str = Field(
        default="huggingface", description="LLM provider (openai, anthropic, huggingface, local)"
    )
    model: str = Field(default="bigcode/starcoder2-3b", description="Model name or HuggingFace model ID")
    api_key: Optional[str] = Field(default=None, description="API key")
    base_url: Optional[str] = Field(default=None, description="Custom base URL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: int = Field(default=2048, gt=0, description="Maximum tokens")
    timeout: int = Field(default=60, gt=0, description="Request timeout in seconds")

    # HuggingFace specific settings
    use_auth_token: bool = Field(default=False, description="Use HuggingFace auth token")
    device_map: str = Field(default="auto", description="Device mapping for model loading")
    torch_dtype: str = Field(default="float16", description="PyTorch data type")
    trust_remote_code: bool = Field(default=True, description="Trust remote code for custom models")


class AgentConfig(BaseModel):
    """Base configuration for agents."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    timeout: int = Field(default=60, gt=0)
    retry_attempts: int = Field(default=3, ge=0)

    # Agent-specific settings
    enabled: bool = Field(default=True)
    debug: bool = Field(default=False)


class GeneratorConfig(AgentConfig):
    """Configuration for the Code Generator Agent."""

    temperature: float = Field(default=0.7)
    include_comments: bool = Field(default=True)
    include_docstrings: bool = Field(default=True)
    optimize_for_readability: bool = Field(default=True)

    # Language-specific settings
    default_language: str = Field(default="python")
    supported_languages: list[str] = Field(
        default=["python", "javascript", "typescript", "java", "cpp", "go", "rust"]
    )


class ReviewerConfig(AgentConfig):
    """Configuration for the Code Reviewer Agent."""

    temperature: float = Field(default=0.3)  # Lower for consistency

    # Review settings
    focus_areas: list[str] = Field(
        default=["correctness", "style", "performance", "security", "maintainability"]
    )
    severity_levels: list[str] = Field(default=["low", "medium", "high", "critical"])
    min_review_score: float = Field(default=0.0, ge=0.0, le=100.0)

    # Analysis settings
    include_suggestions: bool = Field(default=True)
    include_examples: bool = Field(default=True)
    security_focused: bool = Field(default=True)


class ExecutorConfig(AgentConfig):
    """Configuration for the Code Executor Agent."""

    temperature: float = Field(default=0.5)

    # Execution settings
    execution_timeout: int = Field(default=30, gt=0)
    memory_limit: str = Field(default="512m")
    cpu_limit: str = Field(default="1.0")

    # Security settings
    sandboxed_execution: bool = Field(default=True)
    network_disabled: bool = Field(default=True)
    docker_enabled: bool = Field(default=True)
    docker_image_timeout: int = Field(default=300)

    # Supported environments
    supported_languages: list[str] = Field(
        default=["python", "javascript", "typescript", "java", "cpp", "go", "rust"]
    )


class AgentsConfig(BaseModel):
    """Configuration for all agents."""

    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    reviewer: ReviewerConfig = Field(default_factory=ReviewerConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    
    # CUDA-specific agent configurations
    cuda_generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    cuda_optimizer: ReviewerConfig = Field(default_factory=ReviewerConfig) 
    cuda_tester: ExecutorConfig = Field(default_factory=ExecutorConfig)


class WorkflowConfig(BaseModel):
    """Configuration for LangGraph workflow."""

    max_iterations: int = Field(default=3, ge=1, le=10)
    human_in_loop: bool = Field(default=False)

    # Quality thresholds
    min_execution_score: float = Field(default=50.0, ge=0.0, le=100.0)
    target_review_score: float = Field(default=80.0, ge=0.0, le=100.0)
    human_feedback_threshold: float = Field(default=30.0, ge=0.0, le=100.0)

    # Routing settings
    skip_execution_on_security_issues: bool = Field(default=True)
    require_review_before_execution: bool = Field(default=True)
    auto_iterate_on_failure: bool = Field(default=True)


class VERLConfig(BaseModel):
    """VERL-specific configuration parameters."""

    kl_coef: float = Field(default=0.001, description="KL divergence coefficient")
    ppo_epochs: int = Field(default=4, gt=0, description="Number of PPO epochs per update")
    mini_batch_size: int = Field(default=2, gt=0, description="Mini-batch size for PPO updates")
    clip_ratio: float = Field(default=0.2, gt=0, description="PPO clipping ratio")
    value_clip_ratio: float = Field(default=0.2, gt=0, description="Value function clipping ratio")
    max_grad_norm: float = Field(default=1.0, gt=0, description="Gradient clipping norm")
    entropy_coef: float = Field(default=0.01, ge=0, description="Entropy coefficient")
    value_loss_coef: float = Field(default=0.5, ge=0, description="Value loss coefficient")


class TrainingConfig(BaseModel):
    """Configuration for VERL training."""

    # Data settings
    data_path: str = Field(default="./data/training_problems")
    evaluation_data_path: str = Field(default="./data/evaluation_sets")

    # Training parameters
    algorithm: str = Field(default="ppo")
    episodes: int = Field(default=100, gt=0)
    batch_size: int = Field(default=8, gt=0)
    learning_rate: float = Field(default=1e-5, gt=0)

    # Checkpointing
    checkpoint_dir: str = Field(default="./checkpoints")
    save_interval: int = Field(default=10, gt=0)

    # Monitoring
    wandb_project: Optional[str] = Field(default=None)
    log_interval: int = Field(default=1, gt=0)

    # Reward function weights
    reward_weights: dict[str, float] = Field(
        default_factory=lambda: {"correctness": 0.7, "style": 0.2, "efficiency": 0.1},
        description="Weights for composite reward function",
    )
    
    # CUDA-specific reward parameters
    cuda_rewards: dict[str, float] = Field(
        default_factory=lambda: {
            "target_speedup": 2.0,
            "correctness_weight": 0.4,
            "performance_weight": 0.4,
            "improvement_weight": 0.2
        },
        description="CUDA-specific reward configuration"
    )
    
    # Data sources configuration
    data_sources: list[str] = Field(
        default_factory=lambda: ["kernelbench_local"],
        description="Training data sources"
    )
    
    # Multi-turn conversation configuration
    conversation: dict[str, Any] = Field(
        default_factory=lambda: {
            "max_turns": 5,
            "discount_factor": 0.9,
            "early_termination_threshold": 0.8
        },
        description="Multi-turn conversation settings"
    )

    # VERL-specific parameters
    verl: VERLConfig = Field(default_factory=VERLConfig, description="VERL-specific configuration")


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = Field(default="INFO")
    format: str = Field(default="json")  # json or text
    file_path: Optional[str] = Field(default=None)

    # Structured logging
    include_timestamps: bool = Field(default=True)
    include_context: bool = Field(default=True)
    max_message_length: int = Field(default=1000, gt=0)


class SecurityConfig(BaseModel):
    """Security configuration."""

    # Code analysis
    enable_security_scanning: bool = Field(default=True)
    blocked_imports: list[str] = Field(
        default=["subprocess", "os.system", "eval", "exec", "compile"]
    )
    allowed_imports: Optional[list[str]] = Field(default=None)

    # Execution security
    sandbox_all_execution: bool = Field(default=True)
    network_isolation: bool = Field(default=True)
    file_system_isolation: bool = Field(default=True)
    resource_limits: bool = Field(default=True)


class Config(BaseSettings):
    """Main configuration class."""

    # Metadata
    version: str = Field(default="0.1.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)

    # Component configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    # System settings
    max_concurrent_requests: int = Field(default=10, gt=0)
    request_timeout: int = Field(default=300, gt=0)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "env_nested_delimiter": "__",  # For nested config like LLM__API_KEY
        "extra": "ignore"  # Allow extra fields from env vars
    }


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file and environment.

    Args:
        config_path: Path to configuration file (YAML)

    Returns:
        Loaded configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_data = {}

    # Load from file if provided
    if config_path and Path(config_path).exists():
        with open(config_path, encoding="utf-8") as f:
            file_data = yaml.safe_load(f)
            if file_data:
                config_data.update(file_data)

    # Create config with file data and environment variables
    config = Config(**config_data)

    # Override with environment variables for API keys if not set
    if not config.llm.api_key:
        if config.llm.provider == "openai":
            config.llm.api_key = os.getenv("OPENAI_API_KEY")
        elif config.llm.provider == "anthropic":
            config.llm.api_key = os.getenv("ANTHROPIC_API_KEY")

    return config


def save_config(config: Config, config_path: str) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save
        config_path: Path to save configuration file
    """
    config_data = config.dict()

    # Remove sensitive information
    if "api_key" in config_data.get("llm", {}):
        config_data["llm"]["api_key"] = "[REDACTED]"

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, default_flow_style=False, indent=2)


def create_default_config(config_path: str) -> Config:
    """
    Create a default configuration file.

    Args:
        config_path: Path where to create the config file

    Returns:
        Default configuration
    """
    config = Config()

    # Ensure directory exists
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)

    # Save default configuration
    save_config(config, config_path)

    return config


def validate_config(config: Config) -> list[str]:
    """
    Validate configuration and return any issues.

    Args:
        config: Configuration to validate

    Returns:
        List of validation issues (empty if valid)
    """
    issues = []

    # Check required API keys
    if not config.llm.api_key:
        if config.llm.provider in ["openai", "anthropic"]:
            issues.append(f"API key required for {config.llm.provider}")

    # Check file paths
    if config.training.data_path:
        if not Path(config.training.data_path).exists():
            issues.append(f"Training data path does not exist: {config.training.data_path}")

    # Check resource limits
    if config.agents.executor.execution_timeout > 300:
        issues.append("Execution timeout should not exceed 300 seconds for safety")

    # Check iteration limits
    if config.workflow.max_iterations > 10:
        issues.append("Maximum iterations should not exceed 10 to prevent infinite loops")

    return issues
