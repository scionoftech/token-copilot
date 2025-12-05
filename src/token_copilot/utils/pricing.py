"""Model pricing database and cost calculation utilities."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    model_id: str
    provider: str
    context_window: int
    max_output_tokens: int
    input_cost_per_1m: float  # Cost per 1M input tokens in USD
    output_cost_per_1m: float  # Cost per 1M output tokens in USD
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    family: str = ""  # e.g., "gpt-4", "claude-3", "llama2"


# Comprehensive model pricing database (as of 2025)
MODEL_PRICING: Dict[str, ModelConfig] = {
    # OpenAI Models
    "gpt-3.5-turbo": ModelConfig(
        model_id="gpt-3.5-turbo",
        provider="openai",
        context_window=16385,
        max_output_tokens=4096,
        input_cost_per_1m=0.50,
        output_cost_per_1m=1.50,
        supports_function_calling=True,
        family="gpt-3.5",
    ),
    "gpt-3.5-turbo-16k": ModelConfig(
        model_id="gpt-3.5-turbo-16k",
        provider="openai",
        context_window=16385,
        max_output_tokens=4096,
        input_cost_per_1m=0.50,
        output_cost_per_1m=1.50,
        supports_function_calling=True,
        family="gpt-3.5",
    ),
    "gpt-4": ModelConfig(
        model_id="gpt-4",
        provider="openai",
        context_window=8192,
        max_output_tokens=8192,
        input_cost_per_1m=30.00,
        output_cost_per_1m=60.00,
        supports_function_calling=True,
        family="gpt-4",
    ),
    "gpt-4-32k": ModelConfig(
        model_id="gpt-4-32k",
        provider="openai",
        context_window=32768,
        max_output_tokens=32768,
        input_cost_per_1m=60.00,
        output_cost_per_1m=120.00,
        supports_function_calling=True,
        family="gpt-4",
    ),
    "gpt-4-turbo": ModelConfig(
        model_id="gpt-4-turbo",
        provider="openai",
        context_window=128000,
        max_output_tokens=4096,
        input_cost_per_1m=10.00,
        output_cost_per_1m=30.00,
        supports_function_calling=True,
        supports_vision=True,
        family="gpt-4",
    ),
    "gpt-4-turbo-preview": ModelConfig(
        model_id="gpt-4-turbo-preview",
        provider="openai",
        context_window=128000,
        max_output_tokens=4096,
        input_cost_per_1m=10.00,
        output_cost_per_1m=30.00,
        supports_function_calling=True,
        family="gpt-4",
    ),
    "gpt-4o": ModelConfig(
        model_id="gpt-4o",
        provider="openai",
        context_window=128000,
        max_output_tokens=4096,
        input_cost_per_1m=5.00,
        output_cost_per_1m=15.00,
        supports_function_calling=True,
        supports_vision=True,
        family="gpt-4",
    ),
    "gpt-4o-mini": ModelConfig(
        model_id="gpt-4o-mini",
        provider="openai",
        context_window=128000,
        max_output_tokens=16384,
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
        supports_function_calling=True,
        supports_vision=True,
        family="gpt-4",
    ),
    # Anthropic Claude Models
    "claude-2.0": ModelConfig(
        model_id="claude-2.0",
        provider="anthropic",
        context_window=100000,
        max_output_tokens=4096,
        input_cost_per_1m=8.00,
        output_cost_per_1m=24.00,
        family="claude-2",
    ),
    "claude-2.1": ModelConfig(
        model_id="claude-2.1",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=4096,
        input_cost_per_1m=8.00,
        output_cost_per_1m=24.00,
        family="claude-2",
    ),
    "claude-3-opus-20240229": ModelConfig(
        model_id="claude-3-opus-20240229",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=4096,
        input_cost_per_1m=15.00,
        output_cost_per_1m=75.00,
        supports_vision=True,
        family="claude-3",
    ),
    "claude-3-sonnet-20240229": ModelConfig(
        model_id="claude-3-sonnet-20240229",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=4096,
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        supports_vision=True,
        family="claude-3",
    ),
    "claude-3-haiku-20240307": ModelConfig(
        model_id="claude-3-haiku-20240307",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=4096,
        input_cost_per_1m=0.25,
        output_cost_per_1m=1.25,
        supports_vision=True,
        family="claude-3",
    ),
    # Ollama Models (local, free but include for reference)
    "llama2": ModelConfig(
        model_id="llama2",
        provider="ollama",
        context_window=4096,
        max_output_tokens=2048,
        input_cost_per_1m=0.0,
        output_cost_per_1m=0.0,
        family="llama2",
    ),
    "llama2:13b": ModelConfig(
        model_id="llama2:13b",
        provider="ollama",
        context_window=4096,
        max_output_tokens=2048,
        input_cost_per_1m=0.0,
        output_cost_per_1m=0.0,
        family="llama2",
    ),
    "llama2:70b": ModelConfig(
        model_id="llama2:70b",
        provider="ollama",
        context_window=4096,
        max_output_tokens=2048,
        input_cost_per_1m=0.0,
        output_cost_per_1m=0.0,
        family="llama2",
    ),
    "codellama": ModelConfig(
        model_id="codellama",
        provider="ollama",
        context_window=16384,
        max_output_tokens=4096,
        input_cost_per_1m=0.0,
        output_cost_per_1m=0.0,
        family="codellama",
    ),
    "mistral": ModelConfig(
        model_id="mistral",
        provider="ollama",
        context_window=8192,
        max_output_tokens=4096,
        input_cost_per_1m=0.0,
        output_cost_per_1m=0.0,
        family="mistral",
    ),
    "mixtral": ModelConfig(
        model_id="mixtral",
        provider="ollama",
        context_window=32768,
        max_output_tokens=4096,
        input_cost_per_1m=0.0,
        output_cost_per_1m=0.0,
        family="mixtral",
    ),
}


# Model aliases for convenience
MODEL_ALIASES = {
    "gpt-3.5": "gpt-3.5-turbo",
    "gpt-4-preview": "gpt-4-turbo-preview",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-opus": "claude-3-opus-20240229",
    "claude-sonnet": "claude-3-sonnet-20240229",
    "claude-haiku": "claude-3-haiku-20240307",
    # Azure OpenAI versioned models
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-4o-2024-11-20": "gpt-4o",
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4o-2024-05-13": "gpt-4o",
    "gpt-4-turbo-2024-04-09": "gpt-4-turbo",
    "gpt-3.5-turbo-0125": "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106": "gpt-3.5-turbo",
}


def resolve_model_alias(model: str) -> str:
    """Resolve model alias to actual model ID."""
    return MODEL_ALIASES.get(model, model)


def get_model_config(model: str) -> Optional[ModelConfig]:
    """Get model configuration by ID or alias."""
    model = resolve_model_alias(model)
    return MODEL_PRICING.get(model)


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate cost for a given model and token usage.

    Args:
        model: Model ID or alias
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Total cost in USD

    Example:
        >>> calculate_cost("gpt-4", 1000, 500)
        0.06  # $0.03 input + $0.03 output
    """
    config = get_model_config(model)
    if not config:
        return 0.0  # Unknown model, can't calculate cost

    input_cost = (input_tokens / 1_000_000) * config.input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * config.output_cost_per_1m
    return input_cost + output_cost


def list_models(provider: Optional[str] = None) -> List[str]:
    """
    List available models, optionally filtered by provider.

    Args:
        provider: Provider name (e.g., "openai", "anthropic", "ollama")

    Returns:
        List of model IDs

    Example:
        >>> list_models("openai")
        ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', ...]
    """
    if provider:
        return [
            model_id
            for model_id, config in MODEL_PRICING.items()
            if config.provider == provider
        ]
    return list(MODEL_PRICING.keys())


def list_providers() -> List[str]:
    """
    List all available providers.

    Returns:
        List of provider names

    Example:
        >>> list_providers()
        ['openai', 'anthropic', 'ollama']
    """
    return list(set(config.provider for config in MODEL_PRICING.values()))
