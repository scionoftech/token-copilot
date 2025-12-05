"""Azure OpenAI configuration utility for examples.

Loads Azure OpenAI credentials from environment variables.
Copy .env.example to .env and fill in your Azure OpenAI credentials.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_azure_config() -> dict:
    """Get Azure OpenAI configuration from environment variables.

    Returns:
        dict: Azure OpenAI configuration

    Raises:
        ValueError: If required environment variables are not set
    """
    config = {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        "deployment_name": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    }

    # Validate required fields
    missing = [k for k, v in config.items() if not v and k != "api_version"]
    if missing:
        raise ValueError(
            f"Missing required Azure OpenAI environment variables: {', '.join(missing.upper())}. "
            f"Please copy .env.example to .env and fill in your credentials."
        )

    return config


def get_azure_langchain_llm():
    """Get Azure OpenAI LangChain LLM instance.

    Returns:
        AzureChatOpenAI: Configured Azure OpenAI LLM
    """
    try:
        from langchain_openai import AzureChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai is required. Install with: pip install langchain-openai"
        )

    config = get_azure_config()

    return AzureChatOpenAI(
        azure_deployment=config["deployment_name"],
        api_version=config["api_version"],
        azure_endpoint=config["azure_endpoint"],
        api_key=config["api_key"],
        temperature=0.7,
    )


def get_azure_llamaindex_llm():
    """Get Azure OpenAI LlamaIndex LLM instance.

    Returns:
        AzureOpenAI: Configured Azure OpenAI LLM for LlamaIndex
    """
    try:
        from llama_index.llms.azure_openai import AzureOpenAI
    except ImportError:
        raise ImportError(
            "llama-index-llms-azure-openai is required. "
            "Install with: pip install llama-index-llms-azure-openai"
        )

    config = get_azure_config()

    return AzureOpenAI(
        model=config["deployment_name"],
        deployment_name=config["deployment_name"],
        api_key=config["api_key"],
        azure_endpoint=config["azure_endpoint"],
        api_version=config["api_version"],
    )


def print_config_status():
    """Print Azure OpenAI configuration status."""
    try:
        config = get_azure_config()
        print("[OK] Azure OpenAI Configuration Loaded:")
        print(f"  Endpoint: {config['azure_endpoint']}")
        print(f"  Deployment: {config['deployment_name']}")
        print(f"  API Version: {config['api_version']}")
        print(f"  API Key: {'*' * 20}...{config['api_key'][-4:]}")
        return True
    except ValueError as e:
        print(f"[ERROR] Configuration Error: {e}")
        return False


if __name__ == "__main__":
    print_config_status()
