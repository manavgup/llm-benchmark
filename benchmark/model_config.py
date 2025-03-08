"""
Model configuration module for the LLM Benchmark.
Defines available models for each provider and their configurations.
"""

# Default model configurations for each provider
# Format:
# PROVIDER_MODELS = {
#     "model_key": {
#         "name": "display_name",
#         "model_id": "provider_model_id",
#         "description": "model_description"
#     }
# }

ANTHROPIC_MODELS = {
    "claude-3-opus": {
        "name": "Claude 3 Opus",
        "model_id": "claude-3-opus-20240229",
        "description": "Claude 3 Opus by Anthropic (Most powerful)"
    },
    "claude-3-sonnet": {
        "name": "Claude 3 Sonnet",
        "model_id": "claude-3-sonnet-20240229",
        "description": "Claude 3 Sonnet by Anthropic (Balanced)"
    },
    "claude-3-haiku": {
        "name": "Claude 3 Haiku",
        "model_id": "claude-3-haiku-20240307",
        "description": "Claude 3 Haiku by Anthropic (Fast)"
    },
    "claude-2": {
        "name": "Claude 2",
        "model_id": "claude-2.0",
        "description": "Claude 2 by Anthropic (Legacy)"
    }
}

OPENAI_MODELS = {
    "gpt-4-turbo": {
        "name": "GPT-4 Turbo",
        "model_id": "gpt-4-turbo",
        "description": "GPT-4 Turbo by OpenAI (Most powerful)"
    },
    "gpt-4": {
        "name": "GPT-4",
        "model_id": "gpt-4",
        "description": "GPT-4 by OpenAI (Original)"
    },
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo",
        "model_id": "gpt-3.5-turbo",
        "description": "GPT-3.5 Turbo by OpenAI (Fast)"
    }
}

WATSONX_MODELS = {
    "granite-13b": {
        "name": "Granite 13B",
        "model_id": "ibm/granite-13b-instruct-v2",
        "description": "Granite 13B Instruct v2 by IBM"
    },
    "granite-20b": {
        "name": "Granite 20B",
        "model_id": "ibm/granite-20b-instruct-v1",
        "description": "Granite 20B Instruct v1 by IBM"
    },
    "flan-ul2": {
        "name": "Flan-UL2",
        "model_id": "google/flan-ul2",
        "description": "Flan-UL2 by Google (via IBM)"
    },
    "llama-2-70b": {
        "name": "Llama 2 70B",
        "model_id": "meta-llama/llama-2-70b-chat",
        "description": "Llama 2 70B Chat by Meta (via IBM)"
    }
}

OLLAMA_MODELS = {
    "llama3": {
        "name": "Llama 3",
        "model_id": "llama3",
        "description": "Llama 3 by Meta (via Ollama)"
    },
    "mistral": {
        "name": "Mistral",
        "model_id": "mistral",
        "description": "Mistral 7B by Mistral AI (via Ollama)"
    },
    "mixtral": {
        "name": "Mixtral",
        "model_id": "mixtral",
        "description": "Mixtral 8x7B by Mistral AI (via Ollama)"
    },
    "llama2": {
        "name": "Llama 2",
        "model_id": "llama2",
        "description": "Llama 2 by Meta (via Ollama)"
    },
    "codellama": {
        "name": "CodeLlama",
        "model_id": "codellama",
        "description": "CodeLlama by Meta (via Ollama)"
    }
}

# Combine all models into a single dictionary
PROVIDER_MODELS = {
    "anthropic": ANTHROPIC_MODELS,
    "openai": OPENAI_MODELS,
    "watsonx": WATSONX_MODELS,
    "ollama": OLLAMA_MODELS
}

def get_all_provider_models():
    """Get a dictionary of all available models grouped by provider."""
    return PROVIDER_MODELS

def get_provider_models(provider: str):
    """
    Get available models for a specific provider.
    
    Args:
        provider: Provider name (anthropic, openai, watsonx, ollama)
        
    Returns:
        Dictionary of models for the provider or empty dict if provider not found
    """
    return PROVIDER_MODELS.get(provider, {})

def list_available_models():
    """
    Get a formatted string listing all available models.
    
    Returns:
        String with formatted list of all models
    """
    result = "Available Models:\n"
    
    for provider, models in PROVIDER_MODELS.items():
        result += f"\n{provider.upper()}\n"
        result += "-" * len(provider) + "\n"
        
        for key, model in models.items():
            result += f"  - {key}: {model['name']} ({model['description']})\n"
    
    return result

def get_model_info(provider: str, model_key: str):
    """
    Get information for a specific model.
    
    Args:
        provider: Provider name
        model_key: Model key/identifier
        
    Returns:
        Dictionary with model information or None if not found
    """
    provider_models = get_provider_models(provider)
    return provider_models.get(model_key)