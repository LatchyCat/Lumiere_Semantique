# In backend/lumire_core/services/llm_service.py

from typing import List, Dict, Any

from . import ollama_service
from . import gemini_service

# --- Public API for the LLM Service ---

def generate_text(prompt: str, model_identifier: str) -> str:
    """
    Generates text using the specified model provider.
    This is the single entry point for all other services.

    Args:
        prompt: The text prompt for the model.
        model_identifier: A string like "provider/model-name"
                          (e.g., "ollama/qwen3:4b" or "gemini/gemini-1.5-pro-latest").

    Returns:
        The generated text from the model.
    """
    parts = model_identifier.split('/', 1)
    if len(parts) != 2:
        return f"Error: Invalid model identifier format '{model_identifier}'. Expected 'provider/model-name'."

    provider, model_name = parts

    if provider == "ollama":
        return ollama_service.generate_text(prompt, model_name)
    elif provider == "gemini":
        return gemini_service.generate_text(prompt, model_name)
    else:
        return f"Error: Unknown LLM provider '{provider}'."


def list_available_models() -> List[Dict[str, Any]]:
    """
    Aggregates available models from all configured providers.
    """
    all_models = []
    all_models.extend(ollama_service.list_models())
    all_models.extend(gemini_service.list_models())
    return all_models
