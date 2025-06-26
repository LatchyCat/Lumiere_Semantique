# backend/lumiere_core/services/llm_service.py

from typing import List, Dict, Any

from . import ollama_service
from . import gemini_service

# --- The Lumière Task Router Configuration ---
# This is our Python equivalent of the config.json from claude-code-router.
# We will start with models we know work in our system.
TASK_ROUTER_CONFIG = {
    # For simple, non-critical tasks that require instruction following (like JSON output).
    "ROUTER_TASK_SIMPLE": "gemini/gemini-1.5-flash-latest",

    # For complex reasoning, code generation, and architectural analysis.
    "ROUTER_TASK_COMPLEX_REASONING": "gemini/gemini-1.5-flash-latest",

    # For tasks requiring a huge context window, like summarizing a whole repo.
    "ROUTER_TASK_LONG_CONTEXT": "gemini/gemini-1.5-pro-latest",

    # For code-specific generation tasks.
    "ROUTER_TASK_CODE_GENERATION": "gemini/gemini-1.5-flash-latest",
}


class TaskType:
    """Defines the types of tasks our system can perform."""
    SIMPLE = "ROUTER_TASK_SIMPLE"
    COMPLEX_REASONING = "ROUTER_TASK_COMPLEX_REASONING"
    LONG_CONTEXT = "ROUTER_TASK_LONG_CONTEXT"
    CODE_GENERATION = "ROUTER_TASK_CODE_GENERATION"


def generate_text(prompt: str, task_type: str = TaskType.SIMPLE) -> str:
    """
    The new intelligent entry point. It routes the prompt to the best model for the job.

    Args:
        prompt: The text prompt for the model.
        task_type: The type of task, used to select the right model from the router config.

    Returns:
        The generated text from the chosen model.
    """
    # 1. Select the model based on the task type
    model_identifier = TASK_ROUTER_CONFIG.get(task_type)
    if not model_identifier:
        # Fallback to a simple model if the task type is unknown
        model_identifier = TASK_ROUTER_CONFIG[TaskType.SIMPLE]
        print(f"Warning: Unknown task_type '{task_type}'. Defaulting to {model_identifier}.")

    print(f"Task Router: Routing task '{task_type}' to model '{model_identifier}'")

    # 2. Split the identifier to find the provider
    parts = model_identifier.split('/', 1)
    if len(parts) != 2:
        return f"Error: Invalid model identifier format '{model_identifier}'."

    provider, model_name = parts

    # 3. Call the appropriate provider service
    if provider == "ollama":
        return ollama_service.generate_text(prompt, model_name)
    elif provider == "gemini":
        return gemini_service.generate_text(prompt, model_name)
    else:
        return f"Error: Unknown LLM provider '{provider}'."

def list_available_models() -> List[Dict[str, Any]]:
    """
    Aggregates available models from all configured providers. (Unchanged)
    """
    all_models = []
    all_models.extend(ollama_service.list_models())
    all_models.extend(gemini_service.list_models())
    return all_models
