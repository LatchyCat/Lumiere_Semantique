# In lumiere_core/services/ollama_service.py

import ollama
from typing import Dict, List

def generate_text(prompt: str, model_name: str = 'qwen3:4b') -> str:
    """
    Sends a prompt to a local Ollama model and returns the response.

    Args:
        prompt: The full prompt to send to the model.
        model_name: The name of the Ollama model to use for generation.

    Returns:
        The generated text content from the model.
    """
    print(f"Sending prompt to local Ollama model: '{model_name}'...")
    try:
        client = ollama.Client()
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return (f"An error occurred while communicating with the Ollama server: {e}\n"
                f"Please ensure the Ollama server is running and the model '{model_name}' is available.")

def list_models() -> List[Dict[str, str]]:
    """
    Fetches the list of locally available Ollama models.
    """
    print("Fetching available local Ollama models...")
    try:
        client = ollama.Client()
        response = client.list()

        available = []
        for model_data in response.get('models', []):
            model_id = f"ollama/{model_data['name']}"
            available.append({
                "id": model_id,
                "provider": "ollama",
                "name": model_data['name']
            })
        return available
    except Exception as e:
        print(f"Could not fetch Ollama models. Is the Ollama server running? Error: {e}")
        return []
