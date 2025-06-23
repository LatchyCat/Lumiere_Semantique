# In backend/lumiere_core/services/ollama_service.py

import ollama
import requests 
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

# === REPLACE THE ENTIRE list_models FUNCTION WITH THIS NEW VERSION ===
def list_models() -> List[Dict[str, str]]:
    """
    Fetches the list of locally available Ollama models by calling the API directly.
    This is more robust than relying on the library's internal list() parsing.
    """
    print("Fetching available local Ollama models via direct API call...")
    try:
        # Use requests to call the /api/tags endpoint directly
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()

        available = []
        # The structure from the API is {"models": [...]}, so we access the 'models' key
        for model_data in data.get('models', []):
            # The model's full name (e.g., "qwen3:4b") is in the 'name' key
            full_name = model_data.get('name')
            if not full_name:
                continue  # Skip if a model entry is malformed

            model_id = f"ollama/{full_name}"
            available.append({
                "id": model_id,
                "provider": "ollama",
                "name": full_name,  # Use the full name for display as well
            })

        if not available:
            print("Ollama API responded, but no local models were found.")
        else:
            print(f"✓ Found {len(available)} local Ollama models.")

        return available

    except requests.exceptions.ConnectionError:
        print("Could not fetch Ollama models. Is the Ollama server running?")
        return []
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the Ollama API: {e}")
        return []
