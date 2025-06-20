# In lumiere_core/services/llm.py

import ollama
from typing import Dict

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
        # The ollama client automatically connects to http://localhost:11434
        client = ollama.Client()

        # Use the 'chat' method for conversational models like qwen3
        response = client.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )

        # Extract the text content from the response object
        return response['message']['content']

    except Exception as e:
        # Provide a more specific error message for Ollama
        return (f"An error occurred while communicating with the Ollama server: {e}\n"
                f"Please ensure the Ollama server is running and the model '{model_name}' is available.")
