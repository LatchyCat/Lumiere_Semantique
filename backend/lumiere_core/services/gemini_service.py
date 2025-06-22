# In backend/lumiere_core/services/gemini_service.py

import os
import google.generativeai as genai
from typing import List, Dict

# Configure the Gemini client from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def is_configured() -> bool:
    """Check if the Gemini service is ready to be used."""
    return GEMINI_API_KEY is not None

def generate_text(prompt: str, model_name: str) -> str:
    """
    Sends a prompt to the Google Gemini API.

    Args:
        prompt: The full prompt to send.
        model_name: The specific Gemini model to use (e.g., 'gemini-1.5-pro-latest').
    """
    if not is_configured():
        return "Error: GEMINI_API_KEY is not configured in the environment."

    print(f"Sending prompt to Google Gemini model: '{model_name}'...")
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred while communicating with the Gemini API: {e}")
        return f"Error from Gemini API: {e}"

def list_models() -> List[Dict[str, str]]:
    """Lists available Gemini models that support text generation."""
    if not is_configured():
        return []

    print("Fetching available Google Gemini models...")
    available = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                model_id = f"gemini/{m.name.replace('models/', '')}"
                available.append({
                    "id": model_id,
                    "provider": "gemini",
                    "name": m.display_name
                })
        return available
    except Exception as e:
        print(f"Could not fetch Gemini models: {e}")
        return []
