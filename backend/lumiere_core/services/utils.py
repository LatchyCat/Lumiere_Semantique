# In ~/lumiere_semantique/backend/lumiere_core/services/utils.py
# In lumiere_core/services/utils.py
import re

def clean_llm_code_output(raw_code: str) -> str:
    """
    [Robustness] Removes Markdown code fences and extraneous whitespace from LLM output.

    This function uses a regular expression to find and remove
    common Markdown code block fences (like ```python or ```) from the start
    and end of the string. It also strips any leading or trailing whitespace.
    This is a shared utility to ensure all code-generating agents produce
    clean, machine-readable output.
    """
    # This regex matches an optional language specifier (like 'python', 'toml')
    # and the code fences themselves at the start and end of the string.
    code_fence_pattern = r"^\s*```[a-zA-Z]*\n?|```\s*$"
    cleaned_code = re.sub(code_fence_pattern, '', raw_code)
    return cleaned_code.strip()
