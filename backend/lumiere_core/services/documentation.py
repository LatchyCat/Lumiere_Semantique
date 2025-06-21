# In lumiere_core/services/documentation.py
from typing import Dict

from .ollama import search_index
from .llm import generate_text
from .utils import clean_llm_code_output

def generate_docstring_for_code(repo_id: str, new_code: str, instruction: str) -> Dict[str, str]:
    """
    The core logic for the Chronicler Agent (Documentation).

    It finds existing docstring patterns in the repo and uses them as a style guide
    to generate a new docstring for the provided code.
    """
    print(f"Initiating Chronicler Agent for repo '{repo_id}'")

    # --- Step 1: Find Existing Documentation Patterns with RAG ---
    print("   -> Step 1: Finding existing docstring patterns with RAG...")
    index_path, map_path = f"{repo_id}_faiss.index", f"{repo_id}_id_map.json"
    search_query = f"Example docstrings in Python code for a function about: {instruction}"

    context_chunks = search_index(
        query_text=search_query,
        model_name='snowflake-arctic-embed2:latest',
        index_path=index_path,
        map_path=map_path,
        k=5
    )

    doc_context_string = ""
    found_files = set()
    for chunk in context_chunks:
        # Check if the text contains function or class definitions with docstrings
        # We look for patterns that likely contain good docstring examples
        if chunk['text'].strip().startswith(('def ', 'class ')) and chunk['file_path'] not in found_files:
            doc_context_string += f"--- Example from file \"{chunk['file_path']}\" ---\n{chunk['text']}\n\n"
            found_files.add(chunk['file_path'])

    if not doc_context_string:
        doc_context_string = "No specific docstring styles found. Please generate a standard Google-style docstring."
        print("   -> Warning: No existing docstring examples found via RAG.")
    else:
        print(f"   -> Found docstring patterns from files: {list(found_files)}")

    # --- Step 2: Construct the Docstring Generation Prompt ---
    print("   -> Step 2: Constructing docstring generation prompt...")
    prompt = f"""You are an expert technical writer specializing in Python documentation.

**YOUR INSTRUCTIONS:**
1.  **Analyze "EXISTING DOCSTRING EXAMPLES"** to learn the project's documentation style (e.g., Google, reStructuredText, numpy). Pay attention to sections like `Args:`, `Returns:`, `Raises:`.
2.  **Analyze the "CODE TO BE DOCUMENTED"** to understand its parameters, logic, and what it returns.
3.  **Write a complete and professional docstring** for the provided code. It is CRITICAL that you exactly match the style of the examples.
4.  **Output ONLY the docstring itself.** Do not include the function definition or any other text, just the `\"\"\"...\"\"\"` block.

---
### EXISTING DOCSTRING EXAMPLES
{doc_context_string}
---
### CODE TO BE DOCUMENTED
```python
{new_code}
```

Now, generate ONLY the docstring for the code above."""

    # --- Step 3: Generate and Clean the Docstring ---
    print(f"   -> Step 3: Sending request to code generation model 'qwen2.5-coder:3b'...")
    raw_docstring = generate_text(prompt, model_name='qwen2.5-coder:3b')

    print("   -> Step 4: Cleaning and finalizing the docstring...")
    final_docstring = clean_llm_code_output(raw_docstring)

    # A common LLM artifact is to wrap the docstring in triple quotes. Let's ensure it's clean.
    if final_docstring.startswith('"""') and final_docstring.endswith('"""'):
        final_docstring = final_docstring[3:-3].strip()

    return {"docstring": final_docstring}
