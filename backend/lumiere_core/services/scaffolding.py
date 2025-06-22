# In backend/lumiere_core/services/scaffolding.py

import json
from pathlib import Path
from typing import Dict, Optional, List

from .ollama import search_index
from . import llm_service
from .utils import clean_llm_code_output

def _get_file_content_from_cortex(repo_id: str, target_file_path: str) -> Optional[str]:
    """Get file content from the cortex JSON file."""
    cortex_path = Path(f"{repo_id}_cortex.json")
    if not cortex_path.exists():
        print(f"Error: Cortex file not found at {cortex_path}")
        return None

    try:
        with open(cortex_path, 'r', encoding='utf-8') as f:
            cortex_data = json.load(f)

        for file_data in cortex_data.get('files', []):
            if file_data.get('file_path') == target_file_path:
                return file_data.get('raw_content')

    except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
        print(f"Error reading cortex file: {e}")
        return None

    return None

def generate_scaffold(
    repo_id: str,
    target_file: str,
    instruction: str,
    model_identifier: str,
    refinement_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, str]:
    """
    The core logic for the Code Scaffolding Agent, now with iterative refinement.
    """
    print(f"Initiating Scaffolding Agent for '{target_file}' in repo '{repo_id}'")
    print(f"Using model: {model_identifier}")

    # Step 1: Retrieve original file content
    print("   -> Step 1: Retrieving original file content...")
    original_content = _get_file_content_from_cortex(repo_id, target_file)
    if original_content is None:
        return {"error": f"File '{target_file}' not found in the indexed context."}

    # Step 2: Gather additional context with RAG
    print("   -> Step 2: Gathering additional context with RAG...")
    index_path = f"{repo_id}_faiss.index"
    map_path = f"{repo_id}_id_map.json"
    search_query = f"How to implement: {instruction}"

    try:
        context_chunks = search_index(
            query_text=search_query,
            model_name='snowflake-arctic-embed2:latest',
            index_path=index_path,
            map_path=map_path,
            k=5
        )
        rag_context_string = "\n\n".join([
            f"--- Context from file: {chunk['file_path']} ---\n{chunk['text']}"
            for chunk in context_chunks
        ])
    except Exception as e:
        print(f"Warning: Error in RAG search: {e}")
        rag_context_string = "No additional context available due to search error."

    # Step 3: Build refinement history section
    print("   -> Step 3: Constructing advanced scaffolding prompt...")
    refinement_prompt_section = ""
    if refinement_history:
        print("   -> Refinement history detected. Adding to prompt.")
        refinement_prompt_section = "\n--- PREVIOUS ATTEMPTS AND USER FEEDBACK ---\n"
        for i, turn in enumerate(refinement_history):
            refinement_prompt_section += f"### Attempt #{i+1}\n"
            refinement_prompt_section += f"I generated this code:\n```python\n{turn['code']}\n```\n"
            refinement_prompt_section += f"The user provided this feedback: '{turn['feedback']}'\n\n"
        refinement_prompt_section += "Analyze the feedback and generate a new, improved version.\n---\n"

    # Step 4: Construct the prompt
    prompt = f"""You are an expert AI pair programmer.

### GOAL
Modify '{target_file}' to accomplish this: "{instruction}"

### ORIGINAL FILE CONTENT
```python
{original_content}
```

{refinement_prompt_section}

### RELEVANT CONTEXT
{rag_context_string}

### INSTRUCTIONS
You MUST provide the ENTIRE, new content for the file '{target_file}'.
Your response MUST be ONLY raw Python code. Do not add commentary or markdown fences.

Now, generate the full, modified content for '{target_file}'."""

    # Step 5: Generate code with error handling
    print("   -> Step 4: Sending request to the LLM...")
    try:
        raw_generated_code = llm_service.generate_text(prompt, model_identifier=model_identifier)
        if not raw_generated_code:
            return {"error": "Failed to generate code - empty response from LLM"}
    except Exception as e:
        return {"error": f"Failed to generate code: {str(e)}"}

    # Step 6: Clean and finalize the generated code
    print("   -> Step 5: Cleaning and finalizing the generated code...")
    try:
        final_code = clean_llm_code_output(raw_generated_code)
    except Exception as e:
        print(f"Warning: Error cleaning code output: {e}")
        final_code = raw_generated_code  # Use raw output if cleaning fails

    return {
        "generated_code": final_code,
        "original_content": original_content
    }
