# In lumiere_core/services/scaffolding.py
import json
from pathlib import Path
from typing import Dict, Optional

from .ollama import search_index
from .llm import generate_text
# --- NEW: Import the shared cleaning utility ---
from .utils import clean_llm_code_output

def _get_file_content_from_cortex(repo_id: str, target_file_path: str) -> Optional[str]:
    cortex_path = Path(f"{repo_id}_cortex.json")
    if not cortex_path.exists():
        print(f"Error: Cortex file not found at {cortex_path}")
        return None
    with open(cortex_path, 'r', encoding='utf-8') as f:
        cortex_data = json.load(f)
    for file_data in cortex_data.get('files', []):
        if file_data.get('file_path') == target_file_path:
            return file_data.get('raw_content')
    return None

def generate_scaffold(repo_id: str, target_file: str, instruction: str) -> Dict[str, str]:
    """
    The core logic for the Code Scaffolding Agent.
    """
    print(f"Initiating Scaffolding Agent for '{target_file}' in repo '{repo_id}'")

    print("   -> Step 1: Retrieving original file content...")
    original_content = _get_file_content_from_cortex(repo_id, target_file)
    if original_content is None:
        return {"error": f"File '{target_file}' not found in the indexed context for repo '{repo_id}'."}

    print("   -> Step 2: Gathering additional context with RAG...")
    index_path, map_path = f"{repo_id}_faiss.index", f"{repo_id}_id_map.json"
    search_query = f"How to implement the following change in the file {target_file}: {instruction}"
    context_chunks = search_index(query_text=search_query, model_name='snowflake-arctic-embed2:latest', index_path=index_path, map_path=map_path, k=5)
    rag_context_string = "\n\n".join([f"--- Context from file: {chunk['file_path']} ---\n{chunk['text']}" for chunk in context_chunks])

    print("   -> Step 3: Constructing advanced scaffolding prompt...")
    prompt = f"""You are an expert AI pair programmer...
---
### USER INSTRUCTION
{instruction}
---
### ORIGINAL FILE CONTENT: {target_file}
{original_content}
---
### ADDITIONAL RELEVANT CONTEXT
{rag_context_string}
---
Now, provide the full, modified content for the file '{target_file}'.
"""

    print(f"   -> Step 4: Sending request to the default code generation model...")
    raw_generated_code = generate_text(prompt)

    print("   -> Step 5: Cleaning and finalizing the generated code...")
    final_code = clean_llm_code_output(raw_generated_code)

    return {"generated_code": final_code}
