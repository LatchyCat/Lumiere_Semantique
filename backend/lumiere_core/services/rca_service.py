# In backend/lumiere_core/services/rca_service.py
import json
from pathlib import Path
from typing import Dict, Any

from . import llm_service, github
from .ollama import search_index

def _get_repo_id_from_url(repo_url: str) -> str:
    """Helper to derive repo_id from a URL."""
    return repo_url.replace("https://github.com/", "").replace("/", "_")

def _get_file_content_from_cortex(repo_id: str, file_path: str) -> str | None:
    """Helper to get file content from the cortex JSON."""
    cortex_path = Path(f"{repo_id}_cortex.json")
    if not cortex_path.exists():
        return None
    try:
        with open(cortex_path, 'r', encoding='utf-8') as f:
            cortex_data = json.load(f)
        for file_data in cortex_data.get('files', []):
            if file_data.get('file_path') == file_path:
                return file_data.get('raw_content')
    except (json.JSONDecodeError, FileNotFoundError):
        return None
    return None

def generate_briefing(issue_url: str, model_identifier: str) -> Dict[str, Any]:
    """Generates a 'Pre-flight Briefing' for a GitHub issue using RAG."""
    print(f"Generating briefing for issue: {issue_url}")
    issue_data = github.scrape_github_issue(issue_url)
    if not issue_data:
        return {"error": "Could not retrieve issue details from GitHub."}

    repo_id = _get_repo_id_from_url(issue_data['repo_url'])
    query = issue_data['full_text_query']
    index_path = f"{repo_id}_faiss.index"
    map_path = f"{repo_id}_id_map.json"

    try:
        context_chunks = search_index(
            query_text=query, model_name='snowflake-arctic-embed2:latest',
            index_path=index_path, map_path=map_path, k=7
        )
    except FileNotFoundError:
        return {"error": f"Index files for repo '{repo_id}' not found. Please run the crawler and indexer first."}
    except Exception as e:
        return {"error": f"Failed to retrieve context from vector index: {e}"}

    context_string = "\n\n".join([f"--- Context from file '{chunk['file_path']}' ---\n{chunk['text']}" for chunk in context_chunks])

    prompt = f"""
    You are Lumière Sémantique, an expert AI programming assistant.
    Your mission is to provide a "Pre-flight Briefing" for a developer about to work on a task.
    Analyze the user's query and the provided context from the codebase to generate your report.

    The report must be clear, concise, and structured in Markdown. Include:
    1.  **Task Summary:** Briefly rephrase the user's request.
    2.  **Core Analysis:** Based on context, explain how the system currently works in relation to the query.
    3.  **Key Files & Code:** Point out the most important files or functions from the context.
    4.  **Suggested Approach or Potential Challenges:** Offer a high-level plan or mention potential issues.

    --- PROVIDED CONTEXT FROM THE CODEBASE ---
    {context_string}
    --- END OF CONTEXT ---

    USER'S QUERY: "{query}"

    Now, generate the Pre-flight Briefing.
    """
    briefing_report = llm_service.generate_text(prompt, model_identifier)
    return {"briefing": briefing_report}

def perform_rca(repo_url: str, bug_description: str, target_file: str, model_identifier: str) -> Dict[str, Any]:
    """Performs Root Cause Analysis on a specific file."""
    print(f"Performing RCA on '{target_file}' for bug: '{bug_description}'")
    repo_id = _get_repo_id_from_url(repo_url)

    file_content = _get_file_content_from_cortex(repo_id, target_file)
    if file_content is None:
        return {"error": f"File '{target_file}' not found in the indexed context for repo '{repo_id}'."}

    index_path = f"{repo_id}_faiss.index"
    map_path = f"{repo_id}_id_map.json"

    try:
        context_chunks = search_index(
            query_text=bug_description, model_name='snowflake-arctic-embed2:latest',
            index_path=index_path, map_path=map_path, k=5
        )
    except Exception as e:
        print(f"Warning: RAG search failed during RCA: {e}. Proceeding without extra context.")
        context_chunks = []

    rag_context = "\n\n".join([f"--- Context from file '{chunk['file_path']}' ---\n{chunk['text']}" for chunk in context_chunks])

    prompt = f"""
    You are a master debugger and senior engineer. Your task is to perform a Root Cause Analysis (RCA).

    **Bug Description:** "{bug_description}"
    **Target File under Scrutiny:** `{target_file}`

    **Content of `{target_file}`:**
    ```python
    {file_content}
    ```

    **Potentially Related Code from Other Files:**
    {rag_context}

    **Your Analysis:**
    Based on all the provided information, analyze the code and explain the likely root cause of the bug.
    Be specific. Point to exact line numbers or functions if possible. Explain the faulty logic.
    Structure your analysis in Markdown.
    """
    analysis_report = llm_service.generate_text(prompt, model_identifier)
    return {"analysis": analysis_report}
