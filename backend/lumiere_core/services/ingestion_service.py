# backend/lumiere_core/services/ingestion_service.py

import json
import traceback
import os
import re
from pathlib import Path
from typing import Dict, Any

from ingestion.crawler import IntelligentCrawler
from ingestion.jsonifier import Jsonifier
from ingestion.indexing import EmbeddingIndexer

def generate_repo_id(repo_url: str, max_length: int = 100) -> str:
    """
    Generate a safe repository ID from a GitHub URL with intelligent truncation.

    Args:
        repo_url: The GitHub repository URL
        max_length: Maximum length for the generated repo_id (default: 100)

    Returns:
        A safe, filesystem-friendly repository identifier
    """
    # Extract the repo path (owner/repo-name)
    repo_path = repo_url.replace("https://github.com/", "").replace("/", "_")

    # If it's already within limits, return as-is (backward compatibility)
    if len(repo_path) <= max_length:
        return repo_path

    # Split into owner and repo name for intelligent truncation
    original_path = repo_url.replace("https://github.com/", "")
    parts = original_path.split("/", 1)

    if len(parts) != 2:
        # Fallback: just truncate the whole thing
        return repo_path[:max_length]

    owner, repo_name = parts

    # Calculate available space for repo name after owner and underscore
    available_space = max_length - len(owner) - 1  # -1 for the underscore

    if available_space <= 10:  # If owner name is too long, truncate both
        # Use first 40 chars for owner, rest for repo (with underscore)
        truncated_owner = owner[:40]
        available_for_repo = max_length - len(truncated_owner) - 1
        truncated_repo = repo_name[:available_for_repo] if available_for_repo > 0 else ""
        return f"{truncated_owner}_{truncated_repo}".rstrip("_")

    # Truncate repo name intelligently
    if len(repo_name) > available_space:
        # Try to keep meaningful parts of the repo name
        # Remove common words and separators, keep important parts
        cleaned_repo = re.sub(r'[-_\s]+', '-', repo_name)
        words = cleaned_repo.split('-')

        if len(words) > 1:
            # Keep first and last words, add middle words until we hit the limit
            truncated_parts = [words[0]]
            remaining_space = available_space - len(words[0])

            # Add words from the end working backwards
            for word in reversed(words[1:]):
                if remaining_space >= len(word) + 1:  # +1 for separator
                    truncated_parts.insert(-1, word)
                    remaining_space -= len(word) + 1
                else:
                    break

            truncated_repo = '-'.join(truncated_parts)
        else:
            # Single word, just truncate
            truncated_repo = repo_name[:available_space]

        return f"{owner}_{truncated_repo}"

    return repo_path


def clone_and_embed_repository(repo_url: str, embedding_model: str = 'snowflake-arctic-embed2:latest') -> Dict[str, Any]:
    """
    Orchestrates the entire ingestion pipeline, saving all artifacts into a
    repository-specific subdirectory. This is the single source of truth for ingestion.
    """
    # Use the enhanced repo_id generation
    repo_id = generate_repo_id(repo_url)

    # Define the output directory structure inside the backend
    backend_dir = Path(__file__).resolve().parent.parent.parent
    artifacts_base_dir = backend_dir / "cloned_repositories"
    repo_output_dir = artifacts_base_dir / repo_id

    # Ensure the final destination directory exists
    repo_output_dir.mkdir(parents=True, exist_ok=True)

    # Define the full path for the cortex file within the new directory
    output_cortex_path = repo_output_dir / f"{repo_id}_cortex.json"

    print(f"--- INGESTION SERVICE: Starting for {repo_id} ---")
    print(f"   -> Original URL: {repo_url}")
    print(f"   -> Generated repo_id: {repo_id}")
    print(f"   -> Artifacts will be saved to: {repo_output_dir}")

    try:
        # --- Step 1: Crawl & Jsonify ---
        print(f"[1/3] Cloning repository and generating Project Cortex file...")
        with IntelligentCrawler(repo_url=repo_url) as crawler:
            files_to_process = crawler.get_file_paths()
            if not files_to_process:
                return {"status": "failed", "error": "No files found to process in the repository."}

            jsonifier = Jsonifier(
                file_paths=files_to_process,
                repo_root=crawler.repo_path,
                repo_id=repo_id
            )
            project_cortex = jsonifier.generate_cortex()

            with open(output_cortex_path, 'w', encoding='utf-8') as f:
                json.dump(project_cortex, f, indent=2)
            print(f"✓ Project Cortex created successfully: {output_cortex_path}")

        # --- Step 2: Index ---
        print(f"[2/3] Starting vector indexing with model '{embedding_model}'...")
        # The indexer will now automatically save its files alongside the cortex file.
        indexer = EmbeddingIndexer(model_name=embedding_model)
        indexer.process_cortex(str(output_cortex_path))
        print(f"✓ Vector indexing complete.")

        print(f"[3/3] Ingestion complete. Artifacts preserved in '{repo_output_dir}'.")

        return {
            "status": "success",
            "message": f"Repository '{repo_id}' was successfully cloned, embedded, and indexed.",
            "repo_id": repo_id,
            "original_url": repo_url
        }

    except Exception as e:
        print(f"--- INGESTION FAILED for {repo_id} ---")
        traceback.print_exc()
        if output_cortex_path.exists():
            os.remove(output_cortex_path)
        return {"status": "failed", "error": str(e), "details": traceback.format_exc(), "repo_id": repo_id}
