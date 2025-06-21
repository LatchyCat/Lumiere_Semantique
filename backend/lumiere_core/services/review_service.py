# In lumiere_core/services/review_service.py
import uuid
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Optional, List

from .llm import generate_text

# This is a simple in-memory cache for our development server.
REVIEW_ENVIRONMENTS = {}

def _resolve_git_ref(repo_path: Path, ref_name: str) -> Optional[str]:
    """
    [TRUE FINAL VERSION] Verifies a git reference by trying a list of candidates
    to handle common naming variations (e.g., tags with/without 'v' prefix).
    """
    # Build a list of potential candidates to check.
    candidates: List[str] = []

    # 1. Add the ref_name as a potential remote branch.
    candidates.append(f"origin/{ref_name}")

    # 2. Add the ref_name as a direct reference (exact match for a tag, commit, etc.).
    candidates.append(ref_name)

    # 3. Handle the 'v' prefix for version tags, which is the source of the issue.
    if ref_name.startswith('v') and len(ref_name) > 1:
        # If the user provided 'v1.26.15', try '1.26.15' as a direct ref and as a branch.
        ref_without_v = ref_name[1:]
        candidates.append(f"origin/{ref_without_v}")
        candidates.append(ref_without_v)

    # We will now iterate through our candidates and return the first one that is valid.
    print(f"   -> Attempting to resolve '{ref_name}' with candidates: {candidates}")
    for candidate in candidates:
        try:
            # `rev-parse --verify` is the correct, simple tool for this.
            # It exits 0 if the ref is valid and can be resolved, 1 otherwise.
            subprocess.run(
                ['git', 'rev-parse', '--verify', '--quiet', candidate],
                cwd=repo_path, check=True, capture_output=True
            )
            # SUCCESS: The candidate is a valid git object.
            print(f"   -> SUCCESS: Resolved '{ref_name}' as valid git object '{candidate}'")
            return candidate
        except subprocess.CalledProcessError:
            # This candidate failed, continue to the next one.
            continue

    # If the loop completes without finding a valid candidate, the ref does not exist.
    print(f"   -> FAILED: Could not resolve '{ref_name}' with any of the candidates.")
    return None

def prepare_review_environment(repo_url: str, ref_name: str) -> Dict[str, str]:
    """
    [TRUE FINAL VERSION] Clones a repo, fetches all refs, then uses the truly robust
    _resolve_git_ref function to validate and store them for the review.
    """
    review_id = str(uuid.uuid4())
    temp_dir_handle = tempfile.TemporaryDirectory()
    repo_path = Path(temp_dir_handle.name)
    print(f"Preparing review environment {review_id} at {repo_path}")

    try:
        print(f"   -> Cloning {repo_url}...")
        subprocess.run(
            ['git', 'clone', repo_url, str(repo_path)],
            check=True, capture_output=True, text=True
        )

        print("   -> Fetching all data from remote 'origin'...")
        subprocess.run(
            ['git', 'fetch', 'origin', '--prune', '--tags', '--force'],
             cwd=repo_path, check=True, capture_output=True, text=True
        )

        print(f"   -> Resolving feature ref '{ref_name}'...")
        resolved_feature_ref = _resolve_git_ref(repo_path, ref_name)
        if not resolved_feature_ref:
            raise Exception(f"The branch or tag '{ref_name}' could not be found.")

        print("   -> Resolving base branch...")
        resolved_base_ref = _resolve_git_ref(repo_path, 'main')
        if not resolved_base_ref:
            resolved_base_ref = _resolve_git_ref(repo_path, 'master')
        if not resolved_base_ref:
            raise Exception("Could not find a valid base branch ('main' or 'master').")

        REVIEW_ENVIRONMENTS[review_id] = {
            "path": repo_path, "temp_dir_handle": temp_dir_handle,
            "base_ref": resolved_base_ref, "feature_ref": resolved_feature_ref
        }
        print(f"✓ Review environment '{review_id}' is ready.")
        return {"review_id": review_id}

    except Exception as e:
        temp_dir_handle.cleanup()
        error_details = str(e)
        if hasattr(e, 'stderr') and e.stderr and isinstance(e.stderr, str):
            error_details = e.stderr.strip()
        print(f"Error preparing environment: {error_details}")
        raise Exception(error_details)

def get_diff_for_review(review_id: str) -> str:
    env = REVIEW_ENVIRONMENTS.get(review_id)
    if not env: raise FileNotFoundError(f"Review ID '{review_id}' not found or has expired.")
    repo_path, base_ref, feature_ref = env['path'], env['base_ref'], env['feature_ref']
    print(f"   -> Calculating diff for review '{review_id}' between '{base_ref}' and '{feature_ref}'...")
    try:
        diff_command = ['git', 'diff', f'{base_ref}...{feature_ref}']
        result = subprocess.run(diff_command, cwd=repo_path, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to generate diff: {e.stderr.strip() if e.stderr else str(e)}")

def cleanup_review_environment(review_id: str) -> bool:
    env = REVIEW_ENVIRONMENTS.pop(review_id, None)
    if env:
        try:
            env['temp_dir_handle'].cleanup()
            print(f"✓ Cleaned up review environment '{review_id}'.")
            return True
        except Exception as e:
            print(f"Warning: Error during cleanup of review environment '{review_id}': {e}")
            return False
    else:
        print(f"Warning: Review environment '{review_id}' not found for cleanup.")
        return False

def review_code_diff(diff_text: str) -> Dict[str, str]:
    if not diff_text.strip(): return {"review": "No changes detected between the references. The review is complete."}
    prompt = f"You are an expert Senior Software Engineer...\n---\nGIT DIFF\n```diff\n{diff_text}\n```\nNow, provide your review."
    try:
        return {"review": generate_text(prompt, model_name='qwen3:4b')}
    except Exception as e:
        return {"review": f"Error occurred during code review analysis: {str(e)}"}

def get_active_review_count() -> int: return len(REVIEW_ENVIRONMENTS)
def list_active_reviews() -> Dict[str, Dict[str, str]]:
    summary = {}
    for r_id, env in REVIEW_ENVIRONMENTS.items():
        summary[r_id] = {"base": env["base_ref"], "feat": env["feature_ref"], "path": str(env["path"])}
    return summary
