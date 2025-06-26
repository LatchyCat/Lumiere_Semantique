# In backend/lumiere_core/services/ambassador.py

import os
import re
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv

from github import Github, GithubException

from .github import scrape_github_issue, _parse_github_issue_url

from . import llm_service
from .llm_service import TaskType
from ingestion.crawler import IntelligentCrawler

# --- Configuration ---
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')
GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_FORK_USERNAME")

if not GITHUB_TOKEN or not GITHUB_USERNAME:
    raise ValueError("GITHUB_ACCESS_TOKEN and GITHUB_FORK_USERNAME must be set in the .env file.")

g = Github(GITHUB_TOKEN)
user = g.get_user(GITHUB_USERNAME)

def _sanitize_branch_name(text: str) -> str:
    """Creates a URL- and git-safe branch name from a string."""
    text = text.lower()
    text = re.sub(r'[\s/]+', '-', text)
    text = re.sub(r'[^a-z0-9-]', '', text)
    text = text.strip('-')
    return text[:60]

def _validate_file_changes(modified_files: Dict[str, str]) -> List[str]:
    """Validates the file changes dictionary and returns any validation errors."""
    errors = []

    if not modified_files:
        errors.append("No files provided for modification")
        return errors

    for file_path, content in modified_files.items():
        if not isinstance(file_path, str) or not file_path.strip():
            errors.append(f"Invalid file path: {repr(file_path)}")

        if not isinstance(content, str):
            errors.append(f"Invalid content type for {file_path}: expected string, got {type(content)}")

        # Check for potentially dangerous paths
        if file_path.startswith('/') or '..' in file_path:
            errors.append(f"Potentially unsafe file path: {file_path}")

    return errors

def _write_files_safely(repo_path: Path, modified_files: Dict[str, str]) -> List[str]:
    """
    Safely writes files to the repository with proper error handling.
    Returns a list of successfully written files.
    """
    successfully_written = []

    for file_path, new_content in modified_files.items():
        try:
            full_target_path = repo_path / file_path

            # Ensure parent directories exist
            full_target_path.parent.mkdir(parents=True, exist_ok=True)

            # Create backup if file exists
            backup_path = None
            if full_target_path.exists():
                backup_path = full_target_path.with_suffix(full_target_path.suffix + '.bak')
                full_target_path.rename(backup_path)

            try:
                # Write new content with explicit encoding
                full_target_path.write_text(new_content, encoding='utf-8')

                # Stage the file
                subprocess.run(
                    ['git', 'add', str(full_target_path)],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                )

                successfully_written.append(file_path)
                print(f"✓ Staged changes for: {file_path}")

                # Remove backup if successful
                if backup_path and backup_path.exists():
                    backup_path.unlink()

            except Exception as e:
                # Restore backup if write failed
                if backup_path and backup_path.exists():
                    if full_target_path.exists():
                        full_target_path.unlink()
                    backup_path.rename(full_target_path)
                raise e

        except Exception as e:
            print(f"⚠ Failed to write and stage {file_path}: {e}")
            # Continue with other files rather than failing completely
            continue

    return successfully_written

def _run_git_command(command: List[str], repo_path: Path, description: str) -> bool:
    """
    Runs a git command with proper error handling and logging.
    Returns True if successful, False otherwise.
    """
    try:
        subprocess.run(
            command,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e.stderr.strip()}")
        return False

def _generate_pr_body(issue_data: Dict[str, Any], issue_number: int, modified_files: List[str]) -> str:
    """Generates a comprehensive PR body with file change summary."""
    pr_body = f"""
This pull request was automatically generated and approved by the user via the Lumière Sémantique 'Socratic Dialogue' interface to address Issue #{issue_number}.

## Issue Summary
> {issue_data.get('body', 'No description provided.')[:500]}{'...' if len(issue_data.get('body', '')) > 500 else ''}

## Changes in this PR
This PR modifies **{len(modified_files)}** file(s) to resolve the issue:

"""

    for file_path in sorted(modified_files):
        pr_body += f"- `{file_path}`\n"

    pr_body += "\n---\n*This fix was validated by The Crucible against the project's existing test suite.*"

    return pr_body

def dispatch_pr(
    issue_url: str,
    modified_files: Dict[str, str],
    custom_commit_message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Orchestrates the git operations and PR creation for a multi-file change set.
    """
    print("--- AMBASSADOR AGENT ACTIVATED (MULTI-FILE MODE) ---")
    validation_errors = _validate_file_changes(modified_files)
    if validation_errors:
        return {"error": f"Validation failed: {'; '.join(validation_errors)}"}
    print("\n[Step 1/4] Gathering Intel...")
    try:
        issue_data = scrape_github_issue(issue_url)
        if not issue_data: raise ValueError("Failed to scrape issue data.")
        parsed_url = _parse_github_issue_url(issue_url)
        if not parsed_url: raise ValueError("Could not parse issue URL.")
        owner, repo_name, issue_number = parsed_url
        repo_full_name = f"{owner}/{repo_name}"
    except Exception as e:
        return {"error": f"Intel gathering failed: {e}"}
    print("\n[Step 2/4] Preparing Repository...")
    try:
        upstream_repo = g.get_repo(repo_full_name)
        fork_name = f"{GITHUB_USERNAME}/{repo_name}"
        try:
            working_repo = g.get_repo(fork_name)
            print(f"✓ Using existing fork: {fork_name}")
        except GithubException:
            print("Creating fork...")
            working_repo = upstream_repo.create_fork()
            time.sleep(15)
        with IntelligentCrawler(repo_url=working_repo.clone_url) as crawler:
            repo_path = crawler.repo_path
            branch_name = f"lumiere-fix/{issue_number}-{_sanitize_branch_name(issue_data['title'])}"
            default_branch = upstream_repo.default_branch
            print(f"\n[Step 3/4] Applying Fix on new branch '{branch_name}'...")
            if not _run_git_command(['git', 'checkout', default_branch], repo_path, f"Switched to {default_branch}"):
                return {"error": "Failed to checkout default branch"}
            if not _run_git_command(['git', 'pull', upstream_repo.clone_url, default_branch], repo_path, f"Pulled latest from upstream {default_branch}"):
                 return {"error": "Failed to pull latest upstream changes"}
            if not _run_git_command(['git', 'checkout', '-b', branch_name], repo_path, f"Created branch {branch_name}"):
                _run_git_command(['git', 'checkout', branch_name], repo_path, f"Switched to existing branch {branch_name}")
            successfully_written = _write_files_safely(repo_path, modified_files)
            if not successfully_written:
                return {"error": "No files were successfully written"}
            if custom_commit_message:
                commit_message = custom_commit_message
            else:
                commit_prompt = f"Based on the issue title '{issue_data['title']}' and modified files: {', '.join(successfully_written)}, write a concise one-line Conventional Commits message. Output ONLY the message line."

                # --- THE CHANGE IS HERE ---
                commit_message = llm_service.generate_text(
                    commit_prompt,
                    task_type=TaskType.SIMPLE
                ).strip()

            if not _run_git_command(['git', 'commit', '-m', commit_message], repo_path, "Committed changes"):
                return {"error": "Failed to commit changes. Nothing to commit or git error."}
            if not _run_git_command(['git', 'push', '--set-upstream', 'origin', branch_name, '--force'], repo_path, "Pushed branch"):
                return {"error": "Failed to push branch"}
            print("\n[Step 4/4] Creating Pull Request...")
            pr_title = f"fix: {issue_data['title']} (resolves #{issue_number})"
            pr_body = _generate_pr_body(issue_data, issue_number, successfully_written)
            head_branch = f"{working_repo.owner.login}:{branch_name}"
            pull_request = upstream_repo.create_pull(
                title=pr_title, body=pr_body, head=head_branch, base=default_branch
            )
            print(f"✓ Pull Request created: {pull_request.html_url}")
            return {"status": "success", "pull_request_url": pull_request.html_url}
    except Exception as e:
        error_details = str(e)
        if isinstance(e, GithubException): error_details = e.data.get('message', str(e))
        return {"error": f"Operation failed: {error_details}"}
