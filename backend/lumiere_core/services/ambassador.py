# In backend/lumiere_core/services/ambassador.py

import os
import re
import time
import subprocess
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

from github import Github, GithubException

from .github import scrape_github_issue, _parse_github_issue_url
from . import llm_service
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

# --- CORRECTED FUNCTION SIGNATURE ---
def dispatch_pr(issue_url: str, target_file: str, fixed_code: str, model_identifier: str) -> Dict[str, str]:
    """
    Orchestrates the git operations and PR creation using user-approved code.
    """
    print("--- AMBASSADOR AGENT ACTIVATED ---")
    print(f"Target Issue: {issue_url}")
    print(f"Using model for commit message: {model_identifier}")

    print("\n[Step 1/4] Gathering Intel...")
    try:
        issue_data = scrape_github_issue(issue_url)
        if not issue_data:
            raise ValueError("Failed to scrape issue data from GitHub.")

        parsed_url = _parse_github_issue_url(issue_url)
        if not parsed_url:
            raise ValueError("Could not parse the provided issue URL.")
        owner, repo_name, issue_number = parsed_url
        repo_full_name = f"{owner}/{repo_name}"
        print(f"✓ Intel gathered for issue #{issue_number}.")
    except Exception as e:
        return {"error": f"Failed during intel gathering: {e}"}

    print("\n[Step 2/4] Preparing Repository...")
    try:
        upstream_repo = g.get_repo(repo_full_name)
        working_repo = None

        if upstream_repo.owner.login == GITHUB_USERNAME:
            print(f"✓ Target repo is owned by '{GITHUB_USERNAME}'. Skipping fork.")
            working_repo = upstream_repo
        else:
            print(f"Target repo owned by '{upstream_repo.owner.login}'. Finding or creating fork.")
            try:
                working_repo = g.get_repo(f"{GITHUB_USERNAME}/{repo_name}")
                print(f"✓ Fork '{working_repo.full_name}' already exists.")
            except GithubException:
                print("Fork not found. Creating fork...")
                working_repo = upstream_repo.create_fork()
                print(f"✓ Fork created at '{working_repo.full_name}'. Waiting for it to be ready...")
                time.sleep(15)

        with IntelligentCrawler(repo_url=working_repo.clone_url) as crawler:
            repo_path = crawler.repo_path
            sanitized_title = _sanitize_branch_name(issue_data['title'])
            branch_name = f"lumiere-fix/{issue_number}-{sanitized_title}"
            print(f"✓ Cloned {working_repo.full_name} to '{repo_path}'")

            print(f"\n[Step 3/4] Applying Fix on new branch '{branch_name}'...")
            default_branch = upstream_repo.default_branch
            subprocess.run(['git', 'checkout', default_branch], cwd=repo_path, check=True)
            subprocess.run(['git', 'pull', 'origin', default_branch], cwd=repo_path, check=True)
            subprocess.run(['git', 'checkout', '-b', branch_name], cwd=repo_path, check=True)

            full_target_path = repo_path / target_file
            full_target_path.parent.mkdir(parents=True, exist_ok=True)
            full_target_path.write_text(fixed_code, encoding='utf-8')

            commit_prompt = f"Based on the issue title '{issue_data['title']}', write a concise, one-line commit message following the Conventional Commits specification (e.g., 'fix: ...'). Output ONLY the commit message line."

            # --- CORRECTED LLM CALL ---
            commit_message = llm_service.generate_text(commit_prompt, model_identifier=model_identifier).strip()

            subprocess.run(['git', 'add', target_file], cwd=repo_path, check=True)
            subprocess.run(['git', 'commit', '-m', commit_message], cwd=repo_path, check=True)
            print(f"✓ Committed changes with message: \"{commit_message}\"")
            subprocess.run(['git', 'push', '--set-upstream', 'origin', branch_name], cwd=repo_path, check=True)
            print("✓ Pushed new branch to the working repository.")

            print("\n[Step 4/4] Creating Pull Request...")
            pr_title = f"Fix: {issue_data['title']} (Resolves #{issue_number})"
            pr_body = f"""
This pull request was automatically generated and approved by the user via the Lumière Sémantique 'Socratic Dialogue' interface to address Issue #{issue_number}.
This fix was validated by The Crucible against the project's existing test suite.
"""
            head_branch = f"{working_repo.owner.login}:{branch_name}"
            time.sleep(5)

            pull_request = upstream_repo.create_pull(
                title=pr_title, body=pr_body, head=head_branch, base=default_branch
            )
            print(f"✓ Pull Request created successfully: {pull_request.html_url}")
            return {"status": "success", "pull_request_url": pull_request.html_url}

    except Exception as e:
        error_details = str(e)
        if isinstance(e, subprocess.CalledProcessError): error_details = f"Git Command Failed: {e.stderr}"
        elif isinstance(e, GithubException): error_details = f"GitHub API Error ({e.status}): {e.data}"
        print(f"Error during Git Operations or PR Creation: {error_details}")
        return {"error": f"Failed during repository operations: {error_details}"}
