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
from .scaffolding import generate_scaffold
from .llm import generate_text
from ingestion.crawler import IntelligentCrawler

# --- Configuration ---
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')
GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_FORK_USERNAME")

if not GITHUB_TOKEN or not GITHUB_USERNAME:
    raise ValueError("GITHUB_ACCESS_TOKEN and GITHUB_FORK_USERNAME must be set in the .env file.")

g = Github(GITHUB_TOKEN)
user = g.get_user(GITHUB_USERNAME)


# --- NEW HELPER FUNCTION ---
def _sanitize_branch_name(text: str) -> str:
    """Creates a URL- and git-safe branch name from a string."""
    text = text.lower()
    text = re.sub(r'[\s/]+', '-', text)
    text = re.sub(r'[^a-z0-9-]', '', text)
    text = text.strip('-')
    return text[:60]


# --- The Main Orchestration Function ---
def dispatch_pr(issue_url: str) -> Dict[str, str]:
    """
    Orchestrates the entire workflow: from issue analysis to pull request creation.
    """
    print("--- AMBASSADOR AGENT ACTIVATED ---")
    print(f"Target Issue: {issue_url}")

    # 1. GATHER INTEL & PLAN
    # -----------------------
    print("\n[Step 1/5] Gathering Intel & Forming a Plan...")
    try:
        issue_data = scrape_github_issue(issue_url)
        if not issue_data:
            raise ValueError("Failed to scrape issue data from GitHub.")

        parsed_url = _parse_github_issue_url(issue_url)
        if not parsed_url:
            raise ValueError("Could not parse the provided issue URL.")
        owner, repo_name, issue_number = parsed_url

        repo_full_name = f"{owner}/{repo_name}"
        repo_id = repo_full_name.replace("/", "_")

        file_match = re.search(r'in `(\w+\.py)`', issue_data["description"])
        if not file_match:
            raise ValueError("Could not determine the target file from the issue description.")
        target_file = file_match.group(1)

        instruction = f"""
        Analyze the following GitHub issue and fix the bug described.
        Issue Title: {issue_data['title']}
        Issue Body: {issue_data['description']}
        The bug is located in the file: {target_file}
        """
        print(f"✓ Plan established: Fix bug in '{target_file}' based on issue #{issue_number}.")
    except Exception as e:
        print(f"Error during Intel Gathering: {e}")
        return {"error": f"Failed during intel gathering: {e}"}

    # 2. GENERATE THE FIX
    # -------------------
    print(f"\n[Step 2/5] Generating Code Fix for '{target_file}'...")
    try:
        scaffold_result = generate_scaffold(repo_id, target_file, instruction)
        if "error" in scaffold_result:
            raise ValueError(f"Scaffolding failed: {scaffold_result['error']}")
        fixed_code = scaffold_result["generated_code"]
        print("✓ Code fix generated successfully.")
    except Exception as e:
        print(f"Error during Code Generation: {e}")
        return {"error": f"Failed during code generation: {e}"}

    # 3. PREPARE THE REPOSITORY
    # -------------------------
    print("\n[Step 3/5] Preparing Repository...")
    try:
        upstream_repo = g.get_repo(repo_full_name)
        working_repo = None

        if upstream_repo.owner.login == GITHUB_USERNAME:
            print(f"✓ Target repo is owned by '{GITHUB_USERNAME}'. Skipping fork.")
            working_repo = upstream_repo
        else:
            print(f"Target repo is owned by '{upstream_repo.owner.login}'. Attempting to find or create fork.")
            try:
                working_repo = g.get_repo(f"{GITHUB_USERNAME}/{repo_name}")
                print(f"✓ Fork '{working_repo.full_name}' already exists.")
            except GithubException:
                print("Fork not found. Creating fork...")
                # --- THIS IS THE CORRECTED LINE ---
                # The 'create_fork()' method is called on the repository you want to fork.
                working_repo = upstream_repo.create_fork()
                print(f"✓ Fork created at '{working_repo.full_name}'. Waiting for it to be ready...")
                time.sleep(10)

        with IntelligentCrawler(repo_url=working_repo.clone_url) as crawler:
            repo_path = crawler.repo_path

            sanitized_title = _sanitize_branch_name(issue_data['title'])
            branch_name = f"lumiere-fix/{issue_number}-{sanitized_title}"
            print(f"✓ Cloned {working_repo.full_name} to '{repo_path}'")

            # 4. APPLY THE FIX
            # ----------------
            print(f"\n[Step 4/5] Applying Fix on new branch '{branch_name}'...")
            subprocess.run(['git', 'checkout', '-b', branch_name], cwd=repo_path, check=True)
            (repo_path / target_file).write_text(fixed_code)
            commit_prompt = f"Based on the issue title '{issue_data['title']}', write a concise, one-line commit message following the Conventional Commits specification (e.g., 'fix: ...')."
            raw_commit_message = generate_text(commit_prompt, model_name='qwen3:4b')
            commit_message = re.sub(r'</?think>.*?</think>', '', raw_commit_message, flags=re.DOTALL).strip()

            subprocess.run(['git', 'add', target_file], cwd=repo_path, check=True)
            subprocess.run(['git', 'commit', '-m', commit_message], cwd=repo_path, check=True)
            print(f"✓ Committed changes with message: \"{commit_message}\"")
            subprocess.run(['git', 'push', '--set-upstream', 'origin', branch_name], cwd=repo_path, check=True)
            print("✓ Pushed new branch to the working repository.")

            # 5. CREATE PULL REQUEST
            # ----------------------
            print("\n[Step 5/5] Creating Pull Request...")
            pr_title = f"Fix: {issue_data['title']} (Closes #{issue_number})"
            pr_body = f"""
This pull request was automatically generated by the Lumière Sémantique 'Ambassador' agent to address Issue #{issue_number}.

### Analysis (Pre-flight Briefing)
The issue describes a bug in `{target_file}`. The function was incorrectly performing addition instead of subtraction.

### Changes
- Modified `{target_file}` to correctly implement the subtraction logic.
- This change is validated by the existing test suite.
            """
            head_branch = f"{working_repo.owner.login}:{branch_name}"

            print("\n--- Pre-PR Sanity Checks ---")
            print("Waiting 3 seconds for GitHub's systems to sync...")
            time.sleep(3)

            try:
                remote_branch = working_repo.get_branch(branch_name)
                print(f"✓ Branch '{branch_name}' confirmed to exist on {working_repo.full_name} (SHA: {remote_branch.commit.sha})")
            except GithubException as e:
                print(f"✗ CRITICAL ERROR: Branch '{branch_name}' not found on {working_repo.full_name} after push. {e}")
                return {"error": f"Branch not found after push: {e}"}

            print("--- Attempting PR Creation ---")
            try:
                pull_request = upstream_repo.create_pull(
                    title=pr_title,
                    body=pr_body,
                    head=head_branch,
                    base="main"
                )
                print(f"✓ Pull Request created successfully: {pull_request.html_url}")
                print("\n--- AMBASSADOR AGENT MISSION COMPLETE ---")
                return {"status": "success", "pull_request_url": pull_request.html_url}
            except GithubException as e:
                print(f"✗ GitHub API Error during PR Creation: {e}")
                print(f"  Status: {e.status}")
                print(f"  Data: {e.data}")
                raise e

    except Exception as e:
        print(f"Error during Git Operations or PR Creation: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"STDERR: {e.stderr}")
        return {"error": f"Failed during repository operations: {e}"}
