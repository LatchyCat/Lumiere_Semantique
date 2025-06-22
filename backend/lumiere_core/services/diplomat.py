# In backend/lumiere_core/services/diplomat.py

import os
import re
import requests
from typing import Dict, Any, List
from github import Github, GithubException, Issue

from . import llm_service
from .utils import clean_llm_code_output

# --- Configuration ---
GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_ACCESS_TOKEN must be set in the .env file.")

g = Github(GITHUB_TOKEN)

def _get_pr_for_issue(issue: Issue) -> Dict[str, Any] | None:
    """Finds the Pull Request that closed a given issue."""
    try:
        for event in issue.get_timeline():
            if event.event == "closed" and event.source and event.source.issue:
                pr = event.source.issue
                return {
                    "url": pr.html_url,
                    "title": pr.title,
                    "diff_url": pr.diff_url,
                }
    except GithubException as e:
        print(f"   -> API error while fetching timeline for {issue.html_url}: {e}")
    return None

def find_similar_solved_issues(issue_title: str, issue_body: str, model_identifier: str) -> Dict[str, Any]:
    """
    The main logic for The Diplomat agent.
    Searches GitHub for similar, solved issues and synthesizes the findings.
    """
    print("--- DIPLOMAT AGENT ACTIVATED ---")
    print(f"Using model: {model_identifier}")

    print("\n[Step 1/3] Generating a targeted search query from issue details...")
    query_generation_prompt = f"""
You are an expert GitHub search querycrafter. Based on the following issue title and body, generate a concise, powerful search query for finding similar issues.
Focus on extracting key library names, error messages, and critical function names.
For example, for a "TypeError" in "requests", the query might be: `requests "TypeError: timeout value must be a float"`.

ISSUE TITLE: {issue_title}
ISSUE BODY:
{issue_body}

Now, provide ONLY the search query string. Do not include any of your own commentary or XML tags.
"""
    raw_query = llm_service.generate_text(query_generation_prompt, model_identifier)
    search_query = clean_llm_code_output(raw_query).replace('"', '')
    print(f"✓ Generated Search Query: '{search_query}'")

    print("\n[Step 2/3] Searching GitHub for similar, solved issues...")
    qualified_query = f'{search_query} is:issue is:closed stars:>100 in:body'

    try:
        issues = g.search_issues(query=qualified_query, order="desc")
        print(f"✓ Found {issues.totalCount} potential matches. Analyzing the top 5...")

        evidence = []
        for issue in issues[:5]:
            print(f"   -> Analyzing: {issue.html_url}")
            closing_pr = _get_pr_for_issue(issue)
            if closing_pr:
                evidence.append({
                    "issue_title": issue.title,
                    "issue_url": issue.html_url,
                    "repo_name": issue.repository.full_name,
                    "solution_url": closing_pr['url'],
                    "diff_url": closing_pr['diff_url'],
                })

        if not evidence:
            return {
                "summary": "The Diplomat was unable to find relevant, solved issues on GitHub for this specific problem.",
                "evidence": []
            }

    except GithubException as e:
        return {"error": f"An error occurred while searching GitHub: {e.data.get('message', str(e))}"}

    print("\n[Step 3/3] Synthesizing findings into an intelligence briefing...")
    evidence_str = ""
    for item in evidence:
        evidence_str += f"- Issue in **{item['repo_name']}**: \"{item['issue_title']}\"\n"
        evidence_str += f"  - Issue Link: {item['issue_url']}\n"
        evidence_str += f"  - Solved by PR: {item['solution_url']}\n"

    synthesis_prompt = f"""
You are "The Diplomat," an AI agent for Lumière Sémantique.
You have found several solved issues on GitHub that are similar to the user's current problem.
Your mission is to write a concise intelligence briefing summarizing your findings. Do NOT tell the user how to fix their code. Instead, highlight the PATTERNS you found in the solutions.

Example summary format:
"This appears to be a known configuration issue. I found similar reports in `psf/requests` and `org/project` that were solved by changing a specific parameter. This strengthens the case for a configuration-based fix."

Here is the evidence you collected:
{evidence_str}

Now, generate the "Diplomat Intelligence Briefing" in Markdown.
"""
    summary = llm_service.generate_text(synthesis_prompt, model_identifier)

    print("--- DIPLOMAT AGENT MISSION COMPLETE ---")
    return {"summary": summary, "evidence": evidence}
