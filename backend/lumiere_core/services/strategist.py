# In backend/lumiere_core/services/strategist.py

import json
import re
from typing import Dict, List, Any

from . import github

from . import llm_service
from .llm_service import TaskType

def analyze_and_prioritize(repo_url: str) -> Dict[str, Any]:
    """
    The core logic for The Strategist agent.
    Fetches all open issues and uses an LLM to prioritize them.
    The 'model_identifier' is no longer passed in.
    """
    print("--- STRATEGIST AGENT ACTIVATED ---")
    print(f"Analyzing repository: {repo_url}")
    # The specific model used is now decided by the Task Router.

    # Step 1: Fetch and Enrich All Open Issues (Unchanged)
    print("\n[Step 1/3] Fetching and enriching all open issues...")
    match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
    if not match:
        return {"error": f"Could not parse repository name from URL: {repo_url}"}
    repo_full_name = f"{match.group(1)}/{match.group(2)}"
    raw_issues = github.list_open_issues(repo_full_name)
    if not raw_issues:
        return {"analysis_summary": "No open issues found for this repository.", "prioritized_issues": []}
    enriched_issues = []
    issues_for_prompt = ""
    for issue_stub in raw_issues:
        issue_details = github.scrape_github_issue(issue_stub['url'])
        if issue_details:
            enriched_issue_data = {**issue_stub, **issue_details}
            enriched_issues.append(enriched_issue_data)
            description = issue_details.get('description') or ""
            issues_for_prompt += f"### Issue #{issue_stub['number']}: {issue_stub['title']}\n{description}\n\n---\n\n"
    print(f"✓ Found and enriched {len(enriched_issues)} open issues.")

    # Step 2: Use LLM to score and justify prioritization
    print("\n[Step 2/3] Submitting issues to LLM for prioritization analysis...")
    prompt = f"""You are "The Strategist", an expert engineering manager. Your mission is to analyze a list of open GitHub issues and prioritize them.
You MUST produce a valid JSON array as your output. For each issue, create a JSON object with these exact fields:
- "issue_number": The integer issue number.
- "score": An integer from 0 to 100, where 100 is most critical.
- "justification": A concise, one-sentence explanation for your score.
SCORING CRITERIA:
- Critical (90-100): Crashes, data corruption, security vulnerabilities.
- High (70-89): Major feature bugs, performance problems.
- Medium (40-69): Minor bugs, UI/UX issues.
- Low (0-39): Feature requests, documentation, refactoring.
Analyze the following issues and provide ONLY the JSON array as your response.
--- START OF ISSUES ---
{issues_for_prompt}
--- END OF ISSUES ---
"""
    # --- THE CHANGE IS HERE ---
    llm_response_str = llm_service.generate_text(
        prompt,
        task_type=TaskType.COMPLEX_REASONING
    )

    try:
        cleaned_str = re.sub(r'<think>.*?</think>', '', llm_response_str, flags=re.DOTALL)
        json_str_match = re.search(r'\[.*\]', cleaned_str, re.DOTALL)
        if not json_str_match:
            raise json.JSONDecodeError("No JSON array found in the LLM's cleaned response.", llm_response_str, 0)
        prioritization_data = json.loads(json_str_match.group(0))
        priority_map = {item['issue_number']: item for item in prioritization_data}
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing LLM response: {e}\nLLM Response was:\n{llm_response_str}")
        return {"error": "Failed to parse prioritization data from LLM.", "llm_response": llm_response_str}

    print("✓ LLM analysis complete.")

    # Step 3: Merge data and sort (Unchanged)
    print("\n[Step 3/3] Finalizing report...")
    final_ranked_list = []
    for issue in enriched_issues:
        issue_number = issue['number']
        if issue_number in priority_map:
            issue.update(priority_map[issue_number])
            final_ranked_list.append(issue)
    final_ranked_list.sort(key=lambda x: x.get('score', 0), reverse=True)
    summary = f"Analyzed {len(final_ranked_list)} open issues."
    for i, issue in enumerate(final_ranked_list):
        issue['rank'] = i + 1
    return {
        "repository": repo_full_name,
        "analysis_summary": summary,
        "prioritized_issues": final_ranked_list
    }
