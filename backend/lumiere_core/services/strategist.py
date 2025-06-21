# In backend/lumiere_core/services/strategist.py

import json
import re  # <--- THE MISSING IMPORT
from typing import Dict, List, Any

from . import github
from . import ambassador
from .llm import generate_text

def analyze_and_prioritize(repo_url: str, auto_dispatch_config: dict) -> Dict[str, Any]:
    """
    The core logic for The Strategist agent.
    Fetches all open issues, uses an LLM to prioritize them, and optionally
    dispatches the Ambassador agent to fix the top-priority issues.
    """
    print("--- STRATEGIST AGENT ACTIVATED ---")
    print(f"Analyzing repository: {repo_url}")

    # Step 1 & 2: Fetch and Enrich All Open Issues
    print("\n[Step 1/3] Fetching and enriching all open issues...")

    # A more robust way to get the repo_full_name from any valid repo URL
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
            # The 'labels' are part of the raw issue data from list_open_issues, let's ensure they are added.
            # Correction: get_issues() provides label objects, not simple strings. We need to extract their names.
            # For now, this part of the enrichment logic can be improved later.
            enriched_issues.append(enriched_issue_data)
            issues_for_prompt += f"### Issue #{issue_stub['number']}: {issue_stub['title']}\n{issue_details['description']}\n\n---\n\n"

    print(f"✓ Found and enriched {len(enriched_issues)} open issues.")

    # Step 3: Use LLM to score and justify prioritization
    print("\n[Step 2/3] Submitting issues to LLM for prioritization analysis...")

    prompt = f"""You are "The Strategist", an expert engineering manager for the Lumière Sémantique project. Your mission is to analyze a list of open GitHub issues and prioritize them.

You MUST produce a valid JSON array as your output. For each issue, create a JSON object with these exact fields:
- "issue_number": The integer issue number.
- "score": An integer from 0 to 100, where 100 is most critical.
- "justification": A concise, one-sentence explanation for your score.

SCORING CRITERIA:
- **Critical (90-100):** Crashes, data corruption, security vulnerabilities, broken core features.
- **High (70-89):** Major feature bugs, performance problems, incorrect calculations.
- **Medium (40-69):** Minor bugs, UI/UX issues, dependency updates.
- **Low (0-39):** Feature requests, documentation, questions, refactoring.

Analyze the following issues and provide ONLY the JSON array as your response.

--- START OF ISSUES ---
{issues_for_prompt}
--- END OF ISSUES ---
"""

    llm_response_str = generate_text(prompt, model_name='qwen3:4b')

    try:
        # Clean the response to ensure it's a valid JSON array
        json_str_match = re.search(r'\[.*\]', llm_response_str, re.DOTALL)
        if not json_str_match:
            raise json.JSONDecodeError("No JSON array found in LLM response", ll_response_str, 0)

        prioritization_data = json.loads(json_str_match.group(0))

        priority_map = {item['issue_number']: item for item in prioritization_data}

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing LLM response: {e}\nLLM Response was:\n{llm_response_str}")
        return {"error": "Failed to parse prioritization data from LLM.", "llm_response": llm_response_str}

    print("✓ LLM analysis complete.")

    # Step 4 & 5: Merge data, sort, and auto-dispatch
    print("\n[Step 3/3] Finalizing report and handling auto-dispatch...")

    final_ranked_list = []
    for issue in enriched_issues:
        issue_number = issue['number']
        if issue_number in priority_map:
            issue.update(priority_map[issue_number])
            issue['dispatch_status'] = 'manual_review_required'
            final_ranked_list.append(issue)

    final_ranked_list.sort(key=lambda x: x.get('score', 0), reverse=True)

    dispatches_made = 0
    if auto_dispatch_config.get("enabled", False):
        print(f"Auto-dispatch is ENABLED. Looking for issues with labels: {auto_dispatch_config.get('dispatch_labels', [])}")
        for issue in final_ranked_list:
            if dispatches_made >= auto_dispatch_config.get("max_dispatches", 0):
                break

            # This logic will be improved later when we properly fetch labels
            issue_labels = [label for label in auto_dispatch_config.get('dispatch_labels', []) if label.lower() in issue['title'].lower()]

            if issue_labels:
                print(f"Found matching issue #{issue['number']} for auto-dispatch. Dispatching Ambassador...")
                dispatch_result = ambassador.dispatch_pr(issue['url'])

                if 'error' not in dispatch_result:
                    issue['dispatch_status'] = 'dispatched'
                    issue['pr_url'] = dispatch_result.get('pull_request_url')
                    dispatches_made += 1
                else:
                    issue['dispatch_status'] = 'dispatch_failed'
                    issue['error_message'] = dispatch_result.get('error')

    summary = f"Analyzed {len(final_ranked_list)} open issues. "
    if dispatches_made > 0:
        summary += f"Automatically dispatched Ambassador for {dispatches_made} issue(s)."

    for i, issue in enumerate(final_ranked_list):
        issue['rank'] = i + 1

    return {
        "repository": repo_full_name,
        "analysis_summary": summary,
        "prioritized_issues": final_ranked_list
    }
