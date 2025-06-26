# backend/lumiere_core/services/review_service.py

import logging
import json
from pathlib import Path

from . import (
    github, llm_service, graph_differ, oracle_service, cartographer,
    diff_parser, scaffolding
)
from .llm_service import TaskType
from ingestion.crawler import IntelligentCrawler
from ingestion.jsonifier import Jsonifier

logger = logging.getLogger(__name__)

def _generate_graph_for_ref(crawler: IntelligentCrawler) -> dict:
    """Helper function to generate a full architectural graph for the current state of the crawler."""
    files_to_process = crawler.get_file_paths()
    if not files_to_process:
        return {}
    jsonifier = Jsonifier(
        file_paths=files_to_process,
        repo_root=crawler.repo_path,
        repo_id="dynamic_analysis"
    )
    project_cortex = jsonifier.generate_cortex()
    return project_cortex.get("architectural_graph", {})


def inquire_pr(pr_url: str) -> dict:
    """
    Main orchestration logic for "The Inquisitor" dynamic code review agent.
    The model_identifier is no longer needed.
    """
    logger.info(f"--- INQUISITOR AGENT ACTIVATED for PR: {pr_url} ---")

    # 1. On-Demand Workspace Creation & PR Detail Fetching
    try:
        pr_details = github.scrape_github_issue(pr_url)
        if not pr_details:
            return {"error": "Failed to scrape PR details."}
        repo_url = pr_details['repo_url']
        parsed_url = github._parse_github_issue_url(pr_url)
        if not parsed_url:
            return {"error": "Could not parse PR URL."}
        owner, repo_name, pr_number = parsed_url
        gh_repo = github.g.get_repo(f"{owner}/{repo_name}")
        pr_obj = gh_repo.get_pull(pr_number)
        base_ref = pr_obj.base.ref
        head_ref = pr_obj.head.ref
        logger.info(f"Analyzing PR #{pr_number}: merging '{head_ref}' into '{base_ref}'")
    except Exception as e:
        logger.error(f"Failed to get PR details from GitHub API: {e}", exc_info=True)
        return {"error": f"Failed to get PR details from GitHub API: {e}"}

    with IntelligentCrawler(repo_url=repo_url) as crawler:
        # 2. Analyze Base Branch
        logger.info(f"[1/3] Checking out base branch '{base_ref}' and generating graph...")
        if not crawler.checkout_ref(base_ref):
            return {"error": f"Could not checkout base branch '{base_ref}'."}
        base_graph_data = _generate_graph_for_ref(crawler)
        base_graph = oracle_service._build_knowledge_graph(base_graph_data)

        # 3. Analyze Head Branch
        head_sha = pr_obj.head.sha
        logger.info(f"[2/3] Checking out head commit '{head_sha}' and generating graph...")

        # --- FIX: Fetch PR commits from origin to handle forks ---
        try:
            # This fetches the specific commits associated with the PR's head.
            # Using the private method is acceptable here to avoid major refactoring.
            crawler._run_git_command(['git', 'fetch', 'origin', f'pull/{pr_number}/head'], check=True)
            logger.info(f"Successfully fetched commits for PR #{pr_number}")
        except Exception as e:
            logger.warning(f"Could not fetch PR head directly: {e}. The commit might already exist locally.")
            pass # We can still try to checkout the sha

        # Now, checkout the specific commit hash, which is more reliable than branch names.
        if not crawler.checkout_ref(head_sha):
            return {"error": f"Could not checkout head commit SHA '{head_sha}'. The branch might have been deleted or force-pushed."}

        head_graph_data = _generate_graph_for_ref(crawler)
        head_graph = oracle_service._build_knowledge_graph(head_graph_data)

        # 4. Perform Architectural Graph Differencing
        logger.info("[3/3] Comparing architectural graphs...")
        architectural_delta = graph_differ.compare_graphs(base_graph, head_graph)
        delta_str = json.dumps(architectural_delta, indent=2)
        # --- FIX: Use the reliable SHA for the diff calculation ---
        text_diff = crawler.get_diff_for_branch(ref_name=head_sha, base_ref=base_ref)
        if "Error from crawler" in text_diff:
            return {"error": f"Failed to get git diff. {text_diff}"}

        # 5. Synthesize the Inquisitive Review
        prompt_parts = [
            "You are The Inquisitor, an AI agent expert in software architecture. Your task is to review a Pull Request by analyzing how it changes the project's structure.",
            f"\n\n**Pull Request Details:**\n- Title: \"{pr_details.get('title', '')}\"\n- Description: \"{pr_details.get('description', '')}\"",
            "\n\n**ARCHITECTURAL DELTA (Summary of Structural Changes):**\nThis report shows what was added to or removed from the codebase's architecture.",
            f"```json\n{delta_str}\n```",
            f"\n\n**CODE CHANGES (Text Diff):**\n```diff\n{text_diff[:4000]}\n```",
            "\n\n---\n\n**YOUR TASK:**\nBased on the **Architectural Delta**, write a code review.",
            "- Focus on the *structural implications* of the changes.",
            "- Highlight any new dependencies, removed functions that might be used elsewhere, or significant changes in how components connect.",
            "- Use the text diff for specific line context.",
            "- Start with a summary of the most important architectural changes.",
            "- Provide clear, constructive feedback."
        ]
        synthesis_prompt = "\n".join(prompt_parts)

        # --- THE CHANGE IS HERE ---
        review_text = llm_service.generate_text(
            synthesis_prompt,
            task_type=TaskType.COMPLEX_REASONING
        )
        logger.info("--- INQUISITOR AGENT MISSION COMPLETE ---")

        return {"review": review_text, "metadata": {"delta": architectural_delta}}

def harmonize_pr_fix(pr_url: str, review_text: str) -> dict:
    """
    Orchestrates the fix-generation process based on a review from the Inquisitor.
    """
    logger.info(f"--- HARMONIZER AGENT ACTIVATED for PR: {pr_url} ---")

    # 1. Get PR details
    try:
        pr_details = github.scrape_github_issue(pr_url)
        if not pr_details:
            return {"error": "Failed to scrape PR details."}
        repo_url = pr_details['repo_url']
        parsed_url = github._parse_github_issue_url(pr_url)
        owner, repo_name, pr_number = parsed_url
        gh_repo = github.g.get_repo(f"{owner}/{repo_name}")
        pr_obj = gh_repo.get_pull(pr_number)
        base_ref = pr_obj.base.ref
        head_ref = pr_obj.head.ref
        repo_id = repo_url.replace("https://github.com/", "").replace("/", "_")
    except Exception as e:
        logger.error(f"Failed to get PR details for Harmonizer: {e}", exc_info=True)
        return {"error": "Failed to get PR details for Harmonizer.", "details": str(e)}

    # 2. Get working copy of the code and diff
    with IntelligentCrawler(repo_url=repo_url) as crawler:
        if not crawler.checkout_ref(head_ref):
            return {"error": f"Could not checkout head branch '{head_ref}' for fixing."}
        all_files_in_pr = {str(p.relative_to(crawler.repo_path)): p.read_text(encoding='utf-8', errors='ignore') for p in crawler.get_file_paths()}
        text_diff = crawler.get_diff_for_branch(ref_name=head_ref, base_ref=base_ref)
        if "Error from crawler" in text_diff:
            return {"error": "Failed to get git diff for Harmonizer.", "details": text_diff}
        target_files = diff_parser.get_changed_files_from_diff(text_diff)
        if not target_files:
            return {"error": "Harmonizer could not determine which files to fix from the PR diff."}
        original_contents_for_fix = {fp: all_files_in_pr.get(fp, "") for fp in target_files}

    # 3. Call the Scaffolder
    logger.info(f"Harmonizer is calling the Scaffolder with {len(target_files)} target files.")

    # Scaffolder no longer needs a model passed in; it will use the Task Router.
    scaffold_result = scaffolding.generate_scaffold(
        repo_id=repo_id,
        target_files=target_files,
        instruction=f"Fix the issues identified in the following code review for PR titled '{pr_details.get('title')}'.",
        rca_report=review_text,
        refinement_history=[]
    )

    if "error" in scaffold_result:
        return scaffold_result

    scaffold_result["original_contents"] = original_contents_for_fix
    logger.info("--- HARMONIZER MISSION COMPLETE ---")
    return scaffold_result
