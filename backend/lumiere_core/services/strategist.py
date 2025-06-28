# In backend/lumiere_core/services/strategist.py

import json
import re
from typing import Dict, List, Any

from . import github
from . import ollama
from . import impact_analyzer

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

    # Step 1: Fetch and Enrich All Open Issues with Blast Radius Analysis
    print("\n[Step 1/4] Fetching and enriching all open issues...")
    match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
    if not match:
        return {"error": f"Could not parse repository name from URL: {repo_url}"}
    repo_full_name = f"{match.group(1)}/{match.group(2)}"
    
    # Extract repo_id for blast radius analysis (assuming repo_id follows pattern owner_repo)
    repo_id = repo_full_name.replace('/', '_')
    
    raw_issues = github.list_open_issues(repo_full_name)
    if not raw_issues:
        return {"analysis_summary": "No open issues found for this repository.", "prioritized_issues": []}
    
    enriched_issues = []
    issues_for_prompt = ""
    
    print(f"\n[Step 2/4] Performing blast radius analysis for onboarding suitability...")
    
    for issue_stub in raw_issues:
        issue_details = github.scrape_github_issue(issue_stub['url'])
        if issue_details:
            enriched_issue_data = {**issue_stub, **issue_details}
            
            # Perform blast radius analysis for onboarding suitability
            blast_analysis = _analyze_issue_blast_radius(
                repo_id, 
                enriched_issue_data.get('title', ''), 
                enriched_issue_data.get('description', '')
            )
            
            # Add blast radius data to the issue
            enriched_issue_data.update(blast_analysis)
            
            enriched_issues.append(enriched_issue_data)
            description = issue_details.get('description') or ""
            onboarding_info = f" [Onboarding Score: {blast_analysis.get('onboarding_suitability_score', 0):.1f}]"
            issues_for_prompt += f"### Issue #{issue_stub['number']}: {issue_stub['title']}{onboarding_info}\n{description}\n\n---\n\n"
    
    print(f"✓ Found and enriched {len(enriched_issues)} open issues with blast radius analysis.")

    # Step 3: Use LLM to score and justify prioritization
    print("\n[Step 3/4] Submitting issues to LLM for prioritization analysis...")
    prompt = f"""You are "The Strategist", an expert engineering manager. Your mission is to analyze a list of open GitHub issues and prioritize them.

You MUST produce a valid JSON array as your output. For each issue, create a JSON object with these exact fields:
- "issue_number": The integer issue number.
- "score": An integer from 0 to 100, where 100 is most critical.
- "justification": A concise, one-sentence explanation for your score.

ENHANCED SCORING CRITERIA:
- Critical (90-100): Crashes, data corruption, security vulnerabilities.
- High (70-89): Major feature bugs, performance problems.
- Medium (40-69): Minor bugs, UI/UX issues.
- Low (0-39): Feature requests, documentation, refactoring.

ONBOARDING CONSIDERATIONS:
Each issue includes an "Onboarding Score" (0-100) in brackets. Higher scores indicate better suitability for newcomers:
- Score 80-100: Excellent for newcomers (small blast radius, isolated changes)
- Score 60-79: Good for newcomers (moderate complexity)
- Score 40-59: Moderate difficulty (requires some experience)
- Score 0-39: Advanced (high complexity, wide impact)

When scoring issues that appear to be minor bugs or feature requests (typically low priority), 
consider boosting their score slightly if they have high onboarding suitability (score > 70) 
as these make excellent "good first issues" for new contributors.

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

    # Step 4: Merge data and sort with enhanced metrics
    print("\n[Step 4/4] Finalizing report...")
    final_ranked_list = []
    for issue in enriched_issues:
        issue_number = issue['number']
        if issue_number in priority_map:
            issue.update(priority_map[issue_number])
            final_ranked_list.append(issue)
    
    final_ranked_list.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    # Enhanced summary with onboarding metrics
    onboarding_suitable = [issue for issue in final_ranked_list 
                          if issue.get('onboarding_suitability_score', 0) > 70]
    
    summary = f"Analyzed {len(final_ranked_list)} open issues. Found {len(onboarding_suitable)} issues suitable for newcomers."
    
    for i, issue in enumerate(final_ranked_list):
        issue['rank'] = i + 1
        
    return {
        "repository": repo_full_name,
        "analysis_summary": summary,
        "prioritized_issues": final_ranked_list,
        "onboarding_suitable_count": len(onboarding_suitable),
        "total_issues_analyzed": len(final_ranked_list)
    }


def _analyze_issue_blast_radius(repo_id: str, title: str, description: str) -> Dict[str, Any]:
    """
    Analyze the blast radius for a GitHub issue by using RAG search to find relevant files
    and then calculating the impact scope.
    
    Args:
        repo_id: Repository identifier
        title: Issue title
        description: Issue description
        
    Returns:
        Dictionary with blast radius analysis results
    """
    try:
        # Combine title and description for RAG search
        issue_text = f"{title}\n{description}"
        
        # Use RAG search to find top 3-5 relevant code chunks
        try:
            search_results = ollama.search_index(
                query_text=issue_text,
                model_name="snowflake-arctic-embed",  # Use default embedding model
                repo_id=repo_id,
                k=5  # Get top 5 relevant chunks
            )
            
            # Extract file paths from search results
            seed_files = []
            for result in search_results:
                # Extract file path from chunk_id (typically format: repo_id_filepath_chunk_index)
                chunk_id = result.get('chunk_id', '')
                if chunk_id:
                    # Remove repo_id prefix and chunk index suffix to get file path
                    parts = chunk_id.replace(f"{repo_id}_", "", 1).rsplit('_', 1)
                    if parts:
                        file_path = parts[0]
                        if file_path not in seed_files:
                            seed_files.append(file_path)
            
        except Exception as e:
            print(f"  ⚠️  RAG search failed for issue analysis: {e}")
            # Fallback: use empty seed files which will result in minimal blast radius
            seed_files = []
        
        # If we found relevant files, analyze blast radius
        if seed_files:
            blast_analysis = impact_analyzer.analyze_blast_radius(
                repo_id=repo_id,
                seed_files=seed_files[:3],  # Use top 3 files
                max_depth=2
            )
            
            blast_radius = blast_analysis.get('blast_radius', len(seed_files))
            onboarding_score = impact_analyzer.calculate_onboarding_suitability(blast_radius)
            
            return {
                'blast_radius': blast_radius,
                'onboarding_suitability_score': onboarding_score,
                'affected_files': blast_analysis.get('affected_nodes', []),
                'seed_files_found': seed_files,
                'analysis_method': 'rag_search'
            }
        else:
            # No relevant files found - assume minimal impact
            return {
                'blast_radius': 1,
                'onboarding_suitability_score': 95.0,  # High suitability for unknown scope
                'affected_files': [],
                'seed_files_found': [],
                'analysis_method': 'fallback_minimal'
            }
            
    except Exception as e:
        print(f"  ⚠️  Blast radius analysis failed: {e}")
        # Safe fallback
        return {
            'blast_radius': 10,  # Moderate assumption
            'onboarding_suitability_score': 50.0,  # Neutral score
            'affected_files': [],
            'seed_files_found': [],
            'analysis_method': 'error_fallback',
            'error': str(e)
        }
