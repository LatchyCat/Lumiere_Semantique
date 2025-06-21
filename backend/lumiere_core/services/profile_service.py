# In backend/lumiere_core/services/profile_service.py
from typing import Dict, Any
from . import github
from .llm import generate_text

def _format_data_for_llm(profile_data: Dict[str, Any]) -> str:
    """Formats the aggregated GitHub data into a text block for the LLM."""

    user = profile_data['user_profile']
    text = f"""
# GitHub User Profile Analysis for: {user['login']} ({user.get('name', 'N/A')})
Bio: {user.get('bio', 'N/A')}
Followers: {user.get('followers', 0)} | Following: {user.get('following', 0)}
Public Repos: {user.get('public_repos', 0)}

---
## Owned Repositories (Sample)
"""
    if profile_data['repositories']:
        for repo in profile_data['repositories'][:5]:
            text += f"- **{repo['name']}**: {repo.get('language', 'N/A')} | ☆{repo['stargazers_count']} | {repo.get('description', 'No description')}\n"
    else:
        text += "No public repositories found.\n"

    text += "\n---"
    text += "\n## Starred Repositories (Sample)\n"
    if profile_data['starred_repositories']:
        for repo in profile_data['starred_repositories'][:5]:
             text += f"- **{repo['full_name']}**: {repo.get('description', 'No description')}\n"
    else:
        text += "No starred repositories found.\n"

    text += "\n---"
    text += "\n## Recent Issue/PR Comments & Replies by {user['login']}\n"
    if profile_data['comment_threads']:
        for thread in profile_data['comment_threads']:
            text += f"\nOn repo `{thread['repo_name']}` (Issue/PR #{thread['issue_number']}):\n"
            text += f"  - **Their Comment**: \"{thread['user_comment']['body']}\"\n"
            if thread['replies']:
                for reply in thread['replies']:
                    text += f"    - **Reply from {reply['user']}**: \"{reply['body']}\"\n"
            else:
                text += "    - No replies to this comment found.\n"
    else:
        text += "No recent comments found.\n"

    return text

def generate_profile_review(username: str) -> Dict[str, Any]:
    """
    The core logic for the Chronicler Agent.
    Fetches a user's GitHub activity and generates a narrative summary.
    """
    print(f"Initiating Chronicler Agent for user '{username}'")

    print("   -> Step 1: Fetching profile data from GitHub API...")
    user_profile = github.get_user_profile(username)
    if not user_profile:
        raise FileNotFoundError(f"User '{username}' not found on GitHub.")

    repositories = github.get_user_repos(username)
    starred = github.get_user_starred(username)
    comment_threads = github.get_user_comment_threads(username)

    raw_data = {
        "user_profile": user_profile, "repositories": repositories,
        "starred_repositories": starred, "comment_threads": comment_threads,
    }

    print("   -> Step 2: Formatting data and constructing FINAL prompt for LLM...")
    context_string = _format_data_for_llm(raw_data)

    # --- FINAL, MOST DIRECT PROMPT ---
    prompt = f"""You are an expert GitHub profile analyst. Your task is to analyze the user '{username}' based ONLY on the provided data.

**CRITICAL INSTRUCTION: Your entire analysis MUST be about the user '{username}'. Do NOT summarize the technical problems in the comments. Instead, use the comments to understand the USER'S BEHAVIOR.**

Generate a "Developer Profile Briefing" in Markdown with these exact sections:

### 1. Identity & Technical Focus
*   Based on their bio, owned repos, and starred repos, what are '{username}'s primary technical interests?
*   What are their main programming languages? (e.g., JavaScript, C++, Python)

### 2. Community Engagement Style
*   Based on their comments, what is '{username}'s role in the community? Are they reporting bugs, asking for help, or providing solutions?
*   Analyze the tone and content of THEIR comments. For example: `The user provides detailed debugging reports ("Debugging Report: itzzzme/anime-api Integration Issues") suggesting a methodical approach to problem-solving.`

### 3. Community Reception
*   Look at the replies to '{username}'s comments. Are others engaging with them? Are they receiving help and feedback?
*   Briefly summarize the nature of the replies they receive (e.g., "The user receives helpful replies from other developers, who offer suggestions and updated decryption keys.").

---
### RAW GITHUB DATA FOR {username}
{context_string}
---

Now, generate the Developer Profile Briefing about the user '{username}'.
"""

    print("   -> Step 3: Sending request to LLM for narrative generation...")
    summary = generate_text(prompt, model_name='qwen3:4b')

    final_response = { "profile_summary": summary, "raw_data": raw_data }

    return final_response
