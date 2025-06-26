# backend/lumiere_core/services/suggester_service.py

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def suggest_next_actions(last_action: str, result_data: Dict = None) -> Dict:
    """
    Generates a list of suggested next actions based on the last command's result.

    Args:
        last_action: The command that was just executed (e.g., "review", "analyze").
        result_data: The JSON data returned by the last command.

    Returns:
        A dictionary containing a list of suggestions and a recommended choice.
    """
    if result_data is None:
        result_data = {}

    suggestions = []
    recommended_choice = None

    if last_action == "review":
        suggestions.append({
            "key": "1",
            "text": "🎵 Harmonize: Attempt an automated fix based on this review.",
            "command": "harmonize"
        })
        suggestions.append({
            "key": "2",
            "text": "🔮 Oracle: Ask a follow-up question about the PR.",
            "command": "ask"
        })
        recommended_choice = "1"

    elif last_action == "analyze":
        suggestions.append({
            "key": "1",
            "text": "🎯 List Issues: View the prioritized list of issues for this repo.",
            "command": "list"
        })
        suggestions.append({
            "key": "2",
            "text": "🗺️ Graph: Visualize the repository's architecture.",
            "command": "graph"
        })
        suggestions.append({
            "key": "3",
            "text": "🔮 Oracle: Ask a high-level question about the codebase.",
            "command": "ask"
        })
        recommended_choice = "1"

    elif last_action == "dashboard":
        # Check if the briefing text suggests negative trends
        briefing_text = result_data.get("briefing", "").lower()
        if "increase" in briefing_text or "jumped" in briefing_text or "dropped" in briefing_text:
            suggestions.append({
                "key": "1",
                "text": "🔍 Re-analyze: Run a fresh analysis to get the latest graph and issues.",
                "command": "analyze"
            })
            recommended_choice = "1"

    # Always add a way to go back
    back_key = str(len(suggestions) + 1)
    suggestions.append({
        "key": back_key,
        "text": "↩️ Main Menu: Return to the main command prompt.",
        "command": "back"
    })

    # If no specific recommendation, recommend going back.
    if not recommended_choice:
        recommended_choice = back_key

    return {
        "suggestions": suggestions,
        "recommended_choice": recommended_choice
    }
