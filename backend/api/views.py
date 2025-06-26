# In backend/api/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
import re
import traceback
import json
from pathlib import Path

# --- Correctly import services from lumiere_core ---
from lumiere_core.services import (
    llm_service, github, ambassador, crucible, diplomat,
    documentation, profile_service, rca_service, scaffolding, strategist,
    testing, review_service, ingestion_service, cortex_service, oracle_service,
    suggester_service
)
from lumiere_core.services.llm_service import TaskType

# A sensible default model for old workflows, though it's now mostly unused.
DEFAULT_MODEL = "ollama/qwen3:4b"

class HealthCheckView(APIView):
    """A simple view to confirm the server is running."""
    def get(self, request, *args, **kwargs):
        return Response({"status": "ok"}, status=status.HTTP_200_OK)

class ListModelsView(APIView):
    """Returns a list of all available LLM models from configured providers."""
    def get(self, request, *args, **kwargs):
        try:
            models = llm_service.list_available_models()
            return Response(models, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": "Failed to list models.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class StrategistPrioritizeView(APIView):
    def post(self, request, *args, **kwargs):
        repo_url = request.data.get('repo_url')
        if not repo_url:
            return Response(
                {"error": "'repo_url' is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            # The 'model' parameter is no longer needed. The Task Router handles it.
            result = strategist.analyze_and_prioritize(repo_url)
            if "error" in result:
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "An internal server error occurred.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class DiplomatView(APIView):
    def post(self, request, *args, **kwargs):
        issue_title = request.data.get('issue_title')
        issue_body = request.data.get('issue_body')
        if not all([issue_title, issue_body]):
            return Response(
                {"error": "'issue_title' and 'issue_body' are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            # This service should also be updated to not require model_identifier
            # Assuming diplomat_service is updated internally to use the Task Router
            result = diplomat.find_similar_solved_issues(issue_title, issue_body)
            if "error" in result:
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "An internal server error occurred.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class BriefingView(APIView):
    def post(self, request, *args, **kwargs):
        issue_url = request.data.get('issue_url')
        if not issue_url:
            return Response(
                {"error": "'issue_url' is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            # Assuming rca_service.generate_briefing is updated to use the Task Router
            result = rca_service.generate_briefing(issue_url)
            if "error" in result:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "An internal server error occurred.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class RcaView(APIView):
    def post(self, request, *args, **kwargs):
        repo_url = request.data.get('repo_url')
        bug_description = request.data.get('bug_description')
        advanced_analysis = request.data.get('advanced_analysis', False)
        confidence_threshold = request.data.get('confidence_threshold', 0.7)

        if not all([repo_url, bug_description]):
            return Response(
                {"error": "'repo_url' and 'bug_description' are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            # Assuming rca_service.perform_rca is updated for the Task Router
            result = rca_service.perform_rca(
                repo_url=repo_url,
                bug_description=bug_description,
                advanced_analysis=advanced_analysis,
                confidence_threshold=confidence_threshold
            )
            if "error" in result:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "An internal server error occurred.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ScaffoldView(APIView):
    def post(self, request, *args, **kwargs):
        repo_id = request.data.get('repo_id')
        target_files = request.data.get('target_files')
        instruction = request.data.get('instruction')
        rca_report = request.data.get('rca_report')
        refinement_history = request.data.get('refinement_history')

        if not all([repo_id, target_files, instruction, rca_report]):
            return Response(
                {"error": "'repo_id', 'target_files' (list), 'instruction', and 'rca_report' are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        if not isinstance(target_files, list):
            return Response(
                {"error": "'target_files' must be a list of strings."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            # Scaffolding service uses the Task Router internally now
            result = scaffolding.generate_scaffold(
                repo_id, target_files, instruction, rca_report, refinement_history
            )
            if "error" in result:
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "An internal server error occurred.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class CrucibleValidateView(APIView):
    def post(self, request, *args, **kwargs):
        repo_url = request.data.get('repo_url')
        target_file = request.data.get('target_file')
        modified_code = request.data.get('modified_code')
        if not all([repo_url, target_file, modified_code]):
            return Response(
                {"error": "'repo_url', 'target_file', 'modified_code' are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            result = crucible.validate_fix(repo_url, target_file, modified_code)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            error_details = traceback.format_exc()
            return Response(
                {
                    "error": "An unexpected internal server error occurred in The Crucible.",
                    "details": str(e),
                    "traceback": error_details
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class AmbassadorDispatchView(APIView):
    def post(self, request, *args, **kwargs):
        issue_url = request.data.get('issue_url')
        modified_files = request.data.get('modified_files')

        if not all([issue_url, modified_files]):
            return Response(
                {"error": "'issue_url' and 'modified_files' (dict) are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        if not isinstance(modified_files, dict):
            return Response(
                {"error": "'modified_files' must be a dictionary."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            # Ambassador uses the Task Router internally now
            result = ambassador.dispatch_pr(issue_url, modified_files)
            if "error" in result:
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(result, status=status.HTTP_201_CREATED)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "An internal server error occurred.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ProfileReviewView(APIView):
    def post(self, request, *args, **kwargs):
        username = request.data.get('username')
        if not username:
            return Response(
                {"error": "'username' is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            # Profile service uses the Task Router internally
            result = profile_service.generate_profile_review(username)
            if "error" in result:
                return Response(result, status=status.HTTP_404_NOT_FOUND)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "An internal server error occurred.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class IssueListView(APIView):
    def get(self, request, *args, **kwargs):
        repo_url = request.query_params.get('repo_url')
        if not repo_url:
            return Response(
                {"error": "'repo_url' query parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
        if not match:
            return Response(
                {"error": "Invalid 'repo_url' format."},
                status=status.HTTP_400_BAD_REQUEST
            )
        repo_full_name = f"{match.group(1)}/{match.group(2)}"
        try:
            issues = github.list_open_issues(repo_full_name)
            return Response(issues, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "An internal server error occurred.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class FileContentView(APIView):
    def post(self, request, *args, **kwargs):
        repo_id = request.data.get('repo_id')
        file_path = request.data.get('file_path')
        if not all([repo_id, file_path]):
            return Response(
                {"error": "'repo_id' and 'file_path' are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            content = cortex_service.get_file_content(repo_id, file_path)
            if content is None:
                return Response(
                    {"error": f"File '{file_path}' not found for repo '{repo_id}'."},
                    status=status.HTTP_404_NOT_FOUND
                )
            return Response({"content": content}, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "An internal server error occurred.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class TestGenerationView(APIView):
    def post(self, request, *args, **kwargs):
        repo_id = request.data.get('repo_id')
        new_code = request.data.get('new_code')
        instruction = request.data.get('instruction')
        if not all([repo_id, new_code, instruction]):
            return Response(
                {"error": "'repo_id', 'new_code', and 'instruction' are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            # Testing service uses the Task Router internally
            result = testing.generate_tests_for_code(repo_id, new_code, instruction)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "An internal server error occurred.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class DocstringGenerationView(APIView):
    def post(self, request, *args, **kwargs):
        repo_id = request.data.get('repo_id')
        new_code = request.data.get('new_code')
        instruction = request.data.get('instruction')
        if not all([repo_id, new_code, instruction]):
            return Response(
                {"error": "'repo_id', 'new_code', and 'instruction' are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            # Documentation service uses the Task Router internally
            result = documentation.generate_docstring_for_code(repo_id, new_code, instruction)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "An internal server error occurred.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class IngestRepositoryView(APIView):
    def post(self, request, *args, **kwargs):
        repo_url = request.data.get('repo_url')
        if not repo_url:
            return Response(
                {"error": "'repo_url' is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            result = ingestion_service.clone_and_embed_repository(repo_url)
            if result.get("status") == "failed":
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(result, status=status.HTTP_201_CREATED)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "An unexpected internal server error occurred during ingestion.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class GraphDataView(APIView):
    def get(self, request, *args, **kwargs):
        repo_id = request.query_params.get('repo_id')
        if not repo_id:
            return Response(
                {"error": "'repo_id' query parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        backend_dir = Path(__file__).resolve().parent.parent
        cortex_file = backend_dir / "cloned_repositories" / repo_id / f"{repo_id}_cortex.json"

        if not cortex_file.exists():
            return Response(
                {"error": f"Cortex file for repo '{repo_id}' not found."},
                status=status.HTTP_404_NOT_FOUND
            )

        try:
            with open(cortex_file, 'r', encoding='utf-8') as f:
                cortex_data = json.load(f)

            graph_data = cortex_data.get('architectural_graph')
            if not graph_data:
                new_message = ("Architectural graph is not available. The Cartographer feature currently only supports "
                             "Python projects (.py files). This repository does not appear to contain Python code suitable for graphing.")
                return Response({"message": new_message, "graph": None}, status=status.HTTP_200_OK)

            return Response({"graph": graph_data, "repo_id": repo_id}, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "Failed to read or parse graph data.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class OracleView(APIView):
    def post(self, request, *args, **kwargs):
        repo_id = request.data.get('repo_id')
        question = request.data.get('question')
        if not all([repo_id, question]):
            return Response(
                {"error": "'repo_id' and 'question' are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            # Oracle service uses the Task Router internally
            result = oracle_service.answer_question(repo_id, question)
            if "error" in result:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "An internal server error occurred in The Oracle.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class AdjudicateView(APIView):
    def post(self, request, *args, **kwargs):
        pr_url = request.data.get('pr_url')
        if not pr_url:
            return Response(
                {"error": "'pr_url' is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            # Inquire service uses the Task Router internally
            result = review_service.inquire_pr(pr_url)
            if "error" in result:
                # Safe check for ingestion error with proper string handling
                if result.get("error") and "ingested" in str(result.get("error", "")):
                    return Response(result, status=status.HTTP_412_PRECONDITION_FAILED)
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "An internal server error occurred in The Inquisitor.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class HarmonizeView(APIView):
    def post(self, request, *args, **kwargs):
        pr_url = request.data.get('pr_url')
        review_text = request.data.get('review_text')
        if not all([pr_url, review_text]):
            return Response(
                {"error": "'pr_url' and 'review_text' are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            # Harmonizer uses services that use the Task Router
            result = review_service.harmonize_pr_fix(pr_url, review_text)
            if "error" in result:
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "An internal server error occurred in The Harmonizer.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class SentinelBriefingView(APIView):
    def get(self, request, *args, **kwargs):
        repo_id = request.query_params.get('repo_id')
        if not repo_id:
            return Response(
                {"error": "'repo_id' query parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        backend_dir = Path(__file__).resolve().parent.parent
        metrics_path = backend_dir / "cloned_repositories" / repo_id / "metrics.json"

        if not metrics_path.exists():
            return Response(
                {"error": f"No metrics file found for repo '{repo_id}'. Please run ingestion/analysis first."},
                status=status.HTTP_404_NOT_FOUND
            )

        try:
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)

            if len(metrics_data) < 2:
                return Response({
                    "briefing": "Not enough historical data to generate a trend analysis. At least two data points are needed."
                })

            latest = metrics_data[-1]
            previous = metrics_data[-2]
            trends_str = "Key Trends:\n"

            for key, value in latest.items():
                if isinstance(value, (int, float)) and key in previous:
                    prev_val = previous[key]
                    if prev_val != 0:
                        change = ((value - prev_val) / prev_val) * 100
                        trends_str += f"- {key.replace('_', ' ').title()}: {value:.2f} ({change:+.1f}% change)\n"

            # Note: The prompt is truncated for brevity but should include the full Sentinel prompt
            prompt = f"You are The Sentinel, analyzing repository metrics trends...\n{trends_str}"
            briefing = llm_service.generate_text(prompt, task_type=TaskType.SIMPLE)

            return Response({"briefing": briefing, "latest_metrics": latest})
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "Failed to generate Sentinel briefing.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

# --- NEW VIEW FOR THE MISSION CONTROLLER ---
class SuggestActionsView(APIView):
    """
    Takes context from the last action and suggests next steps.
    """
    def post(self, request, *args, **kwargs):
        last_action = request.data.get("last_action")
        result_data = request.data.get("result_data", {})

        if not last_action:
            return Response(
                {"error": "'last_action' is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            suggestions = suggester_service.suggest_next_actions(last_action, result_data)
            return Response(suggestions, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "Failed to generate suggestions.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
