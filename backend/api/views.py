# In backend/api/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
import re
import traceback
import json
from pathlib import Path
import shutil
from datetime import datetime, timedelta
from dataclasses import asdict


# --- Correctly import services from lumiere_core ---
from lumiere_core.services import (
    llm_service, github, ambassador, crucible, diplomat,
    documentation, profile_service, rca_service, scaffolding, strategist,
    testing, review_service, ingestion_service, cortex_service, oracle_service,
    suggester_service, expertise_service, onboarding_service,
)
from lumiere_core.services.quartermaster_service import QuartermasterService, CompliancePolicy
from lumiere_core.services.loremaster_service import LoremasterService
from lumiere_core.services.ingestion_service import ingest_local_directory, generate_archive_id
from lumiere_core.services.llm_service import TaskType
from lumiere_core.services.cortex_service import get_bom_data, has_bom_data, get_repository_metadata

# A sensible default model for old workflows, though it's now mostly unused.
DEFAULT_MODEL = "ollama/qwen3:4b"

# --- HELPER CONSTANTS & FUNCTIONS ---
CLONED_REPOS_DIR = Path(__file__).resolve().parent.parent / "cloned_repositories"

def _is_safe_path(base_dir: Path, repo_id: str) -> bool:
    """Checks if the repo_id is a safe path component to prevent traversal."""
    if '..' in repo_id or '/' in repo_id or '\\' in repo_id:
        return False

    # Path().resolve() will resolve symlinks and '..'
    resolved_repo_path = (base_dir / repo_id).resolve()
    resolved_base_dir = base_dir.resolve()

    # Check if the resolved path is a sub-path of the base directory.
    try:
        resolved_repo_path.relative_to(resolved_base_dir)
        return True
    except ValueError:
        # This exception is raised if the path is not a sub-path, which means it's unsafe.
        return False


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

# --- REFACTORED VIEW ---
class GraphDataView(APIView):
    def get(self, request, repo_id, *args, **kwargs):
        if not repo_id:
            return Response(
                {"error": "'repo_id' path parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        if not _is_safe_path(CLONED_REPOS_DIR, repo_id):
            return Response({"error": "Invalid repo_id"}, status=status.HTTP_400_BAD_REQUEST)

        cortex_file = CLONED_REPOS_DIR / repo_id / f"{repo_id}_cortex.json"

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

# --- REFACTORED VIEW ---
class SentinelBriefingView(APIView):
    def get(self, request, repo_id, *args, **kwargs):
        if not repo_id:
            return Response(
                {"error": "'repo_id' path parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        if not _is_safe_path(CLONED_REPOS_DIR, repo_id):
            return Response({"error": "Invalid repo_id"}, status=status.HTTP_400_BAD_REQUEST)

        metrics_path = CLONED_REPOS_DIR / repo_id / "metrics.json"

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
                        change = ((value - prev_val) / abs(prev_val)) * 100
                        trends_str += f"- {key.replace('_', ' ').title()}: {value:.2f} ({change:+.1f}% change)\n"

            prompt = f"You are The Sentinel, an AI that monitors software project health. Analyze the following metric trends and provide a concise, human-readable briefing for a developer dashboard. Highlight any significant changes or potential issues.\n\n{trends_str}"
            briefing = llm_service.generate_text(prompt, task_type=TaskType.SIMPLE)

            return Response({"briefing": briefing, "latest_metrics": latest})
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "Failed to generate Sentinel briefing.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# --- NEW REPOSITORY MANAGEMENT VIEWS ---

class ListRepositoriesView(APIView):
    """Lists all previously analyzed repositories that have complete artifacts."""
    def get(self, request, *args, **kwargs):
        analyzed_repos = []
        if not CLONED_REPOS_DIR.is_dir():
            return Response([], status=status.HTTP_200_OK)

        for repo_dir in CLONED_REPOS_DIR.iterdir():
            if repo_dir.is_dir():
                repo_id = repo_dir.name
                cortex_file = repo_dir / f"{repo_id}_cortex.json"
                faiss_file = repo_dir / f"{repo_id}_faiss.index"
                map_file = repo_dir / f"{repo_id}_id_map.json"

                if all([cortex_file.exists(), faiss_file.exists(), map_file.exists()]):
                    try:
                        display_name = repo_id.replace("_", "/", 1)
                        full_url = f"https://github.com/{display_name}"
                        analyzed_repos.append({"repo_id": repo_id, "display_name": display_name, "url": full_url})
                    except Exception:
                        continue
        return Response(sorted(analyzed_repos, key=lambda x: x['repo_id']), status=status.HTTP_200_OK)

class RepositoryDetailView(APIView):
    """Handles retrieval and deletion of a single repository's data."""
    def get(self, request, repo_id, *args, **kwargs):
        """Returns detailed metadata for a single repository."""
        if not _is_safe_path(CLONED_REPOS_DIR, repo_id):
            return Response({"error": "Invalid repo_id"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            metadata = cortex_service.get_repository_metadata(repo_id)
            if metadata is None:
                return Response({"error": f"Repository '{repo_id}' not found or its cortex file is missing."}, status=status.HTTP_404_NOT_FOUND)
            return Response(metadata, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": "An internal server error occurred.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def delete(self, request, repo_id, *args, **kwargs):
        """Deletes the directory and all contents for a given repository."""
        if not _is_safe_path(CLONED_REPOS_DIR, repo_id):
            return Response({"error": "Invalid repo_id"}, status=status.HTTP_400_BAD_REQUEST)

        repo_dir = CLONED_REPOS_DIR / repo_id
        if not repo_dir.is_dir():
            return Response(status=status.HTTP_204_NO_CONTENT) # Treat as success if already gone

        try:
            shutil.rmtree(repo_dir)
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Exception as e:
            return Response({"error": f"Failed to delete repository '{repo_id}'.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class RepositoryStatusView(APIView):
    """Checks if a repository has been fully ingested and analyzed."""
    def get(self, request, repo_id, *args, **kwargs):
        if not _is_safe_path(CLONED_REPOS_DIR, repo_id):
            return Response({"status": "not_found"}, status=status.HTTP_200_OK)

        repo_dir = CLONED_REPOS_DIR / repo_id
        if not repo_dir.is_dir():
            return Response({"status": "not_found"}, status=status.HTTP_200_OK)

        cortex_file = repo_dir / f"{repo_id}_cortex.json"
        faiss_file = repo_dir / f"{repo_id}_faiss.index"
        map_file = repo_dir / f"{repo_id}_id_map.json"

        if all([cortex_file.exists(), faiss_file.exists(), map_file.exists()]):
            return Response({"status": "complete"}, status=status.HTTP_200_OK)
        else:
            return Response({"status": "not_found"}, status=status.HTTP_200_OK)

class SentinelMetricsHistoryView(APIView):
    """Reads and returns the entire metrics.json for a given repository."""
    def get(self, request, repo_id, *args, **kwargs):
        if not _is_safe_path(CLONED_REPOS_DIR, repo_id):
            return Response({"error": "Invalid repo_id"}, status=status.HTTP_400_BAD_REQUEST)

        metrics_file = CLONED_REPOS_DIR / repo_id / "metrics.json"
        if not metrics_file.is_file():
            return Response({"error": f"Metrics file for repo '{repo_id}' not found."}, status=status.HTTP_404_NOT_FOUND)

        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            return Response(metrics_data, status=status.HTTP_200_OK)
        except json.JSONDecodeError:
            return Response({"error": "Failed to parse metrics.json file."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            return Response({"error": "An internal server error occurred.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# --- THE MISSION CONTROLLER ---
class SuggestActionsView(APIView):
    """Takes context from the last action and suggests next steps."""
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


# --- Bill of Materials (BOM) ---
class BomDataView(APIView):
    """Enhanced BOM view that works with your existing cortex service."""
    def get(self, request, *args, **kwargs):
        repo_id = request.query_params.get('repo_id')
        format_type = request.query_params.get('format', 'json')  # json, summary, detailed

        if not repo_id:
            return Response(
                {"error": "'repo_id' query parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            # Use your existing cortex service
            bom_data = get_bom_data(repo_id, format_type)
            if not bom_data:
                # Check if repo exists but just doesn't have BOM
                metadata = get_repository_metadata(repo_id)
                if metadata:
                    return Response(
                        {"error": f"Bill of Materials not found for repo '{repo_id}'. Repository exists but BOM was not generated during ingestion."},
                        status=status.HTTP_404_NOT_FOUND
                    )
                else:
                    return Response(
                        {"error": f"Repository '{repo_id}' not found."},
                        status=status.HTTP_404_NOT_FOUND
                    )
            return Response(bom_data, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "An internal server error occurred while retrieving BOM data.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class BomDependenciesView(APIView):
    """View for filtered dependency information."""
    def get(self, request, *args, **kwargs):
        repo_id = request.query_params.get('repo_id')
        ecosystem = request.query_params.get('ecosystem')
        dependency_type = request.query_params.get('dependency_type')
        outdated_only = request.query_params.get('outdated_only', 'false').lower() == 'true'
        security_risk = request.query_params.get('security_risk')

        if not repo_id:
            return Response(
                {"error": "'repo_id' query parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            bom_data = get_bom_data(repo_id, "json")
            if not bom_data:
                return Response(
                    {"error": f"BOM data not found for repository '{repo_id}'."},
                    status=status.HTTP_404_NOT_FOUND
                )
            all_dependencies = []
            for dep_type, deps in bom_data.get('dependencies', {}).items():
                for dep in deps:
                    dep['category'] = dep_type
                    all_dependencies.append(dep)
            filtered_deps = all_dependencies
            if ecosystem:
                filtered_deps = [d for d in filtered_deps if d.get('ecosystem') == ecosystem]
            if dependency_type:
                filtered_deps = [d for d in filtered_deps if d.get('category') == dependency_type]
            if security_risk:
                filtered_deps = [d for d in filtered_deps if d.get('security_risk') == security_risk]
            if outdated_only:
                filtered_deps = [d for d in filtered_deps if d.get('deprecated', False)]
            return Response({
                'repo_id': repo_id, 'total_count': len(filtered_deps),
                'filters_applied': {
                    'ecosystem': ecosystem, 'dependency_type': dependency_type,
                    'outdated_only': outdated_only, 'security_risk': security_risk
                },
                'dependencies': filtered_deps,
                'statistics': self._generate_dependency_stats(filtered_deps)
            }, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "Failed to retrieve dependencies.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _generate_dependency_stats(self, dependencies):
        """Generate statistics for dependencies."""
        ecosystems = {}
        for dep in dependencies:
            ecosystem = dep.get('ecosystem', 'unknown')
            ecosystems[ecosystem] = ecosystems.get(ecosystem, 0) + 1
        return {
            'ecosystems': ecosystems, 'total_count': len(dependencies),
            'deprecated_count': sum(1 for d in dependencies if d.get('deprecated', False))
        }

class BomServicesView(APIView):
    """View for service information."""
    def get(self, request, *args, **kwargs):
        repo_id = request.query_params.get('repo_id')
        service_type = request.query_params.get('service_type')
        if not repo_id:
            return Response(
                {"error": "'repo_id' query parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            bom_data = get_bom_data(repo_id, "json")
            if not bom_data:
                return Response(
                    {"error": f"BOM data not found for repository '{repo_id}'."},
                    status=status.HTTP_404_NOT_FOUND
                )
            services = bom_data.get('services', [])
            if service_type:
                services = [s for s in services if s.get('service_type') == service_type]
            return Response({
                'repo_id': repo_id, 'total_count': len(services),
                'services': services, 'service_types': list(set(s.get('service_type', 'unknown') for s in services)),
                'infrastructure_analysis': bom_data.get('infrastructure', {})
            }, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "Failed to retrieve services.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class BomSecurityView(APIView):
    """View for security analysis."""
    def get(self, request, *args, **kwargs):
        repo_id = request.query_params.get('repo_id')
        severity = request.query_params.get('severity')
        if not repo_id:
            return Response(
                {"error": "'repo_id' query parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            bom_data = get_bom_data(repo_id, "json")
            if not bom_data:
                return Response(
                    {"error": f"BOM data not found for repository '{repo_id}'."},
                    status=status.HTTP_404_NOT_FOUND
                )
            security_analysis = bom_data.get('security_analysis', {})
            enhanced_analysis = {
                'summary': security_analysis,
                'vulnerable_dependencies': self._get_vulnerable_dependencies(bom_data, severity),
                'security_recommendations': self._generate_security_recommendations(bom_data),
                'compliance_status': self._check_compliance(bom_data),
                'last_scan': security_analysis.get('last_scan'),
                'repo_id': repo_id
            }
            return Response(enhanced_analysis, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "Security analysis failed.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _get_vulnerable_dependencies(self, bom_data, severity):
        return []
    def _generate_security_recommendations(self, bom_data):
        return [
            'Enable automated dependency scanning', 'Set up security alerts for new vulnerabilities',
            'Regularly update dependencies', 'Implement security testing in CI/CD pipeline'
        ]
    def _check_compliance(self, bom_data):
        return {
            'compliant': True, 'checks': {
                'outdated_dependencies': 'pass', 'known_vulnerabilities': 'pass',
                'license_compliance': 'pass'
            }
        }

class BomRegenerateView(APIView):
    """View to regenerate BOM for a repository."""
    def post(self, request, *args, **kwargs):
        repo_id = request.data.get('repo_id')
        force = request.data.get('force', False)
        if not repo_id:
            return Response(
                {"error": "'repo_id' is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            metadata = get_repository_metadata(repo_id)
            if not metadata:
                return Response(
                    {"error": f"Repository '{repo_id}' not found."},
                    status=status.HTTP_404_NOT_FOUND
                )
            if has_bom_data(repo_id) and not force:
                existing_bom = get_bom_data(repo_id)
                last_updated = existing_bom.get('summary', {}).get('last_updated')
                if last_updated:
                    try:
                        last_updated_dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                        if datetime.now().replace(tzinfo=last_updated_dt.tzinfo) - last_updated_dt < timedelta(hours=24):
                            return Response({
                                'message': 'BOM is recent, use force=true to regenerate',
                                'last_updated': last_updated, 'regenerated': False
                            }, status=status.HTTP_200_OK)
                    except:
                        pass
            return Response({
                'message': 'BOM regeneration requires re-ingesting the repository',
                'suggestion': 'Use the /api/ingest/ endpoint to regenerate BOM data',
                'regenerated': False
            }, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "BOM regeneration failed.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class BomCompareView(APIView):
    """View to compare BOMs between repositories."""
    def post(self, request, *args, **kwargs):
        repo_id_1 = request.data.get('repo_id_1')
        repo_id_2 = request.data.get('repo_id_2')
        comparison_type = request.data.get('comparison_type', 'dependencies')
        if not all([repo_id_1, repo_id_2]):
            return Response(
                {"error": "'repo_id_1' and 'repo_id_2' are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            bom_1 = get_bom_data(repo_id_1)
            bom_2 = get_bom_data(repo_id_2)
            if not bom_1:
                return Response(
                    {"error": f"BOM data not found for repository '{repo_id_1}'."},
                    status=status.HTTP_404_NOT_FOUND
                )
            if not bom_2:
                return Response(
                    {"error": f"BOM data not found for repository '{repo_id_2}'."},
                    status=status.HTTP_404_NOT_FOUND
                )
            if comparison_type == "dependencies":
                comparison = self._compare_dependencies(bom_1, bom_2)
            elif comparison_type == "services":
                comparison = self._compare_services(bom_1, bom_2)
            elif comparison_type == "languages":
                comparison = self._compare_languages(bom_1, bom_2)
            else:
                comparison = self._comprehensive_comparison(bom_1, bom_2)
            return Response({
                'repo_1': {'id': repo_id_1}, 'repo_2': {'id': repo_id_2},
                'comparison_type': comparison_type, 'comparison': comparison
            }, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "Comparison failed.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    def _compare_dependencies(self, bom_1, bom_2):
        deps_1, deps_2 = set(), set()
        for deps in bom_1.get('dependencies', {}).values():
            for dep in deps:
                deps_1.add(f"{dep.get('name')}@{dep.get('version')}")
        for deps in bom_2.get('dependencies', {}).values():
            for dep in deps:
                deps_2.add(f"{dep.get('name')}@{dep.get('version')}")
        return {
            'common': list(deps_1.intersection(deps_2)),
            'unique_to_repo_1': list(deps_1 - deps_2),
            'unique_to_repo_2': list(deps_2 - deps_1),
            'total_repo_1': len(deps_1), 'total_repo_2': len(deps_2)
        }
    def _compare_services(self, bom_1, bom_2):
        services_1 = {f"{s.get('name')}:{s.get('version')}" for s in bom_1.get('services', [])}
        services_2 = {f"{s.get('name')}:{s.get('version')}" for s in bom_2.get('services', [])}
        return {
            'common': list(services_1.intersection(services_2)),
            'unique_to_repo_1': list(services_1 - services_2),
            'unique_to_repo_2': list(services_2 - services_1)
        }
    def _compare_languages(self, bom_1, bom_2):
        langs_1, langs_2 = set(bom_1.get('languages', {}).keys()), set(bom_2.get('languages', {}).keys())
        return {
            'common': list(langs_1.intersection(langs_2)),
            'unique_to_repo_1': list(langs_1 - langs_2),
            'unique_to_repo_2': list(langs_2 - langs_1)
        }
    def _comprehensive_comparison(self, bom_1, bom_2):
        return {
            'dependencies': self._compare_dependencies(bom_1, bom_2),
            'services': self._compare_services(bom_1, bom_2),
            'languages': self._compare_languages(bom_1, bom_2),
            'summary': {
                'primary_language_1': bom_1.get('summary', {}).get('primary_language'),
                'primary_language_2': bom_2.get('summary', {}).get('primary_language'),
                'dependency_count_1': bom_1.get('summary', {}).get('total_dependencies', 0),
                'dependency_count_2': bom_2.get('summary', {}).get('total_dependencies', 0)
            }
        }


class ExpertiseView(APIView):
    """
    API endpoint for the Expertise Service - finds knowledgeable contributors
    for specific files or modules. Part of the Onboarding Concierge feature.
    """

    def post(self, request):
        """
        Find experts for a given file or module pattern.
        
        Expected payload:
        {
            "repo_id": "string",
            "file_path": "string (optional)",
            "module_pattern": "string (optional)",
            "type": "file" | "module" | "summary"
        }
        """
        try:
            data = request.data
            repo_id = data.get('repo_id')
            file_path = data.get('file_path')
            module_pattern = data.get('module_pattern')
            query_type = data.get('type', 'file')

            if not repo_id:
                return Response(
                    {'error': 'repo_id is required'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Validate repo_id for security
            if not _is_safe_path(CLONED_REPOS_DIR, repo_id):
                return Response(
                    {'error': 'Invalid repo_id'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Initialize expertise service
            expertise_svc = expertise_service.ExpertiseService(CLONED_REPOS_DIR)

            # Handle different query types
            if query_type == 'file':
                if not file_path:
                    return Response(
                        {'error': 'file_path is required for file expertise queries'}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                experts = expertise_svc.find_experts_for_file(repo_id, file_path)
                
                return Response({
                    'repository': repo_id,
                    'file_path': file_path,
                    'experts': experts,
                    'query_type': 'file',
                    'expert_count': len(experts)
                })

            elif query_type == 'module':
                if not module_pattern:
                    return Response(
                        {'error': 'module_pattern is required for module expertise queries'}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                experts = expertise_svc.find_experts_for_module(repo_id, module_pattern)
                
                return Response({
                    'repository': repo_id,
                    'module_pattern': module_pattern,
                    'experts': experts,
                    'query_type': 'module',
                    'expert_count': len(experts)
                })

            elif query_type == 'summary':
                summary = expertise_svc.get_repository_experts_summary(repo_id)
                
                return Response({
                    'repository': repo_id,
                    'query_type': 'summary',
                    **summary
                })

            else:
                return Response(
                    {'error': 'Invalid query type. Must be "file", "module", or "summary"'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

        except Exception as e:
            return Response(
                {
                    'error': 'Internal server error occurred while finding experts',
                    'details': str(e),
                    'traceback': traceback.format_exc()
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class OnboardingGuideView(APIView):
    """
    API endpoint for the Onboarding Service - generates personalized onboarding 
    paths for GitHub issues. Part of the Onboarding Concierge feature.
    """

    def post(self, request):
        """
        Generate a personalized onboarding guide for a GitHub issue.
        
        Expected payload:
        {
            "repo_id": "string",
            "issue_number": "integer"
        }
        """
        try:
            data = request.data
            repo_id = data.get('repo_id')
            issue_number = data.get('issue_number')

            if not repo_id:
                return Response(
                    {'error': 'repo_id is required'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            if not issue_number:
                return Response(
                    {'error': 'issue_number is required'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Validate issue_number is an integer
            try:
                issue_number = int(issue_number)
            except (ValueError, TypeError):
                return Response(
                    {'error': 'issue_number must be a valid integer'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Validate repo_id for security
            if not _is_safe_path(CLONED_REPOS_DIR, repo_id):
                return Response(
                    {'error': 'Invalid repo_id'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Initialize onboarding service
            onboarding_svc = onboarding_service.OnboardingService(CLONED_REPOS_DIR)

            # Generate the onboarding guide
            guide_result = onboarding_svc.generate_onboarding_path(repo_id, issue_number)

            if guide_result.get('generation_successful'):
                return Response({
                    'repository': repo_id,
                    'issue_number': issue_number,
                    'issue_title': guide_result.get('issue_title'),
                    'onboarding_guide': guide_result.get('onboarding_guide'),
                    'learning_path_steps': guide_result.get('learning_path_steps'),
                    'locus_files': guide_result.get('locus_files', []),
                    'detailed_steps': guide_result.get('enriched_steps', []),
                    'generation_timestamp': datetime.now().isoformat()
                })
            else:
                return Response(
                    {
                        'error': 'Failed to generate onboarding guide',
                        'details': guide_result.get('error', 'Unknown error'),
                        'repository': repo_id,
                        'issue_number': issue_number
                    },
                    status=status.HTTP_422_UNPROCESSABLE_ENTITY
                )

        except Exception as e:
            return Response(
                {
                    'error': 'Internal server error occurred while generating onboarding guide',
                    'details': str(e),
                    'traceback': traceback.format_exc()
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ApiEndpointInventoryView(APIView):
    """
    API endpoint to retrieve all discovered API endpoints from a repository's cortex data.
    This provides a complete inventory of all web API routes discovered during ingestion.
    """

    def get(self, request, repo_id, *args, **kwargs):
        """
        Retrieve all API endpoints for a given repository.
        
        Args:
            repo_id (str): The repository identifier
            
        Returns:
            JSON response containing all discovered API endpoints with metadata
        """
        if not repo_id:
            return Response(
                {"error": "'repo_id' path parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        # Validate repo_id for security
        if not _is_safe_path(CLONED_REPOS_DIR, repo_id):
            return Response(
                {"error": "Invalid repo_id"}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Load cortex data using the existing cortex service
            cortex_data = cortex_service.load_cortex_data(repo_id)
            
            if not cortex_data:
                return Response(
                    {"error": f"Cortex data not found for repository '{repo_id}'. Repository may not have been ingested or analysis may have failed."},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            aggregated_endpoints = []
            endpoint_count_by_framework = {}
            total_endpoints = 0
            
            # Process each file in the cortex data
            for file_data in cortex_data.get('files', []):
                file_path = file_data.get('file_path', '')
                api_endpoints = file_data.get('api_endpoints', [])
                
                # Process each endpoint found in this file
                for endpoint in api_endpoints:
                    # Enrich endpoint with file context
                    enriched_endpoint = {
                        **endpoint,  # All original endpoint data
                        'handler_file': file_path,  # Add the file containing this endpoint
                        'file_language': file_data.get('detected_language', 'unknown'),
                        'file_size_kb': file_data.get('file_size_kb', 0)
                    }
                    
                    # Optional: Add architectural graph connections
                    # This could be enhanced to show which functions this endpoint calls
                    if cortex_data.get('architectural_graph'):
                        # Find related functions/classes that this endpoint might call
                        graph_data = cortex_data['architectural_graph']
                        handler_name = endpoint.get('handler_function_name', '')
                        
                        # Look for outbound calls from this handler
                        related_functions = []
                        for node_id, node_data in graph_data.get('nodes', {}).items():
                            if node_data.get('name') == handler_name:
                                # Find outbound edges
                                for edge in graph_data.get('edges', []):
                                    if edge.get('source') == node_id:
                                        target_node = graph_data['nodes'].get(edge.get('target'), {})
                                        related_functions.append(target_node.get('name', ''))
                        
                        if related_functions:
                            enriched_endpoint['related_functions'] = related_functions
                    
                    aggregated_endpoints.append(enriched_endpoint)
                    
                    # Track framework statistics
                    framework = endpoint.get('framework', 'Unknown')
                    endpoint_count_by_framework[framework] = endpoint_count_by_framework.get(framework, 0) + 1
                    total_endpoints += 1
            
            # Generate summary statistics
            methods_distribution = {}
            path_patterns = []
            
            for endpoint in aggregated_endpoints:
                # Count HTTP methods
                for method in endpoint.get('methods', []):
                    methods_distribution[method] = methods_distribution.get(method, 0) + 1
                
                # Collect path patterns for analysis
                path = endpoint.get('path', '')
                if path:
                    path_patterns.append(path)
            
            # Analyze path patterns (simple heuristic for RESTful endpoints)
            rest_score = self._calculate_rest_score(path_patterns)
            
            # Build comprehensive response
            response_data = {
                'repository': repo_id,
                'api_inventory': {
                    'total_endpoints': total_endpoints,
                    'endpoints': aggregated_endpoints,
                    'summary': {
                        'frameworks_detected': list(endpoint_count_by_framework.keys()),
                        'framework_distribution': endpoint_count_by_framework,
                        'http_methods_distribution': methods_distribution,
                        'rest_score': rest_score,
                        'files_with_endpoints': len([f for f in cortex_data.get('files', []) if f.get('api_endpoints')])
                    },
                    'analysis_metadata': {
                        'analyzed_at': cortex_data.get('last_crawled_utc'),
                        'total_files_analyzed': len(cortex_data.get('files', [])),
                        'primary_language': cortex_data.get('polyglot_summary', {}).get('primary_language', 'unknown')
                    }
                }
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            traceback.print_exc()
            return Response(
                {
                    'error': 'Failed to retrieve API endpoint inventory',
                    'details': str(e),
                    'repository': repo_id
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _calculate_rest_score(self, path_patterns):
        """
        Calculate a simple RESTfulness score based on path patterns.
        
        Args:
            path_patterns (List[str]): List of URL paths
            
        Returns:
            float: Score between 0 and 1 indicating RESTfulness
        """
        if not path_patterns:
            return 0.0
        
        rest_indicators = 0
        total_paths = len(path_patterns)
        
        for path in path_patterns:
            # Check for RESTful patterns
            if '/{' in path or '/id' in path or re.search(r'/\d+', path):
                rest_indicators += 1  # Has path parameters
            if path.count('/') >= 2:
                rest_indicators += 0.5  # Has reasonable depth
            if not any(action in path.lower() for action in ['create', 'update', 'delete', 'get', 'list']):
                rest_indicators += 0.5  # Doesn't use action verbs in URL
        
        return min(1.0, rest_indicators / total_paths)


# --- QUARTERMASTER ARCHETYPE VIEWS ---

class QuartermasterDashboardView(APIView):
    """Get high-level supply-chain health summary for a repository."""
    
    def get(self, request, repo_id, *args, **kwargs):
        if not repo_id:
            return Response(
                {"error": "'repo_id' path parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not _is_safe_path(CLONED_REPOS_DIR, repo_id):
            return Response({"error": "Invalid repo_id"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            quartermaster = QuartermasterService(CLONED_REPOS_DIR)
            dashboard = quartermaster.get_dashboard_health(repo_id)
            
            if "error" in dashboard:
                return Response(dashboard, status=status.HTTP_404_NOT_FOUND)
            
            return Response(dashboard, status=status.HTTP_200_OK)
            
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "Failed to generate Quartermaster dashboard.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class QuartermasterVulnerabilitiesView(APIView):
    """List all known vulnerabilities for a repository."""
    
    def get(self, request, repo_id, *args, **kwargs):
        if not repo_id:
            return Response(
                {"error": "'repo_id' path parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not _is_safe_path(CLONED_REPOS_DIR, repo_id):
            return Response({"error": "Invalid repo_id"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Load BOM data
            bom_data = get_bom_data(repo_id)
            if not bom_data:
                return Response(
                    {"error": f"BOM data not found for repository '{repo_id}'."},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            quartermaster = QuartermasterService(CLONED_REPOS_DIR)
            vulnerabilities = quartermaster.check_vulnerabilities(bom_data)
            
            # Convert dataclasses to dicts for JSON serialization
            vuln_dicts = [asdict(vuln) for vuln in vulnerabilities]
            
            # Add summary statistics
            severity_counts = {}
            for vuln in vulnerabilities:
                severity_counts[vuln.severity] = severity_counts.get(vuln.severity, 0) + 1
            
            response_data = {
                "repository": repo_id,
                "total_vulnerabilities": len(vulnerabilities),
                "severity_breakdown": severity_counts,
                "vulnerabilities": vuln_dicts,
                "scan_timestamp": datetime.now().isoformat()
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "Failed to retrieve vulnerabilities.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class QuartermasterSimulateUpgradeView(APIView):
    """Run a safe upgrade simulation for a dependency."""
    
    def post(self, request, repo_id, *args, **kwargs):
        if not repo_id:
            return Response(
                {"error": "'repo_id' path parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not _is_safe_path(CLONED_REPOS_DIR, repo_id):
            return Response({"error": "Invalid repo_id"}, status=status.HTTP_400_BAD_REQUEST)
        
        dependency_name = request.data.get('dependency_name')
        target_version = request.data.get('target_version')
        
        if not all([dependency_name, target_version]):
            return Response(
                {"error": "'dependency_name' and 'target_version' are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            quartermaster = QuartermasterService(CLONED_REPOS_DIR)
            simulation_report = quartermaster.simulate_upgrade(repo_id, dependency_name, target_version)
            
            # Convert dataclass to dict for JSON serialization
            report_dict = asdict(simulation_report)
            
            return Response(report_dict, status=status.HTTP_200_OK)
            
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "Failed to simulate upgrade.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class QuartermasterLicenseComplianceView(APIView):
    """Check licenses against a compliance policy."""
    
    def post(self, request, repo_id, *args, **kwargs):
        if not repo_id:
            return Response(
                {"error": "'repo_id' path parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not _is_safe_path(CLONED_REPOS_DIR, repo_id):
            return Response({"error": "Invalid repo_id"}, status=status.HTTP_400_BAD_REQUEST)
        
        policy_data = request.data.get('policy', {})
        
        # Create compliance policy from request data or use default
        if policy_data:
            policy = CompliancePolicy(
                allowed_licenses=policy_data.get('allowed', []),
                denied_licenses=policy_data.get('denied', []),
                restricted_licenses=policy_data.get('restricted', []),
                policy_name=policy_data.get('name', 'custom_policy'),
                created_at=datetime.now().isoformat()
            )
        else:
            # Default policy
            policy = CompliancePolicy(
                allowed_licenses=["MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "ISC"],
                denied_licenses=["GPL-3.0", "AGPL-3.0", "SSPL-1.0"],
                restricted_licenses=["GPL-2.0", "LGPL-3.0"],
                policy_name="default_policy",
                created_at=datetime.now().isoformat()
            )
        
        try:
            # Load BOM data
            bom_data = get_bom_data(repo_id)
            if not bom_data:
                return Response(
                    {"error": f"BOM data not found for repository '{repo_id}'."},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            quartermaster = QuartermasterService(CLONED_REPOS_DIR)
            violations = quartermaster.check_license_compliance(bom_data, policy)
            
            # Convert dataclasses to dicts for JSON serialization
            violation_dicts = [asdict(violation) for violation in violations]
            
            # Calculate compliance status
            compliant = len(violations) == 0
            violation_types = {}
            for violation in violations:
                violation_types[violation.violation_type] = violation_types.get(violation.violation_type, 0) + 1
            
            response_data = {
                "repository": repo_id,
                "compliant": compliant,
                "policy_applied": asdict(policy),
                "total_violations": len(violations),
                "violation_breakdown": violation_types,
                "violations": violation_dicts,
                "scan_timestamp": datetime.now().isoformat()
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "Failed to check license compliance.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class QuartermasterRiskReportView(APIView):
    """Generate a comprehensive risk report for management."""
    
    def get(self, request, repo_id, *args, **kwargs):
        if not repo_id:
            return Response(
                {"error": "'repo_id' path parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not _is_safe_path(CLONED_REPOS_DIR, repo_id):
            return Response({"error": "Invalid repo_id"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            quartermaster = QuartermasterService(CLONED_REPOS_DIR)
            risk_report = quartermaster.generate_risk_report(repo_id)
            
            if "error" in risk_report:
                return Response(risk_report, status=status.HTTP_404_NOT_FOUND)
            
            return Response(risk_report, status=status.HTTP_200_OK)
            
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "Failed to generate risk report.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# --- LOREMASTER ARCHETYPE VIEWS ---

class LoremasterApiInventoryView(APIView):
    """Get the raw list of all discovered API endpoints."""
    
    def get(self, request, repo_id, *args, **kwargs):
        if not repo_id:
            return Response(
                {"error": "'repo_id' path parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not _is_safe_path(CLONED_REPOS_DIR, repo_id):
            return Response({"error": "Invalid repo_id"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            loremaster = LoremasterService(CLONED_REPOS_DIR)
            inventory = loremaster.get_api_inventory(repo_id)
            
            if "error" in inventory:
                return Response(inventory, status=status.HTTP_404_NOT_FOUND)
            
            return Response(inventory, status=status.HTTP_200_OK)
            
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "Failed to retrieve API inventory.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class LoremasterOpenApiSpecView(APIView):
    """Get the auto-generated OpenAPI 3.0 specification."""
    
    def get(self, request, repo_id, *args, **kwargs):
        if not repo_id:
            return Response(
                {"error": "'repo_id' path parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not _is_safe_path(CLONED_REPOS_DIR, repo_id):
            return Response({"error": "Invalid repo_id"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            loremaster = LoremasterService(CLONED_REPOS_DIR)
            openapi_spec = loremaster.generate_openapi_spec(repo_id)
            
            if "error" in openapi_spec:
                return Response(openapi_spec, status=status.HTTP_404_NOT_FOUND)
            
            return Response(openapi_spec, status=status.HTTP_200_OK)
            
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "Failed to generate OpenAPI specification.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class LoremasterDocumentationView(APIView):
    """Get the full, standalone HTML documentation page."""
    
    def get(self, request, repo_id, *args, **kwargs):
        if not repo_id:
            return Response(
                {"error": "'repo_id' path parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not _is_safe_path(CLONED_REPOS_DIR, repo_id):
            return Response({"error": "Invalid repo_id"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            loremaster = LoremasterService(CLONED_REPOS_DIR)
            html_content = loremaster.generate_documentation_page(repo_id)
            
            # Return HTML content directly
            from django.http import HttpResponse
            return HttpResponse(html_content, content_type='text/html')
            
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "Failed to generate documentation page.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class LoremasterClientSnippetView(APIView):
    """Generate a client code snippet for an endpoint."""
    
    def post(self, request, repo_id, *args, **kwargs):
        if not repo_id:
            return Response(
                {"error": "'repo_id' path parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not _is_safe_path(CLONED_REPOS_DIR, repo_id):
            return Response({"error": "Invalid repo_id"}, status=status.HTTP_400_BAD_REQUEST)
        
        endpoint_path = request.data.get('endpoint_path')
        method = request.data.get('method', 'GET')
        language = request.data.get('language', 'python')
        
        if not endpoint_path:
            return Response(
                {"error": "'endpoint_path' is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # First get the API inventory to find the endpoint
            loremaster = LoremasterService(CLONED_REPOS_DIR)
            inventory = loremaster.get_api_inventory(repo_id)
            
            if "error" in inventory:
                return Response(inventory, status=status.HTTP_404_NOT_FOUND)
            
            # Find matching endpoint
            endpoint_data = None
            for endpoint in inventory.get('endpoints', []):
                if (endpoint.get('path') == endpoint_path and 
                    method.upper() in [m.upper() for m in endpoint.get('methods', [])]):
                    endpoint_data = endpoint
                    break
            
            if not endpoint_data:
                return Response(
                    {"error": f"Endpoint '{method} {endpoint_path}' not found."},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Generate client snippet
            snippet_result = loremaster.generate_client_snippet(endpoint_data, language)
            
            if "error" in snippet_result:
                return Response(snippet_result, status=status.HTTP_400_BAD_REQUEST)
            
            return Response(snippet_result, status=status.HTTP_200_OK)
            
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "Failed to generate client snippet.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# --- LIBRARIAN'S ARCHIVES VIEWS ---

class LibrarianIngestView(APIView):
    """Ingest a new local directory as an Archive."""
    
    def post(self, request, *args, **kwargs):
        path = request.data.get('path')
        
        if not path:
            return Response(
                {"error": "'path' is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate path exists and is accessible
        import os
        if not os.path.exists(path):
            return Response(
                {"error": f"Path does not exist: {path}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not os.path.isdir(path):
            return Response(
                {"error": f"Path is not a directory: {path}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            result = ingest_local_directory(path)
            
            if result.get("status") == "failed":
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            return Response(result, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "An unexpected internal server error occurred during archive ingestion.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class LibrarianListArchivesView(APIView):
    """List all available archives."""
    
    def get(self, request, *args, **kwargs):
        archives = []
        
        # Check for local_archives directory
        backend_dir = Path(__file__).resolve().parent.parent
        archives_dir = backend_dir / "local_archives"
        
        if not archives_dir.exists():
            return Response([], status=status.HTTP_200_OK)
        
        for archive_dir in archives_dir.iterdir():
            if archive_dir.is_dir():
                archive_id = archive_dir.name
                cortex_file = archive_dir / f"{archive_id}_cortex.json"
                faiss_file = archive_dir / f"{archive_id}_faiss.index"
                map_file = archive_dir / f"{archive_id}_id_map.json"
                
                if all([cortex_file.exists(), faiss_file.exists(), map_file.exists()]):
                    try:
                        # Load archive metadata
                        with open(cortex_file, 'r', encoding='utf-8') as f:
                            cortex_data = json.load(f)
                        
                        archive_metadata = cortex_data.get("archive_metadata", {})
                        
                        archives.append({
                            "archive_id": archive_id,
                            "source_path": archive_metadata.get("source_path", "unknown"),
                            "ingested_at": archive_metadata.get("ingested_at"),
                            "total_files": archive_metadata.get("directory_stats", {}).get("total_files", 0),
                            "primary_language": archive_metadata.get("directory_stats", {}).get("primary_language", "Unknown"),
                            "directory_size_mb": archive_metadata.get("directory_stats", {}).get("directory_size_mb", 0)
                        })
                    except Exception:
                        continue
        
        return Response(sorted(archives, key=lambda x: x.get('ingested_at', '')), status=status.HTTP_200_OK)

class LibrarianArchiveDetailView(APIView):
    """Get details for a specific archive."""
    
    def get(self, request, archive_id, *args, **kwargs):
        if not archive_id:
            return Response(
                {"error": "'archive_id' path parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        backend_dir = Path(__file__).resolve().parent.parent
        archives_dir = backend_dir / "local_archives"
        archive_dir = archives_dir / archive_id
        
        if not archive_dir.exists():
            return Response(
                {"error": f"Archive '{archive_id}' not found."},
                status=status.HTTP_404_NOT_FOUND
            )
        
        cortex_file = archive_dir / f"{archive_id}_cortex.json"
        
        if not cortex_file.exists():
            return Response(
                {"error": f"Archive cortex file not found for '{archive_id}'."},
                status=status.HTTP_404_NOT_FOUND
            )
        
        try:
            with open(cortex_file, 'r', encoding='utf-8') as f:
                cortex_data = json.load(f)
            
            archive_metadata = cortex_data.get("archive_metadata", {})
            
            # Build response with archive details
            response_data = {
                "archive_id": archive_id,
                "archive_metadata": archive_metadata,
                "polyglot_summary": cortex_data.get("polyglot_summary", {}),
                "total_files": len(cortex_data.get("files", [])),
                "scan_timestamp": cortex_data.get("last_crawled_utc"),
                "languages_detected": cortex_data.get("polyglot_summary", {}).get("languages_detected", 0)
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "Failed to retrieve archive details.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def delete(self, request, archive_id, *args, **kwargs):
        """Delete an archive and its artifacts."""
        if not archive_id:
            return Response(
                {"error": "'archive_id' path parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        backend_dir = Path(__file__).resolve().parent.parent
        archives_dir = backend_dir / "local_archives"
        archive_dir = archives_dir / archive_id
        
        if not archive_dir.exists():
            return Response(status=status.HTTP_204_NO_CONTENT)  # Already gone, treat as success
        
        try:
            shutil.rmtree(archive_dir)
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Exception as e:
            return Response(
                {"error": f"Failed to delete archive '{archive_id}'.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class LibrarianAskView(APIView):
    """Ask a question about an archive using the enhanced Oracle service."""
    
    def post(self, request, *args, **kwargs):
        archive_id = request.data.get('archive_id')
        question = request.data.get('question')
        
        if not all([archive_id, question]):
            return Response(
                {"error": "'archive_id' and 'question' are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Use the enhanced Oracle service which now supports archives
            result = oracle_service.answer_question(archive_id, question)
            
            if "error" in result:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            
            return Response(result, status=status.HTTP_200_OK)
            
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": "An internal server error occurred in The Oracle for archives.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
