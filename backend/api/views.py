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
    testing, review_service, ingestion_service, cortex_service,
)

# A sensible default model, preferably a fast one.
# It will be used if the client doesn't specify a model.
DEFAULT_MODEL = "ollama/qwen3:4b"

# --- NEW VIEW: Health Check ---
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
            return Response({"error": "Failed to list models.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class StrategistPrioritizeView(APIView):
    def post(self, request, *args, **kwargs):
        repo_url = request.data.get('repo_url')
        model = request.data.get('model', DEFAULT_MODEL)
        if not repo_url:
            return Response({"error": "'repo_url' is required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            result = strategist.analyze_and_prioritize(repo_url, model_identifier=model)
            if "error" in result:
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": "An internal server error occurred.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DiplomatView(APIView):
    def post(self, request, *args, **kwargs):
        issue_title = request.data.get('issue_title')
        issue_body = request.data.get('issue_body')
        model = request.data.get('model', DEFAULT_MODEL)
        if not all([issue_title, issue_body]):
            return Response({"error": "'issue_title' and 'issue_body' are required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            result = diplomat.find_similar_solved_issues(issue_title, issue_body, model_identifier=model)
            if "error" in result:
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": "An internal server error occurred.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class BriefingView(APIView):
    def post(self, request, *args, **kwargs):
        issue_url = request.data.get('issue_url')
        model = request.data.get('model', DEFAULT_MODEL)
        if not issue_url:
            return Response({"error": "'issue_url' is required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            result = rca_service.generate_briefing(issue_url, model_identifier=model)
            if "error" in result:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": "An internal server error occurred.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RcaView(APIView):
    def post(self, request, *args, **kwargs):
        repo_url = request.data.get('repo_url')
        bug_description = request.data.get('bug_description')
        model = request.data.get('model', DEFAULT_MODEL)

        # New optional parameters
        advanced_analysis = request.data.get('advanced_analysis', False)
        confidence_threshold = request.data.get('confidence_threshold', 0.7)

        if not all([repo_url, bug_description]):
            return Response(
                {"error": "'repo_url' and 'bug_description' are required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            result = rca_service.perform_rca(
                repo_url=repo_url,
                bug_description=bug_description,
                model_identifier=model,
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
        # --- UPDATED TO MATCH NEW SERVICE ---
        repo_id = request.data.get('repo_id')
        target_files = request.data.get('target_files') # Expects a list
        instruction = request.data.get('instruction')
        rca_report = request.data.get('rca_report')
        model = request.data.get('model', DEFAULT_MODEL)
        refinement_history = request.data.get('refinement_history')

        if not all([repo_id, target_files, instruction, rca_report]):
            return Response({"error": "'repo_id', 'target_files' (list), 'instruction', and 'rca_report' are required."}, status=status.HTTP_400_BAD_REQUEST)
        if not isinstance(target_files, list):
            return Response({"error": "'target_files' must be a list of strings."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            result = scaffolding.generate_scaffold(
                repo_id, target_files, instruction, model, rca_report, refinement_history
            )
            if "error" in result:
                # Use 500 for server-side issues like missing cortex file
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": "An internal server error occurred.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class CrucibleValidateView(APIView):
    def post(self, request, *args, **kwargs):
        repo_url = request.data.get('repo_url')
        target_file = request.data.get('target_file')
        modified_code = request.data.get('modified_code')
        if not all([repo_url, target_file, modified_code]):
            return Response({"error": "'repo_url', 'target_file', 'modified_code' are required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            result = crucible.validate_fix(repo_url, target_file, modified_code)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            error_details = traceback.format_exc()
            return Response({"error": "An unexpected internal server error occurred in The Crucible.", "details": str(e), "traceback": error_details}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AmbassadorDispatchView(APIView):
    def post(self, request, *args, **kwargs):
        # --- MODIFIED: Expect a dictionary of files ---
        issue_url = request.data.get('issue_url')
        modified_files = request.data.get('modified_files') # Expects dict
        model = request.data.get('model', DEFAULT_MODEL)

        if not all([issue_url, modified_files]):
            return Response({"error": "'issue_url' and 'modified_files' (dict) are required."}, status=status.HTTP_400_BAD_REQUEST)
        if not isinstance(modified_files, dict):
            return Response({"error": "'modified_files' must be a dictionary."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            result = ambassador.dispatch_pr(issue_url, modified_files, model)
            if "error" in result:
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(result, status=status.HTTP_201_CREATED)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": "An internal server error occurred.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ProfileReviewView(APIView):
    def post(self, request, *args, **kwargs):
        username = request.data.get('username')
        model = request.data.get('model', DEFAULT_MODEL)
        if not username:
            return Response({"error": "'username' is required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            result = profile_service.generate_profile_review(username, model_identifier=model)
            if "error" in result:
                return Response(result, status=status.HTTP_404_NOT_FOUND)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": "An internal server error occurred.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class IssueListView(APIView):
    def get(self, request, *args, **kwargs):
        repo_url = request.query_params.get('repo_url')
        if not repo_url:
            return Response({"error": "'repo_url' query parameter is required."}, status=status.HTTP_400_BAD_REQUEST)
        match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
        if not match:
            return Response({"error": "Invalid 'repo_url' format."}, status=status.HTTP_400_BAD_REQUEST)
        repo_full_name = f"{match.group(1)}/{match.group(2)}"
        try:
            issues = github.list_open_issues(repo_full_name)
            return Response(issues, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": "An internal server error occurred.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FileContentView(APIView):
    def post(self, request, *args, **kwargs):
        repo_id = request.data.get('repo_id')
        file_path = request.data.get('file_path')
        if not all([repo_id, file_path]):
            return Response({"error": "'repo_id' and 'file_path' are required."}, status=status.HTTP_400_BAD_REQUEST)

        # --- THE FIX: Call the new, correct service ---
        content = cortex_service.get_file_content(repo_id, file_path)

        if content is None:
            return Response({"error": f"File '{file_path}' not found for repo '{repo_id}'."}, status=status.HTTP_404_NOT_FOUND)

        return Response({"content": content}, status=status.HTTP_200_OK)


class TestGenerationView(APIView):
    def post(self, request, *args, **kwargs):
        repo_id = request.data.get('repo_id')
        new_code = request.data.get('new_code')
        instruction = request.data.get('instruction')
        if not all([repo_id, new_code, instruction]):
            return Response({"error": "'repo_id', 'new_code', and 'instruction' are required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            result = testing.generate_tests_for_code(repo_id, new_code, instruction)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": "An internal server error occurred.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class DocstringGenerationView(APIView):
    def post(self, request, *args, **kwargs):
        repo_id = request.data.get('repo_id')
        new_code = request.data.get('new_code')
        instruction = request.data.get('instruction')
        # --- ADDED: Get the model from the request ---
        model = request.data.get('model', DEFAULT_MODEL)

        if not all([repo_id, new_code, instruction]):
            return Response({"error": "'repo_id', 'new_code', and 'instruction' are required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            # --- MODIFIED: Pass the model to the service ---
            result = documentation.generate_docstring_for_code(repo_id, new_code, instruction, model)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": "An internal server error occurred.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class PrepareReviewView(APIView):
    def post(self, request, *args, **kwargs):
        repo_url = request.data.get('repo_url')
        ref_name = request.data.get('ref_name') # e.g., a branch or tag name
        if not all([repo_url, ref_name]):
            return Response({"error": "'repo_url' and 'ref_name' are required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            result = review_service.prepare_review_environment(repo_url, ref_name)
            return Response(result, status=status.HTTP_201_CREATED)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": "Failed to prepare review environment.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ExecuteReviewView(APIView):
    def post(self, request, *args, **kwargs):
        review_id = request.data.get('review_id')
        model = request.data.get('model', DEFAULT_MODEL)
        if not review_id:
            return Response({"error": "'review_id' is required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            diff_text = review_service.get_diff_for_review(review_id)
            result = review_service.review_code_diff(diff_text) # Assumes review_code_diff uses default model
            review_service.cleanup_review_environment(review_id)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": "Failed to execute review.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# --- Ingestion ---
class IngestRepositoryView(APIView):
    """
    Triggers the full ingestion pipeline for a given repository URL.
    This is a long-running, synchronous operation.
    """
    def post(self, request, *args, **kwargs):
        repo_url = request.data.get('repo_url')
        if not repo_url:
            return Response({"error": "'repo_url' is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # We can also get the embedding model from the request in the future
            result = ingestion_service.clone_and_embed_repository(repo_url)
            if result.get("status") == "failed":
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(result, status=status.HTTP_201_CREATED)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": "An unexpected internal server error occurred during ingestion.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GraphDataView(APIView):
    """
    Retrieves the architectural graph data from a repository's Cortex file.
    """
    def get(self, request, *args, **kwargs):
        repo_id = request.query_params.get('repo_id')
        if not repo_id:
            return Response({"error": "'repo_id' query parameter is required."}, status=status.HTTP_400_BAD_REQUEST)

        # Construct the path to the cortex file
        backend_dir = Path(__file__).resolve().parent.parent
        cortex_file = backend_dir / "cloned_repositories" / repo_id / f"{repo_id}_cortex.json"

        if not cortex_file.exists():
            return Response({"error": f"Cortex file for repo '{repo_id}' not found."}, status=status.HTTP_404_NOT_FOUND)

        try:
            with open(cortex_file, 'r', encoding='utf-8') as f:
                cortex_data = json.load(f)

            graph_data = cortex_data.get('architectural_graph')
            if not graph_data:
                new_message = "Architectural graph is not available. The Cartographer feature currently only supports Python projects (.py files). This repository does not appear to contain Python code suitable for graphing."
                return Response({
                    "message": new_message,
                    "graph": None
                }, status=status.HTTP_200_OK)

            return Response({"graph": graph_data, "repo_id": repo_id}, status=status.HTTP_200_OK)

        except Exception as e:
            traceback.print_exc()
            return Response({"error": "Failed to read or parse graph data.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
