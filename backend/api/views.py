# In api/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
import json
import traceback
import re

from lumiere_core.services.ollama import search_index
from lumiere_core.services.llm import generate_text
from lumiere_core.services.github import scrape_github_issue, list_open_issues
from lumiere_core.services.scaffolding import generate_scaffold
from lumiere_core.services.testing import generate_tests_for_code
from lumiere_core.services.documentation import generate_docstring_for_code
from ingestion.crawler import IntelligentCrawler
from lumiere_core.services import review_service
from lumiere_core.services import profile_service
from lumiere_core.services import ambassador
from lumiere_core.services import strategist

STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in', 'into', 'is', 'it', 'its', 'no', 'not', 'of', 'on', 'or', 'such', 'that', 'the', 'their', 'then', 'there', 'these', 'they', 'this', 'to', 'was', 'will', 'with', 'i', 'im', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'herself', 'it', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'shouldn', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'hi', 'everyone'
}

def _filter_blame_by_keywords(blame_output: str, keywords: set) -> str:
    if not keywords: return ""
    filtered_lines = [line for line in blame_output.splitlines() if any(keyword.lower() in line.lower() for keyword in keywords)]
    return "\n".join(filtered_lines)

class BriefingView(APIView):
    def post(self, request, *args, **kwargs):
        issue_url = request.data.get('issue_url')
        if not issue_url: return Response({"error": "A 'issue_url' field is required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            issue_data = scrape_github_issue(issue_url)
            if not issue_data: return Response({"error": "Failed to fetch data from the GitHub API."}, status=status.HTTP_400_BAD_REQUEST)
            repo_url, rag_query, repo_id = issue_data['repo_url'], issue_data['full_text_query'], issue_data['repo_url'].replace("https://github.com/", "").replace("/", "_")
            index_path, map_path = f"{repo_id}_faiss.index", f"{repo_id}_id_map.json"
            if not os.path.exists(index_path): return Response({"error": f"The repository '{repo_id}' has not been indexed yet."}, status=status.HTTP_404_NOT_FOUND)
            raw_keywords = set(re.findall(r'\b([A-Z][a-zA-Z0-9_]+|[a-z_]{3,})\b', rag_query))
            keywords = {kw for kw in raw_keywords if kw.lower() not in STOP_WORDS}
            initial_chunks = search_index(query_text=rag_query, model_name='snowflake-arctic-embed2:latest', index_path=index_path, map_path=map_path, k=15)
            ranked_chunks = sorted([{"chunk": chunk, "score": sum(len(kw) for kw in keywords if kw.lower() in chunk['text'].lower())} for chunk in initial_chunks], key=lambda x: x['score'], reverse=True)
            context_string = "\n\n".join([f"--- Context from file: {item['chunk']['file_path']} ---\n{item['chunk']['text']}" for item in ranked_chunks[:7]])
            briefing_prompt = f"You are an expert Principal Engineer... GITHUB ISSUE\n{rag_query}\n---\nRELEVANT CONTEXT\n{context_string}\n---"
            final_report = generate_text(briefing_prompt, model_name='qwen3:4b')
            return Response({"briefing": final_report}, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": "An internal server error occurred.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ScaffoldView(APIView):
    def post(self, request, *args, **kwargs):
        repo_id, target_file, instruction = request.data.get('repo_id'), request.data.get('target_file'), request.data.get('instruction')
        if not all([repo_id, target_file, instruction]): return Response({"error": "The 'repo_id', 'target_file', and 'instruction' fields are required."}, status=status.HTTP_400_BAD_REQUEST)
        index_path, cortex_path = f"{repo_id}_faiss.index", f"{repo_id}_cortex.json"
        if not os.path.exists(index_path) or not os.path.exists(cortex_path): return Response({"error": f"Index and/or Cortex file for '{repo_id}' not found."}, status=status.HTTP_404_NOT_FOUND)
        try:
            result = generate_scaffold(repo_id=repo_id, target_file=target_file, instruction=instruction)
            if "error" in result: return Response(result, status=status.HTTP_404_NOT_FOUND)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": "An internal server error occurred.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class TestGenerationView(APIView):
    def post(self, request, *args, **kwargs):
        repo_id, instruction, new_code = request.data.get('repo_id'), request.data.get('instruction'), request.data.get('new_code')
        if not all([repo_id, instruction, new_code]):
            return Response({"error": "'repo_id', 'instruction', and 'new_code' are required."}, status=status.HTTP_400_BAD_REQUEST)
        index_path = f"{repo_id}_faiss.index"
        if not os.path.exists(index_path):
            return Response({"error": f"Index for '{repo_id}' not found. Please ingest the repository first."}, status=status.HTTP_404_NOT_FOUND)
        try:
            result = generate_tests_for_code(repo_id=repo_id, new_code=new_code, instruction=instruction)
            if "error" in result: return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            print(f"An unexpected error occurred in TestGenerationView: {e}"); traceback.print_exc()
            return Response({"error": "Internal server error.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class RcaView(APIView):
    def post(self, request, *args, **kwargs):
        repo_url, bug_description, target_file = request.data.get('repo_url'), request.data.get('bug_description'), request.data.get('target_file')
        if not all([repo_url, bug_description, target_file]):
            return Response({"error": "'repo_url', 'bug_description', 'target_file' are required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            with IntelligentCrawler(repo_url=repo_url) as crawler:
                resolved_target_path = crawler.find_file_path(target_file)
                if isinstance(resolved_target_path, dict): return Response(resolved_target_path, status=status.HTTP_409_CONFLICT)
                if not resolved_target_path: return Response({"error": f"Lumière Sémantique could not find '{target_file}'."}, status=status.HTTP_404_NOT_FOUND)
                blame_output = crawler.get_blame_for_file(resolved_target_path)
                if "Error from" in blame_output: return Response({"error": blame_output}, status=status.HTTP_400_BAD_REQUEST)
                raw_keywords = set(re.findall(r'`([^`]+)`|\b([a-zA-Z]{3,})\b', bug_description))
                keywords = {item.lower() for tpl in raw_keywords for item in tpl if item and item.lower() not in STOP_WORDS}
                filtered_blame = _filter_blame_by_keywords(blame_output, keywords)
                if not filtered_blame:
                    return Response({"rca_report": "### Root Cause Analysis\n\nNo relevant lines could be found in the git blame history for the keywords in the bug description. Unable to perform analysis."}, status=status.HTTP_200_OK)
                prompt = f"""You are a software detective...
### BUG DESCRIPTION
{bug_description}
---
### PRE-FILTERED BLAME LOG for {resolved_target_path}
This log only contains lines that include the keywords: {', '.join(keywords)}
{filtered_blame}
---
Now, generate the Root Cause Analysis report.
"""
                raw_report = generate_text(prompt, model_name='qwen3:4b')
                final_report = raw_report.split("</think>", 1)[-1].strip()
                return Response({"rca_report": final_report}, status=status.HTTP_200_OK)
        except Exception as e:
            print(f"An unexpected error occurred in RcaView: {e}"); traceback.print_exc()
            return Response({"error": "An internal server error.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class DocstringGenerationView(APIView):
    def post(self, request, *args, **kwargs):
        repo_id, instruction, new_code = request.data.get('repo_id'), request.data.get('instruction'), request.data.get('new_code')
        if not all([repo_id, instruction, new_code]):
            return Response({"error": "'repo_id', 'instruction', and 'new_code' are required."}, status=status.HTTP_400_BAD_REQUEST)
        index_path = f"{repo_id}_faiss.index"
        if not os.path.exists(index_path):
            return Response({"error": f"Index for '{repo_id}' not found. Please ingest the repository first."}, status=status.HTTP_404_NOT_FOUND)
        try:
            result = generate_docstring_for_code(repo_id=repo_id, new_code=new_code, instruction=instruction)
            if "error" in result: return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            print("✓ Docstring generation complete. Sending response to client.")
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            print(f"An unexpected error occurred in DocstringGenerationView: {e}"); traceback.print_exc()
            return Response({"error": "Internal server error.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class PrepareReviewView(APIView):
    def post(self, request, *args, **kwargs):
        repo_url = request.data.get('repo_url')
        branch_name = request.data.get('branch_name')
        if not all([repo_url, branch_name]):
            return Response({"error": "'repo_url' and 'branch_name' are required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            result = review_service.prepare_review_environment(repo_url, branch_name)
            return Response(result, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({"error": "Failed to prepare review environment.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ExecuteReviewView(APIView):
    def post(self, request, *args, **kwargs):
        review_id = request.data.get('review_id')
        if not review_id:
            return Response({"error": "'review_id' is required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            diff_text = review_service.get_diff_for_review(review_id)
            result = review_service.review_code_diff(diff_text)
            return Response(result, status=status.HTTP_200_OK)
        except FileNotFoundError as e:
            return Response({"error": str(e)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            print(f"An unexpected error occurred in ExecuteReviewView: {e}"); traceback.print_exc()
            return Response({"error": "Internal server error during review execution.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            if review_id:
                review_service.cleanup_review_environment(review_id)

class ProfileReviewView(APIView):
    def post(self, request, *args, **kwargs):
        username = request.data.get('username')
        if not username:
            return Response({"error": "A 'username' field is required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            result = profile_service.generate_profile_review(username)
            return Response(result, status=status.HTTP_200_OK)
        except FileNotFoundError as e:
            return Response({"error": str(e)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            print(f"An unexpected error occurred in ProfileReviewView: {e}")
            traceback.print_exc()
            return Response({"error": "An internal server error occurred.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class AmbassadorDispatchView(APIView):
    def post(self, request, *args, **kwargs):
        issue_url = request.data.get('issue_url')
        if not issue_url:
            return Response({"error": "A 'issue_url' field is required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            result = ambassador.dispatch_pr(issue_url)
            if "error" in result:
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(result, status=status.HTTP_201_CREATED)
        except Exception as e:
            print(f"An unexpected error occurred in AmbassadorDispatchView: {e}")
            traceback.print_exc()
            return Response({"error": "An internal server error occurred.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class IssueListView(APIView):
    def get(self, request, *args, **kwargs):
        repo_url = request.query_params.get('repo_url')
        if not repo_url:
            return Response({"error": "A 'repo_url' query parameter is required."}, status=status.HTTP_400_BAD_REQUEST)

        match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
        if not match:
            return Response({"error": "Invalid 'repo_url' format. Should be like https://github.com/owner/repo"}, status=status.HTTP_400_BAD_REQUEST)

        repo_full_name = f"{match.group(1)}/{match.group(2)}"

        try:
            issues = list_open_issues(repo_full_name)
            return Response(issues, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": "An internal server error occurred while fetching issues.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class StrategistPrioritizeView(APIView):
    def post(self, request, *args, **kwargs):
        repo_url = request.data.get('repo_url')
        if not repo_url:
            return Response({"error": "A 'repo_url' field is required."}, status=status.HTTP_400_BAD_REQUEST)

        auto_dispatch_config = request.data.get('auto_dispatch_config', {"enabled": False})

        try:
            result = strategist.analyze_and_prioritize(repo_url, auto_dispatch_config)
            if "error" in result:
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            print(f"An unexpected error occurred in StrategistPrioritizeView: {e}")
            traceback.print_exc()
            return Response({"error": "An internal server error occurred.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
