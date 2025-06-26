# In backend/api/urls.py

from django.urls import path
from .views import (
    BriefingView, ScaffoldView, TestGenerationView, RcaView,
    DocstringGenerationView,
    ProfileReviewView, AmbassadorDispatchView, IssueListView,
    StrategistPrioritizeView, FileContentView, DiplomatView,
    CrucibleValidateView, ListModelsView, HealthCheckView,
    IngestRepositoryView, GraphDataView, OracleView,
    AdjudicateView, HarmonizeView, SentinelBriefingView,
    SuggestActionsView,
)

urlpatterns = [
    # --- HEALTH CHECK ENDPOINT ---
    path('health/', HealthCheckView.as_view(), name='health_check'),

    # --- SUGGESTER ENDPOINT ---
    path('suggest-actions/', SuggestActionsView.as_view(), name='suggest_actions'),

    # --- MODEL MANAGEMENT ENDPOINT ---
    path('models/list/', ListModelsView.as_view(), name='list_models'),

     # --- INGESTION ENDPOINT ---
    path('ingest/', IngestRepositoryView.as_view(), name='ingest_repository'),

     # --- ORACLE (Q&A) ENDPOINT ---
    path('oracle/ask/', OracleView.as_view(), name='oracle_ask'),

    # --- Graph ENDPOINT ---
    path('graph/', GraphDataView.as_view(), name='graph_data'),

    # --- "Triage & Strategy" ENDPOINTS ---
    path('issues/list/', IssueListView.as_view(), name='list_issues'),
    path('strategist/prioritize/', StrategistPrioritizeView.as_view(), name='strategist_prioritize'),
    path('diplomat/find-similar-issues/', DiplomatView.as_view(), name='diplomat_find_similar'),

    # --- "Execution & Validation" ENDPOINTS ---
    path('briefing/', BriefingView.as_view(), name='briefing'),
    path('scaffold/', ScaffoldView.as_view(), name='scaffold'),
    path('crucible/validate/', CrucibleValidateView.as_view(), name='crucible_validate'),
    path('file-content/', FileContentView.as_view(), name='file_content'),
    path('generate-tests/', TestGenerationView.as_view(), name='generate_tests'),
    path('generate-docstring/', DocstringGenerationView.as_view(), name='generate_docstring'),
    path('rca/', RcaView.as_view(), name='rca'),
    path('ambassador/dispatch/', AmbassadorDispatchView.as_view(), name='ambassador_dispatch'),

    # --- "Review" ENDPOINTS ---
    path('profile/review/', ProfileReviewView.as_view(), name='profile_review'),
    path('review/adjudicate/', AdjudicateView.as_view(), name='adjudicate_pr'),
    path('review/harmonize/', HarmonizeView.as_view(), name='harmonize_pr_fix'),

    # --- SENTINEL ENDPOINT ---
    path('sentinel/briefing/', SentinelBriefingView.as_view(), name='sentinel_briefing'),
]
