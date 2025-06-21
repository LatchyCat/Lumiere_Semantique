# In api/urls.py
from django.urls import path
from .views import (
    BriefingView, ScaffoldView, TestGenerationView, RcaView,
    DocstringGenerationView, PrepareReviewView, ExecuteReviewView,
    ProfileReviewView, AmbassadorDispatchView, IssueListView,
    # --- NEW IMPORT ---
    StrategistPrioritizeView,
)

urlpatterns = [
    # --- "Triage & Strategy" ENDPOINTS ---
    path('issues/list/', IssueListView.as_view(), name='list_issues'),
    path('strategist/prioritize/', StrategistPrioritizeView.as_view(), name='strategist_prioritize'),

    # --- "Execution" ENDPOINTS ---
    path('briefing/', BriefingView.as_view(), name='briefing'),
    path('scaffold/', ScaffoldView.as_view(), name='scaffold'),
    path('generate-tests/', TestGenerationView.as_view(), name='generate_tests'),
    path('generate-docstring/', DocstringGenerationView.as_view(), name='generate_docstring'),
    path('rca/', RcaView.as_view(), name='rca'),
    path('ambassador/dispatch/', AmbassadorDispatchView.as_view(), name='ambassador_dispatch'),

    # --- "Review" ENDPOINTS ---
    path('review/prepare', PrepareReviewView.as_view(), name='prepare_review'),
    path('review/execute', ExecuteReviewView.as_view(), name='execute_review'),
    path('profile/review/', ProfileReviewView.as_view(), name='profile_review'),
]
