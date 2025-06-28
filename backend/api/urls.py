# Enhanced backend/api/urls.py

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

    # Enhanced BOM Views
    BomDataView, BomDependenciesView, BomServicesView,
    BomSecurityView, BomRegenerateView, BomCompareView,

    # Quartermaster Views
    QuartermasterDashboardView, QuartermasterVulnerabilitiesView,
    QuartermasterSimulateUpgradeView, QuartermasterLicenseComplianceView,
    QuartermasterRiskReportView,

    # Loremaster Views
    LoremasterApiInventoryView, LoremasterOpenApiSpecView,
    LoremasterDocumentationView, LoremasterClientSnippetView,

    # Librarian Views
    LibrarianIngestView, LibrarianListArchivesView,
    LibrarianArchiveDetailView, LibrarianAskView,

    # Onboarding Concierge Views
    ExpertiseView, OnboardingGuideView,

    # API Endpoint Inventory View
    ApiEndpointInventoryView,

    # GET and DELETE methods for the /api/v1/repositories/{repo_id}/
    ListRepositoriesView, RepositoryDetailView,
    RepositoryStatusView, SentinelMetricsHistoryView
)

urlpatterns = [
    # --- HEALTH CHECK ENDPOINT ---
    path('health/', HealthCheckView.as_view(), name='health_check'),

    # --- SUGGESTER ENDPOINT ---
    path('suggest-actions/', SuggestActionsView.as_view(), name='suggest_actions'),


    # --- BILL OF MATERIALS (BOM) ENDPOINTS ---
    path('bom/', BomDataView.as_view(), name='bom_data'),
    path('bom/dependencies/', BomDependenciesView.as_view(), name='bom_dependencies'),
    path('bom/services/', BomServicesView.as_view(), name='bom_services'),
    path('bom/security/', BomSecurityView.as_view(), name='bom_security'),
    path('bom/regenerate/', BomRegenerateView.as_view(), name='bom_regenerate'),
    path('bom/compare/', BomCompareView.as_view(), name='bom_compare'),

    # --- QUARTERMASTER ENDPOINTS ---
    path('quartermaster/<str:repo_id>/dashboard/', QuartermasterDashboardView.as_view(), name='quartermaster_dashboard'),
    path('quartermaster/<str:repo_id>/vulnerabilities/', QuartermasterVulnerabilitiesView.as_view(), name='quartermaster_vulnerabilities'),
    path('quartermaster/<str:repo_id>/simulate-upgrade/', QuartermasterSimulateUpgradeView.as_view(), name='quartermaster_simulate_upgrade'),
    path('quartermaster/<str:repo_id>/check-license-compliance/', QuartermasterLicenseComplianceView.as_view(), name='quartermaster_license_compliance'),
    path('quartermaster/<str:repo_id>/risk-report/', QuartermasterRiskReportView.as_view(), name='quartermaster_risk_report'),

    # --- LOREMASTER ENDPOINTS ---
    path('loremaster/<str:repo_id>/inventory/', LoremasterApiInventoryView.as_view(), name='loremaster_inventory'),
    path('loremaster/<str:repo_id>/openapi-spec/', LoremasterOpenApiSpecView.as_view(), name='loremaster_openapi_spec'),
    path('loremaster/<str:repo_id>/documentation/', LoremasterDocumentationView.as_view(), name='loremaster_documentation'),
    path('loremaster/<str:repo_id>/client-snippet/', LoremasterClientSnippetView.as_view(), name='loremaster_client_snippet'),

    # --- LIBRARIAN ENDPOINTS ---
    path('librarian/ingest/', LibrarianIngestView.as_view(), name='librarian_ingest'),
    path('librarian/archives/', LibrarianListArchivesView.as_view(), name='librarian_archives'),
    path('librarian/archives/<str:archive_id>/', LibrarianArchiveDetailView.as_view(), name='librarian_archive_detail'),
    path('librarian/ask/', LibrarianAskView.as_view(), name='librarian_ask'),


    # --- MODEL MANAGEMENT ENDPOINT ---
    path('models/list/', ListModelsView.as_view(), name='list_models'),

     # --- INGESTION ENDPOINT ---
    path('ingest/', IngestRepositoryView.as_view(), name='ingest_repository'),

     # --- ORACLE (Q&A) ENDPOINT ---
    path('oracle/ask/', OracleView.as_view(), name='oracle_ask'),

    # --- ONBOARDING CONCIERGE ENDPOINTS ---
    path('expertise/find/', ExpertiseView.as_view(), name='find_experts'),
    path('onboarding/guide/', OnboardingGuideView.as_view(), name='onboarding_guide'),

    # --- REPOSITORY MANAGEMENT ---
    path('repositories/', ListRepositoriesView.as_view(), name='repository-list'),
    path('repositories/<str:repo_id>/', RepositoryDetailView.as_view(), name='repository-detail'),
    path('repositories/<str:repo_id>/status/', RepositoryStatusView.as_view(), name='repository-status'),
    path('repositories/<str:repo_id>/api-endpoints/', ApiEndpointInventoryView.as_view(), name='api-endpoint-inventory'),


    # --- Graph ENDPOINT ---
    path('graph/<str:repo_id>/', GraphDataView.as_view(), name='graph_data'),

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

    # --- SENTINEL ENDPOINTS ---
    path('sentinel/briefing/<str:repo_id>/', SentinelBriefingView.as_view(), name='sentinel_briefing'),
    path('sentinel/metrics/<str:repo_id>/', SentinelMetricsHistoryView.as_view(), name='sentinel-metrics-history'),
]
