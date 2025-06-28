# backend/lumiere_core/services/cortex_service.py

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Constants
CORTEX_FILENAME_TEMPLATE = "{repo_id}_cortex.json"
CLONED_REPOS_DIR = "cloned_repositories"


class CortexFileNotFound(Exception):
    """Raised when the Cortex file is missing for a given repository."""


class CortexFileMalformed(Exception):
    """Raised when the Cortex file is unreadable or not valid JSON."""


def _get_cortex_path(repo_id: str) -> Path:
    """
    Constructs the full path to a repository's Cortex file.

    Args:
        repo_id: The unique ID of the repository.

    Returns:
        Path object pointing to the expected Cortex file location.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    return base_dir / CLONED_REPOS_DIR / repo_id / CORTEX_FILENAME_TEMPLATE.format(repo_id=repo_id)


def load_cortex_data(repo_id: str) -> Dict[str, Any]:
    """
    Loads and parses the cortex JSON file for a given repository.

    Args:
        repo_id: The unique ID of the repository.

    Returns:
        Parsed JSON data as a dictionary.

    Raises:
        CortexFileNotFound: If the cortex file does not exist.
        CortexFileMalformed: If the file is not valid JSON.
    """
    cortex_path = _get_cortex_path(repo_id)

    if not cortex_path.exists():
        logger.error(f"Cortex file not found at: {cortex_path}")
        raise CortexFileNotFound(f"Cortex file not found for repo: {repo_id}")

    try:
        with cortex_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.exception(f"Failed to parse cortex file: {cortex_path}")
        raise CortexFileMalformed(f"Failed to load or parse cortex file: {e}") from e


def get_file_content(repo_id: str, file_path: str) -> Optional[str]:
    """
    Retrieves the raw content of a specific file from the repository's Cortex data.

    Args:
        repo_id: The unique ID of the repository.
        file_path: The relative path of the file within the repo.

    Returns:
        The raw file content as a string, or None if the file isn't found.
    """
    try:
        cortex_data = load_cortex_data(repo_id)
        for file_entry in cortex_data.get("files", []):
            if file_entry.get("file_path") == file_path:
                return file_entry.get("raw_content")
    except (CortexFileNotFound, CortexFileMalformed):
        return None

    return None


def get_node_line_map(repo_id: str) -> Dict[str, List[Dict]]:
    """
    Creates a map from file paths to a list of their contained nodes with line numbers.
    This is a critical helper for the Adjudicator's diff parser.

    Args:
        repo_id: The unique ID of the repository.

    Returns:
        A dictionary mapping filenames to lists of node info.
        e.g. {'src/main.py': [{'id': '...', 'start_line': 10, 'end_line': 25}]}
    """
    node_map = {}
    try:
        cortex_data = load_cortex_data(repo_id)
        graph = cortex_data.get("architectural_graph", {})
        nodes = graph.get("nodes", {})

        for node_id, node_data in nodes.items():
            # We are interested in nodes that have line numbers and a file path.
            # This typically means functions, classes, and methods.
            start_line = node_data.get("start_line")
            end_line = node_data.get("end_line")
            # The 'file' attribute is present on class/function/method nodes.
            file_path = node_data.get("file")

            if start_line and end_line and file_path:
                if file_path not in node_map:
                    node_map[file_path] = []

                node_map[file_path].append({
                    "id": node_id,
                    "start_line": start_line,
                    "end_line": end_line,
                })
    except (CortexFileNotFound, CortexFileMalformed) as e:
        logger.error(f"Could not generate node line map for {repo_id}: {e}")
        return {}

    return node_map


def get_bom_data(repo_id: str, format_type: str = "json") -> Optional[Dict[str, Any]]:
    """
    Get BOM data for a repository with different format options.

    Args:
        repo_id: Repository identifier
        format_type: Format type (json, summary, detailed)

    Returns:
        Dict containing BOM data in requested format or None if not available
    """
    try:
        cortex_data = load_cortex_data(repo_id)
        bom_data = cortex_data.get('tech_stack_bom')

        if not bom_data:
            return None

        if format_type == "summary":
            return {
                "repo_id": repo_id,
                "summary": bom_data.get('summary', {}),
                "primary_ecosystems": list(set(dep.get('ecosystem', 'unknown')
                                            for deps in bom_data.get('dependencies', {}).values()
                                            for dep in deps)),
                "service_count": len(bom_data.get('services', [])),
                "last_updated": bom_data.get('summary', {}).get('last_updated'),
                "generation_status": cortex_data.get('bom_generation_status', 'unknown')
            }

        elif format_type == "detailed":
            # Add enhanced analysis
            enhanced_bom = bom_data.copy()
            enhanced_bom['analysis'] = {
                'dependency_health': _analyze_dependency_health(bom_data),
                'security_insights': _generate_security_insights(bom_data),
                'architecture_patterns': _detect_architecture_patterns(bom_data),
                'modernization_opportunities': _suggest_modernization(bom_data)
            }
            enhanced_bom['repo_id'] = repo_id
            enhanced_bom['generation_status'] = cortex_data.get('bom_generation_status', 'unknown')
            return enhanced_bom

        # Default json format
        bom_data_copy = bom_data.copy()
        bom_data_copy['repo_id'] = repo_id
        bom_data_copy['generation_status'] = cortex_data.get('bom_generation_status', 'unknown')
        return bom_data_copy

    except (CortexFileNotFound, CortexFileMalformed):
        return None

def has_bom_data(repo_id: str) -> bool:
    """
    Check if repository has BOM data available.

    Args:
        repo_id: Repository identifier

    Returns:
        True if BOM data exists, False otherwise
    """
    try:
        cortex_data = load_cortex_data(repo_id)
        return 'tech_stack_bom' in cortex_data
    except (CortexFileNotFound, CortexFileMalformed):
        return False

def get_repository_metadata(repo_id: str) -> Optional[Dict[str, Any]]:
    """
    Get repository metadata including BOM summary if available.

    Args:
        repo_id: Repository identifier

    Returns:
        Metadata dictionary or None if repository not found
    """
    try:
        cortex_data = load_cortex_data(repo_id)

        metadata = {
            'repo_id': repo_id,
            'has_bom': 'tech_stack_bom' in cortex_data,
            'bom_status': cortex_data.get('bom_generation_status', 'not_available'),
            'version': cortex_data.get('version', '1.0.0'),
            'file_count': len(cortex_data.get('files', []))
        }

        # Add BOM summary if available
        if 'bom_summary' in cortex_data:
            metadata['bom_summary'] = cortex_data['bom_summary']
        elif 'tech_stack_bom' in cortex_data:
            bom_data = cortex_data['tech_stack_bom']
            metadata['bom_summary'] = {
                'primary_language': bom_data.get('summary', {}).get('primary_language', 'Unknown'),
                'total_dependencies': bom_data.get('summary', {}).get('total_dependencies', 0),
                'total_services': bom_data.get('summary', {}).get('total_services', 0),
                'ecosystems': bom_data.get('summary', {}).get('ecosystems', [])
            }

        return metadata

    except (CortexFileNotFound, CortexFileMalformed):
        return None

# Helper functions for BOM analysis
def _analyze_dependency_health(bom_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the health of dependencies."""
    all_deps = []
    for deps in bom_data.get('dependencies', {}).values():
        all_deps.extend(deps)

    total_deps = len(all_deps)
    outdated_count = sum(1 for dep in all_deps if dep.get('deprecated', False))

    return {
        'total_dependencies': total_deps,
        'potentially_outdated': outdated_count,
        'health_score': max(0, 100 - (outdated_count / total_deps * 100)) if total_deps > 0 else 100,
        'ecosystems': list(set(dep.get('ecosystem', 'unknown') for dep in all_deps))
    }

def _generate_security_insights(bom_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate security insights from BOM data."""
    security_analysis = bom_data.get('security_analysis', {})
    return {
        'risk_level': 'low',
        'vulnerable_dependencies': security_analysis.get('high_risk_dependencies', 0),
        'recommendations': [
            'Enable automated dependency scanning',
            'Set up security alerts for new vulnerabilities',
            'Regularly update dependencies'
        ]
    }

def _detect_architecture_patterns(bom_data: Dict[str, Any]) -> List[str]:
    """Detect architectural patterns from BOM data."""
    patterns = []
    services = bom_data.get('services', [])
    service_types = [s.get('service_type') for s in services]

    if 'database' in service_types and 'cache' in service_types:
        patterns.append('Caching Layer')
    if 'message_queue' in service_types:
        patterns.append('Message-Driven Architecture')
    if len([s for s in services if s.get('service_type') == 'database']) > 1:
        patterns.append('Polyglot Persistence')
    if len(services) > 3:
        patterns.append('Microservices Architecture')

    return patterns

def _suggest_modernization(bom_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Suggest modernization opportunities."""
    suggestions = []
    all_deps = []
    for deps in bom_data.get('dependencies', {}).values():
        all_deps.extend(deps)

    # Check for outdated frameworks
    python_deps = [d for d in all_deps if d.get('ecosystem') == 'python']
    if any(d.get('name') == 'Django' and d.get('version', '').startswith('2.') for d in python_deps):
        suggestions.append({
            'type': 'framework_upgrade',
            'title': 'Upgrade Django to version 4.x',
            'description': 'Django 2.x is no longer supported. Consider upgrading to Django 4.x for security updates and new features.'
        })

    # Check for containerization opportunities
    if not bom_data.get('infrastructure', {}).get('containerized', False):
        suggestions.append({
            'type': 'containerization',
            'title': 'Consider containerizing your application',
            'description': 'Containerization can improve deployment consistency and scalability.'
        })

    return suggestions
