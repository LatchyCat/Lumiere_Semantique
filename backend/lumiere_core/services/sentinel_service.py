# backend/lumiere_core/services/sentinel_service.py

import os
import ast
import logging
import networkx as nx
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

# --- Constants ---
TEST_KEYWORDS = ("test", "spec")
IMPL_PREFIXES = ("src/", "lib/")
IMPL_SUFFIXES = (".py", ".js", ".rs", ".go")


# --- Metric Calculation Helpers ---

def _calculate_code_to_test_ratio(files_data: List[Dict[str, str]]) -> float:
    """Calculate the ratio of test files to implementation files."""
    test_files = 0
    impl_files = 0

    for file_info in files_data:
        path = file_info.get("file_path", "").lower()
        if any(keyword in path for keyword in TEST_KEYWORDS):
            test_files += 1
        elif path.startswith(IMPL_PREFIXES) or path.endswith(IMPL_SUFFIXES):
            impl_files += 1

    if impl_files == 0:
        return 0.0
    return round(test_files / impl_files, 3)


def _has_docstring(node: ast.AST) -> bool:
    """Check if the first statement of a node is a docstring."""
    if hasattr(node, "body") and node.body:
        first_stmt = node.body[0]
        return isinstance(first_stmt, ast.Expr) and isinstance(getattr(first_stmt, "value", None), ast.Str)
    return False


def _calculate_documentation_coverage(graph: nx.DiGraph) -> float:
    """Calculate the percentage of functions, methods, and classes with a docstring."""
    documented = 0
    total = 0

    for _, data in graph.nodes(data=True):
        if data.get("type") in ("function", "method", "class"):
            total += 1
            raw = data.get("raw_content", "")
            if isinstance(raw, str):
                try:
                    parsed = ast.parse(raw)
                    if parsed.body and _has_docstring(parsed.body[0]):
                        documented += 1
                except (SyntaxError, IndexError, TypeError) as e:
                    logger.debug(f"Docstring parse error: {e}")

    if total == 0:
        return 0.0
    return round((documented / total) * 100, 2)


def _build_graph(graph_data: Dict[str, Any]) -> nx.DiGraph:
    """Builds a NetworkX directed graph from graph_data dictionary."""
    graph = nx.DiGraph()
    nodes = graph_data.get("nodes", {})
    edges = graph_data.get("edges", [])

    graph.add_nodes_from(nodes.keys())
    graph.add_edges_from((e["source"], e["target"]) for e in edges if "source" in e and "target" in e)

    return graph


# --- Main Service Function ---

def calculate_snapshot_metrics(repo_path: Path, graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates a snapshot of project health metrics at a given point in time.

    Args:
        repo_path: Root path of the repository.
        graph_data: Architectural graph data from the cartographer.

    Returns:
        A dictionary with calculated health metrics.
    """
    logger.info(f"Sentinel: Calculating snapshot metrics for {repo_path}")
    metrics: Dict[str, Any] = {}

    # --- Architectural Metrics ---
    graph = _build_graph(graph_data) if graph_data else nx.DiGraph()

    metrics["total_nodes"] = graph.number_of_nodes()
    metrics["total_edges"] = graph.number_of_edges()
    metrics["coupling_factor"] = (
        round(metrics["total_edges"] / metrics["total_nodes"], 3)
        if metrics["total_nodes"] > 0 else 0
    )
    metrics["average_cyclomatic_complexity"] = 5.0  # Placeholder

    # --- Code Quality Metrics ---
    all_files_info = [
        {"file_path": str(Path(root) / file)}
        for root, _, files in os.walk(repo_path)
        for file in files
    ]

    metrics["code_to_test_ratio"] = _calculate_code_to_test_ratio(all_files_info)
    metrics["documentation_coverage"] = _calculate_documentation_coverage(graph)
    metrics["timestamp"] = datetime.utcnow().isoformat()

    logger.info(f"Sentinel: Metrics calculated: {metrics}")
    return metrics
