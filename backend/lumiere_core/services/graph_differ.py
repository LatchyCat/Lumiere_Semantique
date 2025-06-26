# backend/lumiere_core/services/graph_differ.py

import networkx as nx
from typing import Dict, List, Tuple, Iterable


def compare_graphs(base_graph: nx.DiGraph, head_graph: nx.DiGraph) -> Dict[str, List[str]]:
    """
    Compares two architectural graphs (base vs. head) and identifies changes.

    Args:
        base_graph (nx.DiGraph): Graph from the base branch.
        head_graph (nx.DiGraph): Graph from the head branch (e.g. a PR).

    Returns:
        Dict[str, List[str]]: A dictionary with:
            - 'nodes_added': New nodes in the head graph.
            - 'nodes_removed': Nodes missing from the head graph.
            - 'edges_added': New edges (formatted as 'A -> B').
            - 'edges_removed': Missing edges (formatted as 'A -> B').
    """
    def format_edges(edges: Iterable[Tuple[str, str]]) -> List[str]:
        return [f"{u} -> {v}" for u, v in sorted(edges)]

    base_nodes = set(base_graph.nodes)
    head_nodes = set(head_graph.nodes)

    base_edges = set(base_graph.edges)
    head_edges = set(head_graph.edges)

    return {
        "nodes_added": sorted(head_nodes - base_nodes),
        "nodes_removed": sorted(base_nodes - head_nodes),
        "edges_added": format_edges(head_edges - base_edges),
        "edges_removed": format_edges(base_edges - head_edges),
    }
