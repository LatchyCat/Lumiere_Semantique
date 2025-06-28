# backend/lumiere_core/services/dance_service.py

import logging
import json
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, deque
from pathlib import Path
import xml.etree.ElementTree as ET
import time

from . import cortex_service, oracle_service
from .oracle_service import OracleService

logger = logging.getLogger(__name__)


class DanceService:
    """The Masked Dancer - reveals the secret choreography of running applications."""
    
    def __init__(self):
        self.oracle = OracleService()
    
    def trace_dance(self, repo_id: str, starting_node_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """
        Core function to trace execution flow starting from a given node.
        
        Args:
            repo_id: Repository identifier
            starting_node_id: The node to start tracing from
            max_depth: Maximum depth for the trace to prevent infinite recursion
            
        Returns:
            Dictionary containing trace steps and metadata
        """
        try:
            # Load cortex data
            cortex_data = cortex_service.load_cortex_data(repo_id)
            graph_data = cortex_data.get("architectural_graph")
            
            if not graph_data:
                return {"error": "No architectural graph found for this repository"}
            
            # Build networkx graph
            graph = self._build_graph(graph_data)
            
            if starting_node_id not in graph:
                return {"error": f"Starting node '{starting_node_id}' not found in graph"}
            
            # Perform depth-first search to trace execution flow
            trace_steps = []
            visited_in_path = set()
            recursion_stack = []
            
            self._dfs_trace(
                graph, starting_node_id, trace_steps, 
                visited_in_path, recursion_stack, 0, max_depth
            )
            
            return {
                "starting_node": starting_node_id,
                "trace_steps": trace_steps,
                "total_steps": len(trace_steps),
                "max_depth_reached": max(step.get("depth", 0) for step in trace_steps) if trace_steps else 0,
                "nodes_visited": len(set(step["target"] for step in trace_steps)),
                "has_recursion": any(step.get("is_recursive", False) for step in trace_steps)
            }
            
        except Exception as e:
            logger.error(f"Error tracing dance for {repo_id}: {e}")
            return {"error": str(e)}
    
    def _build_graph(self, graph_data: Dict[str, Any]) -> nx.DiGraph:
        """Build networkx graph from cortex data."""
        G = nx.DiGraph()
        
        # Add nodes
        nodes = graph_data.get("nodes", {})
        for node_id, node_data in nodes.items():
            G.add_node(node_id, **node_data)
        
        # Add edges
        edges = graph_data.get("edges", [])
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            edge_type = edge.get("type", "unknown")
            
            if source and target and source in G:
                G.add_edge(source, target, type=edge_type)
        
        return G
    
    def _dfs_trace(self, graph: nx.DiGraph, current_node: str, trace_steps: List[Dict],
                   visited_in_path: Set[str], recursion_stack: List[str], 
                   depth: int, max_depth: int):
        """
        Depth-first search to trace execution flow.
        """
        if depth >= max_depth:
            return
        
        if current_node in visited_in_path:
            # Recursion detected
            trace_steps.append({
                "source": recursion_stack[-1] if recursion_stack else current_node,
                "target": current_node,
                "depth": depth,
                "type": "RECURSIVE_CALL",
                "is_recursive": True
            })
            return
        
        visited_in_path.add(current_node)
        recursion_stack.append(current_node)
        
        # Get outgoing edges (function calls, dependencies, etc.)
        for target_node in graph.successors(current_node):
            edge_data = graph.get_edge_data(current_node, target_node)
            edge_type = edge_data.get("type", "CALLS") if edge_data else "CALLS"
            
            # Only follow certain edge types for execution flow
            if edge_type in ["CALLS", "INHERITS_FROM", "IMPLEMENTS", "USES"]:
                trace_steps.append({
                    "source": current_node,
                    "target": target_node,
                    "depth": depth + 1,
                    "type": edge_type,
                    "is_recursive": False
                })
                
                # Continue tracing
                self._dfs_trace(
                    graph, target_node, trace_steps, 
                    visited_in_path, recursion_stack, depth + 1, max_depth
                )
        
        visited_in_path.remove(current_node)
        recursion_stack.pop()
    
    def find_starting_node(self, repo_id: str, query: str) -> Dict[str, Any]:
        """
        Use The Oracle to identify the best starting function/method for a dance.
        
        Args:
            repo_id: Repository identifier
            query: Natural language description of what to trace
            
        Returns:
            Dictionary with suggested starting node and confidence
        """
        oracle_prompt = f"""Identify the single best starting function/method node in the graph for this concept: '{query}'.
        
        Look for:
        - API endpoints or route handlers
        - Main entry points
        - Controller methods
        - Service functions
        - Event handlers
        
        Return ONLY the node identifier (function name or class.method format).
        If you're not confident, return the most likely candidate with an explanation."""
        
        try:
            # Use Oracle to search for relevant code
            search_result = oracle_service.perform_semantic_search(
                repo_id, query, {"k": 10, "use_graph": True}
            )
            
            # Extract potential starting points from search results
            candidates = []
            for result in search_result.get("results", []):
                text = result.get("text", "")
                file_path = result.get("file_path", "")
                
                # Look for function definitions, class methods, API routes
                function_matches = self._extract_function_names(text)
                for func_name in function_matches:
                    candidates.append({
                        "node_id": func_name,
                        "file_path": file_path,
                        "confidence": result.get("relevance_score", 0.5),
                        "context": text[:200]
                    })
            
            if not candidates:
                return {"error": "No suitable starting points found for the query"}
            
            # Sort by confidence and return best match
            candidates.sort(key=lambda x: x["confidence"], reverse=True)
            best_candidate = candidates[0]
            
            return {
                "suggested_node": best_candidate["node_id"],
                "file_path": best_candidate["file_path"],
                "confidence": best_candidate["confidence"],
                "context": best_candidate["context"],
                "alternatives": candidates[1:5]  # Top 5 alternatives
            }
            
        except Exception as e:
            logger.error(f"Error finding starting node: {e}")
            return {"error": str(e)}
    
    def _extract_function_names(self, text: str) -> List[str]:
        """Extract function and method names from code text."""
        import re
        
        function_names = []
        
        # Python function definitions
        python_funcs = re.findall(r'def\s+([a-zA-Z_]\w*)\s*\(', text)
        function_names.extend(python_funcs)
        
        # Python class methods
        class_methods = re.findall(r'class\s+(\w+).*?def\s+([a-zA-Z_]\w*)', text, re.DOTALL)
        for class_name, method_name in class_methods:
            function_names.append(f"{class_name}.{method_name}")
        
        # JavaScript/TypeScript functions
        js_funcs = re.findall(r'function\s+([a-zA-Z_]\w*)\s*\(', text)
        function_names.extend(js_funcs)
        
        # API route decorators
        routes = re.findall(r'@app\.route\([\'"]([^\'\"]+)[\'"].*?\ndef\s+([a-zA-Z_]\w*)', text, re.DOTALL)
        for route, func_name in routes:
            function_names.append(func_name)
        
        return list(set(function_names))  # Remove duplicates
    
    def render_tree_cli(self, trace_data: Dict[str, Any]) -> str:
        """
        Render trace as a beautiful rich.Tree for CLI display.
        
        Args:
            trace_data: Result from trace_dance()
            
        Returns:
            Formatted tree string for console display
        """
        if "error" in trace_data:
            return f"❌ Dance failed: {trace_data['error']}"
        
        starting_node = trace_data.get("starting_node", "unknown")
        trace_steps = trace_data.get("trace_steps", [])
        
        if not trace_steps:
            return f"💃 The Dance of '{starting_node}' - No execution flow found"
        
        # Build tree structure
        tree_lines = [f"💃 The Dance of '{starting_node}'"]
        
        # Group steps by depth for proper tree rendering
        depth_groups = defaultdict(list)
        for step in trace_steps:
            depth_groups[step["depth"]].append(step)
        
        # Render tree with proper indentation
        for depth in sorted(depth_groups.keys()):
            for step in depth_groups[depth]:
                indent = "│   " * (depth - 1) + "├── " if depth > 0 else "└── "
                
                # Choose emoji based on edge type
                emoji = "📞"
                if step["type"] == "INHERITS_FROM":
                    emoji = "🧬"
                elif step["type"] == "IMPLEMENTS":
                    emoji = "⚙️"
                elif step["type"] == "USES":
                    emoji = "🔗"
                elif step.get("is_recursive", False):
                    emoji = "🔄"
                
                target = step["target"]
                edge_type = step["type"].lower().replace("_", " ")
                
                tree_lines.append(f"{indent}{emoji} {edge_type} {target}")
        
        # Add metadata
        metadata = [
            f"",
            f"📊 Dance Statistics:",
            f"   • Total steps: {trace_data.get('total_steps', 0)}",
            f"   • Max depth: {trace_data.get('max_depth_reached', 0)}",
            f"   • Nodes visited: {trace_data.get('nodes_visited', 0)}",
            f"   • Has recursion: {'Yes' if trace_data.get('has_recursion', False) else 'No'}"
        ]
        
        return "\n".join(tree_lines + metadata)
    
    def generate_svg_animation(self, trace_data: Dict[str, Any], 
                              output_file: Optional[str] = None) -> str:
        """
        Generate an animated SVG visualization of the execution flow.
        
        Args:
            trace_data: Result from trace_dance()
            output_file: Optional file path to save the SVG
            
        Returns:
            SVG content as string
        """
        if "error" in trace_data:
            return f"<svg><text>Error: {trace_data['error']}</text></svg>"
        
        starting_node = trace_data.get("starting_node", "unknown")
        trace_steps = trace_data.get("trace_steps", [])
        
        # Collect all unique nodes
        all_nodes = set([starting_node])
        for step in trace_steps:
            all_nodes.add(step["source"])
            all_nodes.add(step["target"])
        
        # Layout nodes in a hierarchical structure
        node_positions = self._calculate_node_positions(list(all_nodes), trace_steps)
        
        # Generate SVG
        svg_content = self._build_svg_content(
            starting_node, trace_steps, node_positions
        )
        
        # Save to file if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(svg_content)
                logger.info(f"SVG animation saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save SVG: {e}")
        
        return svg_content
    
    def _calculate_node_positions(self, nodes: List[str], 
                                 trace_steps: List[Dict]) -> Dict[str, Tuple[int, int]]:
        """Calculate positions for nodes in the SVG layout."""
        positions = {}
        
        # Simple hierarchical layout
        levels = defaultdict(list)
        node_depths = {}
        
        # Calculate depth for each node based on trace
        for step in trace_steps:
            depth = step.get("depth", 0)
            target = step["target"]
            if target not in node_depths or depth < node_depths[target]:
                node_depths[target] = depth
        
        # Group nodes by depth level
        for node in nodes:
            depth = node_depths.get(node, 0)
            levels[depth].append(node)
        
        # Position nodes
        y_spacing = 80
        x_spacing = 200
        
        for depth, level_nodes in levels.items():
            y = depth * y_spacing + 50
            x_start = 50
            
            for i, node in enumerate(level_nodes):
                x = x_start + (i * x_spacing)
                positions[node] = (x, y)
        
        return positions
    
    def _build_svg_content(self, starting_node: str, trace_steps: List[Dict],
                          node_positions: Dict[str, Tuple[int, int]]) -> str:
        """Build the complete SVG content with animations."""
        
        # Calculate SVG dimensions
        max_x = max(pos[0] for pos in node_positions.values()) + 150
        max_y = max(pos[1] for pos in node_positions.values()) + 100
        
        svg_parts = [
            f'<svg width="{max_x}" height="{max_y}" xmlns="http://www.w3.org/2000/svg">',
            '<style>',
            '.node { fill: #4a90e2; stroke: #2c5aa0; stroke-width: 2; }',
            '.node-text { fill: white; font-family: Arial; font-size: 12px; text-anchor: middle; }',
            '.edge { stroke: #666; stroke-width: 2; fill: none; opacity: 0; }',
            '.edge-animated { stroke-dasharray: 5,5; }',
            '@keyframes flow { from { stroke-dashoffset: 10; } to { stroke-dashoffset: 0; } }',
            '</style>',
            '',
            '<!-- Nodes -->'
        ]
        
        # Add nodes
        for node, (x, y) in node_positions.items():
            node_id = node.replace('.', '_').replace('/', '_')
            svg_parts.extend([
                f'<g id="node_{node_id}">',
                f'  <circle cx="{x}" cy="{y}" r="25" class="node"/>',
                f'  <text x="{x}" y="{y + 5}" class="node-text">{node[:10]}</text>',
                '</g>'
            ])
        
        svg_parts.append('\n<!-- Edges -->')
        
        # Add edges with animation
        for i, step in enumerate(trace_steps):
            source = step["source"]
            target = step["target"]
            
            if source in node_positions and target in node_positions:
                x1, y1 = node_positions[source]
                x2, y2 = node_positions[target]
                
                edge_id = f"edge_{i}"
                animation_delay = i * 0.5  # Stagger animations
                
                svg_parts.extend([
                    f'<path id="{edge_id}" d="M {x1} {y1} L {x2} {y2}" class="edge edge-animated">',
                    f'  <animate attributeName="opacity" values="0;1;1" dur="0.5s" begin="{animation_delay}s" fill="freeze"/>',
                    f'  <animateTransform attributeName="stroke-dashoffset" values="10;0" dur="1s" begin="{animation_delay}s" fill="freeze"/>',
                    '</path>'
                ])
        
        # Add JavaScript for interactivity
        svg_parts.extend([
            '',
            '<script type="text/javascript"><![CDATA[',
            '// SVG is self-contained and animated',
            'console.log("Lumière Sémantique - Dance visualization loaded");',
            ']]></script>',
            '</svg>'
        ])
        
        return '\n'.join(svg_parts)


# Global instance
_dance_service = None

def get_dance_service() -> DanceService:
    """Get or create the global Dance service instance."""
    global _dance_service
    if _dance_service is None:
        _dance_service = DanceService()
    return _dance_service

# Public API
def trace_execution_flow(repo_id: str, starting_point: str, max_depth: int = 10) -> Dict[str, Any]:
    """
    Public API to trace execution flow starting from a given point.
    """
    service = get_dance_service()
    return service.trace_dance(repo_id, starting_point, max_depth)

def find_entry_point(repo_id: str, description: str) -> Dict[str, Any]:
    """
    Public API to find the best entry point for tracing based on description.
    """
    service = get_dance_service()
    return service.find_starting_node(repo_id, description)

def visualize_dance(repo_id: str, starting_point: str, format_type: str = "cli", 
                   output_file: Optional[str] = None) -> str:
    """
    Public API to visualize execution flow.
    
    Args:
        repo_id: Repository identifier
        starting_point: Function/method to start tracing from
        format_type: "cli" for rich.Tree or "svg" for animated SVG
        output_file: Optional file path for SVG output
        
    Returns:
        Formatted visualization string
    """
    service = get_dance_service()
    
    # Get trace data
    trace_data = service.trace_dance(repo_id, starting_point)
    
    if format_type == "svg":
        return service.generate_svg_animation(trace_data, output_file)
    else:
        return service.render_tree_cli(trace_data)