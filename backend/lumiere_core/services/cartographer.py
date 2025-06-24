# backend/lumiere_core/services/cartographer.py

import ast
import logging
from collections import defaultdict
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# --- Tree-sitter imports ---
try:
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logging.warning("Tree-sitter not available. JavaScript analysis will be limited.")

# --- Setup Tree-sitter for JavaScript ---
if TREE_SITTER_AVAILABLE:
    try:
        LANGUAGE_LIBRARY_PATH = 'build/my-languages.so'
        JS_LANGUAGE = Language(LANGUAGE_LIBRARY_PATH, 'javascript')
        js_parser = Parser()
        js_parser.set_language(JS_LANGUAGE)
        JAVASCRIPT_PARSER_READY = True
    except Exception as e:
        JAVASCRIPT_PARSER_READY = False
        logging.warning(f"Failed to initialize JavaScript parser: {e}")
else:
    JAVASCRIPT_PARSER_READY = False

# Configure logging
logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 0: ENHANCED DATA STRUCTURES AND ENUMS
# ==============================================================================

class NodeType(Enum):
    """Enumeration of node types in the architecture graph."""
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    MODULE = "module"
    INTERFACE = "interface"
    NAMESPACE = "namespace"


class EdgeType(Enum):
    """Enumeration of relationship types between nodes."""
    IMPORTS = "IMPORTS"
    INHERITS_FROM = "INHERITS_FROM"
    IMPLEMENTS = "IMPLEMENTS"
    CALLS = "CALLS"
    USES = "USES"
    EXTENDS = "EXTENDS"
    COMPOSES = "COMPOSES"
    DEPENDS_ON = "DEPENDS_ON"


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    UNKNOWN = "unknown"


@dataclass
class AnalysisMetadata:
    """Metadata for analysis results."""
    file_path: str
    language: Language
    lines_of_code: int = 0
    complexity_score: float = 0.0
    last_modified: Optional[str] = None
    analysis_timestamp: Optional[str] = None
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class GraphNode:
    """Enhanced node representation with metadata."""
    id: str
    type: NodeType
    name: str
    file_path: str
    language: Language
    metadata: Dict[str, Any] = None
    children: List[str] = None
    complexity: float = 0.0

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.children is None:
            self.children = []


@dataclass
class GraphEdge:
    """Enhanced edge representation with metadata."""
    source: str
    target: str
    type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = None
    confidence: float = 1.0

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ==============================================================================
# SECTION 1: ENHANCED LANGUAGE-SPECIFIC MAPPERS
# ==============================================================================

def _detect_language(file_path: str, content: str = "") -> Language:
    """Enhanced language detection based on file extension and content."""
    path = Path(file_path)
    extension = path.suffix.lower()

    extension_map = {
        '.py': Language.PYTHON,
        '.js': Language.JAVASCRIPT,
        '.jsx': Language.JAVASCRIPT,
        '.ts': Language.TYPESCRIPT,
        '.tsx': Language.TYPESCRIPT,
        '.java': Language.JAVA,
        '.cs': Language.CSHARP,
    }

    detected = extension_map.get(extension, Language.UNKNOWN)

    # Content-based detection for ambiguous cases
    if detected == Language.UNKNOWN and content:
        if content.strip().startswith('#!/usr/bin/env python') or 'import ' in content[:200]:
            detected = Language.PYTHON
        elif 'function ' in content[:200] or 'const ' in content[:200] or 'let ' in content[:200]:
            detected = Language.JAVASCRIPT

    return detected


def _calculate_complexity(tree: ast.AST) -> float:
    """Calculate cyclomatic complexity for Python AST."""
    complexity = 1  # Base complexity

    class ComplexityVisitor(ast.NodeVisitor):
        def __init__(self):
            self.complexity = 1

        def visit_If(self, node):
            self.complexity += 1
            self.generic_visit(node)

        def visit_While(self, node):
            self.complexity += 1
            self.generic_visit(node)

        def visit_For(self, node):
            self.complexity += 1
            self.generic_visit(node)

        def visit_Try(self, node):
            self.complexity += len(node.handlers)
            self.generic_visit(node)

        def visit_With(self, node):
            self.complexity += 1
            self.generic_visit(node)

    visitor = ComplexityVisitor()
    visitor.visit(tree)
    return visitor.complexity


def _map_python_ast(file_path: str, tree: ast.AST, nodes: defaultdict, edges: list) -> AnalysisMetadata:
    """Enhanced Python AST mapping with improved error handling and metadata."""
    metadata = AnalysisMetadata(file_path=file_path, language=Language.PYTHON)

    try:
        visitor = _PythonCartographerVisitor()
        visitor.visit(tree)

        # Calculate complexity
        complexity = _calculate_complexity(tree)
        metadata.complexity_score = complexity

        # Enhanced node creation with metadata
        nodes[file_path].update({
            "type": NodeType.FILE.value,
            "language": Language.PYTHON.value,
            "classes": [],
            "functions": [],
            "complexity": complexity,
            "imports_count": len(visitor.imports),
            "metadata": {
                "total_classes": len(visitor.class_defs),
                "total_functions": len(visitor.function_defs),
                "total_imports": len(visitor.imports)
            }
        })

        # Process imports with enhanced metadata
        for imp in visitor.imports:
            edge_metadata = {"import_type": imp['type']}
            if imp['type'] == 'from_import':
                edge_metadata["module"] = imp.get('module', '')

            edges.append({
                "source": file_path,
                "target": imp['name'],
                "type": EdgeType.IMPORTS.value,
                "metadata": edge_metadata,
                "confidence": 0.9
            })

        # Process classes with inheritance tracking
        for c_def in visitor.class_defs:
            class_node_id = f"{file_path}::{c_def['name']}"
            nodes[class_node_id].update({
                "type": NodeType.CLASS.value,
                "name": c_def['name'],
                "file": file_path,
                "methods": [],
                "language": Language.PYTHON.value,
                "metadata": {
                    "base_classes": c_def['inherits_from'],
                    "method_count": len([f for f in visitor.function_defs if f['class_context'] == c_def['name']])
                }
            })
            nodes[file_path]['classes'].append(c_def['name'])

            # Enhanced inheritance relationships
            for base in c_def['inherits_from']:
                edges.append({
                    "source": class_node_id,
                    "target": base,
                    "type": EdgeType.INHERITS_FROM.value,
                    "confidence": 0.95
                })

        # Process functions with context awareness
        for f_def in visitor.function_defs:
            func_metadata = {
                "is_method": bool(f_def['class_context']),
                "is_private": f_def['name'].startswith('_'),
                "is_dunder": f_def['name'].startswith('__') and f_def['name'].endswith('__')
            }

            if f_def['class_context']:
                parent_node_id = f"{file_path}::{f_def['class_context']}"
                if parent_node_id in nodes:
                    nodes[parent_node_id]['methods'].append(f_def['name'])

                    # Create method node
                    method_node_id = f"{parent_node_id}::{f_def['name']}"
                    nodes[method_node_id].update({
                        "type": NodeType.METHOD.value,
                        "name": f_def['name'],
                        "parent_class": f_def['class_context'],
                        "file": file_path,
                        "metadata": func_metadata
                    })
            else:
                nodes[file_path]['functions'].append(f_def['name'])

                # Create function node
                func_node_id = f"{file_path}::{f_def['name']}"
                nodes[func_node_id].update({
                    "type": NodeType.FUNCTION.value,
                    "name": f_def['name'],
                    "file": file_path,
                    "metadata": func_metadata
                })

        # Process function calls with improved context tracking
        for call in visitor.function_calls:
            source_context = f"::{call['class_context']}" if call['class_context'] else ""
            source_id = f"{file_path}{source_context}"
            target_name = call['name'].split('.')[-1]

            edges.append({
                "source": source_id,
                "target": target_name,
                "type": EdgeType.CALLS.value,
                "metadata": {
                    "full_call": call['name'],
                    "context": call['class_context']
                },
                "confidence": 0.8
            })

    except Exception as e:
        error_msg = f"Error processing Python AST for {file_path}: {str(e)}"
        logger.error(error_msg)
        metadata.errors.append(error_msg)

    return metadata


def _map_javascript_ast(file_path: str, content: str, nodes: defaultdict, edges: list) -> AnalysisMetadata:
    """Enhanced JavaScript AST mapping with fallback parsing."""
    metadata = AnalysisMetadata(file_path=file_path, language=Language.JAVASCRIPT)
    metadata.lines_of_code = len(content.splitlines())

    try:
        if JAVASCRIPT_PARSER_READY:
            _map_javascript_with_treesitter(file_path, content, nodes, edges, metadata)
        else:
            _map_javascript_with_regex(file_path, content, nodes, edges, metadata)
            metadata.warnings.append("Using regex-based fallback for JavaScript parsing")

    except Exception as e:
        error_msg = f"Error processing JavaScript for {file_path}: {str(e)}"
        logger.error(error_msg)
        metadata.errors.append(error_msg)

        # Fallback to regex parsing
        try:
            _map_javascript_with_regex(file_path, content, nodes, edges, metadata)
            metadata.warnings.append("Fell back to regex parsing due to Tree-sitter error")
        except Exception as fallback_error:
            metadata.errors.append(f"Fallback parsing also failed: {str(fallback_error)}")

    return metadata


def _map_javascript_with_treesitter(file_path: str, content: str, nodes: defaultdict, edges: list, metadata: AnalysisMetadata):
    """Tree-sitter based JavaScript parsing (original logic preserved)."""
    tree = js_parser.parse(bytes(content, "utf8"))
    root_node = tree.root_node

    nodes[file_path].update({
        "type": NodeType.FILE.value,
        "language": Language.JAVASCRIPT.value,
        "classes": [],
        "functions": [],
        "lines_of_code": metadata.lines_of_code
    })

    # Enhanced query patterns with better error handling
    query_patterns = {
        "requires": '(call_expression function: (identifier) @func (#eq? @func "require") arguments: (arguments (string (string_fragment) @module)))',
        "imports": '(import_statement source: (string (string_fragment) @module))',
        "functions": '(function_declaration name: (identifier) @name)',
        "arrow_functions": '(variable_declarator id: (identifier) @name value: (arrow_function))',
        "classes": '(class_declaration name: (identifier) @name)',
        "calls": '(call_expression function: [ (identifier) @name (member_expression property: (property_identifier) @name) ] )',
        "exports": '(export_statement)',
        "methods": '(method_definition name: (property_identifier) @name)'
    }

    function_count = 0
    class_count = 0
    import_count = 0

    for query_name, pattern in query_patterns.items():
        try:
            query = JS_LANGUAGE.query(pattern)
            captures = query.captures(root_node)

            for node, capture_name in captures:
                text = node.text.decode('utf8')

                if capture_name == 'module':
                    import_count += 1
                    edges.append({
                        "source": file_path,
                        "target": text,
                        "type": EdgeType.IMPORTS.value,
                        "metadata": {"import_style": query_name},
                        "confidence": 0.9
                    })
                elif capture_name == 'name':
                    if query_name in ['functions', 'arrow_functions']:
                        function_count += 1
                        nodes[file_path]['functions'].append(text)

                        # Create function node
                        func_node_id = f"{file_path}::{text}"
                        nodes[func_node_id].update({
                            "type": NodeType.FUNCTION.value,
                            "name": text,
                            "file": file_path,
                            "language": Language.JAVASCRIPT.value,
                            "metadata": {"function_type": "arrow" if "arrow" in query_name else "declaration"}
                        })

                    elif query_name == 'classes':
                        class_count += 1
                        nodes[file_path]['classes'].append(text)

                        # Create class node
                        class_node_id = f"{file_path}::{text}"
                        nodes[class_node_id].update({
                            "type": NodeType.CLASS.value,
                            "name": text,
                            "file": file_path,
                            "language": Language.JAVASCRIPT.value,
                            "methods": []
                        })

                    elif query_name == 'calls':
                        edges.append({
                            "source": file_path,
                            "target": text,
                            "type": EdgeType.CALLS.value,
                            "confidence": 0.8
                        })

                    elif query_name == 'methods':
                        # Find parent class for method
                        parent = node.parent
                        while parent and parent.type != 'class_declaration':
                            parent = parent.parent

                        if parent:
                            class_name_node = parent.child_by_field_name('name')
                            if class_name_node:
                                class_name = class_name_node.text.decode('utf8')
                                class_node_id = f"{file_path}::{class_name}"
                                if class_node_id in nodes:
                                    nodes[class_node_id]['methods'].append(text)

                                    # Create method node
                                    method_node_id = f"{class_node_id}::{text}"
                                    nodes[method_node_id].update({
                                        "type": NodeType.METHOD.value,
                                        "name": text,
                                        "parent_class": class_name,
                                        "file": file_path,
                                        "language": Language.JAVASCRIPT.value
                                    })

        except Exception as query_error:
            metadata.warnings.append(f"Query '{query_name}' failed: {str(query_error)}")
            continue

    # Update metadata
    nodes[file_path]['metadata'] = {
        "total_functions": function_count,
        "total_classes": class_count,
        "total_imports": import_count
    }

    metadata.complexity_score = function_count + class_count * 2  # Simple heuristic


def _map_javascript_with_regex(file_path: str, content: str, nodes: defaultdict, edges: list, metadata: AnalysisMetadata):
    """Regex-based fallback JavaScript parsing."""
    import re

    nodes[file_path].update({
        "type": NodeType.FILE.value,
        "language": Language.JAVASCRIPT.value,
        "classes": [],
        "functions": []
    })

    # Basic regex patterns for fallback parsing
    patterns = {
        'imports': [
            r"import\s+.*?from\s+['\"]([^'\"]+)['\"]",
            r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
        ],
        'functions': [
            r"function\s+(\w+)\s*\(",
            r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>"
        ],
        'classes': [r"class\s+(\w+)"],
        'calls': [r"(\w+)\s*\("]
    }

    function_count = 0
    class_count = 0
    import_count = 0

    for pattern_type, regex_list in patterns.items():
        for regex_pattern in regex_list:
            matches = re.findall(regex_pattern, content, re.MULTILINE)

            for match in matches:
                if pattern_type == 'imports':
                    import_count += 1
                    edges.append({
                        "source": file_path,
                        "target": match,
                        "type": EdgeType.IMPORTS.value,
                        "confidence": 0.7  # Lower confidence for regex
                    })
                elif pattern_type == 'functions':
                    function_count += 1
                    nodes[file_path]['functions'].append(match)
                elif pattern_type == 'classes':
                    class_count += 1
                    nodes[file_path]['classes'].append(match)
                elif pattern_type == 'calls':
                    edges.append({
                        "source": file_path,
                        "target": match,
                        "type": EdgeType.CALLS.value,
                        "confidence": 0.6  # Lower confidence for regex calls
                    })

    metadata.complexity_score = function_count + class_count * 2


# ==============================================================================
# SECTION 2: ENHANCED MAIN `generate_graph` FUNCTION
# ==============================================================================

def generate_graph(analyzed_files: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Enhanced orchestration of multi-language architectural analysis.

    Args:
        analyzed_files: Dictionary of file paths to analysis data
        config: Optional configuration for analysis behavior

    Returns:
        Enhanced graph structure with nodes, edges, and metadata
    """
    if config is None:
        config = {}

    # Configuration options
    include_private = config.get('include_private', True)
    min_confidence = config.get('min_confidence', 0.0)
    enable_metrics = config.get('enable_metrics', True)

    logger.info("--- ENHANCED POLYGLOT CARTOGRAPHER AGENT ACTIVATED ---")
    logger.info(f"   -> Mapping architecture for {len(analyzed_files)} files...")

    nodes = defaultdict(dict)
    edges = []
    analysis_metadata = []
    supported_languages = set()
    total_complexity = 0.0

    # Language-specific mappers registry
    mappers = {
        Language.PYTHON: _map_python_ast,
        Language.JAVASCRIPT: _map_javascript_ast,
        # Future languages can be registered here
    }

    for file_path, analysis_data in analyzed_files.items():
        # Enhanced language detection
        content = analysis_data.get('content', '')
        detected_language = _detect_language(file_path, content)

        # Override with provided language if available
        if 'language' in analysis_data:
            try:
                detected_language = Language(analysis_data['language'])
            except ValueError:
                logger.warning(f"Unknown language '{analysis_data['language']}' for {file_path}")

        supported_languages.add(detected_language)
        logger.info(f"      - Analyzing: {file_path} ({detected_language.value})")

        try:
            # Dispatch to appropriate mapper
            if detected_language in mappers:
                if detected_language == Language.PYTHON and analysis_data.get('ast'):
                    metadata = mappers[detected_language](file_path, analysis_data['ast'], nodes, edges)
                elif detected_language == Language.JAVASCRIPT and content:
                    metadata = mappers[detected_language](file_path, content, nodes, edges)
                else:
                    # Create basic node for unsupported analysis
                    metadata = AnalysisMetadata(file_path=file_path, language=detected_language)
                    metadata.warnings.append("Limited analysis - missing required data")
                    nodes[file_path].update({
                        "type": NodeType.FILE.value,
                        "language": detected_language.value,
                        "status": "limited_analysis"
                    })
            else:
                # Unsupported language - create basic node
                metadata = AnalysisMetadata(file_path=file_path, language=detected_language)
                metadata.warnings.append(f"Language {detected_language.value} not fully supported")
                nodes[file_path].update({
                    "type": NodeType.FILE.value,
                    "language": detected_language.value,
                    "status": "unsupported_language"
                })

            analysis_metadata.append(metadata)
            total_complexity += metadata.complexity_score

        except Exception as e:
            error_msg = f"Failed to process {file_path}: {str(e)}"
            logger.error(error_msg)

            # Create error node
            error_metadata = AnalysisMetadata(file_path=file_path, language=detected_language)
            error_metadata.errors.append(error_msg)
            analysis_metadata.append(error_metadata)

            nodes[file_path].update({
                "type": NodeType.FILE.value,
                "language": detected_language.value,
                "status": "analysis_failed",
                "error": str(e)
            })

    # Filter edges based on confidence threshold
    if min_confidence > 0.0:
        edges = [edge for edge in edges if edge.get('confidence', 1.0) >= min_confidence]
        logger.info(f"   -> Filtered edges with confidence >= {min_confidence}")

    # Filter private elements if requested
    if not include_private:
        filtered_nodes = {}
        for node_id, node_data in nodes.items():
            if not node_data.get('name', '').startswith('_'):
                filtered_nodes[node_id] = node_data
        nodes = filtered_nodes
        logger.info("   -> Filtered private elements")

    # Calculate project-wide metrics
    project_metrics = {}
    if enable_metrics:
        project_metrics = {
            "total_files": len(analyzed_files),
            "supported_languages": [lang.value for lang in supported_languages],
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "average_complexity": total_complexity / len(analyzed_files) if analyzed_files else 0.0,
            "total_complexity": total_complexity,
            "analysis_errors": sum(len(m.errors) for m in analysis_metadata),
            "analysis_warnings": sum(len(m.warnings) for m in analysis_metadata)
        }

    logger.info("✓ Enhanced Polyglot Cartographer mapping complete.")
    logger.info(f"   -> Generated {len(nodes)} nodes and {len(edges)} edges")
    logger.info(f"   -> Analyzed languages: {', '.join(lang.value for lang in supported_languages)}")

    # Enhanced return structure
    result = {
        "nodes": dict(nodes),
        "edges": edges,
        "metadata": {
            "analysis_metadata": [
                {
                    "file_path": m.file_path,
                    "language": m.language.value,
                    "complexity_score": m.complexity_score,
                    "lines_of_code": m.lines_of_code,
                    "errors": m.errors,
                    "warnings": m.warnings
                }
                for m in analysis_metadata
            ],
            "project_metrics": project_metrics,
            "config": config,
            "version": "2.0.0"
        }
    }

    return result


# ==============================================================================
# SECTION 3: ENHANCED HELPER CLASSES AND FUNCTIONS
# ==============================================================================

def _safe_unparse(node: ast.AST) -> str:
    """Enhanced robust version of ast.unparse with better type handling."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{_safe_unparse(node.value)}.{node.attr}"
    elif isinstance(node, ast.Constant):
        return str(node.value)
    elif isinstance(node, ast.Subscript):
        return f"{_safe_unparse(node.value)}[{_safe_unparse(node.slice)}]"
    elif isinstance(node, ast.Call):
        func_name = _safe_unparse(node.func)
        return f"{func_name}()"

    try:
        return ast.unparse(node)
    except Exception:
        return f"ComplexType_{type(node).__name__}"


class _PythonCartographerVisitor(ast.NodeVisitor):
    """Enhanced AST visitor for Python with improved analysis capabilities."""

    def __init__(self):
        self.imports = []
        self.function_calls = []
        self.class_defs = []
        self.function_defs = []
        self.current_class = None
        self.current_function = None
        self.decorators = []
        self.variables = []
        self.call_stack = []  # Track nested contexts

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            import_info = {
                "type": "direct_import",
                "name": alias.name,
                "alias": alias.asname,
                "lineno": node.lineno
            }
            self.imports.append(import_info)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or '.'
        for alias in node.names:
            import_info = {
                "type": "from_import",
                "module": module,
                "name": alias.name,
                "alias": alias.asname,
                "level": node.level,
                "lineno": node.lineno
            }
            self.imports.append(import_info)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        original_class = self.current_class
        self.current_class = node.name

        # Extract base classes and decorators
        base_classes = [_safe_unparse(b) for b in node.bases]
        decorators = [_safe_unparse(d) for d in node.decorator_list]

        class_info = {
            "name": node.name,
            "inherits_from": base_classes,
            "decorators": decorators,
            "lineno": node.lineno,
            "methods": [],
            "is_abstract": any("abstract" in dec.lower() for dec in decorators)
        }
        self.class_defs.append(class_info)

        self.generic_visit(node)
        self.current_class = original_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._visit_function_def(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._visit_function_def(node, is_async=True)

    def _visit_function_def(self, node, is_async=False):
        original_function = self.current_function
        self.current_function = node.name

        # Extract decorators and arguments
        decorators = [_safe_unparse(d) for d in node.decorator_list]
        args = [arg.arg for arg in node.args.args]

        function_info = {
            "name": node.name,
            "class_context": self.current_class,
            "decorators": decorators,
            "args": args,
            "is_async": is_async,
            "lineno": node.lineno,
            "is_property": any("property" in dec for dec in decorators),
            "is_staticmethod": any("staticmethod" in dec for dec in decorators),
            "is_classmethod": any("classmethod" in dec for dec in decorators)
        }
        self.function_defs.append(function_info)

        self.generic_visit(node)
        self.current_function = original_function

    def visit_Call(self, node: ast.Call):
        call_name = _safe_unparse(node.func)

        # Extract more call context
        call_info = {
            "name": call_name,
            "class_context": self.current_class,
            "function_context": self.current_function,
            "lineno": node.lineno,
            "args_count": len(node.args),
            "kwargs_count": len(node.keywords)
        }
        self.function_calls.append(call_info)

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        """Track variable assignments for better context."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_info = {
                    "name": target.id,
                    "class_context": self.current_class,
                    "function_context": self.current_function,
                    "lineno": node.lineno,
                    "value_type": type(node.value).__name__
                }
                self.variables.append(var_info)
        self.generic_visit(node)


# ==============================================================================
# SECTION 4: ADDITIONAL UTILITY FUNCTIONS
# ==============================================================================

def analyze_project_structure(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the project structure and provide architectural insights.

    Args:
        graph_data: The graph data returned by generate_graph()

    Returns:
        Dictionary containing structural analysis and insights
    """
    nodes = graph_data.get('nodes', {})
    edges = graph_data.get('edges', [])

    # Calculate various metrics
    file_nodes = {k: v for k, v in nodes.items() if v.get('type') == NodeType.FILE.value}
    class_nodes = {k: v for k, v in nodes.items() if v.get('type') == NodeType.CLASS.value}
    function_nodes = {k: v for k, v in nodes.items() if v.get('type') == NodeType.FUNCTION.value}

    # Language distribution
    language_dist = {}
    for node in file_nodes.values():
        lang = node.get('language', 'unknown')
        language_dist[lang] = language_dist.get(lang, 0) + 1

    # Dependency analysis
    import_edges = [e for e in edges if e.get('type') == EdgeType.IMPORTS.value]
    call_edges = [e for e in edges if e.get('type') == EdgeType.CALLS.value]
    inheritance_edges = [e for e in edges if e.get('type') == EdgeType.INHERITS_FROM.value]

    # Find highly connected nodes (potential architectural hotspots)
    node_connections = {}
    for edge in edges:
        source = edge.get('source', '')
        target = edge.get('target', '')
        node_connections[source] = node_connections.get(source, 0) + 1
        node_connections[target] = node_connections.get(target, 0) + 1

    hotspots = sorted(node_connections.items(), key=lambda x: x[1], reverse=True)[:10]

    # Calculate modularity metrics
    total_files = len(file_nodes)
    total_imports = len(import_edges)
    coupling_ratio = total_imports / total_files if total_files > 0 else 0

    # Identify potential issues
    issues = []
    if coupling_ratio > 5:
        issues.append("High coupling detected - consider reducing dependencies")

    circular_deps = _detect_circular_dependencies(edges)
    if circular_deps:
        issues.extend([f"Circular dependency detected: {' -> '.join(cycle)}" for cycle in circular_deps])

    return {
        "summary": {
            "total_files": total_files,
            "total_classes": len(class_nodes),
            "total_functions": len(function_nodes),
            "language_distribution": language_dist,
            "coupling_ratio": coupling_ratio
        },
        "dependencies": {
            "total_imports": total_imports,
            "total_calls": len(call_edges),
            "inheritance_relationships": len(inheritance_edges)
        },
        "hotspots": hotspots,
        "issues": issues,
        "recommendations": _generate_recommendations(graph_data)
    }


def _detect_circular_dependencies(edges: List[Dict[str, Any]]) -> List[List[str]]:
    """
    Detect circular dependencies in the graph using DFS.

    Args:
        edges: List of edge dictionaries

    Returns:
        List of circular dependency paths
    """
    # Build adjacency list for import relationships
    graph = defaultdict(list)
    for edge in edges:
        if edge.get('type') == EdgeType.IMPORTS.value:
            graph[edge['source']].append(edge['target'])

    def dfs(node, path, visited, rec_stack):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        cycles = []
        for neighbor in graph[node]:
            if neighbor in rec_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycles.append(path[cycle_start:] + [neighbor])
            elif neighbor not in visited:
                cycles.extend(dfs(neighbor, path[:], visited, rec_stack))

        rec_stack.remove(node)
        return cycles

    visited = set()
    all_cycles = []

    for node in graph:
        if node not in visited:
            all_cycles.extend(dfs(node, [], visited, set()))

    return all_cycles


def _generate_recommendations(graph_data: Dict[str, Any]) -> List[str]:
    """
    Generate architectural recommendations based on graph analysis.

    Args:
        graph_data: The complete graph data

    Returns:
        List of recommendation strings
    """
    recommendations = []

    nodes = graph_data.get('nodes', {})
    edges = graph_data.get('edges', [])
    metadata = graph_data.get('metadata', {})

    # Check for high complexity files
    project_metrics = metadata.get('project_metrics', {})
    avg_complexity = project_metrics.get('average_complexity', 0)

    if avg_complexity > 10:
        recommendations.append("Consider refactoring high-complexity files to improve maintainability")

    # Check for files with many dependencies
    file_import_counts = {}
    for edge in edges:
        if edge.get('type') == EdgeType.IMPORTS.value:
            source = edge['source']
            file_import_counts[source] = file_import_counts.get(source, 0) + 1

    max_imports = max(file_import_counts.values()) if file_import_counts else 0
    if max_imports > 15:
        recommendations.append("Some files have excessive imports - consider dependency injection or facade patterns")

    # Check for large classes
    large_classes = []
    for node_id, node_data in nodes.items():
        if node_data.get('type') == NodeType.CLASS.value:
            method_count = len(node_data.get('methods', []))
            if method_count > 20:
                large_classes.append(node_data.get('name', 'Unknown'))

    if large_classes:
        recommendations.append(f"Large classes detected ({', '.join(large_classes[:3])}) - consider splitting responsibilities")

    # Check language diversity
    languages = project_metrics.get('supported_languages', [])
    if len(languages) > 3:
        recommendations.append("Multiple languages detected - ensure proper integration patterns and documentation")

    # Check for missing documentation patterns
    analysis_warnings = project_metrics.get('analysis_warnings', 0)
    if analysis_warnings > len(nodes) * 0.1:  # More than 10% warning rate
        recommendations.append("High analysis warning rate - consider improving code documentation and structure")

    return recommendations


def export_graph_data(graph_data: Dict[str, Any], format_type: str = 'json', output_path: Optional[str] = None) -> str:
    """
    Export graph data in various formats.

    Args:
        graph_data: The graph data to export
        format_type: Export format ('json', 'dot', 'cytoscape')
        output_path: Optional file path to save the export

    Returns:
        Exported data as string
    """
    if format_type == 'json':
        return _export_as_json(graph_data, output_path)
    elif format_type == 'dot':
        return _export_as_dot(graph_data, output_path)
    elif format_type == 'cytoscape':
        return _export_as_cytoscape(graph_data, output_path)
    else:
        raise ValueError(f"Unsupported export format: {format_type}")


def _export_as_json(graph_data: Dict[str, Any], output_path: Optional[str]) -> str:
    """Export as JSON format."""
    import json

    json_str = json.dumps(graph_data, indent=2, ensure_ascii=False)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)

    return json_str


def _export_as_dot(graph_data: Dict[str, Any], output_path: Optional[str]) -> str:
    """Export as Graphviz DOT format."""
    nodes = graph_data.get('nodes', {})
    edges = graph_data.get('edges', [])

    dot_lines = ['digraph ArchitectureGraph {']
    dot_lines.append('  rankdir=TB;')
    dot_lines.append('  node [shape=box];')

    # Add nodes
    for node_id, node_data in nodes.items():
        node_type = node_data.get('type', 'unknown')
        node_name = node_data.get('name', node_id.split('::')[-1])
        language = node_data.get('language', 'unknown')

        # Color by type
        colors = {
            NodeType.FILE.value: 'lightblue',
            NodeType.CLASS.value: 'lightgreen',
            NodeType.FUNCTION.value: 'lightyellow',
            NodeType.METHOD.value: 'lightcoral'
        }
        color = colors.get(node_type, 'lightgray')

        safe_id = node_id.replace(':', '_').replace('/', '_').replace('.', '_')
        label = f"{node_name}\\n({node_type}, {language})"

        dot_lines.append(f'  {safe_id} [label="{label}", fillcolor="{color}", style=filled];')

    # Add edges
    for edge in edges:
        source = edge['source'].replace(':', '_').replace('/', '_').replace('.', '_')
        target = edge['target'].replace(':', '_').replace('/', '_').replace('.', '_')
        edge_type = edge.get('type', 'unknown')

        # Edge styles by type
        styles = {
            EdgeType.IMPORTS.value: 'solid',
            EdgeType.INHERITS_FROM.value: 'dashed',
            EdgeType.CALLS.value: 'dotted'
        }
        style = styles.get(edge_type, 'solid')

        dot_lines.append(f'  {source} -> {target} [label="{edge_type}", style="{style}"];')

    dot_lines.append('}')
    dot_content = '\n'.join(dot_lines)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(dot_content)

    return dot_content


def _export_as_cytoscape(graph_data: Dict[str, Any], output_path: Optional[str]) -> str:
    """Export as Cytoscape.js JSON format."""
    import json

    nodes = graph_data.get('nodes', {})
    edges = graph_data.get('edges', [])

    cytoscape_data = {
        "elements": {
            "nodes": [],
            "edges": []
        }
    }

    # Convert nodes
    for node_id, node_data in nodes.items():
        cyto_node = {
            "data": {
                "id": node_id,
                "label": node_data.get('name', node_id.split('::')[-1]),
                "type": node_data.get('type', 'unknown'),
                "language": node_data.get('language', 'unknown')
            }
        }
        cytoscape_data["elements"]["nodes"].append(cyto_node)

    # Convert edges
    for i, edge in enumerate(edges):
        cyto_edge = {
            "data": {
                "id": f"edge_{i}",
                "source": edge['source'],
                "target": edge['target'],
                "type": edge.get('type', 'unknown'),
                "weight": edge.get('weight', 1.0)
            }
        }
        cytoscape_data["elements"]["edges"].append(cyto_edge)

    json_str = json.dumps(cytoscape_data, indent=2)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)

    return json_str


# ==============================================================================
# SECTION 5: CONFIGURATION AND VALIDATION
# ==============================================================================

def validate_graph_data(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the integrity of graph data and return validation results.

    Args:
        graph_data: The graph data to validate

    Returns:
        Dictionary containing validation results and any issues found
    """
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "statistics": {}
    }

    nodes = graph_data.get('nodes', {})
    edges = graph_data.get('edges', [])

    # Check for orphaned edges
    node_ids = set(nodes.keys())
    orphaned_edges = []

    for i, edge in enumerate(edges):
        source = edge.get('source')
        target = edge.get('target')

        if source and source not in node_ids:
            orphaned_edges.append(f"Edge {i}: source '{source}' not found in nodes")

        # Note: targets might be external dependencies, so we don't validate them as strictly

    if orphaned_edges:
        validation_results["warnings"].extend(orphaned_edges)

    # Check for required node fields
    invalid_nodes = []
    for node_id, node_data in nodes.items():
        if not isinstance(node_data, dict):
            invalid_nodes.append(f"Node '{node_id}': data is not a dictionary")
            continue

        required_fields = ['type']
        missing_fields = [field for field in required_fields if field not in node_data]

        if missing_fields:
            invalid_nodes.append(f"Node '{node_id}': missing fields {missing_fields}")

    if invalid_nodes:
        validation_results["errors"].extend(invalid_nodes)
        validation_results["is_valid"] = False

    # Gather statistics
    validation_results["statistics"] = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "node_types": {},
        "edge_types": {},
        "languages": set()
    }

    # Count node types and languages
    for node_data in nodes.values():
        node_type = node_data.get('type', 'unknown')
        language = node_data.get('language', 'unknown')

        validation_results["statistics"]["node_types"][node_type] = \
            validation_results["statistics"]["node_types"].get(node_type, 0) + 1
        validation_results["statistics"]["languages"].add(language)

    # Count edge types
    for edge in edges:
        edge_type = edge.get('type', 'unknown')
        validation_results["statistics"]["edge_types"][edge_type] = \
            validation_results["statistics"]["edge_types"].get(edge_type, 0) + 1

    # Convert set to list for JSON serialization
    validation_results["statistics"]["languages"] = list(validation_results["statistics"]["languages"])

    return validation_results


def create_analysis_config(**kwargs) -> Dict[str, Any]:
    """
    Create a configuration dictionary for graph analysis.

    Keyword Args:
        include_private: Whether to include private methods/functions
        min_confidence: Minimum confidence threshold for edges
        enable_metrics: Whether to calculate project metrics
        max_depth: Maximum analysis depth for nested structures
        excluded_patterns: List of file patterns to exclude
        language_specific: Dictionary of language-specific options

    Returns:
        Configuration dictionary
    """
    default_config = {
        'include_private': True,
        'min_confidence': 0.0,
        'enable_metrics': True,
        'max_depth': 10,
        'excluded_patterns': ['*.pyc', '*.pyo', '__pycache__/*', 'node_modules/*'],
        'language_specific': {
            'python': {
                'include_decorators': True,
                'track_async_functions': True
            },
            'javascript': {
                'include_arrow_functions': True,
                'track_async_functions': True,
                'parse_jsx': False
            }
        }
    }

    # Update with provided kwargs
    config = default_config.copy()
    config.update(kwargs)

    return config
