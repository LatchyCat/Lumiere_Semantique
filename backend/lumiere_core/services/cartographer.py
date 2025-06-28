# backend/lumiere_core/services/cartographer.py

import ast
import logging
import json
from collections import defaultdict
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


# --- Tree-sitter imports ---
try:
    from tree_sitter import Language, Parser, Node
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logging.warning("Tree-sitter not available. JavaScript analysis will be limited.")

# Configure logging
logger = logging.getLogger(__name__)


class TreeSitterConfig:
    """Centralized Tree-sitter configuration and initialization."""

    def __init__(self):
        # --- FIX: Calculate the absolute path to the library ---
        # This makes the path resilient to where the script is run from.
        # It finds the directory of this file (services/) and then navigates to the build directory.
        current_file_dir = Path(__file__).resolve().parent
        self.library_path = current_file_dir / 'build' / 'my-languages.so'

        self.js_language: Optional[Language] = None
        self.js_parser: Optional[Parser] = None
        self.is_ready = False

        if TREE_SITTER_AVAILABLE:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize Tree-sitter parser for JavaScript."""
        if not self.library_path.exists():
            logger.warning(f"Language library not found at {self.library_path}")
            logger.warning("Please run 'python build_parsers.py' from the 'backend' directory.")
            return

        try:
            self.js_language = Language(str(self.library_path), 'javascript')
            self.js_parser = Parser()
            self.js_parser.set_language(self.js_language)
            self.is_ready = True
            logger.info("✓ Tree-sitter JavaScript parser is ready.")
        except Exception as e:
            logger.error(f"Failed to load JavaScript parser: {e}")
            self.is_ready = False


# Global Tree-sitter configuration
ts_config = TreeSitterConfig()


class PythonCartographerVisitor(ast.NodeVisitor):
    """Enhanced AST visitor for Python with better error handling and organization."""

    def __init__(self):
        self.imports: List[Dict[str, Any]] = []
        self.function_calls: List[Dict[str, Any]] = []
        self.class_defs: List[Dict[str, Any]] = []
        self.function_defs: List[Dict[str, Any]] = []
        self.current_context_stack: List[str] = []  # Track nested contexts

    @property
    def current_class(self) -> Optional[str]:
        """Get the current class context."""
        for context in reversed(self.current_context_stack):
            if context.startswith('class:'):
                return context[6:]  # Remove 'class:' prefix
        return None

    def visit_Import(self, node: ast.Import) -> None:
        """Process direct imports."""
        for alias in node.names:
            self.imports.append({
                'type': 'direct',
                'name': alias.name,
                'alias': alias.asname,
                'line': node.lineno
            })
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Process from imports."""
        module = node.module or '.'
        for alias in node.names:
            self.imports.append({
                'type': 'from',
                'module': module,
                'name': alias.name,
                'alias': alias.asname,
                'line': node.lineno
            })
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Process class definitions with context tracking."""
        self.current_context_stack.append(f'class:{node.name}')

        self.class_defs.append({
            'name': node.name,
            'inherits_from': [safe_unparse(base) for base in node.bases],
            'line': node.lineno,
            'decorators': [safe_unparse(dec) for dec in node.decorator_list]
        })

        self.generic_visit(node)
        self.current_context_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Process function definitions."""
        self.function_defs.append({
            'name': node.name,
            'class_context': self.current_class,
            'line': node.lineno,
            'args': [arg.arg for arg in node.args.args],
            'decorators': [safe_unparse(dec) for dec in node.decorator_list]
        })
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Process function calls."""
        call_name = safe_unparse(node.func)
        self.function_calls.append({
            'name': call_name,
            'class_context': self.current_class,
            'line': node.lineno
        })
        self.generic_visit(node)


def safe_unparse(node: ast.AST) -> str:
    """Enhanced version of ast.unparse with better error handling."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        try:
            return f"{safe_unparse(node.value)}.{node.attr}"
        except Exception:
            return f"<complex>.{node.attr}"
    elif isinstance(node, ast.Constant):
        return str(node.value)

    try:
        return ast.unparse(node)
    except Exception as e:
        logger.debug(f"Failed to unparse AST node: {e}")
        return f"<unparseable:{type(node).__name__}>"


class JavaScriptMapper:
    """Handles JavaScript AST mapping with Tree-sitter."""

    QUERY_PATTERNS = {
        "requires": '(call_expression function: (identifier) @func (#eq? @func "require") arguments: (arguments (string (string_fragment) @module)))',
        "imports": '(import_statement source: (string (string_fragment) @module))',
        "functions": '(function_declaration name: (identifier) @name)',
        "arrow_functions": '(variable_declarator name: (identifier) @name value: (arrow_function))',
        "classes": '(class_declaration name: (identifier) @name)',
        "calls": '(call_expression function: [ (identifier) @name (member_expression property: (property_identifier) @name) ] )',
        "exports": '(export_statement declaration: (function_declaration name: (identifier) @name))'
    }

    @staticmethod
    def map_ast(file_path: str, content: str, nodes: defaultdict, edges: List[Dict]) -> None:
        """Maps JavaScript content to graph nodes and edges using Tree-sitter."""
        if not ts_config.is_ready:
            logger.warning(f"Skipping JavaScript analysis for {file_path} - parser not ready")
            return

        try:
            tree = ts_config.js_parser.parse(bytes(content, "utf8"))
            root_node = tree.root_node

            # Initialize file node
            nodes[file_path].update({
                "type": "file",
                "language": "javascript",
                "classes": [],
                "functions": [],
                "exports": []
            })

            JavaScriptMapper._process_queries(file_path, root_node, nodes, edges)

        except Exception as e:
            logger.error(f"Failed to parse JavaScript file {file_path}: {e}")

    @staticmethod
    def _process_queries(file_path: str, root_node: Node, nodes: defaultdict, edges: List[Dict]) -> None:
        """Process Tree-sitter queries for JavaScript analysis."""
        for query_name, pattern in JavaScriptMapper.QUERY_PATTERNS.items():
            try:
                query = ts_config.js_language.query(pattern)
                captures = query.captures(root_node)

                for node, name in captures:
                    text = node.text.decode('utf8')
                    JavaScriptMapper._handle_capture(query_name, name, text, file_path, nodes, edges)

            except Exception as e:
                logger.warning(f"Tree-sitter query '{query_name}' failed for {file_path}: {e}")

    @staticmethod
    def _handle_capture(query_name: str, capture_name: str, text: str,
                       file_path: str, nodes: defaultdict, edges: List[Dict]) -> None:
        """Handle individual Tree-sitter captures."""
        if capture_name == 'module':
            edges.append({"source": file_path, "target": text, "type": "IMPORTS"})
        elif capture_name == 'name':
            if query_name in ['functions', 'arrow_functions']:
                nodes[file_path]['functions'].append(text)
            elif query_name == 'classes':
                nodes[file_path]['classes'].append(text)
            elif query_name == 'calls':
                edges.append({"source": file_path, "target": text, "type": "CALLS"})
            elif query_name == 'exports':
                nodes[file_path]['exports'].append(text)


def map_python_ast(file_path: str, tree: ast.AST, nodes: defaultdict, edges: List[Dict]) -> None:
    """Enhanced Python AST mapping with better organization."""
    try:
        visitor = PythonCartographerVisitor()
        visitor.visit(tree)

        # Initialize file node
        nodes[file_path].update({
            "type": "file",
            "language": "python",
            "classes": [],
            "functions": [],
            "imports": len(visitor.imports)
        })

        # Process classes
        for class_def in visitor.class_defs:
            class_name = class_def['name']
            class_node_id = f"{file_path}::{class_name}"

            nodes[file_path]['classes'].append(class_name)
            nodes[class_node_id].update({
                "type": "class",
                "name": class_name,
                "file": file_path,
                "methods": [],
                "line": class_def['line'],
                "decorators": class_def['decorators']
            })

            # Add inheritance edges
            for base in class_def['inherits_from']:
                if base != '<unparseable:Name>':  # Skip unparseable bases
                    edges.append({"source": class_node_id, "target": base, "type": "INHERITS_FROM"})

        # Process functions and methods
        for func_def in visitor.function_defs:
            func_name = func_def['name']
            if func_def['class_context']:
                parent_class_id = f"{file_path}::{func_def['class_context']}"
                if parent_class_id in nodes:
                    nodes[parent_class_id]['methods'].append(func_name)
            else:
                nodes[file_path]['functions'].append(func_name)

        # Process imports
        for imp in visitor.imports:
            target = imp.get('module', imp['name'])
            edges.append({
                "source": file_path,
                "target": target,
                "type": "IMPORTS",
                "import_type": imp['type']
            })

        # Process function calls
        for call in visitor.function_calls:
            source_id = file_path
            if call['class_context']:
                source_id = f"{file_path}::{call['class_context']}"

            if source_id in nodes:
                edges.append({
                    "source": source_id,
                    "target": call['name'],
                    "type": "CALLS"
                })

    except Exception as e:
        logger.error(f"Failed to map Python AST for {file_path}: {e}")


def generate_graph(analyzed_files: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced orchestration of multi-language architectural analysis."""
    logger.info("--- ENHANCED POLYGLOT CARTOGRAPHER AGENT ACTIVATED ---")
    logger.info(f"   -> Mapping architecture for {len(analyzed_files)} files...")

    nodes = defaultdict(dict)
    edges = []
    languages_found = set()
    processing_stats = {"success": 0, "errors": 0}

    for file_path, analysis_data in analyzed_files.items():
        try:
            lang = analysis_data.get('language')
            if not lang:
                logger.warning(f"No language specified for {file_path}")
                continue

            languages_found.add(lang)
            logger.debug(f"      - Analyzing: {file_path} ({lang})")

            if lang == 'python' and 'ast' in analysis_data:
                map_python_ast(file_path, analysis_data['ast'], nodes, edges)
                processing_stats["success"] += 1
            elif lang == 'javascript' and 'content' in analysis_data:
                JavaScriptMapper.map_ast(file_path, analysis_data['content'], nodes, edges)
                processing_stats["success"] += 1
            else:
                logger.warning(f"Unsupported language or missing data for {file_path} ({lang})")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            processing_stats["errors"] += 1

    # Generate summary
    total_nodes = len(nodes)
    total_edges = len(edges)

    logger.info("✓ Enhanced Polyglot Cartographer mapping complete.")
    logger.info(f"   -> Generated {total_nodes} nodes and {total_edges} edges")
    logger.info(f"   -> Analyzed languages: {', '.join(filter(None, languages_found))}")
    logger.info(f"   -> Processing stats: {processing_stats['success']} successful, {processing_stats['errors']} errors")

    return {
        "nodes": dict(nodes),
        "edges": edges,
        "metadata": {
            "languages": list(languages_found),
            "stats": processing_stats,
            "tree_sitter_available": TREE_SITTER_AVAILABLE,
            "javascript_parser_ready": ts_config.is_ready
        }
    }


# Backward compatibility aliases
_PythonCartographerVisitor = PythonCartographerVisitor
_safe_unparse = safe_unparse
_map_python_ast = map_python_ast
_map_javascript_ast = JavaScriptMapper.map_ast

# Legacy global variables for backward compatibility
LANGUAGE_LIBRARY_PATH = ts_config.library_path
JS_LANGUAGE = ts_config.js_language
js_parser = ts_config.js_parser
JAVASCRIPT_PARSER_READY = ts_config.is_ready
