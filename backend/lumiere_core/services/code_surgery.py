# backend/lumiere_core/services/code_surgery.py

import re
import ast
from typing import Dict, Tuple, Optional, List, Set, Any

# ==============================================================================
# SECTION 1: CORE EXECUTOR / ROUTER
# ==============================================================================

def execute_surgical_plan(
    original_contents: Dict[str, str],
    plan: List[Dict[str, Any]]
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Main entry point for the Code Surgery agent.
    Iterates through a surgical plan and applies the specified operations.
    """
    modified_contents = original_contents.copy()
    report = {
        "operations_attempted": len(plan),
        "operations_succeeded": 0,
        "operations_failed": 0,
        "errors": []
    }

    for op in plan:
        operation_type = op.get("operation")
        file_path = op.get("file_path")

        try:
            if operation_type == "CREATE_FILE":
                modified_contents[file_path] = _handle_create_file(op)

            elif operation_type == "REPLACE_BLOCK":
                current_content = modified_contents.get(file_path, "")
                modified_contents[file_path] = _handle_replace_block(current_content, op)

            elif operation_type == "ADD_FIELD_TO_STRUCT":
                current_content = modified_contents.get(file_path, "")
                modified_contents[file_path] = _handle_add_field_to_struct(current_content, op)

            else:
                raise ValueError(f"Unknown operation type: '{operation_type}'")

            print(f"  ✓ Operation '{operation_type}' on '{file_path}' succeeded.")
            report["operations_succeeded"] += 1

        except Exception as e:
            error_msg = f"Operation '{operation_type}' on '{file_path}' failed: {e}"
            print(f"  ❌ {error_msg}")
            report["errors"].append(error_msg)
            report["operations_failed"] += 1

    return modified_contents, report


# ==============================================================================
# SECTION 2: OPERATION HANDLERS
# ==============================================================================

def _handle_create_file(operation: Dict[str, Any]) -> str:
    """Handles the CREATE_FILE operation. Returns the new file content."""
    return operation.get("content", "")

def _handle_replace_block(original_content: str, operation: Dict[str, Any]) -> str:
    """Handles the REPLACE_BLOCK operation for functions, methods, or classes."""
    target_id = operation.get("target_identifier")
    new_content = operation.get("content", "")
    file_path = operation.get("file_path", "")

    language = 'python' if file_path.endswith('.py') else 'rust' if file_path.endswith('.rs') else 'markdown' if file_path.endswith('.md') else 'javascript'

    boundaries = _find_block_boundaries(original_content, target_id, language)
    if not boundaries:
        raise ValueError(f"Could not find function/block '{target_id}' to replace.")

    start_pos, end_pos = boundaries
    # Ensure a newline after the replacement, unless it's the end of the file
    new_code_with_newline = new_content + "\n" if end_pos < len(original_content) else new_content
    return original_content[:start_pos] + new_code_with_newline + original_content[end_pos:]

def _handle_add_field_to_struct(original_content: str, operation: Dict[str, Any]) -> str:
    """Handles the ADD_FIELD_TO_STRUCT operation, specifically for Rust/C-like languages."""
    target_id = operation.get("target_identifier")
    new_field_line = operation.get("content", "")

    pattern = re.compile(
        rf"(struct\s+{re.escape(target_id)}\s*{{)(.*?)(\}})",
        re.DOTALL | re.MULTILINE
    )
    match = pattern.search(original_content)

    if not match:
        raise ValueError(f"Could not find struct definition for '{target_id}'.")

    struct_header, struct_body, struct_footer = match.groups()
    lines = struct_body.strip().split('\n')
    indentation = "    "
    if lines and lines[-1].strip():
        last_line = lines[-1]
        indentation = " " * (len(last_line) - len(last_line.lstrip()))

    new_body = struct_body.rstrip() + "\n" + indentation + new_field_line.strip() + "\n"
    new_struct_code = struct_header + new_body + struct_footer
    return original_content.replace(match.group(0), new_struct_code)


# ==============================================================================
# SECTION 3: THE FINAL, UPGRADED "FINDER"
# ==============================================================================

def _find_block_boundaries(content: str, block_name: str, language: str) -> Optional[Tuple[int, int]]:
    """
    Find the start and end text positions of a function/method/block in the content.
    --- THIS IS THE ENHANCED VERSION ---
    """
    if language == 'markdown':
        # ... (markdown logic is fine, no changes needed here) ...
        # For brevity, I'm omitting the markdown part. The code is in your file.
        heading_level = block_name.count('#')
        safe_block_name = re.escape(block_name.replace('#', '').strip())
        pattern = re.compile(rf"^(#{'{'}{heading_level}{'}'}\s*{safe_block_name}.*?)$", re.MULTILINE | re.IGNORECASE)
        match = pattern.search(content)

        if not match:
            return None

        start_pos = match.start()
        next_heading_pattern = re.compile(rf"^(#{'{1,'}{heading_level}{'}'}\s+.*)$", re.MULTILINE)
        next_match = next_heading_pattern.search(content, pos=match.end())

        end_pos = next_match.start() if next_match else len(content)
        return (start_pos, end_pos)

    # --- Start of significant changes ---
    if language == 'python':
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == block_name:
                    # AST parsing logic remains the same
                    start_pos = -1
                    # This logic correctly finds the start and end using AST, which is reliable.
                    # No changes needed here.
                    if hasattr(node, 'decorator_list') and node.decorator_list:
                        # Find the start of the first decorator
                        start_pos = content.find(ast.get_source_segment(content, node.decorator_list[0]))
                    else:
                        # No decorators, find the start of the node itself
                        start_pos = content.find(ast.get_source_segment(content, node))

                    if start_pos != -1:
                        end_pos = start_pos + len(ast.get_source_segment(content, node))
                        return (start_pos, end_pos)
            return None # Explicitly return None if not found in AST walk
        except (SyntaxError, ValueError):
            # Fallback to regex if AST parsing fails
            pass

    # Generic Regex-based search for Rust, JS, etc.
    # Escape the block name to be safe in regex, but keep ` ` as a flexible spacer
    flexible_name = re.escape(block_name).replace(r'\ ', r'\s+')

    # NEW, more robust regex patterns for Rust `impl` blocks
    patterns = [
        # Rust `impl Trait for Struct` (e.g., `impl Default for ConfigFile`)
        rf"^(?:pub(?:\(.*\))?\s+)?impl\s+{flexible_name}\s*{{",
        # Standard impl block (e.g., `impl ConfigFile`)
        rf"^(?:pub(?:\(.*\))?\s+)?impl(?:<[^>]*>)?\s+{flexible_name}\s*{{",
        # Functions
        rf"^(?:pub(?:\(.*\))?\s+)?(?:async\s+)?fn\s+{flexible_name}\s*\(",
        # Structs
        rf"^(?:pub(?:\(.*\))?\s+)?struct\s+{flexible_name}\s*{{",
        # JS/TS/Python Fallbacks
        rf"^(?:export\s+)?(?:async\s+)?function\s+{flexible_name}\s*\(",
        rf"^\s*@?.*\s*def\s+{flexible_name}\s*\(",
        rf"^\s*class\s+{flexible_name}",
    ]

    for pattern_str in patterns:
        pattern = re.compile(pattern_str, re.MULTILINE)
        match = pattern.search(content)
        if match:
            start_pos = match.start()
            # Find the end of the block by matching braces `{}`
            if '{' in content[start_pos:match.end()]:
                pos = content.find('{', start_pos)
                brace_count = 1
                while pos < len(content) - 1:
                    pos += 1
                    if content[pos] == '{':
                        brace_count += 1
                    elif content[pos] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            return (start_pos, pos + 1)
            # Find the end of Python block by indentation
            elif ':' in content[start_pos:match.end()]:
                lines = content[start_pos:].splitlines()
                if not lines: return None
                base_indent = len(lines[0]) - len(lines[0].lstrip())
                end_line_index = 0
                for i, line in enumerate(lines[1:]):
                    if line.strip() and (len(line) - len(line.lstrip()) <= base_indent):
                        end_line_index = i
                        break
                else:
                    end_line_index = len(lines) -1

                end_pos = start_pos + len("\n".join(lines[:end_line_index + 1]))
                return (start_pos, end_pos)

    return None


def get_relevant_code_from_cortex(content: str, rca_report: str, file_path: str) -> str:
    """
    Extracts relevant code sections from a file's content based on an RCA report.
    """
    if not content or not rca_report:
        return content

    keywords = _extract_keywords_from_rca(rca_report)
    if not keywords:
        return content

    if not file_path.endswith('.py'):
        relevant_lines = []
        for line in content.splitlines():
            if any(kw in line for kw in keywords):
                relevant_lines.append(line)
        return "\n".join(relevant_lines) if relevant_lines else content

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return content

    relevant_nodes = []
    import_nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in keywords:
                relevant_nodes.append(node)
                if isinstance(node, ast.ClassDef):
                    for sub_node in node.body:
                        if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub_node.name in keywords and sub_node not in relevant_nodes:
                            relevant_nodes.append(sub_node)

    if not relevant_nodes:
        return content

    code_parts = [ast.get_source_segment(content, n) for n in import_nodes if n]
    if code_parts and relevant_nodes:
        code_parts.append("\n... # (Code imports)\n")

    added_sources = set()
    for node in sorted(relevant_nodes, key=lambda n: n.lineno):
        source_segment = ast.get_source_segment(content, node)
        if source_segment and source_segment not in added_sources:
            code_parts.append(source_segment)
            added_sources.add(source_segment)

    separator = "\n\n... # (Code omitted for brevity)\n\n"
    final_code = separator.join(part for part in code_parts if part)

    print(f"  🧠 Compressed code for '{file_path}'. Original: {len(content)} chars, Compressed: {len(final_code)} chars.")
    return final_code


def _extract_keywords_from_rca(rca_report: str) -> Set[str]:
    """Extracts potential function, class, and variable names from the RCA report."""
    keywords = set()
    keywords.update(re.findall(r'`([^`]+)`', rca_report))
    keywords.update(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', rca_report))
    keywords.update(re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)*|[a-z]+(?:_[a-z]+)+)\b', rca_report))

    cleaned_keywords = set()
    for kw in keywords:
        cleaned = kw.split('.')[-1].split('::')[-1]
        if len(cleaned) > 2:
            cleaned_keywords.add(cleaned)

    return cleaned_keywords
