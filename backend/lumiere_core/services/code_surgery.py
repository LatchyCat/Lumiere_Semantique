# backend/lumiere_core/services/code_surgery.py

import re
import json
import ast
from typing import Dict, Tuple, Optional, List

def _get_function_signature(code_block: str, language: str) -> str:
    """Extracts the function signature from a block of code based on language."""
    if language == 'python':
        match = re.search(r"^\s*(?:async\s+)?def\s+\w+\s*\(.*?\):", code_block, re.MULTILINE)
    else: # Default to JS-like
        match = re.search(r"(?:function\s+\w+\s*\(.*?\)|\w+\s*[:=]\s*\(.*?\)\s*=>)", code_block)

    if match:
        return match.group(0).strip()
    return ""

def _validate_function_code(func_code: str, expected_func_name: str, language: str) -> bool:
    """Validates that the function code contains a proper function definition for the given language."""
    if not func_code or not func_code.strip():
        return False

    if expected_func_name not in func_code:
        return False

    if language == 'python':
        has_def = "def " in func_code
        has_colon = ":" in func_code.split('\n')[0]
        if not (has_def and has_colon):
            return False
        # Basic Python syntax check
        try:
            ast.parse(func_code)
            return True
        except SyntaxError:
            return False
    else: # Default to JS-like
        has_function_keyword = "function" in func_code
        has_arrow_function = "=>" in func_code
        if not (has_function_keyword or has_arrow_function):
            return False
        # Basic brace balance check
        return func_code.count('{') > 0 and func_code.count('{') == func_code.count('}')

def _find_function_boundaries(content: str, func_name: str, language: str) -> Optional[Tuple[int, int]]:
    """
    Find the start and end text positions of a function/method in the content.
    This uses AST for Python for accuracy and regex for JS-like languages.
    """
    if language == 'python':
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # Find function or method in class
                    if isinstance(node, ast.ClassDef):
                        for sub_node in node.body:
                           if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub_node.name == func_name:
                                start_pos = content.find(ast.get_source_segment(content, sub_node))
                                end_pos = start_pos + len(ast.get_source_segment(content, sub_node))
                                return (start_pos, end_pos)
                    # Find top-level function
                    elif node.name == func_name:
                        start_pos = content.find(ast.get_source_segment(content, node))
                        end_pos = start_pos + len(ast.get_source_segment(content, node))
                        return (start_pos, end_pos)
            return None # Function not found
        except (SyntaxError, ValueError):
            # Fallback to regex if AST parsing fails
            pass

    # Regex for JS-like or Python fallback
    escaped_name = re.escape(func_name)
    pattern = rf"(?:function\s+|def\s+){escaped_name}\s*\(.*?\)\s*[:\{{]"
    match = re.search(pattern, content, re.MULTILINE)

    if not match:
        return None
    start_pos = match.start()

    # JS brace counting
    if '{' in match.group(0):
        brace_count = 0
        pos = match.end() - 1
        while pos < len(content):
            char = content[pos]
            if char == '{': brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0: return (start_pos, pos + 1)
            pos += 1
    # Python indentation logic (basic)
    elif ':' in match.group(0):
        lines = content[start_pos:].splitlines()
        base_indent = len(lines[0]) - len(lines[0].lstrip())
        end_line_index = 0
        for i, line in enumerate(lines[1:]):
            if line.strip() == "": continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent:
                end_line_index = i
                break
        else: # Reached end of file
            end_line_index = len(lines) -1

        end_pos = start_pos + len("\n".join(lines[:end_line_index+1]))
        return (start_pos, end_pos)

    return None

def _validate_snippets_response(data: Dict, file_languages: Dict[str, str]) -> Tuple[bool, str]:
    """Validate that the LLM response has the expected structure and content."""
    if not isinstance(data, dict):
        return False, "Response is not a dictionary"

    for file_path, snippets in data.items():
        if not isinstance(snippets, dict):
            return False, f"Snippets for {file_path} is not a dictionary"

        language = file_languages.get(file_path, 'javascript') # Default to JS if lang unknown

        for func_name, func_code in snippets.items():
            if not _validate_function_code(func_code, func_name, language):
                return False, f"Invalid function code for '{func_name}' in '{file_path}'"
    return True, ""

def replace_functions_in_file(original_content: str, changed_snippets: Dict[str, str], file_path: str) -> str:
    """Surgically replaces functions in a file with new versions, now language-aware."""
    language = 'python' if file_path.endswith('.py') else 'javascript'
    modified_content = original_content
    replacement_count = 0

    for func_name, new_func_code in changed_snippets.items():
        if not _validate_function_code(new_func_code, func_name, language):
            print(f"⚠️ Skipping invalid snippet for '{func_name}' in {file_path}")
            continue

        boundaries = _find_function_boundaries(modified_content, func_name, language)
        if not boundaries:
            print(f"⚠️ Could not find function '{func_name}' in {file_path} to replace.")
            continue

        start_pos, end_pos = boundaries
        modified_content = modified_content[:start_pos] + new_func_code + "\n" + modified_content[end_pos:]
        replacement_count += 1
        print(f"✓ Replaced function '{func_name}' in {file_path}")

    if replacement_count == 0:
        print(f"⚠️ No functions were replaced in {file_path}")
    return modified_content

def validate_and_parse_snippets(llm_response_json: str) -> Tuple[Optional[Dict], str]:
    """Validate and parse the LLM response JSON containing function snippets."""
    if not llm_response_json or not llm_response_json.strip():
        return None, "LLM response is empty"
    try:
        data = json.loads(llm_response_json)
    except json.JSONDecodeError as e:
        return None, f"Failed to parse JSON: {e}"

    file_languages = {path: ('python' if path.endswith('.py') else 'javascript') for path in data.keys()}
    is_valid, error_msg = _validate_snippets_response(data, file_languages)
    if not is_valid:
        return None, error_msg

    return data, ""

def get_relevant_code_from_cortex(content: str, rca_report: str) -> str:
    """Extracts relevant code sections from a file based on an RCA report."""
    if not content or not rca_report:
        return content
    # A simple implementation: return the full content.
    # A more sophisticated version could use AST and keywords from the RCA.
    return content
