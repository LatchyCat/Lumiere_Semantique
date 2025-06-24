# backend/lumiere_core/services/scaffolding.py

import json
import re
from pathlib import Path
from typing import Dict, Optional, List, Any

from . import llm_service
from .utils import clean_llm_code_output
from . import code_surgery

# Language-specific configurations
LANGUAGE_CONFIG = {
    '.py': {
        'name': 'Python',
        'function_keywords': ['def ', 'async def ', 'class '],
        'comment_style': '#',
        'example_format': '''{{
  "calculator.py": {{
    "subtract": "def subtract(a, b):\\n    # This is the corrected implementation\\n    return a - b"
  }}
}}'''
    },
    '.js': {
        'name': 'JavaScript',
        'function_keywords': ['function ', 'const ', 'let ', 'var ', '=>', 'class '],
        'comment_style': '//',
        'example_format': '''{{
  "utils.js": {{
    "calculateTotal": "function calculateTotal(items) {{\\n  // New implementation\\n  return items.reduce((acc, item) => acc + item.price, 0);\\n}}"
  }}
}}'''
    },
    '.ts': {
        'name': 'TypeScript',
        'function_keywords': ['function ', 'const ', 'let ', 'var ', '=>', 'class ', 'interface ', 'type '],
        'comment_style': '//',
        'example_format': '''{{
  "utils.ts": {{
    "calculateTotal": "function calculateTotal(items: Item[]): number {{\\n  // New implementation\\n  return items.reduce((acc, item) => acc + item.price, 0);\\n}}"
  }}
}}'''
    },
    '.gs': {
        'name': 'Google Apps Script',
        'function_keywords': ['function ', 'const ', 'let ', 'var ', '=>', 'class '],
        'comment_style': '//',
        'example_format': '''{{
  "dailySync.gs": {{
    "dailySync": "function dailySync() {{\\n  console.log('🔄 Starting daily sync...');\\n  // Improved implementation\\n  try {{\\n    // Your logic here\\n  }} catch (error) {{\\n    console.error('❌ Sync failed:', error);\\n  }}\\n}}"
  }}
}}'''
    },
    '.java': {
        'name': 'Java',
        'function_keywords': ['public ', 'private ', 'protected ', 'static ', 'class ', 'interface '],
        'comment_style': '//',
        'example_format': '''{{
  "Calculator.java": {{
    "subtract": "public int subtract(int a, int b) {{\\n    // This is the corrected implementation\\n    return a - b;\\n}}"
  }}
}}'''
    },
    '.cpp': {
        'name': 'C++',
        'function_keywords': ['int ', 'void ', 'double ', 'float ', 'char ', 'bool ', 'class ', 'struct '],
        'comment_style': '//',
        'example_format': '''{{
  "calculator.cpp": {{
    "subtract": "int subtract(int a, int b) {{\\n    // This is the corrected implementation\\n    return a - b;\\n}}"
  }}
}}'''
    },
    '.c': {
        'name': 'C',
        'function_keywords': ['int ', 'void ', 'double ', 'float ', 'char ', 'struct '],
        'comment_style': '//',
        'example_format': '''{{
  "calculator.c": {{
    "subtract": "int subtract(int a, int b) {{\\n    /* This is the corrected implementation */\\n    return a - b;\\n}}"
  }}
}}'''
    },
    '.go': {
        'name': 'Go',
        'function_keywords': ['func ', 'type ', 'var ', 'const '],
        'comment_style': '//',
        'example_format': '''{{
  "calculator.go": {{
    "Subtract": "func Subtract(a, b int) int {{\\n    // This is the corrected implementation\\n    return a - b\\n}}"
  }}
}}'''
    },
    '.rb': {
        'name': 'Ruby',
        'function_keywords': ['def ', 'class ', 'module '],
        'comment_style': '#',
        'example_format': '''{{
  "calculator.rb": {{
    "subtract": "def subtract(a, b)\\n  # This is the corrected implementation\\n  a - b\\nend"
  }}
}}'''
    },
    '.php': {
        'name': 'PHP',
        'function_keywords': ['function ', 'class ', 'public ', 'private ', 'protected '],
        'comment_style': '//',
        'example_format': '''{{
  "calculator.php": {{
    "subtract": "function subtract($a, $b) {{\\n    // This is the corrected implementation\\n    return $a - $b;\\n}}"
  }}
}}'''
    }
}

def _get_language_config(file_path: str) -> Dict[str, Any]:
    """Get language configuration based on file extension."""
    ext = Path(file_path).suffix.lower()
    return LANGUAGE_CONFIG.get(ext, {
        'name': 'Unknown',
        'function_keywords': ['function ', 'def ', 'class '],
        'comment_style': '//',
        'example_format': '''{{
  "example.txt": {{
    "functionName": "// Language-specific implementation\\nfunction example() {{ return true; }}"
  }}
}}'''
    })

def _detect_primary_language(target_files: List[str]) -> Dict[str, Any]:
    """Detect the primary language from target files."""
    if not target_files:
        return _get_language_config('')

    # Count file extensions
    ext_counts = {}
    for file_path in target_files:
        ext = Path(file_path).suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    # Get most common extension
    primary_ext = max(ext_counts, key=ext_counts.get) if ext_counts else ''
    return _get_language_config(primary_ext)

def _is_valid_json(json_str: str) -> bool:
    """Helper function to validate JSON strings."""
    try:
        parsed = json.loads(json_str)
        return isinstance(parsed, dict) and len(parsed) > 0
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def _extract_json_from_llm(raw_text: str) -> Optional[str]:
    """
    Ultra-robust JSON extraction from LLM responses.
    Enhanced to handle multiple languages and coding patterns.
    """

    # Clean the input text
    raw_text = raw_text.strip()

    # Strategy 1: Try Markdown JSON code block with comprehensive patterns
    markdown_patterns = [
        r"```(?:json|JSON)?\s*(\{.*?\})\s*```",      # Standard markdown
        r"```(?:json|JSON)?\n(\{.*?\})\n```",        # With newlines
        r"```(\{.*?\})```",                          # Simple fences
        r"`(\{.*?\})`",                              # Single backticks
        r"```(?:json|JSON)?\s*(\{.*?)\s*```",        # Incomplete end brace
        r"(?i)```json\s*(\{.*?\})\s*```",            # Case insensitive
    ]

    for pattern in markdown_patterns:
        matches = re.finditer(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            json_candidate = match.group(1).strip()
            if _is_valid_json(json_candidate):
                return json_candidate

    # Strategy 2: Enhanced bracket counting with multiple attempts
    json_candidates = []
    stack = []
    start_indices = []

    for i, char in enumerate(raw_text):
        if char == '{':
            if not stack:
                start_indices.append(i)
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_indices:
                    start_index = start_indices.pop()
                    json_candidate = raw_text[start_index:i+1]
                    json_candidates.append(json_candidate)

    # Test all candidates from bracket counting
    for candidate in json_candidates:
        if _is_valid_json(candidate):
            return candidate

    # Strategy 3: Look for file path patterns (enhanced for multiple languages)
    # Updated to include .gs files and other languages
    file_extensions = '|'.join([
        'py', 'js', 'ts', 'gs', 'java', 'cpp', 'c', 'h', 'html', 'css',
        'json', 'yaml', 'yml', 'xml', 'md', 'txt', 'go', 'rb', 'php',
        'swift', 'kt', 'dart', 'rs', 'scala', 'sql', 'sh', 'bat'
    ])

    file_patterns = [
        rf'\{{[^{{}}]*?"[^"]*\.(?:{file_extensions})"[^{{}}]*?:.*?\}}',
        rf'\{{.*?"[^"]*/"[^"]*\.(?:{file_extensions})".*?:.*?\}}',
        rf'\{{.*?["\'][^"\']*\.(?:{file_extensions})["\'].*?:.*?\}}',
    ]

    for pattern in file_patterns:
        matches = re.finditer(pattern, raw_text, re.DOTALL)
        for match in matches:
            json_candidate = match.group(0)
            if _is_valid_json(json_candidate):
                return json_candidate

    # Strategy 4: Try to extract from common LLM response patterns
    response_patterns = [
        r'(?:Here(?:\'s|s| is)? (?:the )?(?:JSON|json|response|fix|solution|code)?:?\s*)(\{.*?\})',
        r'(?:Response|Answer|Solution|Fix):\s*(\{.*?\})',
        rf'(\{{[^{{}}]*?["\'][^"\']*\.(?:{file_extensions})["\'][^{{}}]*?:.*?\}})',
    ]

    for pattern in response_patterns:
        matches = re.finditer(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            json_candidate = match.group(1).strip()
            if _is_valid_json(json_candidate):
                return json_candidate

    # Strategy 5: Last resort - try to find ANY valid JSON structure
    # Look for balanced braces and try to extract
    brace_positions = []
    for i, char in enumerate(raw_text):
        if char in '{}':
            brace_positions.append((i, char))

    # Try different combinations of brace positions
    for start_pos in range(len(brace_positions)):
        if brace_positions[start_pos][1] == '{':
            brace_count = 0
            for end_pos in range(start_pos, len(brace_positions)):
                if brace_positions[end_pos][1] == '{':
                    brace_count += 1
                elif brace_positions[end_pos][1] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        start_idx = brace_positions[start_pos][0]
                        end_idx = brace_positions[end_pos][0] + 1
                        json_candidate = raw_text[start_idx:end_idx]
                        if _is_valid_json(json_candidate):
                            return json_candidate
                        break

    return None


def _sanitize_json_response(json_str: str) -> str:
    """
    Comprehensive JSON sanitization to fix common LLM formatting issues.
    Enhanced for multiple programming languages.
    """
    # Remove common prefixes/suffixes that LLMs add
    prefixes_to_remove = [
        "Here's the JSON response:",
        "Here is the JSON:",
        "Here's the JSON:",
        "The JSON object is:",
        "Response:",
        "Here's the fix:",
        "Solution:",
        "The fix is:",
        "Here's your fix:",
        "JSON:",
        "```json",
        "```",
        "Answer:",
        "Result:",
        "Here's the updated code:",
        "Here's the corrected implementation:",
    ]

    suffixes_to_remove = [
        "Let me know if you need any clarification!",
        "This should fix the issue.",
        "Hope this helps!",
        "```",
        "Let me know if you have any questions.",
        "Please let me know if you need any modifications.",
        "This addresses the issue mentioned in the RCA report.",
        "The code is now language-agnostic.",
        "This should work for your specific language.",
    ]

    cleaned = json_str.strip()

    # Remove prefixes (case insensitive)
    for prefix in prefixes_to_remove:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
            break  # Only remove one prefix

    # Remove suffixes (case insensitive)
    for suffix in suffixes_to_remove:
        if cleaned.lower().endswith(suffix.lower()):
            cleaned = cleaned[:-len(suffix)].strip()
            break  # Only remove one suffix

    # Fix common JSON syntax issues
    # Replace smart quotes with regular quotes
    cleaned = cleaned.replace('"', '"').replace('"', '"')
    cleaned = cleaned.replace(''', "'").replace(''', "'")

    # Replace backticks that might be used as quotes
    cleaned = cleaned.replace('`', '"')

    # Fix trailing commas (common LLM mistake)
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)

    # Fix missing commas between objects/arrays
    cleaned = re.sub(r'}\s*{', '},{', cleaned)
    cleaned = re.sub(r']\s*\[', '],[', cleaned)

    # Fix unescaped quotes in strings (basic attempt)
    # This is tricky, so we'll just try to fix obvious cases
    cleaned = re.sub(r'([^\\])"([^",:}\]]*)"([^",:}\]]*)"', r'\1"\2\"\3"', cleaned)

    # Fix common spacing issues
    cleaned = re.sub(r'\s*:\s*', ':', cleaned)
    cleaned = re.sub(r'\s*,\s*', ',', cleaned)

    # Remove any remaining markdown code block indicators
    cleaned = re.sub(r'^```[a-zA-Z]*\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)

    return cleaned


def _attempt_json_repair(raw_text: str) -> Optional[str]:
    """
    Last resort JSON repair for severely malformed responses.
    Enhanced to handle multiple programming languages including .gs files.
    """
    try:
        # Enhanced file pattern to include more extensions
        file_extensions = '|'.join([
            'py', 'js', 'ts', 'gs', 'java', 'cpp', 'c', 'h', 'html', 'css',
            'json', 'yaml', 'yml', 'xml', 'md', 'txt', 'go', 'rb', 'php',
            'swift', 'kt', 'dart', 'rs', 'scala', 'sql', 'sh', 'bat'
        ])

        file_pattern = rf'["\']([^"\']*\.(?:{file_extensions}))["\']'
        files_found = re.findall(file_pattern, raw_text)

        if not files_found:
            return None

        # Try to build a JSON structure
        result = {}

        # Simple pattern matching for file content
        for file_path in files_found:
            # Look for content after the file path
            content_patterns = [
                rf'["\']({re.escape(file_path)})["\']:\s*["\']([^"\']*?)["\']',
                rf'["\']({re.escape(file_path)})["\']:\s*"""([^"]*?)"""',
                rf'["\']({re.escape(file_path)})["\']:\s*```([^`]*?)```',
                # Enhanced patterns for function definitions
                rf'["\']({re.escape(file_path)})["\']:\s*{{([^{{}}]*?)}}',
            ]

            for pattern in content_patterns:
                match = re.search(pattern, raw_text, re.DOTALL)
                if match:
                    result[file_path] = match.group(2).strip()
                    break

        if result:
            return json.dumps(result)

    except Exception:
        pass

    return None


def generate_scaffold(
    repo_id: str,
    target_files: List[str],
    instruction: str,
    model_identifier: str,
    rca_report: str,
    refinement_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Core logic for Code Scaffolding using the new "Surgical" Two-Step method.
    Now fully language-agnostic with support for .gs files and multiple languages.

    Returns:
        Dict containing either success data or error information
    """
    print(f"🔧 Initiating Surgical Scaffolding for {target_files} in repo '{repo_id}'")

    try:
        # --- SETUP AND VALIDATION ---
        backend_dir = Path(__file__).resolve().parent.parent.parent
        cortex_path = backend_dir / "cloned_repositories" / repo_id / f"{repo_id}_cortex.json"

        if not cortex_path.exists():
            return {
                "error": "Cortex file not found",
                "details": f"Expected path: {cortex_path}"
            }

        # Load cortex data
        try:
            with open(cortex_path, 'r', encoding='utf-8') as f:
                cortex_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            return {
                "error": "Failed to load cortex file",
                "details": str(e)
            }

        # Build file map and validate target files
        file_map = {file['file_path']: file['raw_content'] for file in cortex_data.get('files', [])}
        original_contents = {}
        missing_files = []

        for fp in target_files:
            if fp in file_map and file_map[fp]:
                original_contents[fp] = file_map[fp]
            else:
                missing_files.append(fp)

        if missing_files:
            return {
                "error": "Some target files not found in cortex",
                "missing_files": missing_files,
                "available_files": list(file_map.keys())
            }

        if not original_contents:
            return {
                "error": "No valid target files found",
                "target_files": target_files
            }

        print(f"✓ Loaded {len(original_contents)} target files")

        # Detect primary language for better prompt generation
        language_config = _detect_primary_language(target_files)
        print(f"📝 Detected primary language: {language_config['name']}")

        # --- STEP 1: GENERATE THE SNIPPETS ---
        print("📝 Step 1: Generating function snippets...")

        # Build relevant code section with language awareness
        file_content_prompt_section = "\n\n### RELEVANT EXISTING CODE\n"
        for path, content in original_contents.items():
            # Get language-specific compressed content
            compressed_content = code_surgery.get_relevant_code_from_cortex(content, rca_report)
            file_content_prompt_section += f"<file path=\"{path}\">\n{compressed_content}\n</file>\n\n"

        # Build refinement context
        refinement_context = ""
        if refinement_history:
            refinement_context = "\n\n### PREVIOUS REFINEMENT ATTEMPTS\n"
            for i, refinement in enumerate(refinement_history[-3:]):  # Last 3 attempts
                feedback = refinement.get("feedback", "No feedback provided.")
                refinement_context += f"Attempt {i+1} Feedback: {feedback}\n"
            refinement_context += "\nPlease learn from the previous feedback and generate a better fix.\n"

        # Enhanced language-agnostic prompt
        snippet_prompt = f"""You are a code generation expert specializing in {language_config['name']} and multi-language codebases.

<Instructions>
1. Read the <Goal> and <RCA_Report> carefully.
2. Based on the <RELEVANT_EXISTING_CODE>, identify and rewrite ONLY the functions/methods that need to be changed to fix the bug.
3. Your response MUST be a single, valid JSON object with no additional text or explanations.
4. The keys of the JSON object MUST be the full file paths (e.g., "calculator.py", "utils.js", "dailySync.gs").
5. The values MUST be JSON objects where keys are function/method names and values are the complete, new code for JUST THAT FUNCTION/METHOD.
6. Preserve existing function signatures unless the bug requires changing them.
7. Ensure all functions are syntactically correct and complete for their respective programming language.
8. For Google Apps Script (.gs) files, maintain proper JavaScript syntax and Google Apps Script conventions.
9. Pay attention to language-specific syntax, indentation, and conventions.
10. Include proper error handling and logging where appropriate for the language.
</Instructions>

<Language_Specific_Notes>
- Primary Language: {language_config['name']}
- Comment Style: {language_config['comment_style']}
- Common Function Keywords: {', '.join(language_config['function_keywords'])}
</Language_Specific_Notes>

<Example_Response_Format>
{language_config['example_format']}
</Example_Response_Format>

<Goal>{instruction}</Goal>

<RCA_Report>{rca_report}</RCA_Report>
{refinement_context}
{file_content_prompt_section}

Generate the JSON object containing only the modified function snippets now:"""

        # Generate snippets with enhanced retry logic
        max_retries = 3
        changed_snippets_data = None
        last_llm_response = ""

        for attempt in range(max_retries):
            print(f"  🤖 Attempt {attempt + 1}: Calling LLM for {language_config['name']}...")
            llm_response = llm_service.generate_text(snippet_prompt, model_identifier)
            last_llm_response = llm_response

            if not llm_response or not llm_response.strip():
                if attempt < max_retries - 1:
                    print("  ⚠️ Empty response, retrying...")
                    continue
                else:
                    return {"error": "LLM returned empty response", "attempts": max_retries}

            # Enhanced JSON extraction and validation
            json_str = _extract_json_from_llm(llm_response)
            if not json_str:
                # Try sanitization first
                sanitized = _sanitize_json_response(llm_response)
                json_str = _extract_json_from_llm(sanitized)

            if not json_str:
                json_str = _attempt_json_repair(llm_response)

            if json_str:
                parsed_data, error_msg = code_surgery.validate_and_parse_snippets(json_str)
                if parsed_data:
                    changed_snippets_data = parsed_data
                    print(f"  ✓ Successfully parsed {len(changed_snippets_data)} file(s) with snippets.")
                    break
                else:
                    print(f"  ⚠️ Attempt {attempt + 1} failed validation: {error_msg}")
                    # Log the JSON for debugging
                    print(f"  📝 Failed JSON: {json_str[:200]}...")
            else:
                print(f"  ⚠️ Attempt {attempt + 1} failed: Could not extract JSON from the response.")
                # Log part of the response for debugging
                print(f"  📝 LLM Response preview: {llm_response[:200]}...")

        if not changed_snippets_data:
            return {
                "error": f"Failed to generate valid snippets after {max_retries} attempts",
                "llm_response": last_llm_response,
                "language_detected": language_config['name']
            }

        # --- STEP 2: PERFORM THE CODE SURGERY ---
        print("🔬 Step 2: Performing code surgery...")

        final_modified_files = {}
        surgery_errors = []

        for file_path, snippets in changed_snippets_data.items():
            if file_path in original_contents:
                print(f"  🔧 Operating on {file_path} ({_get_language_config(file_path)['name']})...")
                try:
                    modified_content = code_surgery.replace_functions_in_file(
                        original_contents[file_path],
                        snippets,
                        file_path # Pass file_path for language-specific logic
                    )
                    final_modified_files[file_path] = modified_content
                    print(f"  ✓ Successfully modified {file_path}")

                except Exception as e:
                    error_msg = f"Surgery failed for {file_path}: {str(e)}"
                    surgery_errors.append(error_msg)
                    print(f"  ❌ {error_msg}")
                    # Keep the original content as fallback
                    final_modified_files[file_path] = original_contents[file_path]
            else:
                warning_msg = f"LLM generated snippets for untracked file: {file_path}"
                surgery_errors.append(warning_msg)
                print(f"  ⚠️ {warning_msg}")

        # Ensure we have results for all target files
        for file_path in target_files:
            if file_path not in final_modified_files:
                final_modified_files[file_path] = original_contents[file_path]

        result = {
            "modified_files": final_modified_files,
            "original_contents": original_contents,
            "snippets_generated": changed_snippets_data,
            "files_modified": len(final_modified_files),
            "functions_targeted": sum(len(snippets) for snippets in changed_snippets_data.values()),
            "primary_language": language_config['name'],
            "languages_detected": list(set(_get_language_config(fp)['name'] for fp in target_files))
        }

        if surgery_errors:
            result["warnings"] = surgery_errors

        print(f"🎉 Surgical scaffolding completed successfully for {language_config['name']} codebase!")
        return result

    except Exception as e:
        print(f"❌ Critical error in generate_scaffold: {str(e)}")
        return {
            "error": "Critical error in generate_scaffold",
            "details": str(e),
            "traceback": str(e.__traceback__) if hasattr(e, '__traceback__') else None
        }
