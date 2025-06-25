# backend/lumiere_core/services/scaffolding.py

import json
import re
import traceback
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

from . import llm_service
from .utils import clean_llm_code_output
from . import code_surgery

# Language-specific configurations
LANGUAGE_CONFIG = {
    '.py': { 'name': 'Python' },
    '.js': { 'name': 'JavaScript' },
    '.ts': { 'name': 'TypeScript' },
    '.gs': { 'name': 'Google Apps Script' },
    '.java': { 'name': 'Java' },
    '.cpp': { 'name': 'C++' },
    '.c': { 'name': 'C' },
    '.go': { 'name': 'Go' },
    '.rb': { 'name': 'Ruby' },
    '.php': { 'name': 'PHP' },
    '.rs': { 'name': 'Rust' },
}

def _get_language_config(file_path: str) -> Dict[str, Any]:
    ext = Path(file_path).suffix.lower()
    return LANGUAGE_CONFIG.get(ext, {'name': 'Unknown'})

def _detect_primary_language(target_files: List[str]) -> Dict[str, Any]:
    if not target_files:
        return _get_language_config('')
    try:
        ext_counts = {}
        for fp in target_files:
            ext = Path(fp).suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        primary_ext = max(ext_counts, key=ext_counts.get)
        return _get_language_config(primary_ext)
    except (ValueError, IndexError):
        return _get_language_config('')


def _extract_json_from_llm(raw_text: str) -> Optional[str]:
    match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', raw_text, re.DOTALL)
    if match:
        return match.group(1)
    try:
        start = raw_text.index('[')
        end = raw_text.rindex(']') + 1
        return raw_text[start:end]
    except ValueError:
        return None

def _validate_and_parse_surgical_plan(json_str: str) -> Tuple[Optional[List[Dict]], str]:
    if not json_str:
        return None, "AI response was empty or did not contain a JSON object."
    try:
        plan = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"AI response was not valid JSON. Parser error: {e}"

    if not isinstance(plan, list):
        return None, f"AI response was not a list of operations. Found type: {type(plan).__name__}."

    for i, op in enumerate(plan):
        op_num = i + 1
        if not isinstance(op, dict):
            return None, f"Operation #{op_num} is not a valid object."

        operation_type = op.get("operation")
        if not operation_type:
            return None, f"Operation #{op_num} is missing the required 'operation' field."

        required_fields = {
            "REPLACE_BLOCK": ["file_path", "target_identifier", "content"],
            "ADD_FIELD_TO_STRUCT": ["file_path", "target_identifier", "content"],
            "CREATE_FILE": ["file_path", "content"],
            "INSERT_CODE_AT": ["file_path", "line_number", "content"],
        }
        if operation_type not in required_fields:
            return None, f"Operation #{op_num} has an unknown operation type: '{operation_type}'."

        missing = [field for field in required_fields[operation_type] if field not in op]
        if missing:
            return None, f"Operation #{op_num} ('{operation_type}') is missing required fields: {', '.join(missing)}."

    return plan, ""

def _scout_expand_scope(
    original_contents: Dict[str, str],
    surgical_plan: List[Dict],
    full_file_map: Dict[str, str]
) -> Dict[str, str]:
    """
    The Scout Service.
    Ensures the file scope matches the AI's plan by loading any missing files.
    """
    print("🛰️  Activating Scout: Verifying and expanding file scope...")

    plan_files = {op['file_path'] for op in surgical_plan if 'file_path' in op}

    updated_contents = original_contents.copy()
    expanded_files_loaded = 0

    for file_path in plan_files:
        if file_path not in updated_contents:
            updated_contents[file_path] = full_file_map.get(file_path, "")
            print(f"  → Scout expanded scope to include: {file_path}")
            expanded_files_loaded += 1

    if expanded_files_loaded > 0:
        print(f"✓ Scout successfully expanded scope with {expanded_files_loaded} new file(s).")
    else:
        print("✓ File scope is consistent with the AI's plan.")

    return updated_contents


def generate_scaffold(
    repo_id: str,
    target_files: List[str],
    instruction: str,
    model_identifier: str,
    rca_report: str,
    refinement_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Core logic for Code Scaffolding with Dynamic Scope Expansion.
    """
    print(f"🔧 Initiating Surgical Scaffolding for {target_files} in repo '{repo_id}'")

    try:
        # --- SETUP AND VALIDATION ---
        backend_dir = Path(__file__).resolve().parent.parent.parent
        cortex_path = backend_dir / "cloned_repositories" / repo_id / f"{repo_id}_cortex.json"
        if not cortex_path.exists():
            return {"error": "Cortex file not found", "details": f"Expected path: {cortex_path}"}

        with open(cortex_path, 'r', encoding='utf-8') as f:
            cortex_data = json.load(f)

        file_map = {file['file_path']: file['raw_content'] for file in cortex_data.get('files', [])}
        original_contents = {fp: file_map.get(fp, "") for fp in target_files}

        print(f"✓ Loaded {len(target_files)} initial target files.")
        language_config = _detect_primary_language(target_files)
        print(f"📝 Detected primary language: {language_config['name']}")

        # --- STEP 1: GENERATE THE SURGICAL PLAN ---
        print("📝 Step 1: Generating surgical plan...")

        file_content_prompt_section = "\n\n### RELEVANT EXISTING CODE\n"
        for path, content in original_contents.items():
            if content:
                compressed_content = code_surgery.get_relevant_code_from_cortex(content, rca_report, path)
                file_content_prompt_section += f"<file path=\"{path}\">\n{compressed_content}\n</file>\n\n"

        refinement_context = ""
        if refinement_history:
            refinement_context = "\n\n### PREVIOUS REFINEMENT ATTEMPTS\n"
            for i, refinement in enumerate(refinement_history[-2:]):
                feedback = refinement.get("feedback", "No feedback provided.")
                refinement_context += f"Attempt {i+1} Feedback: {feedback}\n"
            refinement_context += "\nPlease learn from the previous feedback and generate a better plan.\n"

        surgical_prompt = f"""You are an expert software architect specializing in {language_config['name']}. Your task is to generate a precise surgical plan to fix a bug.

<Goal>{instruction}</Goal>
<RCA_Report>{rca_report}</RCA_Report>
{refinement_context}
{file_content_prompt_section}

### YOUR TASK ###
Based on all the provided information, create a step-by-step surgical plan as a JSON array.

### AVAILABLE OPERATIONS ###
You can use the following operations in your plan:
1.  `"operation": "CREATE_FILE"`: Creates a new file.
    - Required fields: `file_path`, `content`.
2.  `"operation": "REPLACE_BLOCK"`: Replaces an entire function, method, class, or other code block.
    - Required fields: `file_path`, `target_identifier` (the unique name/signature of the block to replace), `content`.
3.  `"operation": "ADD_FIELD_TO_STRUCT"`: (For Rust/C/Go) Adds a new field to a struct.
    - Required fields: `file_path`, `target_identifier` (the name of the struct), `content` (the line(s) for the new field).

### RESPONSE FORMAT ###
- You MUST respond with ONLY a valid JSON array `[...]`.
- Do not include any explanations, markdown fences, or other text.
- If the AI needs to modify a file not in the provided context, it should add an operation for that file. The tool will handle loading it.

<Example_Response_Format>
```json
[
  {{
    "operation": "ADD_FIELD_TO_STRUCT",
    "file_path": "src/config/mod.rs",
    "target_identifier": "ConfigFile",
    "content": "    pub autoplay: bool,"
  }},
  {{
    "operation": "REPLACE_BLOCK",
    "file_path": "src/player.rs",
    "target_identifier": "play_next_song",
    "content": "pub fn play_next_song(config: &Config) {{\\n    if config.autoplay {{\\n        // new logic here...\\n    }}\\n}}"
  }}
]
</Example_Response_Format>
Generate the surgical plan now.
"""
        max_retries = 3
        surgical_plan = None
        last_llm_response = ""

        for attempt in range(max_retries):
            print(f"  🤖 Attempt {attempt + 1}: Calling LLM for surgical plan...")
            llm_response = llm_service.generate_text(surgical_prompt, model_identifier)
            last_llm_response = llm_response

            if not llm_response or not llm_response.strip():
                print("  ⚠️ Empty response, retrying...")
                continue

            json_str = _extract_json_from_llm(llm_response)
            if json_str:
                plan, error_msg = _validate_and_parse_surgical_plan(json_str)
                if plan:
                    surgical_plan = plan
                    print(f"  ✓ Successfully parsed surgical plan with {len(plan)} operations.")
                    break
                else:
                    print(f"  ❌ Blueprint Rejected (Attempt {attempt+1}/{max_retries}): {error_msg}")
                    print(f"  📝 Faulty JSON received: {json_str[:250]}...")
            else:
                print(f"  ❌ Could not extract JSON from LLM response (Attempt {attempt+1}/{max_retries}).")

        if not surgical_plan:
            return {
                "error": f"Failed to generate a valid surgical plan after {max_retries} attempts.",
                "details": "The AI's final proposed plan was malformed or incomplete. Please review the raw response for clues.",
                "llm_response": last_llm_response,
            }

        # --- The Scout Service is called here ---
        final_contents_for_surgery = _scout_expand_scope(
            original_contents,
            surgical_plan,
            file_map
        )

        # --- STEP 2: PERFORM THE CODE SURGERY ---
        print("🔬 Step 2: Dispatching plan to Code Surgery agent...")

        modified_files, surgery_report = code_surgery.execute_surgical_plan(
            final_contents_for_surgery,
            surgical_plan
        )

        if surgery_report.get("errors"):
            return {
                "error": "Code Surgery failed to apply the plan.",
                "details": surgery_report["errors"]
            }

        print("🎉 Surgical scaffolding completed successfully!")
        return {
            "modified_files": modified_files,
            "original_contents": final_contents_for_surgery, # Return the expanded set
            "plan_executed": surgical_plan,
            "surgery_report": surgery_report,
        }

    except Exception as e:
        print(f"❌ Critical error in generate_scaffold: {str(e)}")
        return {
            "error": "A critical error occurred in the scaffolding service.",
            "details": str(e),
            "traceback": traceback.format_exc(),
        }
