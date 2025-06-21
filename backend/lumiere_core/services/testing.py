# In ~/lumiere_semantique/backend/lumiere_core/services/testing.py
# In lumiere_core/services/testing.py
from typing import Dict, List, Optional
import os
import logging

from .ollama import search_index
from .llm import generate_text
# --- NEW: Import the shared cleaning utility ---
from .utils import clean_llm_code_output

# Set up logging for better debugging
logger = logging.getLogger(__name__)

def generate_tests_for_code(repo_id: str, new_code: str, instruction: str) -> Dict[str, str]:
    """
    The core logic for the Test Generation Agent.

    Args:
        repo_id: Identifier for the repository
        new_code: The code that needs tests generated for it
        instruction: Additional instructions for test generation

    Returns:
        Dictionary containing the generated tests
    """
    print(f"Initiating Test Generation Agent for repo '{repo_id}'")

    if not repo_id or not new_code:
        return {"generated_tests": "", "error": "Missing required parameters: repo_id and new_code"}

    try:
        # --- Step 1: Finding existing test patterns with RAG ---
        print("   -> Step 1: Finding existing test patterns with RAG...")
        index_path = f"{repo_id}_faiss.index"
        map_path = f"{repo_id}_id_map.json"
        search_query = f"Example test cases for python code like this: {instruction}"

        try:
            context_chunks = search_index(
                query_text=search_query,
                model_name='snowflake-arctic-embed2:latest',
                index_path=index_path,
                map_path=map_path,
                k=7
            )
        except Exception as e:
            print(f"   -> Warning: RAG search failed: {e}. Proceeding without context.")
            context_chunks = []

        test_context_string = ""
        found_test_files = set()

        for chunk in context_chunks:
            file_path = chunk.get('file_path', '')
            chunk_text = chunk.get('text', '')

            if 'test' in file_path.lower() and file_path not in found_test_files and chunk_text:
                test_context_string += f"--- Example test from file '{file_path}' ---\n{chunk_text}\n\n"
                found_test_files.add(file_path)

        if not test_context_string:
            test_context_string = "No specific test patterns found. Please generate a standard pytest function."
            print("   -> Warning: No existing test files found via RAG. Will generate a generic test.")
        else:
            print(f"   -> Found test patterns from files: {list(found_test_files)}")

        # --- Step 2: The Reinforced Prompt ---
        print("   -> Step 2: Constructing reinforced test generation prompt...")
        prompt = f"""You are an expert QA Engineer and Python programmer. Your task is to write a new unit test for a piece of code.

**YOUR INSTRUCTIONS:**
1.  **Analyze "EXISTING TEST EXAMPLES"** to understand the project's testing style. Pay close attention:
    *   Are the tests inside a `class`?
    *   Do they use `self.assertEqual`, or plain `assert`?
    *   Are there fixtures or `setup` methods?
2.  **Analyze the "NEW CODE TO BE TESTED"** to understand its functionality.
3.  **Write a new test function.** It is CRITICAL that you exactly match the style of the examples. If the examples are standalone functions (e.g., `def test_...():`), your test MUST also be a standalone function. DO NOT invent a class if the examples do not use one.
4.  **Output ONLY raw Python code.** Do not include any explanations, commentary, or Markdown fences.

---
### EXISTING TEST EXAMPLES
{test_context_string}
---
### NEW CODE TO BE TESTED
```python
{new_code}
```

Now, generate ONLY the new, stylistically-consistent test function."""

        # --- Step 3: Generate the Test Code ---
        print("   -> Step 3: Sending request to code generation model 'qwen2.5-coder:3b'...")
        try:
            raw_generated_tests = generate_text(prompt, model_name='qwen2.5-coder:3b')
        except Exception as e:
            print(f"   -> Error: Failed to generate tests with LLM: {e}")
            return {"generated_tests": "", "error": f"LLM generation failed: {str(e)}"}

        # --- Step 4: Clean the output ---
        print("   -> Step 4: Cleaning and finalizing the generated test code...")
        try:
            final_tests = clean_llm_code_output(raw_generated_tests)
        except Exception as e:
            print(f"   -> Warning: Failed to clean LLM output: {e}. Using raw output.")
            final_tests = raw_generated_tests

        print("   -> ✓ Test generation completed successfully.")
        return {"generated_tests": final_tests}

    except Exception as e:
        error_msg = f"Unexpected error in test generation: {str(e)}"
        print(f"   -> Error: {error_msg}")
        logger.error(error_msg, exc_info=True)
        return {"generated_tests": "", "error": error_msg}

def validate_test_code(test_code: str) -> Dict[str, any]:
    """
    Validates the generated test code for basic syntax and structure.

    Args:
        test_code: The generated test code to validate

    Returns:
        Dictionary with validation results
    """
    validation_result = {
        "is_valid": False,
        "has_test_function": False,
        "syntax_errors": [],
        "warnings": []
    }

    if not test_code or not test_code.strip():
        validation_result["syntax_errors"].append("Test code is empty")
        return validation_result

    # Check for basic test function pattern
    if "def test_" in test_code:
        validation_result["has_test_function"] = True
    else:
        validation_result["warnings"].append("No test function found (should start with 'def test_')")

    # Basic syntax validation
    try:
        compile(test_code, '<string>', 'exec')
        validation_result["is_valid"] = True
    except SyntaxError as e:
        validation_result["syntax_errors"].append(f"Syntax error: {str(e)}")
    except Exception as e:
        validation_result["syntax_errors"].append(f"Compilation error: {str(e)}")

    return validation_result

def generate_test_suggestions(code_snippet: str) -> List[str]:
    """
    Generates suggestions for what types of tests should be written for the given code.

    Args:
        code_snippet: The code to analyze for test suggestions

    Returns:
        List of test suggestions
    """
    suggestions = []

    if not code_snippet:
        return suggestions

    code_lower = code_snippet.lower()

    # Basic suggestions based on code patterns
    if "def " in code_lower:
        suggestions.append("Test the function with valid inputs")
        suggestions.append("Test edge cases and boundary conditions")
        suggestions.append("Test with invalid inputs to verify error handling")

    if "class " in code_lower:
        suggestions.append("Test class initialization")
        suggestions.append("Test public methods")
        suggestions.append("Test method interactions")

    if "if " in code_lower or "elif " in code_lower:
        suggestions.append("Test all conditional branches")

    if "for " in code_lower or "while " in code_lower:
        suggestions.append("Test loop behavior with different iteration counts")
        suggestions.append("Test empty collections or zero iterations")

    if "try:" in code_lower or "except" in code_lower:
        suggestions.append("Test exception handling paths")
        suggestions.append("Test successful execution without exceptions")

    if "return " in code_lower:
        suggestions.append("Verify return values for different inputs")

    # Add default suggestions if none found
    if not suggestions:
        suggestions = [
            "Test basic functionality",
            "Test with edge cases",
            "Test error conditions"
        ]

    return suggestions
