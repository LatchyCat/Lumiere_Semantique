# backend/lumiere_core/services/testing.py
import logging
from typing import Dict, List, Optional, Any, Set
from . import llm_service
from .ollama import search_index
from .utils import clean_llm_code_output

# Set up logging for better debugging
logger = logging.getLogger(__name__)

def generate_tests_for_code(repo_id: str, new_code: str, instruction: str) -> Dict[str, str]:
    """
    The core logic for the Test Generation Agent.
    It finds existing test files in the repository to learn the project's testing
    style and then generates a new test function consistent with that style.

    Args:
        repo_id: The unique identifier for the repository.
        new_code: The new code for which tests need to be generated.
        instruction: A natural language description of the code's purpose.

    Returns:
        A dictionary containing the generated test code or an error message.
        Format: {"generated_tests": str, "error": str (optional)}
    """
    logger.info(f"Initiating Test Generation Agent for repo '{repo_id}'")

    # Input validation
    if not all([repo_id, new_code]):
        error_msg = "Missing required parameters: repo_id and new_code."
        logger.error(error_msg)
        return {"generated_tests": "", "error": error_msg}

    if not instruction:
        logger.warning("No instruction provided, using default description")
        instruction = "Generate appropriate tests for the given code"

    try:
        # --- Step 1: Find existing test patterns with RAG ---
        logger.info("Step 1: Finding existing test patterns with RAG...")
        test_context_string = _find_existing_test_patterns(repo_id, instruction)

        # --- Step 2: Construct the Reinforced Prompt ---
        logger.info("Step 2: Constructing reinforced test generation prompt...")
        prompt = _build_test_generation_prompt(test_context_string, new_code)

        # --- Step 3: Generate the Test Code ---
        logger.info("Step 3: Generating test code...")
        raw_generated_tests = _generate_test_code(prompt)

        # --- Step 4: Clean and validate the output ---
        logger.info("Step 4: Cleaning and validating the generated test code...")
        final_tests = clean_llm_code_output(raw_generated_tests)

        # Validate the generated tests
        validation_result = validate_test_code(final_tests)
        if not validation_result["is_valid"]:
            logger.warning(f"Generated test code has validation issues: {validation_result['syntax_errors']}")
            # Still return the code but include warning in logs

        logger.info("✓ Test generation completed successfully.")
        return {"generated_tests": final_tests}

    except Exception as e:
        error_msg = f"An unexpected error occurred during test generation: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"generated_tests": "", "error": error_msg}


def _find_existing_test_patterns(repo_id: str, instruction: str) -> str:
    """
    Find existing test patterns using RAG search.

    Args:
        repo_id: Repository identifier
        instruction: Code description for search context

    Returns:
        String containing formatted test examples or default message
    """
    search_query = f"Example test cases for python code like this: {instruction}"

    try:
        # CORRECTED CALL: Pass repo_id directly to the centralized search function.
        context_chunks = search_index(
            query_text=search_query,
            model_name='snowflake-arctic-embed2:latest',
            repo_id=repo_id,
            k=7  # Look for up to 7 relevant chunks
        )
    except Exception as e:
        logger.warning(f"RAG search failed for test generation in repo '{repo_id}': {e}")
        context_chunks = []

    test_context_string = ""
    found_test_files: Set[str] = set()

    for chunk in context_chunks:
        file_path = chunk.get('file_path', '')
        chunk_text = chunk.get('text', '')

        # Heuristic to identify test files and avoid duplicates
        if (_is_test_file(file_path) and
            file_path not in found_test_files and
            chunk_text and
            len(chunk_text.strip()) > 10):  # Ensure meaningful content

            test_context_string += f"--- Example test from file `{file_path}` ---\n```python\n{chunk_text}\n```\n\n"
            found_test_files.add(file_path)

    if not test_context_string:
        test_context_string = "No specific test patterns found. Please generate a standard `pytest` function using `assert`."
        logger.info("Warning: No existing test files found via RAG. Will generate a generic test.")
    else:
        logger.info(f"Found test patterns from files: {list(found_test_files)}")

    return test_context_string


def _is_test_file(file_path: str) -> bool:
    """Check if a file path indicates a test file."""
    if not file_path:
        return False

    file_path_lower = file_path.lower()
    return (
        'test' in file_path_lower or
        file_path_lower.endswith('_test.py') or
        file_path_lower.startswith('test_') or
        '/tests/' in file_path_lower
    )


def _build_test_generation_prompt(test_context_string: str, new_code: str) -> str:
    """
    Build the prompt for test generation.

    Args:
        test_context_string: Existing test examples
        new_code: Code to generate tests for

    Returns:
        Formatted prompt string
    """
    return f"""You are an expert QA Engineer and Python programmer. Your task is to write a new unit test for a piece of code.

**YOUR INSTRUCTIONS:**
1. **Analyze "EXISTING TEST EXAMPLES"** to understand the project's testing style. Pay close attention to:
   * Imports (e.g., `unittest`, `pytest`).
   * Structure: Are tests inside a `class`?
   * Assertions: Do they use `self.assertEqual` or plain `assert`?
   * Setup: Are there fixtures or `setUp` methods?

2. **Analyze the "NEW CODE TO BE TESTED"** to understand its functionality.

3. **Write a complete test function.** It is CRITICAL that you exactly match the style of the examples.
   If the examples are standalone functions (e.g., `def test_...():`), your test MUST also be a standalone function.
   DO NOT invent a class if the examples do not use one.

4. **Output ONLY raw Python code.** Do not include any explanations, commentary, or Markdown fences like ```python.

---
### EXISTING TEST EXAMPLES
{test_context_string}

---
### NEW CODE TO BE TESTED
```python
{new_code}
```

Now, generate ONLY the new, stylistically-consistent test function."""


def _generate_test_code(prompt: str) -> str:
    """
    Generate test code using the LLM service.

    Args:
        prompt: The formatted prompt for test generation

    Returns:
        Generated test code

    Raises:
        Exception: If LLM generation fails
    """
    model_to_use = "ollama/qwen2.5-coder:3b"
    logger.info(f"Sending request to code generation model '{model_to_use}'...")

    try:
        # Use the master LLM service and pass the full model identifier
        raw_generated_tests = llm_service.generate_text(prompt, model_identifier=model_to_use)

        if not raw_generated_tests or not raw_generated_tests.strip():
            raise ValueError("LLM returned empty response")

        return raw_generated_tests

    except Exception as e:
        logger.error(f"LLM generation failed for tests: {e}")
        raise Exception(f"LLM generation failed: {str(e)}")


def validate_test_code(test_code: str) -> Dict[str, Any]:
    """
    Validates the generated test code for basic syntax and structure.

    Args:
        test_code: The generated test code to validate

    Returns:
        Dictionary with validation results containing:
        - is_valid: bool
        - has_test_function: bool
        - syntax_errors: List[str]
    """
    validation_result = {
        "is_valid": False,
        "has_test_function": False,
        "syntax_errors": [],
    }

    if not test_code or not test_code.strip():
        validation_result["syntax_errors"].append("Test code is empty")
        return validation_result

    # Check for test function presence
    if "def test_" in test_code:
        validation_result["has_test_function"] = True
    else:
        validation_result["syntax_errors"].append("No test function found (should start with 'def test_')")

    # Validate syntax
    try:
        compile(test_code, '<string>', 'exec')
        validation_result["is_valid"] = True
    except SyntaxError as e:
        error_msg = f"Syntax error: {e.msg}"
        if e.lineno:
            error_msg += f" on line {e.lineno}"
        validation_result["syntax_errors"].append(error_msg)
    except Exception as e:
        validation_result["syntax_errors"].append(f"Unexpected compilation error: {str(e)}")

    return validation_result


def generate_test_suggestions(code_snippet: str) -> List[str]:
    """
    Generates high-level suggestions for what types of tests should be written.

    Args:
        code_snippet: The code to analyze for test suggestions

    Returns:
        List of test case suggestions
    """
    if not code_snippet or not code_snippet.strip():
        return ["No code provided for analysis"]

    suggestions = []
    code_lower = code_snippet.lower()

    # Function-based suggestions
    if "def " in code_lower:
        suggestions.extend([
            "Test the happy path with valid inputs",
            "Test edge cases (e.g., empty strings, zero, None)",
            "Test with invalid inputs to verify error handling"
        ])

    # Control flow suggestions
    if "if " in code_lower or "elif " in code_lower:
        suggestions.append("Ensure all conditional branches are tested")

    # Loop suggestions
    if "for " in code_lower or "while " in code_lower:
        suggestions.append("Test loop behavior (e.g., zero, one, and multiple iterations)")

    # Exception handling suggestions
    if "try:" in code_lower and "except" in code_lower:
        suggestions.extend([
            "Verify that expected exceptions are raised correctly",
            "Verify behavior when no exception occurs"
        ])

    # Class-based suggestions
    if "class " in code_lower:
        suggestions.extend([
            "Test object initialization",
            "Test all public methods",
            "Test method interactions and state changes"
        ])

    # Data structure suggestions
    if any(keyword in code_lower for keyword in ["list", "dict", "set", "tuple"]):
        suggestions.append("Test with different data structure sizes and types")

    # Return default suggestions if none found
    if not suggestions:
        suggestions = [
            "Test basic functionality",
            "Test edge cases",
            "Test error conditions"
        ]

    return suggestions


def get_test_coverage_suggestions(code_snippet: str) -> Dict[str, List[str]]:
    """
    Analyze code and provide comprehensive test coverage suggestions.

    Args:
        code_snippet: Code to analyze

    Returns:
        Dictionary categorizing different types of test suggestions
    """
    if not code_snippet or not code_snippet.strip():
        return {"error": ["No code provided for analysis"]}

    suggestions = {
        "unit_tests": generate_test_suggestions(code_snippet),
        "integration_tests": [],
        "edge_cases": [],
        "performance_tests": []
    }

    code_lower = code_snippet.lower()

    # Integration test suggestions
    if any(keyword in code_lower for keyword in ["import", "from", "api", "database", "http"]):
        suggestions["integration_tests"].extend([
            "Test external API interactions",
            "Test database operations if applicable",
            "Test module integration"
        ])

    # Edge case suggestions
    suggestions["edge_cases"].extend([
        "Test with None values",
        "Test with empty collections",
        "Test with maximum/minimum values",
        "Test with malformed input"
    ])

    # Performance test suggestions
    if any(keyword in code_lower for keyword in ["for", "while", "sort", "search"]):
        suggestions["performance_tests"].extend([
            "Test with large datasets",
            "Test execution time constraints",
            "Test memory usage"
        ])

    return suggestions
