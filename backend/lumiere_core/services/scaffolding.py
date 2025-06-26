# backend/lumiere_core/services/scaffolding.py

import json
import re
import traceback
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
from enum import Enum

from . import llm_service
from .utils import clean_llm_code_output
from . import code_surgery

# Enhanced language configuration with comprehensive polyglot support
class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    COMPLEX_REASONING = "complex_reasoning"
    ANALYSIS = "analysis"

POLYGLOT_LANGUAGE_CONFIG = {
    # Web Technologies
    '.js': {
        'name': 'JavaScript',
        'family': 'web',
        'paradigm': 'multi-paradigm',
        'syntax_style': 'c-like',
        'features': ['dynamic', 'interpreted', 'event-driven'],
        'frameworks': ['React', 'Vue', 'Angular', 'Express', 'Node.js'],
        'common_patterns': ['async/await', 'promises', 'closures', 'prototypes']
    },
    '.ts': {
        'name': 'TypeScript',
        'family': 'web',
        'paradigm': 'multi-paradigm',
        'syntax_style': 'c-like',
        'features': ['static-typing', 'compiled', 'object-oriented'],
        'frameworks': ['Angular', 'React', 'Vue', 'NestJS'],
        'common_patterns': ['interfaces', 'generics', 'decorators', 'modules']
    },
    '.jsx': {
        'name': 'JavaScript (React)',
        'family': 'web',
        'paradigm': 'component-based',
        'syntax_style': 'jsx',
        'features': ['virtual-dom', 'component-lifecycle', 'hooks'],
        'frameworks': ['React', 'Next.js', 'Gatsby'],
        'common_patterns': ['JSX', 'hooks', 'state management', 'props']
    },
    '.tsx': {
        'name': 'TypeScript (React)',
        'family': 'web',
        'paradigm': 'component-based',
        'syntax_style': 'jsx',
        'features': ['static-typing', 'virtual-dom', 'component-lifecycle'],
        'frameworks': ['React', 'Next.js'],
        'common_patterns': ['typed props', 'interfaces', 'generic components']
    },
    '.vue': {
        'name': 'Vue.js',
        'family': 'web',
        'paradigm': 'component-based',
        'syntax_style': 'template-based',
        'features': ['reactive', 'template-driven', 'single-file-components'],
        'frameworks': ['Vue', 'Nuxt.js', 'Quasar'],
        'common_patterns': ['template', 'script', 'style', 'composition API']
    },

    # Backend Languages
    '.py': {
        'name': 'Python',
        'family': 'general-purpose',
        'paradigm': 'multi-paradigm',
        'syntax_style': 'indented',
        'features': ['dynamic', 'interpreted', 'duck-typing', 'comprehensive-stdlib'],
        'frameworks': ['Django', 'Flask', 'FastAPI', 'Pyramid'],
        'common_patterns': ['list comprehensions', 'decorators', 'context managers', 'generators']
    },
    '.java': {
        'name': 'Java',
        'family': 'enterprise',
        'paradigm': 'object-oriented',
        'syntax_style': 'c-like',
        'features': ['static-typing', 'compiled', 'garbage-collected', 'platform-independent'],
        'frameworks': ['Spring', 'Spring Boot', 'Hibernate', 'Struts'],
        'common_patterns': ['dependency injection', 'annotations', 'interfaces', 'inheritance']
    },
    '.cs': {
        'name': 'C#',
        'family': 'enterprise',
        'paradigm': 'multi-paradigm',
        'syntax_style': 'c-like',
        'features': ['static-typing', 'compiled', 'garbage-collected', 'linq'],
        'frameworks': ['ASP.NET', '.NET Core', 'Entity Framework', 'Blazor'],
        'common_patterns': ['properties', 'events', 'delegates', 'async/await']
    },
    '.go': {
        'name': 'Go',
        'family': 'systems',
        'paradigm': 'procedural',
        'syntax_style': 'c-like',
        'features': ['static-typing', 'compiled', 'concurrent', 'simple'],
        'frameworks': ['Gin', 'Echo', 'Fiber', 'Buffalo'],
        'common_patterns': ['goroutines', 'channels', 'interfaces', 'composition']
    },
    '.rs': {
        'name': 'Rust',
        'family': 'systems',
        'paradigm': 'multi-paradigm',
        'syntax_style': 'c-like',
        'features': ['memory-safe', 'zero-cost-abstractions', 'ownership', 'pattern-matching'],
        'frameworks': ['Actix', 'Rocket', 'Warp', 'Axum'],
        'common_patterns': ['ownership', 'borrowing', 'traits', 'match expressions']
    },
    '.swift': {
        'name': 'Swift',
        'family': 'mobile',
        'paradigm': 'multi-paradigm',
        'syntax_style': 'modern',
        'features': ['type-safe', 'memory-safe', 'performant', 'expressive'],
        'frameworks': ['SwiftUI', 'UIKit', 'Vapor', 'Perfect'],
        'common_patterns': ['optionals', 'closures', 'protocols', 'extensions']
    },

    # Functional Languages
    '.hs': {
        'name': 'Haskell',
        'family': 'functional',
        'paradigm': 'purely-functional',
        'syntax_style': 'mathematical',
        'features': ['lazy-evaluation', 'immutable', 'type-inference', 'monads'],
        'frameworks': ['Yesod', 'Snap', 'Happstack'],
        'common_patterns': ['monads', 'functors', 'type classes', 'pattern matching']
    },
    '.ex': {
        'name': 'Elixir',
        'family': 'functional',
        'paradigm': 'functional',
        'syntax_style': 'ruby-like',
        'features': ['actor-model', 'fault-tolerant', 'concurrent', 'distributed'],
        'frameworks': ['Phoenix', 'Nerves', 'Broadway'],
        'common_patterns': ['pattern matching', 'pipe operator', 'GenServer', 'supervision trees']
    },
    '.clj': {
        'name': 'Clojure',
        'family': 'functional',
        'paradigm': 'functional',
        'syntax_style': 'lisp',
        'features': ['immutable', 'jvm-hosted', 'concurrent', 'homoiconic'],
        'frameworks': ['Ring', 'Compojure', 'Luminus'],
        'common_patterns': ['s-expressions', 'persistent data structures', 'multimethods', 'macros']
    },

    # Systems Languages
    '.c': {
        'name': 'C',
        'family': 'systems',
        'paradigm': 'procedural',
        'syntax_style': 'c-like',
        'features': ['low-level', 'manual-memory', 'portable', 'efficient'],
        'frameworks': ['glib', 'SDL', 'OpenGL'],
        'common_patterns': ['pointers', 'manual memory management', 'header files', 'macros']
    },
    '.cpp': {
        'name': 'C++',
        'family': 'systems',
        'paradigm': 'multi-paradigm',
        'syntax_style': 'c-like',
        'features': ['object-oriented', 'template-metaprogramming', 'raii', 'zero-overhead'],
        'frameworks': ['Qt', 'Boost', 'POCO', 'FLTK'],
        'common_patterns': ['classes', 'templates', 'RAII', 'smart pointers']
    },

    # Scripting Languages
    '.rb': {
        'name': 'Ruby',
        'family': 'scripting',
        'paradigm': 'object-oriented',
        'syntax_style': 'natural',
        'features': ['dynamic', 'expressive', 'metaprogramming', 'blocks'],
        'frameworks': ['Rails', 'Sinatra', 'Hanami'],
        'common_patterns': ['blocks', 'mixins', 'metaprogramming', 'duck typing']
    },
    '.php': {
        'name': 'PHP',
        'family': 'web',
        'paradigm': 'imperative',
        'syntax_style': 'c-like',
        'features': ['web-focused', 'dynamic', 'interpreted', 'embedded'],
        'frameworks': ['Laravel', 'Symfony', 'CodeIgniter', 'Zend'],
        'common_patterns': ['superglobals', 'include/require', 'associative arrays', 'traits']
    },

    # Mobile/Cross-platform
    '.dart': {
        'name': 'Dart',
        'family': 'mobile',
        'paradigm': 'object-oriented',
        'syntax_style': 'c-like',
        'features': ['widget-based', 'hot-reload', 'ahead-of-time', 'just-in-time'],
        'frameworks': ['Flutter', 'AngularDart'],
        'common_patterns': ['widgets', 'futures', 'streams', 'isolates']
    },
    '.kt': {
        'name': 'Kotlin',
        'family': 'mobile',
        'paradigm': 'multi-paradigm',
        'syntax_style': 'modern',
        'features': ['null-safe', 'interoperable', 'concise', 'expressive'],
        'frameworks': ['Android SDK', 'Ktor', 'Spring'],
        'common_patterns': ['null safety', 'data classes', 'extension functions', 'coroutines']
    },

    # Data Science
    '.r': {
        'name': 'R',
        'family': 'statistical',
        'paradigm': 'functional',
        'syntax_style': 'domain-specific',
        'features': ['statistical', 'vectorized', 'data-analysis', 'visualization'],
        'frameworks': ['Shiny', 'ggplot2', 'dplyr', 'tidyverse'],
        'common_patterns': ['data frames', 'vectorization', 'pipes', 'factors']
    },
    '.jl': {
        'name': 'Julia',
        'family': 'scientific',
        'paradigm': 'multi-paradigm',
        'syntax_style': 'mathematical',
        'features': ['high-performance', 'scientific', 'multiple-dispatch', 'metaprogramming'],
        'frameworks': ['Genie.jl', 'Flux.jl', 'DifferentialEquations.jl'],
        'common_patterns': ['multiple dispatch', 'macros', 'broadcasting', 'type system']
    }
}

def _get_enhanced_language_config(file_path: str) -> Dict[str, Any]:
    """Get comprehensive language configuration including paradigm and features."""
    ext = Path(file_path).suffix.lower()
    config = POLYGLOT_LANGUAGE_CONFIG.get(ext, {
        'name': 'Unknown',
        'family': 'unknown',
        'paradigm': 'unknown',
        'syntax_style': 'unknown',
        'features': [],
        'frameworks': [],
        'common_patterns': []
    })

    # Add file extension for reference
    config['extension'] = ext
    return config

def _detect_polyglot_context(target_files: List[str], cortex_data: Dict) -> Dict[str, Any]:
    """
    Enhanced context detection for polyglot projects.
    Analyzes multiple languages and provides comprehensive project context.
    """
    if not target_files:
        return _get_enhanced_language_config('')

    language_analysis = {}
    framework_hints = set()
    project_patterns = set()

    # Analyze each target file
    for file_path in target_files:
        config = _get_enhanced_language_config(file_path)
        language_name = config['name']

        if language_name not in language_analysis:
            language_analysis[language_name] = {
                'files': [],
                'config': config,
                'weight': 0
            }

        language_analysis[language_name]['files'].append(file_path)
        language_analysis[language_name]['weight'] += 1

        # Collect framework hints from cortex if available
        if cortex_data and 'files' in cortex_data:
            for file_cortex in cortex_data['files']:
                if file_cortex['file_path'] == file_path:
                    if 'framework_hints' in file_cortex:
                        framework_hints.update(file_cortex['framework_hints'])
                    break

        # Add common patterns
        project_patterns.update(config['common_patterns'])

    # Determine primary language by weight and importance
    if language_analysis:
        # Weight by file count and language importance
        language_priority = {
            'Python': 10, 'JavaScript': 9, 'TypeScript': 9, 'Java': 8, 'C#': 8,
            'Go': 7, 'Rust': 7, 'Swift': 6, 'C++': 6, 'Ruby': 5, 'PHP': 5,
            'Dart': 4, 'Kotlin': 4, 'Haskell': 3, 'Elixir': 3, 'Clojure': 3
        }

        weighted_scores = {}
        for lang_name, data in language_analysis.items():
            file_weight = data['weight']
            priority_weight = language_priority.get(lang_name, 1)
            weighted_scores[lang_name] = file_weight * priority_weight

        primary_language = max(weighted_scores, key=weighted_scores.get)
        primary_config = language_analysis[primary_language]['config']
    else:
        primary_language = 'Unknown'
        primary_config = _get_enhanced_language_config('')

    return {
        'primary_language': primary_language,
        'primary_config': primary_config,
        'all_languages': language_analysis,
        'detected_frameworks': list(framework_hints),
        'project_patterns': list(project_patterns),
        'is_polyglot': len(language_analysis) > 1,
        'polyglot_complexity': len(language_analysis)
    }

def _extract_json_from_llm(raw_text: str) -> Optional[str]:
    """Enhanced JSON extraction with better error handling."""
    # Try multiple patterns for JSON extraction
    patterns = [
        r'```(?:json)?\s*(\[.*?\])\s*```',  # Standard markdown code block
        r'```(?:json)?\s*(\{.*?\})\s*```',  # Object in code block
        r'(\[(?:[^[\]]|(?1))*\])',          # Recursive bracket matching
        r'(\{(?:[^{}]|(?1))*\})'            # Recursive brace matching
    ]

    for pattern in patterns:
        match = re.search(pattern, raw_text, re.DOTALL)
        if match:
            return match.group(1)

    # Fallback: try to find JSON-like structures
    try:
        start = raw_text.index('[')
        end = raw_text.rindex(']') + 1
        return raw_text[start:end]
    except ValueError:
        try:
            start = raw_text.index('{')
            end = raw_text.rindex('}') + 1
            return raw_text[start:end]
        except ValueError:
            return None

def _validate_and_parse_surgical_plan(json_str: str, language_context: Dict[str, Any]) -> Tuple[Optional[List[Dict]], str]:
    """Enhanced validation with language-specific checks."""
    if not json_str:
        return None, "AI response was empty or did not contain a JSON object."

    try:
        plan = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"AI response was not valid JSON. Parser error: {e}"

    if not isinstance(plan, list):
        return None, f"AI response was not a list of operations. Found type: {type(plan).__name__}."

    # Enhanced operation validation
    valid_operations = {
        "REPLACE_BLOCK": {
            "required": ["file_path", "target_identifier", "content"],
            "description": "Replace an entire function, method, class, or code block"
        },
        "ADD_FIELD_TO_STRUCT": {
            "required": ["file_path", "target_identifier", "content"],
            "description": "Add a new field to a struct (Rust/C/Go)"
        },
        "CREATE_FILE": {
            "required": ["file_path", "content"],
            "description": "Create a new file"
        },
        "INSERT_CODE_AT": {
            "required": ["file_path", "line_number", "content"],
            "description": "Insert code at a specific line number"
        },
        "ADD_IMPORT": {
            "required": ["file_path", "import_statement"],
            "description": "Add an import/require/using statement"
        },
        "REPLACE_FUNCTION": {
            "required": ["file_path", "function_name", "content"],
            "description": "Replace a specific function"
        },
        "ADD_METHOD_TO_CLASS": {
            "required": ["file_path", "class_name", "method_content"],
            "description": "Add a method to an existing class"
        }
    }

    for i, op in enumerate(plan):
        op_num = i + 1
        if not isinstance(op, dict):
            return None, f"Operation #{op_num} is not a valid object."

        operation_type = op.get("operation")
        if not operation_type:
            return None, f"Operation #{op_num} is missing the required 'operation' field."

        if operation_type not in valid_operations:
            return None, f"Operation #{op_num} has an unknown operation type: '{operation_type}'. Valid operations: {list(valid_operations.keys())}"

        required_fields = valid_operations[operation_type]["required"]
        missing = [field for field in required_fields if field not in op]
        if missing:
            return None, f"Operation #{op_num} ('{operation_type}') is missing required fields: {', '.join(missing)}."

        # Language-specific validation
        file_path = op.get("file_path", "")
        if file_path:
            file_config = _get_enhanced_language_config(file_path)
            language_name = file_config['name']

            # Validate operation compatibility with language
            if operation_type == "ADD_FIELD_TO_STRUCT":
                if language_name not in ['Rust', 'C', 'C++', 'Go']:
                    return None, f"Operation #{op_num}: ADD_FIELD_TO_STRUCT is not applicable to {language_name}. Use ADD_METHOD_TO_CLASS for object-oriented languages."

    return plan, ""

def _scout_expand_scope(
    original_contents: Dict[str, str],
    surgical_plan: List[Dict],
    full_file_map: Dict[str, str]
) -> Dict[str, str]:
    """
    Enhanced Scout Service with polyglot awareness.
    Ensures the file scope matches the AI's plan by loading any missing files.
    """
    print("🛰️  Activating Enhanced Scout: Verifying and expanding polyglot file scope...")

    plan_files = {op['file_path'] for op in surgical_plan if 'file_path' in op}

    updated_contents = original_contents.copy()
    expanded_files_loaded = 0
    language_stats = {}

    for file_path in plan_files:
        if file_path not in updated_contents:
            updated_contents[file_path] = full_file_map.get(file_path, "")
            print(f"  → Scout expanded scope to include: {file_path}")
            expanded_files_loaded += 1

            # Track language diversity
            config = _get_enhanced_language_config(file_path)
            language_name = config['name']
            language_stats[language_name] = language_stats.get(language_name, 0) + 1

    if expanded_files_loaded > 0:
        print(f"✓ Scout successfully expanded scope with {expanded_files_loaded} new file(s).")
        if language_stats:
            print(f"  → Languages in expanded scope: {', '.join(language_stats.keys())}")
    else:
        print("✓ File scope is consistent with the AI's plan.")

    return updated_contents

def _generate_language_aware_prompt(
    target_files: List[str],
    instruction: str,
    rca_report: str,
    language_context: Dict[str, Any],
    file_content_section: str,
    refinement_context: str = ""
) -> str:
    """
    Generate a sophisticated, language-aware prompt for the AI.
    """
    primary_language = language_context['primary_language']
    primary_config = language_context['primary_config']
    is_polyglot = language_context['is_polyglot']

    # Build language expertise section
    language_expertise = f"""You are an expert software architect specializing in {primary_language}"""

    if is_polyglot:
        all_languages = list(language_context['all_languages'].keys())
        language_expertise += f" and polyglot development with {', '.join(all_languages)}"

    language_expertise += "."

    # Add language-specific context
    language_context_section = f"""
### LANGUAGE CONTEXT ###
Primary Language: {primary_language}
- Paradigm: {primary_config.get('paradigm', 'unknown')}
- Syntax Style: {primary_config.get('syntax_style', 'unknown')}
- Key Features: {', '.join(primary_config.get('features', []))}
- Common Patterns: {', '.join(primary_config.get('common_patterns', []))}
"""

    if primary_config.get('frameworks'):
        language_context_section += f"- Popular Frameworks: {', '.join(primary_config['frameworks'])}\n"

    if is_polyglot:
        language_context_section += f"""
This is a polyglot project with {language_context['polyglot_complexity']} languages:
"""
        for lang_name, lang_data in language_context['all_languages'].items():
            files = lang_data['files']
            language_context_section += f"- {lang_name}: {len(files)} file(s) - {', '.join(files)}\n"

    if language_context['detected_frameworks']:
        language_context_section += f"Detected Frameworks: {', '.join(language_context['detected_frameworks'])}\n"

    # Enhanced operation examples based on language
    operation_examples = _get_language_specific_operation_examples(primary_language, primary_config)

    surgical_prompt = f"""{language_expertise} Your task is to generate a precise surgical plan to fix a bug in this {'polyglot' if is_polyglot else primary_language} project.

{language_context_section}

<Goal>{instruction}</Goal>
<RCA_Report>{rca_report}</RCA_Report>
{refinement_context}
{file_content_section}

### YOUR TASK ###
Based on all the provided information, create a step-by-step surgical plan as a JSON array that follows {primary_language} best practices{' and handles the polyglot nature of this project' if is_polyglot else ''}.

### AVAILABLE OPERATIONS ###
You can use the following operations in your plan:

1. `"operation": "CREATE_FILE"`: Creates a new file.
   - Required fields: `file_path`, `content`.
   - Use for: New modules, configuration files, or missing dependencies.

2. `"operation": "REPLACE_BLOCK"`: Replaces an entire function, method, class, or other code block.
   - Required fields: `file_path`, `target_identifier` (the unique name/signature), `content`.
   - Use for: Complete rewrites of functions, classes, or major code blocks.

3. `"operation": "ADD_FIELD_TO_STRUCT"`: (For Rust/C/Go) Adds a new field to a struct.
   - Required fields: `file_path`, `target_identifier` (struct name), `content`.
   - Use for: Adding new data fields to existing structures.

4. `"operation": "INSERT_CODE_AT"`: Inserts code at a specific line number.
   - Required fields: `file_path`, `line_number`, `content`.
   - Use for: Adding code at precise locations.

5. `"operation": "ADD_IMPORT"`: Adds an import/require/using statement.
   - Required fields: `file_path`, `import_statement`.
   - Use for: Adding dependencies or modules.

6. `"operation": "REPLACE_FUNCTION"`: Replaces a specific function.
   - Required fields: `file_path`, `function_name`, `content`.
   - Use for: Function-specific changes.

7. `"operation": "ADD_METHOD_TO_CLASS"`: Adds a method to an existing class.
   - Required fields: `file_path`, `class_name`, `method_content`.
   - Use for: Extending classes with new functionality.

{operation_examples}

### IMPORTANT GUIDELINES ###
- Follow {primary_language} naming conventions and style guidelines
- Ensure type safety and error handling appropriate for {primary_language}
- Use idiomatic {primary_language} patterns: {', '.join(primary_config.get('common_patterns', [])[:3])}
{f'- Consider cross-language compatibility in this polyglot project' if is_polyglot else ''}
- Maintain consistency with existing code architecture
- Include proper documentation/comments in the target language style

### RESPONSE FORMAT ###
- You MUST respond with ONLY a valid JSON array `[...]`.
- Do not include any explanations, markdown fences, or other text.
- If you need to modify a file not in the provided context, add an operation for that file.

Generate the {primary_language} surgical plan now."""

    return surgical_prompt

def _get_language_specific_operation_examples(language: str, config: Dict[str, Any]) -> str:
    """Generate language-specific operation examples."""
    examples = f"\n### {language.upper()} SPECIFIC EXAMPLES ###\n"

    if language == "Python":
        examples += """
<Python_Examples>
```json
[
  {
    "operation": "ADD_IMPORT",
    "file_path": "src/main.py",
    "import_statement": "from typing import Optional, Dict"
  },
  {
    "operation": "REPLACE_FUNCTION",
    "file_path": "src/utils.py",
    "function_name": "process_data",
    "content": "def process_data(data: Dict[str, Any]) -> Optional[str]:\\n    \\"\\"\\"Process data with type safety.\\"\\"\\"\\n    if not data:\\n        return None\\n    return str(data.get('result', ''))"
  }
]
```
</Python_Examples>"""

    elif language in ["JavaScript", "TypeScript"]:
        examples += """
<JavaScript_TypeScript_Examples>
```json
[
  {
    "operation": "ADD_IMPORT",
    "file_path": "src/utils.js",
    "import_statement": "import { validateInput } from './validators';"
  },
  {
    "operation": "REPLACE_FUNCTION",
    "file_path": "src/api.js",
    "function_name": "fetchData",
    "content": "async function fetchData(url) {\\n  try {\\n    const response = await fetch(url);\\n    if (!response.ok) throw new Error('Network error');\\n    return await response.json();\\n  } catch (error) {\\n    console.error('Fetch failed:', error);\\n    throw error;\\n  }\\n}"
  }
]
```
</JavaScript_TypeScript_Examples>"""

    elif language == "Java":
        examples += """
<Java_Examples>
```json
[
  {
    "operation": "ADD_IMPORT",
    "file_path": "src/main/java/Main.java",
    "import_statement": "import java.util.Optional;"
  },
  {
    "operation": "ADD_METHOD_TO_CLASS",
    "file_path": "src/main/java/UserService.java",
    "class_name": "UserService",
    "method_content": "    public Optional<User> findUserById(Long id) {\\n        if (id == null || id <= 0) {\\n            return Optional.empty();\\n        }\\n        return userRepository.findById(id);\\n    }"
  }
]
```
</Java_Examples>"""

    elif language == "Go":
        examples += """
<Go_Examples>
```json
[
  {
    "operation": "ADD_FIELD_TO_STRUCT",
    "file_path": "internal/models/user.go",
    "target_identifier": "User",
    "content": "    Email    string `json:\\"email\\" validate:\\"required,email\\"`"
  },
  {
    "operation": "REPLACE_FUNCTION",
    "file_path": "internal/handlers/user.go",
    "function_name": "CreateUser",
    "content": "func CreateUser(w http.ResponseWriter, r *http.Request) {\\n    var user User\\n    if err := json.NewDecoder(r.Body).Decode(&user); err != nil {\\n        http.Error(w, \\"Invalid JSON\\", http.StatusBadRequest)\\n        return\\n    }\\n    // Process user creation\\n    w.WriteHeader(http.StatusCreated)\\n}"
  }
]
```
</Go_Examples>"""

    elif language == "Rust":
        examples += """
<Rust_Examples>
```json
[
  {
    "operation": "ADD_FIELD_TO_STRUCT",
    "file_path": "src/models/user.rs",
    "target_identifier": "User",
    "content": "    pub email: Option<String>,"
  },
  {
    "operation": "REPLACE_FUNCTION",
    "file_path": "src/lib.rs",
    "function_name": "process_data",
    "content": "pub fn process_data(input: &str) -> Result<String, Box<dyn std::error::Error>> {\\n    if input.is_empty() {\\n        return Err(\\"Input cannot be empty\\".into());\\n    }\\n    Ok(input.to_uppercase())\\n}"
  }
]
```
</Rust_Examples>"""

    else:
        # Generic example for other languages
        examples += f"""
<{language}_Examples>
```json
[
  {{
    "operation": "REPLACE_BLOCK",
    "file_path": "src/main{config.get('extension', '')}",
    "target_identifier": "main_function",
    "content": "// Enhanced main function with proper error handling\\n// TODO: Implement based on {language} best practices"
  }}
]
```
</{language}_Examples>"""

    return examples

def generate_scaffold(
    repo_id: str,
    target_files: List[str],
    instruction: str,
    rca_report: str,
    refinement_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Enhanced Code Scaffolding with comprehensive polyglot support and dynamic scope expansion.

    This version provides:
    - Multi-language project detection and analysis
    - Language-aware prompt generation
    - Framework-specific code generation hints
    - Enhanced operation validation
    - Polyglot project complexity handling
    """
    print(f"🔧 Initiating Enhanced Polyglot Surgical Scaffolding for {target_files} in repo '{repo_id}'")

    try:
        # Load and validate cortex data
        backend_dir = Path(__file__).resolve().parent.parent.parent
        cortex_path = backend_dir / "cloned_repositories" / repo_id / f"{repo_id}_cortex.json"
        if not cortex_path.exists():
            return {"error": "Cortex file not found", "details": f"Expected path: {cortex_path}"}

        with open(cortex_path, 'r', encoding='utf-8') as f:
            cortex_data = json.load(f)

        file_map = {file['file_path']: file['raw_content'] for file in cortex_data.get('files', [])}
        original_contents = {fp: file_map.get(fp, "") for fp in target_files}

        # Enhanced polyglot context detection
        language_context = _detect_polyglot_context(target_files, cortex_data)

        print(f"  🌐 Detected project context:")
        print(f"     • Primary language: {language_context['primary_language']}")
        print(f"     • Project type: {'Polyglot' if language_context['is_polyglot'] else 'Monoglot'}")
        if language_context['is_polyglot']:
            print(f"     • Languages involved: {list(language_context['all_languages'].keys())}")
        if language_context['detected_frameworks']:
            print(f"     • Frameworks detected: {', '.join(language_context['detected_frameworks'])}")

        # --- STEP 1: GENERATE THE ENHANCED SURGICAL PLAN ---
        file_content_prompt_section = "\n\n### RELEVANT EXISTING CODE\n"
        for path, content in original_contents.items():
            if content:
                # Enhanced compression with language awareness
                compressed_content = code_surgery.get_relevant_code_from_cortex(content, rca_report, path)
                language_config = _get_enhanced_language_config(path)
                file_content_prompt_section += f"<file path=\"{path}\" language=\"{language_config['name']}\">\n{compressed_content}\n</file>\n\n"

        refinement_context = ""
        if refinement_history:
            refinement_context = "\n\n### PREVIOUS REFINEMENT ATTEMPTS\n"
            for i, refinement in enumerate(refinement_history[-2:]):
                feedback = refinement.get("feedback", "No feedback provided.")
                refinement_context += f"Attempt {i+1} Feedback: {feedback}\n"
            refinement_context += "\nPlease learn from the previous feedback and generate a better plan that follows language-specific best practices.\n"

        # Generate sophisticated language-aware prompt
        surgical_prompt = _generate_language_aware_prompt(
            target_files, instruction, rca_report, language_context,
            file_content_prompt_section, refinement_context
        )

        max_retries = 3
        surgical_plan = None
        last_llm_response = ""

        for attempt in range(max_retries):
            print(f"  🤖 Attempt {attempt + 1}: Calling LLM for {language_context['primary_language']} surgical plan...")

            # Enhanced LLM call with appropriate task type
            task_type = TaskType.CODE_GENERATION
            if language_context['is_polyglot'] or language_context['polyglot_complexity'] > 2:
                task_type = TaskType.COMPLEX_REASONING

            llm_response = llm_service.generate_text(
                surgical_prompt,
                task_type=task_type
            )
            last_llm_response = llm_response

            if not llm_response or not llm_response.strip():
                print("  ⚠️ Empty response, retrying...")
                continue

            # Enhanced JSON extraction and validation
            json_str = _extract_json_from_llm(llm_response)
            if json_str:
                plan, error_msg = _validate_and_parse_surgical_plan(json_str, language_context)
                if plan:
                    surgical_plan = plan
                    print(f"  ✓ Successfully parsed {language_context['primary_language']} surgical plan with {len(plan)} operations.")

                    # Log plan summary
                    operation_summary = {}
                    for op in plan:
                        op_type = op.get('operation', 'unknown')
                        operation_summary[op_type] = operation_summary.get(op_type, 0) + 1
                    print(f"    → Operations: {', '.join(f'{count}x {op}' for op, count in operation_summary.items())}")
                    break
                else:
                    print(f"  ❌ Blueprint Rejected (Attempt {attempt+1}/{max_retries}): {error_msg}")
                    print(f"  📝 Faulty JSON received: {json_str[:250]}...")
            else:
                print(f"  ❌ Could not extract JSON from LLM response (Attempt {attempt+1}/{max_retries}).")

        if not surgical_plan:
            return {
                "error": f"Failed to generate a valid {language_context['primary_language']} surgical plan after {max_retries} attempts.",
                "details": "The AI's final proposed plan was malformed or incomplete. This may be due to the complexity of the polyglot project or language-specific constraints.",
                "llm_response": last_llm_response,
                "language_context": language_context,
            }

        # Enhanced Scout with polyglot awareness
        final_contents_for_surgery = _scout_expand_scope(
            original_contents,
            surgical_plan,
            file_map
        )

        # Execute surgical plan with enhanced error handling
        print("🔬 Executing enhanced surgical plan...")
        modified_files, surgery_report = code_surgery.execute_surgical_plan(
            final_contents_for_surgery,
            surgical_plan
        )

        if surgery_report.get("errors"):
            return {
                "error": "Enhanced Code Surgery failed to apply the plan.",
                "details": surgery_report["errors"],
                "language_context": language_context,
                "partial_results": surgery_report.get("successes", [])
            }

        # Enhanced success reporting
        print("🎉 Enhanced polyglot surgical scaffolding completed successfully!")
        print(f"  ✓ Modified {len(modified_files)} files")
        print(f"  ✓ Applied {len(surgical_plan)} operations")
        print(f"  ✓ Primary language: {language_context['primary_language']}")

        if language_context['is_polyglot']:
            print(f"  ✓ Handled polyglot complexity: {language_context['polyglot_complexity']} languages")

        return {
            "modified_files": modified_files,
            "original_contents": final_contents_for_surgery,
            "plan_executed": surgical_plan,
            "surgery_report": surgery_report,
            "language_context": language_context,
            "polyglot_summary": {
                "primary_language": language_context['primary_language'],
                "is_polyglot": language_context['is_polyglot'],
                "languages_involved": list(language_context['all_languages'].keys()),
                "frameworks_detected": language_context['detected_frameworks'],
                "complexity_score": language_context['polyglot_complexity']
            }
        }

    except Exception as e:
        print(f"❌ Critical error in enhanced polyglot scaffolding: {str(e)}")
        return {
            "error": "A critical error occurred in the enhanced polyglot scaffolding service.",
            "details": str(e),
            "traceback": traceback.format_exc(),
        }

# --- Enhanced utility functions for polyglot support ---

def analyze_code_complexity(target_files: List[str], cortex_data: Dict) -> Dict[str, Any]:
    """
    Analyze the complexity of a polyglot codebase for better scaffolding decisions.
    """
    if not cortex_data or 'files' not in cortex_data:
        return {"error": "Invalid cortex data"}

    complexity_analysis = {
        "total_files": len(target_files),
        "languages": {},
        "frameworks": set(),
        "complexity_indicators": {},
        "recommendations": []
    }

    for file_path in target_files:
        config = _get_enhanced_language_config(file_path)
        language = config['name']

        if language not in complexity_analysis["languages"]:
            complexity_analysis["languages"][language] = {
                "file_count": 0,
                "total_lines": 0,
                "paradigm": config.get('paradigm', 'unknown'),
                "features": config.get('features', [])
            }

        complexity_analysis["languages"][language]["file_count"] += 1

        # Find corresponding file in cortex
        for file_cortex in cortex_data['files']:
            if file_cortex['file_path'] == file_path:
                lines = len(file_cortex['raw_content'].splitlines())
                complexity_analysis["languages"][language]["total_lines"] += lines

                if 'framework_hints' in file_cortex:
                    complexity_analysis["frameworks"].update(file_cortex['framework_hints'])
                break

    # Convert set to list for JSON serialization
    complexity_analysis["frameworks"] = list(complexity_analysis["frameworks"])

    # Generate complexity indicators
    language_count = len(complexity_analysis["languages"])
    if language_count > 1:
        complexity_analysis["complexity_indicators"]["polyglot"] = True
        complexity_analysis["complexity_indicators"]["language_diversity"] = language_count

    total_lines = sum(lang["total_lines"] for lang in complexity_analysis["languages"].values())
    complexity_analysis["complexity_indicators"]["total_lines"] = total_lines

    if total_lines > 10000:
        complexity_analysis["complexity_indicators"]["large_codebase"] = True

    # Generate recommendations
    if language_count > 3:
        complexity_analysis["recommendations"].append(
            "High language diversity detected. Consider breaking down changes into language-specific phases."
        )

    if len(complexity_analysis["frameworks"]) > 2:
        complexity_analysis["recommendations"].append(
            "Multiple frameworks detected. Ensure cross-framework compatibility."
        )

    return complexity_analysis

def get_language_specific_best_practices(language: str) -> List[str]:
    """
    Return language-specific best practices for code generation.
    """
    practices = {
        "Python": [
            "Follow PEP 8 style guidelines",
            "Use type hints for better code documentation",
            "Implement proper exception handling",
            "Use list comprehensions and generator expressions where appropriate",
            "Follow the principle of 'Pythonic' code"
        ],
        "JavaScript": [
            "Use const and let instead of var",
            "Implement proper error handling with try-catch",
            "Use async/await for asynchronous operations",
            "Follow ESLint recommendations",
            "Use destructuring and modern ES6+ features"
        ],
        "TypeScript": [
            "Leverage strong typing features",
            "Use interfaces and type definitions",
            "Implement proper generic types",
            "Use strict mode configuration",
            "Follow Angular/React specific patterns if applicable"
        ],
        "Java": [
            "Follow Java naming conventions",
            "Use dependency injection patterns",
            "Implement proper exception handling",
            "Use Optional for nullable values",
            "Follow SOLID principles"
        ],
        "Go": [
            "Follow Go formatting standards (gofmt)",
            "Use goroutines and channels for concurrency",
            "Implement proper error handling",
            "Keep interfaces small and focused",
            "Use composition over inheritance"
        ],
        "Rust": [
            "Leverage ownership and borrowing system",
            "Use Result and Option types for error handling",
            "Implement proper trait patterns",
            "Use pattern matching effectively",
            "Follow Rust naming conventions"
        ]
    }

    return practices.get(language, [
        "Follow language-specific style guidelines",
        "Implement proper error handling",
        "Use idiomatic patterns for the language",
        "Ensure code readability and maintainability"
    ])

def detect_framework_patterns(file_content: str, language: str) -> List[str]:
    """
    Detect framework-specific patterns in code for better scaffolding.
    """
    patterns = []
    content_lower = file_content.lower()

    framework_signatures = {
        "react": ["usestate", "useeffect", "jsx", "react.component"],
        "vue": ["vue.component", "template>", "@click", "v-if"],
        "angular": ["@component", "@injectable", "ngmodule", "ngoninit"],
        "express": ["app.get", "app.post", "express()", "router."],
        "flask": ["@app.route", "flask()", "request."],
        "django": ["models.model", "views.view", "urls.py", "django."],
        "spring": ["@controller", "@service", "@autowired", "@requestmapping"],
        "laravel": ["route::", "eloquent", "blade.php", "artisan"]
    }

    for framework, signatures in framework_signatures.items():
        if any(sig in content_lower for sig in signatures):
            patterns.append(framework)

    return patterns

# --- Backward Compatibility ---
def generate_scaffold_legacy(repo_id: str, target_files: List[str], instruction: str, rca_report: str) -> Dict[str, Any]:
    """Legacy interface for backward compatibility."""
    return generate_scaffold(repo_id, target_files, instruction, rca_report)
