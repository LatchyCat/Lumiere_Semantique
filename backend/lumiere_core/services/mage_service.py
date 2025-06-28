# backend/lumiere_core/services/mage_service.py

import logging
import re
import ast
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from . import llm_service, cortex_service, code_surgery
from .llm_service import TaskType

logger = logging.getLogger(__name__)


class MageService:
    """The Mage - master of code transmutation and intelligent transformations."""
    
    def __init__(self):
        self.available_spells = {
            "translate_contract": self._spell_translate_contract,
            "transmute_implementation": self._spell_transmute_implementation,
            "refactor_pattern": self._spell_refactor_pattern,
            "modernize_syntax": self._spell_modernize_syntax,
            "add_type_hints": self._spell_add_type_hints,
            "extract_method": self._spell_extract_method,
            "inline_method": self._spell_inline_method,
            "convert_to_async": self._spell_convert_to_async
        }
    
    def list_spells(self) -> Dict[str, str]:
        """Return available spells and their descriptions."""
        return {
            "translate_contract": "Translate class/interface contracts between languages",
            "transmute_implementation": "Transform implementation style (e.g., procedural to functional)",
            "refactor_pattern": "Apply design patterns or refactor existing patterns",
            "modernize_syntax": "Update code to use modern language features",
            "add_type_hints": "Add type annotations to Python code",
            "extract_method": "Extract code into a separate method",
            "inline_method": "Inline a method's implementation",
            "convert_to_async": "Convert synchronous code to asynchronous"
        }
    
    def cast_spell(self, repo_id: str, spell_name: str, file_path: str, 
                   target_identifier: str, **kwargs) -> Dict[str, Any]:
        """
        Cast a spell to transform code.
        
        Args:
            repo_id: Repository identifier
            spell_name: Name of the spell to cast
            file_path: Path to the file containing the code
            target_identifier: Function/class/method name to transform
            **kwargs: Additional spell-specific parameters
            
        Returns:
            Dictionary with transformation result
        """
        try:
            if spell_name not in self.available_spells:
                return {"error": f"Unknown spell: {spell_name}. Available spells: {list(self.available_spells.keys())}"}
            
            # Get the current code
            file_content = cortex_service.get_file_content(repo_id, file_path)
            if not file_content:
                return {"error": f"Could not retrieve content for file: {file_path}"}
            
            # Find the target code block
            target_code, start_line, end_line = self._find_code_block(file_content, target_identifier)
            if not target_code:
                return {"error": f"Could not find '{target_identifier}' in {file_path}"}
            
            # Cast the spell
            spell_func = self.available_spells[spell_name]
            result = spell_func(target_code, target_identifier, file_path, **kwargs)
            
            if "error" in result:
                return result
            
            # Add metadata
            result.update({
                "original_code": target_code,
                "file_path": file_path,
                "target_identifier": target_identifier,
                "spell_name": spell_name,
                "start_line": start_line,
                "end_line": end_line
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error casting spell {spell_name}: {e}")
            return {"error": str(e)}
    
    def apply_transformation(self, repo_id: str, transformation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a transformation result to the actual file using code surgery.
        
        Args:
            repo_id: Repository identifier
            transformation_result: Result from cast_spell()
            
        Returns:
            Dictionary with application result
        """
        try:
            if "transformed_code" not in transformation_result:
                return {"error": "No transformed code to apply"}
            
            file_path = transformation_result["file_path"]
            target_identifier = transformation_result["target_identifier"]
            new_code = transformation_result["transformed_code"]
            
            # Use code surgery to apply the transformation
            surgery_result = code_surgery.replace_block(repo_id, file_path, target_identifier, new_code)
            
            if surgery_result.get("success"):
                return {
                    "success": True,
                    "message": f"Transformation applied successfully to {file_path}",
                    "changes": surgery_result.get("changes", {})
                }
            else:
                return {"error": f"Failed to apply transformation: {surgery_result.get('error', 'Unknown error')}"}
                
        except Exception as e:
            logger.error(f"Error applying transformation: {e}")
            return {"error": str(e)}
    
    def _find_code_block(self, file_content: str, identifier: str) -> Tuple[str, int, int]:
        """
        Find a code block (function, class, method) in the file content.
        
        Returns:
            Tuple of (code_block, start_line, end_line)
        """
        lines = file_content.split('\n')
        
        # Try to parse as Python first
        try:
            tree = ast.parse(file_content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    if node.name == identifier:
                        start_line = node.lineno
                        end_line = node.end_lineno or start_line
                        code_block = '\n'.join(lines[start_line-1:end_line])
                        return code_block, start_line, end_line
                elif isinstance(node, ast.FunctionDef) and hasattr(node, 'parent'):
                    # Handle class methods
                    if f"{node.parent.name}.{node.name}" == identifier:
                        start_line = node.lineno
                        end_line = node.end_lineno or start_line
                        code_block = '\n'.join(lines[start_line-1:end_line])
                        return code_block, start_line, end_line
        except SyntaxError:
            # Fall back to regex if AST parsing fails
            pass
        
        # Fallback to regex-based search
        patterns = [
            rf'^(\s*)def\s+{re.escape(identifier)}\s*\(',  # Python function
            rf'^(\s*)class\s+{re.escape(identifier)}\s*\(',  # Python class
            rf'^(\s*)async\s+def\s+{re.escape(identifier)}\s*\(',  # Python async function
            rf'^(\s*)function\s+{re.escape(identifier)}\s*\(',  # JavaScript function
            rf'^(\s*){re.escape(identifier)}\s*:\s*function\s*\(',  # JavaScript method
        ]
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    # Find the end of the block by tracking indentation
                    start_line = i + 1
                    base_indent = len(match.group(1)) if match.groups() else 0
                    end_line = start_line
                    
                    # Simple block detection based on indentation
                    for j in range(i + 1, len(lines)):
                        line_content = lines[j].strip()
                        if not line_content:  # Skip empty lines
                            continue
                        
                        current_indent = len(lines[j]) - len(lines[j].lstrip())
                        if current_indent <= base_indent and line_content:
                            end_line = j
                            break
                        end_line = j + 1
                    
                    code_block = '\n'.join(lines[start_line-1:end_line])
                    return code_block, start_line, end_line
        
        return "", 0, 0
    
    def _spell_translate_contract(self, code: str, identifier: str, file_path: str, 
                                 target_language: str = "typescript", **kwargs) -> Dict[str, Any]:
        """Translate class/interface contracts between languages."""
        
        language_prompts = {
            "typescript": """You are an expert polyglot developer. Translate the following Python class into a TypeScript interface/type. 
            Ensure perfect type correspondence and idiomatic TypeScript syntax.
            
            Focus on:
            - Converting Python types to TypeScript equivalents
            - Handling optional vs required fields
            - Using appropriate TypeScript conventions
            - Maintaining the original intent and structure
            
            Python Code:
            {code}
            
            Return ONLY the TypeScript interface/type definition:""",
            
            "python": """You are an expert polyglot developer. Translate the following TypeScript interface/type into a Python class using Pydantic or dataclasses.
            
            Focus on:
            - Converting TypeScript types to Python type hints
            - Using appropriate Python conventions
            - Handling optional vs required fields
            - Maintaining the original intent and structure
            
            TypeScript Code:
            {code}
            
            Return ONLY the Python class definition:""",
            
            "java": """You are an expert polyglot developer. Translate the following code into a Java class/interface.
            
            Focus on:
            - Converting types to Java equivalents
            - Using appropriate Java conventions (getters/setters, builders)
            - Maintaining the original intent and structure
            
            Code:
            {code}
            
            Return ONLY the Java class/interface definition:"""
        }
        
        if target_language not in language_prompts:
            return {"error": f"Unsupported target language: {target_language}"}
        
        prompt = language_prompts[target_language].format(code=code)
        
        try:
            transformed_code = llm_service.generate_text(prompt, task_type=TaskType.COMPLEX_REASONING)
            
            return {
                "transformed_code": self._clean_code_output(transformed_code),
                "transformation_type": f"translate_to_{target_language}",
                "description": f"Translated {identifier} to {target_language}"
            }
        except Exception as e:
            return {"error": f"Translation failed: {str(e)}"}
    
    def _spell_transmute_implementation(self, code: str, identifier: str, file_path: str,
                                      target_style: str = "functional", **kwargs) -> Dict[str, Any]:
        """Transform implementation style (e.g., procedural to functional)."""
        
        style_prompts = {
            "functional": """You are an expert in functional programming. Transform this code to use functional programming principles.
            
            Focus on:
            - Converting loops to map/filter/reduce operations
            - Eliminating side effects where possible
            - Using immutable data structures
            - Applying functional composition
            - Maintaining the same behavior and output
            
            Original Code:
            {code}
            
            Return ONLY the functionally-transformed code:""",
            
            "object_oriented": """You are an expert in object-oriented design. Transform this code to use object-oriented principles.
            
            Focus on:
            - Encapsulating data and behavior in classes
            - Applying appropriate design patterns
            - Using inheritance and composition where beneficial
            - Maintaining single responsibility principle
            - Keeping the same behavior and output
            
            Original Code:
            {code}
            
            Return ONLY the object-oriented code:""",
            
            "async": """You are an expert in asynchronous programming. Transform this code to use async/await patterns.
            
            Focus on:
            - Converting blocking operations to async
            - Using appropriate async libraries
            - Handling concurrent operations efficiently
            - Maintaining error handling
            - Keeping the same behavior and output
            
            Original Code:
            {code}
            
            Return ONLY the asynchronous code:"""
        }
        
        if target_style not in style_prompts:
            return {"error": f"Unsupported target style: {target_style}"}
        
        prompt = style_prompts[target_style].format(code=code)
        
        try:
            transformed_code = llm_service.generate_text(prompt, task_type=TaskType.COMPLEX_REASONING)
            
            return {
                "transformed_code": self._clean_code_output(transformed_code),
                "transformation_type": f"transmute_to_{target_style}",
                "description": f"Transformed {identifier} to {target_style} style"
            }
        except Exception as e:
            return {"error": f"Transmutation failed: {str(e)}"}
    
    def _spell_refactor_pattern(self, code: str, identifier: str, file_path: str,
                               pattern: str = "strategy", **kwargs) -> Dict[str, Any]:
        """Apply design patterns or refactor existing patterns."""
        
        pattern_prompts = {
            "strategy": """Refactor this code to use the Strategy design pattern.
            
            Extract different algorithms/behaviors into separate strategy classes.
            Create a context class that uses these strategies.
            Maintain the same external interface and behavior.
            
            Original Code:
            {code}
            
            Return the refactored code using Strategy pattern:""",
            
            "factory": """Refactor this code to use the Factory design pattern.
            
            Extract object creation logic into factory methods or classes.
            Simplify the main code by delegating object creation.
            Maintain the same functionality and behavior.
            
            Original Code:
            {code}
            
            Return the refactored code using Factory pattern:""",
            
            "observer": """Refactor this code to use the Observer design pattern.
            
            Separate concerns by implementing observer/subject relationships.
            Allow multiple observers to react to changes.
            Maintain loose coupling between components.
            
            Original Code:
            {code}
            
            Return the refactored code using Observer pattern:"""
        }
        
        if pattern not in pattern_prompts:
            return {"error": f"Unsupported pattern: {pattern}"}
        
        prompt = pattern_prompts[pattern].format(code=code)
        
        try:
            transformed_code = llm_service.generate_text(prompt, task_type=TaskType.COMPLEX_REASONING)
            
            return {
                "transformed_code": self._clean_code_output(transformed_code),
                "transformation_type": f"apply_{pattern}_pattern",
                "description": f"Applied {pattern} pattern to {identifier}"
            }
        except Exception as e:
            return {"error": f"Pattern application failed: {str(e)}"}
    
    def _spell_modernize_syntax(self, code: str, identifier: str, file_path: str, **kwargs) -> Dict[str, Any]:
        """Update code to use modern language features."""
        
        # Detect language from file extension
        extension = Path(file_path).suffix.lower()
        
        if extension == ".py":
            prompt = """Modernize this Python code to use the latest Python features and best practices.
            
            Focus on:
            - Using f-strings instead of .format() or %
            - Using pathlib instead of os.path
            - Using type hints where appropriate
            - Using dataclasses or attrs where beneficial
            - Using context managers (with statements)
            - Using list/dict comprehensions where appropriate
            - Using modern Python 3.8+ features (walrus operator, positional-only params, etc.)
            
            Original Code:
            {code}
            
            Return ONLY the modernized Python code:"""
        elif extension in [".js", ".ts"]:
            prompt = """Modernize this JavaScript/TypeScript code to use modern ES6+ features.
            
            Focus on:
            - Using arrow functions where appropriate
            - Using const/let instead of var
            - Using template literals instead of string concatenation
            - Using destructuring assignments
            - Using async/await instead of callbacks
            - Using modules (import/export)
            - Using modern array methods (map, filter, reduce)
            
            Original Code:
            {code}
            
            Return ONLY the modernized JavaScript/TypeScript code:"""
        else:
            return {"error": f"Language modernization not supported for {extension} files"}
        
        prompt = prompt.format(code=code)
        
        try:
            transformed_code = llm_service.generate_text(prompt, task_type=TaskType.COMPLEX_REASONING)
            
            return {
                "transformed_code": self._clean_code_output(transformed_code),
                "transformation_type": "modernize_syntax",
                "description": f"Modernized syntax for {identifier}"
            }
        except Exception as e:
            return {"error": f"Modernization failed: {str(e)}"}
    
    def _spell_add_type_hints(self, code: str, identifier: str, file_path: str, **kwargs) -> Dict[str, Any]:
        """Add type annotations to Python code."""
        
        prompt = """Add comprehensive type hints to this Python code.
        
        Focus on:
        - Adding type hints to function parameters and return values
        - Using appropriate types from typing module (List, Dict, Optional, Union, etc.)
        - Inferring types from the code logic and usage
        - Adding docstrings with type information if not present
        - Using modern type hint syntax (Python 3.9+ where applicable)
        
        Original Code:
        {code}
        
        Return ONLY the code with complete type hints:"""
        
        prompt = prompt.format(code=code)
        
        try:
            transformed_code = llm_service.generate_text(prompt, task_type=TaskType.COMPLEX_REASONING)
            
            return {
                "transformed_code": self._clean_code_output(transformed_code),
                "transformation_type": "add_type_hints",
                "description": f"Added type hints to {identifier}"
            }
        except Exception as e:
            return {"error": f"Type hint addition failed: {str(e)}"}
    
    def _spell_extract_method(self, code: str, identifier: str, file_path: str,
                             method_name: str = "extracted_method", **kwargs) -> Dict[str, Any]:
        """Extract code into a separate method."""
        
        prompt = """Extract the specified logic into a separate method to improve code organization.
        
        Focus on:
        - Identifying cohesive blocks of code that can be extracted
        - Creating a well-named method with appropriate parameters
        - Ensuring the extracted method has a single responsibility
        - Maintaining the same behavior and output
        - Adding appropriate documentation
        
        Method name to create: {method_name}
        
        Original Code:
        {code}
        
        Return the refactored code with the extracted method:"""
        
        prompt = prompt.format(code=code, method_name=method_name)
        
        try:
            transformed_code = llm_service.generate_text(prompt, task_type=TaskType.COMPLEX_REASONING)
            
            return {
                "transformed_code": self._clean_code_output(transformed_code),
                "transformation_type": "extract_method",
                "description": f"Extracted method '{method_name}' from {identifier}"
            }
        except Exception as e:
            return {"error": f"Method extraction failed: {str(e)}"}
    
    def _spell_inline_method(self, code: str, identifier: str, file_path: str, **kwargs) -> Dict[str, Any]:
        """Inline a method's implementation."""
        
        prompt = """Inline the method calls to simplify the code structure.
        
        Focus on:
        - Replacing method calls with the actual implementation
        - Maintaining the same behavior and output
        - Simplifying the code where appropriate
        - Preserving important comments and logic
        
        Original Code:
        {code}
        
        Return the code with methods inlined:"""
        
        prompt = prompt.format(code=code)
        
        try:
            transformed_code = llm_service.generate_text(prompt, task_type=TaskType.COMPLEX_REASONING)
            
            return {
                "transformed_code": self._clean_code_output(transformed_code),
                "transformation_type": "inline_method",
                "description": f"Inlined method calls in {identifier}"
            }
        except Exception as e:
            return {"error": f"Method inlining failed: {str(e)}"}
    
    def _spell_convert_to_async(self, code: str, identifier: str, file_path: str, **kwargs) -> Dict[str, Any]:
        """Convert synchronous code to asynchronous."""
        
        prompt = """Convert this synchronous code to use async/await patterns.
        
        Focus on:
        - Converting blocking operations to async equivalents
        - Using appropriate async libraries (aiohttp, aiofiles, etc.)
        - Handling concurrent operations where beneficial
        - Maintaining proper error handling
        - Adding async/await keywords where needed
        - Keeping the same behavior and output
        
        Original Code:
        {code}
        
        Return ONLY the asynchronous version of the code:"""
        
        prompt = prompt.format(code=code)
        
        try:
            transformed_code = llm_service.generate_text(prompt, task_type=TaskType.COMPLEX_REASONING)
            
            return {
                "transformed_code": self._clean_code_output(transformed_code),
                "transformation_type": "convert_to_async",
                "description": f"Converted {identifier} to async"
            }
        except Exception as e:
            return {"error": f"Async conversion failed: {str(e)}"}
    
    def _clean_code_output(self, llm_output: str) -> str:
        """Clean and format the LLM output to extract just the code."""
        # Remove code fence markers
        cleaned = re.sub(r'^```\w*\n', '', llm_output, flags=re.MULTILINE)
        cleaned = re.sub(r'\n```$', '', cleaned, flags=re.MULTILINE)
        
        # Remove leading/trailing whitespace but preserve internal formatting
        cleaned = cleaned.strip()
        
        # Remove common explanatory text
        lines = cleaned.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # Skip explanatory lines before code
            if not in_code and any(keyword in line.lower() for keyword in 
                                 ['here is', 'here\'s', 'this is', 'the code', 'transformed', 'refactored']):
                continue
            
            # Start collecting code
            if not in_code and (line.strip().startswith(('def ', 'class ', 'function ', 'const ', 'let ', 'var ')) or
                               any(line.strip().startswith(keyword) for keyword in ['import ', 'from ', 'interface ', 'type '])):
                in_code = True
            
            if in_code:
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else cleaned


# Global instance
_mage_service = None

def get_mage_service() -> MageService:
    """Get or create the global Mage service instance."""
    global _mage_service
    if _mage_service is None:
        _mage_service = MageService()
    return _mage_service

# Public API
def list_available_spells() -> Dict[str, str]:
    """Get list of available spells and their descriptions."""
    service = get_mage_service()
    return service.list_spells()

def cast_transformation_spell(repo_id: str, spell_name: str, file_path: str, 
                             target_identifier: str, **kwargs) -> Dict[str, Any]:
    """Cast a transformation spell on code."""
    service = get_mage_service()
    return service.cast_spell(repo_id, spell_name, file_path, target_identifier, **kwargs)

def apply_code_transformation(repo_id: str, transformation_result: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a transformation to the actual source file."""
    service = get_mage_service()
    return service.apply_transformation(repo_id, transformation_result)