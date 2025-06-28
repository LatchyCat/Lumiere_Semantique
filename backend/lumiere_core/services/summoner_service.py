# backend/lumiere_core/services/summoner_service.py

import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from . import llm_service, cortex_service, oracle_service, code_surgery
from .llm_service import TaskType

logger = logging.getLogger(__name__)


class SummonerService:
    """The Summoner - intelligent familiar that materializes entire code patterns."""
    
    def __init__(self):
        self.recipes = {
            "fastapi_endpoint": {
                "name": "FastAPI Endpoint",
                "description": "Create FastAPI endpoint with schemas, handlers, and tests",
                "files": ["schemas", "routes", "tests"],
                "parameters": ["path", "methods", "model_name"]
            },
            "django_rest_viewset": {
                "name": "Django REST ViewSet",
                "description": "Create Django REST framework ViewSet with serializers",
                "files": ["models", "serializers", "views", "urls", "tests"],
                "parameters": ["model_name", "fields"]
            },
            "react_component": {
                "name": "React Component",
                "description": "Create React component with hooks, styles, and tests",
                "files": ["component", "styles", "tests", "types"],
                "parameters": ["component_name", "props"]
            },
            "microservice": {
                "name": "Microservice Structure",
                "description": "Create complete microservice with API, models, and Docker",
                "files": ["api", "models", "config", "docker", "tests"],
                "parameters": ["service_name", "database"]
            },
            "database_model": {
                "name": "Database Model",
                "description": "Create database model with migrations and queries",
                "files": ["model", "migration", "repository", "tests"],
                "parameters": ["model_name", "fields", "relationships"]
            }
        }
        
        self.oracle = oracle_service.OracleService()
    
    def list_recipes(self) -> Dict[str, Dict[str, Any]]:
        """Get available summoning recipes."""
        return self.recipes
    
    def summon_pattern(self, repo_id: str, recipe_name: str, **parameters) -> Dict[str, Any]:
        """
        Summon a complete code pattern based on a recipe.
        
        Args:
            repo_id: Repository identifier
            recipe_name: Name of the recipe to summon
            **parameters: Recipe-specific parameters
            
        Returns:
            Dictionary with summoning result
        """
        try:
            if recipe_name not in self.recipes:
                return {"error": f"Unknown recipe: {recipe_name}. Available: {list(self.recipes.keys())}"}
            
            recipe = self.recipes[recipe_name]
            
            # Step 1: Learn project patterns
            logger.info(f"Learning project patterns for {repo_id}")
            pattern_context = self._learn_project_patterns(repo_id)
            
            if "error" in pattern_context:
                return pattern_context
            
            # Step 2: Generate surgical plan
            logger.info(f"Generating surgical plan for {recipe_name}")
            surgical_plan = self._generate_surgical_plan(recipe, pattern_context, parameters)
            
            if "error" in surgical_plan:
                return surgical_plan
            
            # Step 3: Return plan for approval
            return {
                "recipe_name": recipe_name,
                "recipe_description": recipe["description"],
                "pattern_context": pattern_context,
                "surgical_plan": surgical_plan,
                "ready_to_execute": True
            }
            
        except Exception as e:
            logger.error(f"Error summoning pattern {recipe_name}: {e}")
            return {"error": str(e)}
    
    def execute_summoning(self, repo_id: str, summoning_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the summoning plan using code surgery.
        
        Args:
            repo_id: Repository identifier
            summoning_result: Result from summon_pattern()
            
        Returns:
            Dictionary with execution result
        """
        try:
            if not summoning_result.get("ready_to_execute"):
                return {"error": "Summoning result is not ready for execution"}
            
            surgical_plan = summoning_result["surgical_plan"]
            operations = surgical_plan.get("operations", [])
            
            results = []
            created_files = []
            modified_files = []
            
            for operation in operations:
                op_type = operation.get("type")
                
                if op_type == "CREATE_FILE":
                    result = self._execute_create_file(repo_id, operation)
                    if result.get("success"):
                        created_files.append(operation["file_path"])
                    else:
                        results.append({"operation": operation, "error": result.get("error")})
                        continue
                
                elif op_type == "MODIFY_FILE":
                    result = self._execute_modify_file(repo_id, operation)
                    if result.get("success"):
                        modified_files.append(operation["file_path"])
                    else:
                        results.append({"operation": operation, "error": result.get("error")})
                        continue
                
                elif op_type == "INSERT_CODE":
                    result = self._execute_insert_code(repo_id, operation)
                    if result.get("success"):
                        modified_files.append(operation["file_path"])
                    else:
                        results.append({"operation": operation, "error": result.get("error")})
                        continue
                
                results.append({"operation": operation, "success": True})
            
            # Check if all operations succeeded
            all_success = all(r.get("success", False) for r in results)
            
            return {
                "success": all_success,
                "operations_completed": len([r for r in results if r.get("success")]),
                "total_operations": len(operations),
                "created_files": created_files,
                "modified_files": modified_files,
                "results": results,
                "recipe_name": summoning_result.get("recipe_name"),
                "message": f"Summoning {'completed successfully' if all_success else 'partially completed'}"
            }
            
        except Exception as e:
            logger.error(f"Error executing summoning: {e}")
            return {"error": str(e)}
    
    def _learn_project_patterns(self, repo_id: str) -> Dict[str, Any]:
        """
        Learn the project's architectural patterns and conventions.
        """
        try:
            # Load cortex data
            cortex_data = cortex_service.load_cortex_data(repo_id)
            files_data = cortex_data.get("files", [])
            
            if not files_data:
                return {"error": "No files found in repository cortex data"}
            
            # Analyze different aspects of the project
            patterns = {
                "framework": self._detect_framework(files_data),
                "structure": self._analyze_project_structure(files_data),
                "naming_conventions": self._analyze_naming_conventions(files_data),
                "imports": self._analyze_import_patterns(files_data),
                "testing": self._analyze_testing_patterns(files_data),
                "configuration": self._analyze_config_patterns(files_data)
            }
            
            # Use LLM to enhance pattern analysis
            enhanced_patterns = self._enhance_patterns_with_llm(patterns, files_data[:10])  # Analyze sample files
            
            return {
                "patterns": enhanced_patterns,
                "file_count": len(files_data),
                "analysis_complete": True
            }
            
        except Exception as e:
            logger.error(f"Error learning project patterns: {e}")
            return {"error": str(e)}
    
    def _detect_framework(self, files_data: List[Dict]) -> Dict[str, Any]:
        """Detect the main framework(s) used in the project."""
        framework_indicators = {
            "fastapi": ["from fastapi", "FastAPI", "APIRouter", "@app.route"],
            "django": ["from django", "django.contrib", "models.Model", "views.py"],
            "flask": ["from flask", "Flask", "@app.route", "request."],
            "react": ["import React", "useState", "useEffect", ".jsx", ".tsx"],
            "vue": ["vue", "createApp", ".vue"],
            "angular": ["@angular", "@Component", "angular.json"],
            "express": ["express", "app.get", "app.post", "middleware"],
            "spring": ["@SpringBootApplication", "@RestController", "@Service"],
            "laravel": ["<?php", "Illuminate\\", "artisan"]
        }
        
        detected = defaultdict(int)
        
        for file_data in files_data:
            content = file_data.get("raw_content", "")
            file_path = file_data.get("file_path", "")
            
            for framework, indicators in framework_indicators.items():
                for indicator in indicators:
                    if indicator in content or indicator in file_path:
                        detected[framework] += 1
        
        # Determine primary framework
        if detected:
            primary = max(detected, key=detected.get)
            return {
                "primary": primary,
                "confidence": detected[primary] / len(files_data),
                "all_detected": dict(detected)
            }
        
        return {"primary": "unknown", "confidence": 0, "all_detected": {}}
    
    def _analyze_project_structure(self, files_data: List[Dict]) -> Dict[str, Any]:
        """Analyze the project's directory structure and organization."""
        directories = defaultdict(list)
        file_types = defaultdict(int)
        
        for file_data in files_data:
            file_path = file_data.get("file_path", "")
            path_obj = Path(file_path)
            
            # Track directories
            if len(path_obj.parts) > 1:
                directories[path_obj.parts[0]].append(str(path_obj))
            
            # Track file types
            file_types[path_obj.suffix] += 1
        
        # Common patterns
        common_dirs = {
            "src", "source", "app", "lib", "components", "services", 
            "models", "views", "controllers", "api", "routes",
            "tests", "test", "__tests__", "spec", "schemas",
            "config", "settings", "utils", "helpers", "static"
        }
        
        structure_patterns = {}
        for dir_name in directories:
            if dir_name.lower() in common_dirs:
                structure_patterns[dir_name] = {
                    "file_count": len(directories[dir_name]),
                    "purpose": self._infer_directory_purpose(dir_name, directories[dir_name])
                }
        
        return {
            "directories": dict(directories),
            "file_types": dict(file_types),
            "structure_patterns": structure_patterns,
            "depth": max(len(Path(f.get("file_path", "")).parts) for f in files_data)
        }
    
    def _analyze_naming_conventions(self, files_data: List[Dict]) -> Dict[str, Any]:
        """Analyze naming conventions used in the project."""
        conventions = {
            "file_naming": {"snake_case": 0, "kebab_case": 0, "camelCase": 0, "PascalCase": 0},
            "function_naming": {"snake_case": 0, "camelCase": 0},
            "class_naming": {"PascalCase": 0, "snake_case": 0},
            "variable_naming": {"snake_case": 0, "camelCase": 0}
        }
        
        for file_data in files_data:
            file_path = file_data.get("file_path", "")
            content = file_data.get("raw_content", "")
            
            # Analyze file naming
            file_name = Path(file_path).stem
            if "_" in file_name and file_name.islower():
                conventions["file_naming"]["snake_case"] += 1
            elif "-" in file_name:
                conventions["file_naming"]["kebab_case"] += 1
            elif file_name[0].isupper():
                conventions["file_naming"]["PascalCase"] += 1
            elif file_name[0].islower() and any(c.isupper() for c in file_name):
                conventions["file_naming"]["camelCase"] += 1
            
            # Analyze code naming (simplified)
            # Function definitions
            func_matches = re.findall(r'def\s+([a-zA-Z_]\w*)', content)
            for func_name in func_matches:
                if "_" in func_name:
                    conventions["function_naming"]["snake_case"] += 1
                elif any(c.isupper() for c in func_name[1:]):
                    conventions["function_naming"]["camelCase"] += 1
            
            # Class definitions
            class_matches = re.findall(r'class\s+([a-zA-Z_]\w*)', content)
            for class_name in class_matches:
                if class_name[0].isupper():
                    conventions["class_naming"]["PascalCase"] += 1
                else:
                    conventions["class_naming"]["snake_case"] += 1
        
        # Determine dominant conventions
        dominant = {}
        for category, counts in conventions.items():
            if counts:
                dominant[category] = max(counts, key=counts.get)
            else:
                dominant[category] = "unknown"
        
        return {
            "conventions": conventions,
            "dominant": dominant
        }
    
    def _analyze_import_patterns(self, files_data: List[Dict]) -> Dict[str, Any]:
        """Analyze import and dependency patterns."""
        import_patterns = {
            "relative_imports": 0,
            "absolute_imports": 0,
            "common_modules": defaultdict(int),
            "import_styles": defaultdict(int)
        }
        
        for file_data in files_data:
            content = file_data.get("raw_content", "")
            
            # Python imports
            import_matches = re.findall(r'^(from\s+[.\w]+\s+)?import\s+([.\w, ]+)', content, re.MULTILINE)
            for from_part, import_part in import_matches:
                if from_part and from_part.strip().startswith("from ."):
                    import_patterns["relative_imports"] += 1
                else:
                    import_patterns["absolute_imports"] += 1
                
                # Track common modules
                modules = [m.strip() for m in import_part.split(",")]
                for module in modules:
                    import_patterns["common_modules"][module.split(".")[0]] += 1
            
            # JavaScript/TypeScript imports
            js_imports = re.findall(r'import\s+.*?\s+from\s+[\'"]([^\'\"]+)[\'"]', content)
            for imp in js_imports:
                if imp.startswith("."):
                    import_patterns["relative_imports"] += 1
                else:
                    import_patterns["absolute_imports"] += 1
                    import_patterns["common_modules"][imp.split("/")[0]] += 1
        
        return {
            "patterns": dict(import_patterns),
            "top_modules": dict(sorted(import_patterns["common_modules"].items(), 
                                     key=lambda x: x[1], reverse=True)[:10])
        }
    
    def _analyze_testing_patterns(self, files_data: List[Dict]) -> Dict[str, Any]:
        """Analyze testing patterns and conventions."""
        testing_patterns = {
            "test_frameworks": defaultdict(int),
            "test_file_patterns": [],
            "test_directories": set(),
            "assertion_styles": defaultdict(int)
        }
        
        test_indicators = {
            "pytest": ["import pytest", "def test_", "@pytest"],
            "unittest": ["import unittest", "unittest.TestCase", "def test"],
            "jest": ["describe(", "it(", "test(", "expect("],
            "mocha": ["describe(", "it(", "beforeEach("],
            "jasmine": ["describe(", "it(", "beforeEach(", "jasmine"]
        }
        
        for file_data in files_data:
            file_path = file_data.get("file_path", "")
            content = file_data.get("raw_content", "")
            
            # Check if it's a test file
            if any(pattern in file_path.lower() for pattern in ["test", "spec", "__tests__"]):
                testing_patterns["test_file_patterns"].append(file_path)
                
                # Check directory
                test_dir = Path(file_path).parts[0] if Path(file_path).parts else ""
                if "test" in test_dir.lower():
                    testing_patterns["test_directories"].add(test_dir)
                
                # Detect framework
                for framework, indicators in test_indicators.items():
                    for indicator in indicators:
                        if indicator in content:
                            testing_patterns["test_frameworks"][framework] += 1
        
        return {
            "frameworks": dict(testing_patterns["test_frameworks"]),
            "file_patterns": testing_patterns["test_file_patterns"],
            "directories": list(testing_patterns["test_directories"]),
            "test_file_count": len(testing_patterns["test_file_patterns"])
        }
    
    def _analyze_config_patterns(self, files_data: List[Dict]) -> Dict[str, Any]:
        """Analyze configuration patterns."""
        config_files = []
        config_patterns = {
            "environment_files": [],
            "package_managers": [],
            "build_tools": [],
            "docker": False
        }
        
        config_indicators = {
            "package.json": "npm",
            "yarn.lock": "yarn", 
            "requirements.txt": "pip",
            "Pipfile": "pipenv",
            "poetry.lock": "poetry",
            "Dockerfile": "docker",
            "docker-compose.yml": "docker-compose",
            ".env": "environment",
            "config.py": "python-config",
            "settings.py": "django-settings"
        }
        
        for file_data in files_data:
            file_path = file_data.get("file_path", "")
            file_name = Path(file_path).name
            
            if file_name in config_indicators:
                config_files.append(file_path)
                
                indicator_type = config_indicators[file_name]
                if indicator_type in ["npm", "yarn", "pip", "pipenv", "poetry"]:
                    config_patterns["package_managers"].append(indicator_type)
                elif indicator_type in ["docker", "docker-compose"]:
                    config_patterns["docker"] = True
                elif indicator_type == "environment":
                    config_patterns["environment_files"].append(file_path)
        
        return {
            "config_files": config_files,
            "patterns": config_patterns
        }
    
    def _infer_directory_purpose(self, dir_name: str, files: List[str]) -> str:
        """Infer the purpose of a directory based on its name and contents."""
        purpose_map = {
            "src": "source code",
            "app": "application code", 
            "api": "API endpoints",
            "models": "data models",
            "views": "view components/templates",
            "controllers": "controllers",
            "services": "business logic services",
            "components": "reusable components",
            "utils": "utility functions",
            "helpers": "helper functions",
            "tests": "test files",
            "config": "configuration files",
            "static": "static assets",
            "templates": "template files",
            "migrations": "database migrations"
        }
        
        return purpose_map.get(dir_name.lower(), "unknown purpose")
    
    def _enhance_patterns_with_llm(self, patterns: Dict[str, Any], sample_files: List[Dict]) -> Dict[str, Any]:
        """Use LLM to enhance pattern analysis with deeper insights."""
        try:
            # Prepare sample code for analysis
            code_samples = []
            for file_data in sample_files[:5]:  # Limit to avoid token limits
                file_path = file_data.get("file_path", "")
                content = file_data.get("raw_content", "")[:1000]  # Truncate long files
                
                code_samples.append(f"File: {file_path}\n{content}")
            
            prompt = f"""Analyze these code samples and existing pattern analysis to provide enhanced insights.

Existing Analysis:
{json.dumps(patterns, indent=2)}

Code Samples:
{'---'.join(code_samples)}

Provide enhanced insights in JSON format with these keys:
- architecture_style: (e.g., "MVC", "microservices", "layered")
- dependency_injection: boolean
- async_patterns: boolean  
- error_handling_style: (e.g., "exceptions", "result_types", "callbacks")
- api_style: (e.g., "REST", "GraphQL", "RPC")
- database_access: (e.g., "ORM", "raw_sql", "query_builder")
- recommended_conventions: object with naming, structure recommendations

JSON:"""

            response = llm_service.generate_text(prompt, task_type=TaskType.COMPLEX_REASONING)
            
            try:
                enhanced = json.loads(response.strip())
                patterns.update(enhanced)
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM enhancement response")
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Failed to enhance patterns with LLM: {e}")
            return patterns
    
    def _generate_surgical_plan(self, recipe: Dict[str, Any], pattern_context: Dict[str, Any], 
                               parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a detailed surgical plan for the summoning."""
        try:
            recipe_name = recipe["name"]
            patterns = pattern_context.get("patterns", {})
            
            # Choose appropriate generator based on recipe
            if recipe_name == "FastAPI Endpoint":
                return self._generate_fastapi_plan(patterns, parameters)
            elif recipe_name == "Django REST ViewSet":
                return self._generate_django_plan(patterns, parameters)
            elif recipe_name == "React Component":
                return self._generate_react_plan(patterns, parameters)
            elif recipe_name == "Microservice Structure":
                return self._generate_microservice_plan(patterns, parameters)
            elif recipe_name == "Database Model":
                return self._generate_database_model_plan(patterns, parameters)
            else:
                return {"error": f"No plan generator for recipe: {recipe_name}"}
                
        except Exception as e:
            logger.error(f"Error generating surgical plan: {e}")
            return {"error": str(e)}
    
    def _generate_fastapi_plan(self, patterns: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate plan for FastAPI endpoint creation."""
        path = parameters.get("path", "/items")
        methods = parameters.get("methods", "get,post").split(",")
        model_name = parameters.get("model_name", "Item")
        
        # Determine file locations based on project structure
        structure = patterns.get("structure", {})
        directories = structure.get("structure_patterns", {})
        
        # Find appropriate directories
        schema_dir = "schemas" if "schemas" in directories else "app/schemas"
        routes_dir = "routes" if "routes" in directories else "app/routes"
        tests_dir = "tests" if "tests" in directories else "test"
        
        operations = []
        
        # 1. Create schema file
        schema_content = self._generate_fastapi_schema(model_name, patterns)
        operations.append({
            "type": "CREATE_FILE",
            "file_path": f"{schema_dir}/{model_name.lower()}_schemas.py",
            "content": schema_content,
            "description": f"Create Pydantic schemas for {model_name}"
        })
        
        # 2. Create or modify routes file
        route_content = self._generate_fastapi_routes(path, methods, model_name, patterns)
        route_file = f"{routes_dir}/{model_name.lower()}_routes.py"
        
        operations.append({
            "type": "CREATE_FILE",
            "file_path": route_file,
            "content": route_content,
            "description": f"Create API routes for {model_name}"
        })
        
        # 3. Create test file
        test_content = self._generate_fastapi_tests(path, methods, model_name, patterns)
        operations.append({
            "type": "CREATE_FILE", 
            "file_path": f"{tests_dir}/test_{model_name.lower()}_api.py",
            "content": test_content,
            "description": f"Create tests for {model_name} API"
        })
        
        # 4. Update main app file to include new router
        operations.append({
            "type": "INSERT_CODE",
            "file_path": "main.py",
            "target": "# Router includes",
            "content": f"from {routes_dir.replace('/', '.')} import {model_name.lower()}_routes\napp.include_router({model_name.lower()}_routes.router)",
            "description": "Add new router to main app"
        })
        
        return {
            "operations": operations,
            "summary": f"Create FastAPI endpoint for {model_name} at {path}",
            "files_created": len([op for op in operations if op["type"] == "CREATE_FILE"]),
            "files_modified": len([op for op in operations if op["type"] in ["MODIFY_FILE", "INSERT_CODE"]])
        }
    
    def _generate_fastapi_schema(self, model_name: str, patterns: Dict[str, Any]) -> str:
        """Generate FastAPI Pydantic schema content."""
        return f'''from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class {model_name}Base(BaseModel):
    """Base schema for {model_name}."""
    name: str
    description: Optional[str] = None


class {model_name}Create({model_name}Base):
    """Schema for creating a new {model_name}."""
    pass


class {model_name}Update(BaseModel):
    """Schema for updating an existing {model_name}."""
    name: Optional[str] = None
    description: Optional[str] = None


class {model_name}InDB({model_name}Base):
    """Schema for {model_name} as stored in database."""
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class {model_name}Response({model_name}InDB):
    """Schema for {model_name} API responses."""
    pass
'''
    
    def _generate_fastapi_routes(self, path: str, methods: List[str], model_name: str, patterns: Dict[str, Any]) -> str:
        """Generate FastAPI routes content."""
        route_methods = []
        
        if "get" in [m.lower() for m in methods]:
            route_methods.append(f'''@router.get("{path}", response_model=List[{model_name}Response])
async def get_{model_name.lower()}s():
    """Get all {model_name.lower()}s."""
    # TODO: Implement database query
    return []


@router.get("{path}/{{item_id}}", response_model={model_name}Response)
async def get_{model_name.lower()}(item_id: int):
    """Get a specific {model_name.lower()} by ID."""
    # TODO: Implement database query
    pass''')
        
        if "post" in [m.lower() for m in methods]:
            route_methods.append(f'''@router.post("{path}", response_model={model_name}Response, status_code=201)
async def create_{model_name.lower()}(item: {model_name}Create):
    """Create a new {model_name.lower()}."""
    # TODO: Implement database creation
    pass''')
        
        if "put" in [m.lower() for m in methods]:
            route_methods.append(f'''@router.put("{path}/{{item_id}}", response_model={model_name}Response)
async def update_{model_name.lower()}(item_id: int, item: {model_name}Update):
    """Update an existing {model_name.lower()}."""
    # TODO: Implement database update
    pass''')
        
        if "delete" in [m.lower() for m in methods]:
            route_methods.append(f'''@router.delete("{path}/{{item_id}}", status_code=204)
async def delete_{model_name.lower()}(item_id: int):
    """Delete a {model_name.lower()}."""
    # TODO: Implement database deletion
    pass''')
        
        return f'''from fastapi import APIRouter, HTTPException, Depends
from typing import List

from .{model_name.lower()}_schemas import (
    {model_name}Create,
    {model_name}Update,
    {model_name}Response
)

router = APIRouter(prefix="{path.rstrip('/')}", tags=["{model_name.lower()}s"])


{chr(10).join(route_methods)}
'''
    
    def _generate_fastapi_tests(self, path: str, methods: List[str], model_name: str, patterns: Dict[str, Any]) -> str:
        """Generate FastAPI test content."""
        test_methods = []
        
        for method in methods:
            method_lower = method.lower()
            if method_lower == "get":
                test_methods.append(f'''def test_get_{model_name.lower()}s(client):
    """Test getting all {model_name.lower()}s."""
    response = client.get("{path}")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_{model_name.lower()}_by_id(client):
    """Test getting a {model_name.lower()} by ID."""
    # TODO: Create test data first
    response = client.get("{path}/1")
    # Update assertion based on expected behavior
    assert response.status_code in [200, 404]''')
            
            elif method_lower == "post":
                test_methods.append(f'''def test_create_{model_name.lower()}(client):
    """Test creating a new {model_name.lower()}."""
    test_data = {{
        "name": "Test {model_name}",
        "description": "A test {model_name.lower()}"
    }}
    response = client.post("{path}", json=test_data)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == test_data["name"]''')
        
        return f'''import pytest
from fastapi.testclient import TestClient


{chr(10).join(test_methods)}
'''
    
    def _generate_django_plan(self, patterns: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate plan for Django REST ViewSet creation."""
        # Simplified Django plan - would be expanded in full implementation
        return {
            "operations": [
                {
                    "type": "CREATE_FILE",
                    "file_path": "models.py",
                    "content": "# Django model placeholder",
                    "description": "Create Django model"
                }
            ],
            "summary": "Create Django REST ViewSet",
            "files_created": 1,
            "files_modified": 0
        }
    
    def _generate_react_plan(self, patterns: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate plan for React component creation."""
        # Simplified React plan - would be expanded in full implementation
        return {
            "operations": [
                {
                    "type": "CREATE_FILE",
                    "file_path": "Component.tsx",
                    "content": "// React component placeholder",
                    "description": "Create React component"
                }
            ],
            "summary": "Create React component",
            "files_created": 1,
            "files_modified": 0
        }
    
    def _generate_microservice_plan(self, patterns: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate plan for microservice structure creation."""
        # Simplified microservice plan - would be expanded in full implementation
        return {
            "operations": [
                {
                    "type": "CREATE_FILE",
                    "file_path": "service.py",
                    "content": "# Microservice placeholder",
                    "description": "Create microservice structure"
                }
            ],
            "summary": "Create microservice structure",
            "files_created": 1,
            "files_modified": 0
        }
    
    def _generate_database_model_plan(self, patterns: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate plan for database model creation."""
        # Simplified database model plan - would be expanded in full implementation
        return {
            "operations": [
                {
                    "type": "CREATE_FILE",
                    "file_path": "models.py",
                    "content": "# Database model placeholder",
                    "description": "Create database model"
                }
            ],
            "summary": "Create database model",
            "files_created": 1,
            "files_modified": 0
        }
    
    def _execute_create_file(self, repo_id: str, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a CREATE_FILE operation."""
        try:
            file_path = operation["file_path"]
            content = operation["content"]
            
            # Use code surgery to create the file
            result = code_surgery.create_file(repo_id, file_path, content)
            
            return {"success": result.get("success", False), "error": result.get("error")}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_modify_file(self, repo_id: str, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a MODIFY_FILE operation."""
        try:
            file_path = operation["file_path"]
            target = operation["target"]
            new_content = operation["content"]
            
            # Use code surgery to modify the file
            result = code_surgery.replace_block(repo_id, file_path, target, new_content)
            
            return {"success": result.get("success", False), "error": result.get("error")}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_insert_code(self, repo_id: str, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an INSERT_CODE operation."""
        try:
            file_path = operation["file_path"]
            target = operation.get("target", "")
            content = operation["content"]
            
            # Use code surgery to insert code
            result = code_surgery.insert_code(repo_id, file_path, target, content)
            
            return {"success": result.get("success", False), "error": result.get("error")}
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# Global instance
_summoner_service = None

def get_summoner_service() -> SummonerService:
    """Get or create the global Summoner service instance."""
    global _summoner_service
    if _summoner_service is None:
        _summoner_service = SummonerService()
    return _summoner_service

# Public API
def list_summoning_recipes() -> Dict[str, Dict[str, Any]]:
    """Get available summoning recipes."""
    service = get_summoner_service()
    return service.list_recipes()

def summon_code_pattern(repo_id: str, recipe_name: str, **parameters) -> Dict[str, Any]:
    """Summon a complete code pattern."""
    service = get_summoner_service()
    return service.summon_pattern(repo_id, recipe_name, **parameters)

def execute_summoning_ritual(repo_id: str, summoning_result: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the summoning ritual to create the code."""
    service = get_summoner_service()
    return service.execute_summoning(repo_id, summoning_result)