# backend/lumiere_core/services/loremaster_service.py

import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

from .cortex_service import load_cortex_data
from .llm_service import generate_text, TaskType

logger = logging.getLogger(__name__)

@dataclass
class ApiEndpoint:
    """Represents an API endpoint discovered in the codebase."""
    path: str
    methods: List[str]
    handler_function: str
    handler_file: str
    framework: str
    parameters: List[Dict[str, Any]]
    description: Optional[str] = None
    request_body_schema: Optional[Dict[str, Any]] = None
    response_schema: Optional[Dict[str, Any]] = None

class LoremasterService:
    """
    The Loremaster's Codex - Transmutes raw endpoint data into illuminated API documentation.
    Evolved from the Jsonifier to automatically generate interactive API documentation.
    """
    
    def __init__(self, cloned_repos_dir: Path):
        self.cloned_repos_dir = Path(cloned_repos_dir)
    
    def get_api_inventory(self, repo_id: str) -> Dict[str, Any]:
        """
        Get a simple, clean list of all API endpoints for the repository.
        
        Args:
            repo_id: Repository identifier
            
        Returns:
            Dictionary containing all discovered API endpoints
        """
        try:
            logger.info(f"Loremaster: Getting API inventory for {repo_id}")
            
            cortex_data = load_cortex_data(repo_id)
            if not cortex_data:
                return {"error": "Cortex data not found for repository"}
            
            endpoints = []
            
            # Process each file in the cortex data
            for file_data in cortex_data.get('files', []):
                file_path = file_data.get('file_path', '')
                api_endpoints = file_data.get('api_endpoints', [])
                
                for endpoint in api_endpoints:
                    # Create standardized endpoint object
                    api_endpoint = ApiEndpoint(
                        path=endpoint.get('path', ''),
                        methods=endpoint.get('methods', []),
                        handler_function=endpoint.get('handler_function_name', ''),
                        handler_file=file_path,
                        framework=endpoint.get('framework', 'unknown'),
                        parameters=endpoint.get('parameters', []),
                        description=endpoint.get('description')
                    )
                    endpoints.append(asdict(api_endpoint))
            
            # Generate summary statistics
            framework_counts = {}
            method_counts = {}
            
            for endpoint in endpoints:
                # Count by framework
                framework = endpoint.get('framework', 'unknown')
                framework_counts[framework] = framework_counts.get(framework, 0) + 1
                
                # Count by HTTP methods
                for method in endpoint.get('methods', []):
                    method_counts[method] = method_counts.get(method, 0) + 1
            
            return {
                "repository": repo_id,
                "total_endpoints": len(endpoints),
                "endpoints": endpoints,
                "framework_distribution": framework_counts,
                "method_distribution": method_counts,
                "scan_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Loremaster: Error getting API inventory for {repo_id}: {e}")
            return {"error": f"Failed to get API inventory: {str(e)}"}
    
    def generate_openapi_spec(self, repo_id: str) -> Dict[str, Any]:
        """
        Generate a complete OpenAPI 3.0 specification for the repository's API endpoints.
        
        Args:
            repo_id: Repository identifier
            
        Returns:
            OpenAPI 3.0 specification dictionary
        """
        try:
            logger.info(f"Loremaster: Generating OpenAPI spec for {repo_id}")
            
            cortex_data = load_cortex_data(repo_id)
            if not cortex_data:
                return {"error": "Cortex data not found for repository"}
            
            # Extract metadata
            metadata = cortex_data.get('repository_metadata', {})
            repo_url = metadata.get('repo_url', f'https://github.com/{repo_id.replace("_", "/")}')
            
            # Initialize OpenAPI spec structure
            openapi_spec = {
                "openapi": "3.0.3",
                "info": {
                    "title": f"{repo_id.replace('_', '/')} API",
                    "description": f"Auto-generated API documentation for {repo_id}",
                    "version": "1.0.0",
                    "contact": {
                        "url": repo_url
                    }
                },
                "servers": [
                    {
                        "url": "http://localhost:8000",
                        "description": "Development server"
                    }
                ],
                "paths": {},
                "components": {
                    "schemas": {},
                    "responses": {
                        "NotFound": {
                            "description": "Resource not found",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        },
                        "InternalServerError": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"},
                                            "details": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            # Process endpoints from cortex data
            for file_data in cortex_data.get('files', []):
                api_endpoints = file_data.get('api_endpoints', [])
                
                for endpoint in api_endpoints:
                    path = endpoint.get('path', '')
                    methods = endpoint.get('methods', [])
                    
                    if not path:
                        continue
                    
                    # Convert path format (e.g., Django <str:id> to OpenAPI {id})
                    openapi_path = self._convert_path_format(path)
                    
                    if openapi_path not in openapi_spec["paths"]:
                        openapi_spec["paths"][openapi_path] = {}
                    
                    # Add each HTTP method
                    for method in methods:
                        method_lower = method.lower()
                        openapi_spec["paths"][openapi_path][method_lower] = self._generate_endpoint_spec(
                            endpoint, method, repo_id
                        )
            
            return openapi_spec
            
        except Exception as e:
            logger.error(f"Loremaster: Error generating OpenAPI spec for {repo_id}: {e}")
            return {"error": f"Failed to generate OpenAPI spec: {str(e)}"}
    
    def generate_documentation_page(self, repo_id: str) -> str:
        """
        Generate a complete, standalone HTML documentation page.
        
        Args:
            repo_id: Repository identifier
            
        Returns:
            HTML string containing the documentation page
        """
        try:
            logger.info(f"Loremaster: Generating documentation page for {repo_id}")
            
            openapi_spec = self.generate_openapi_spec(repo_id)
            if "error" in openapi_spec:
                return f"<html><body><h1>Error</h1><p>{openapi_spec['error']}</p></body></html>"
            
            # Generate HTML page with embedded Swagger UI
            html_template = self._get_swagger_ui_template(repo_id, openapi_spec)
            
            return html_template
            
        except Exception as e:
            logger.error(f"Loremaster: Error generating documentation page for {repo_id}: {e}")
            return f"<html><body><h1>Error</h1><p>Failed to generate documentation: {str(e)}</p></body></html>"
    
    def generate_client_snippet(self, endpoint_data: Dict[str, Any], language: str) -> Dict[str, Any]:
        """
        Generate client code snippet for a specific endpoint in the requested language.
        
        Args:
            endpoint_data: Endpoint information dictionary
            language: Programming language for the snippet (python, javascript, curl, etc.)
            
        Returns:
            Dictionary containing the generated code snippet
        """
        try:
            path = endpoint_data.get('path', '')
            methods = endpoint_data.get('methods', [])
            parameters = endpoint_data.get('parameters', [])
            
            if not path or not methods:
                return {"error": "Invalid endpoint data provided"}
            
            # Use the first method if multiple are available
            method = methods[0].upper()
            
            # Generate language-specific snippet using LLM
            prompt = self._create_client_snippet_prompt(path, method, parameters, language)
            
            snippet = generate_text(prompt, task_type=TaskType.CODE_GENERATION)
            
            return {
                "language": language,
                "method": method,
                "path": path,
                "snippet": snippet,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Loremaster: Error generating client snippet: {e}")
            return {"error": f"Failed to generate client snippet: {str(e)}"}
    
    # Private helper methods
    
    def _convert_path_format(self, path: str) -> str:
        """Convert Django/Flask path format to OpenAPI format."""
        # Convert Django <str:param> to OpenAPI {param}
        path = re.sub(r'<(?:str:)?(\w+)>', r'{\1}', path)
        # Convert Flask <param> to OpenAPI {param}
        path = re.sub(r'<(\w+)>', r'{\1}', path)
        return path
    
    def _generate_endpoint_spec(self, endpoint: Dict[str, Any], method: str, repo_id: str) -> Dict[str, Any]:
        """Generate OpenAPI specification for a single endpoint method."""
        handler_function = endpoint.get('handler_function_name', 'unknown')
        description = endpoint.get('description', f"API endpoint handled by {handler_function}")
        parameters = endpoint.get('parameters', [])
        
        spec = {
            "summary": f"{method.upper()} {endpoint.get('path', '')}",
            "description": description,
            "operationId": f"{method.lower()}_{handler_function}",
            "tags": [endpoint.get('framework', 'api')],
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "status": {"type": "string"},
                                    "data": {"type": "object"}
                                }
                            }
                        }
                    }
                },
                "404": {"$ref": "#/components/responses/NotFound"},
                "500": {"$ref": "#/components/responses/InternalServerError"}
            }
        }
        
        # Add parameters if any
        if parameters:
            spec["parameters"] = []
            for param in parameters:
                param_spec = {
                    "name": param.get('name', 'unknown'),
                    "in": param.get('location', 'query'),  # query, path, header
                    "required": param.get('required', False),
                    "schema": {
                        "type": param.get('type', 'string')
                    }
                }
                if param.get('description'):
                    param_spec["description"] = param['description']
                spec["parameters"].append(param_spec)
        
        # Add request body for POST/PUT/PATCH methods
        if method.upper() in ['POST', 'PUT', 'PATCH']:
            spec["requestBody"] = {
                "description": "Request payload",
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "data": {"type": "object"}
                            }
                        }
                    }
                }
            }
        
        return spec
    
    def _get_swagger_ui_template(self, repo_id: str, openapi_spec: Dict[str, Any]) -> str:
        """Generate HTML template with embedded Swagger UI."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Documentation - {repo_id}</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
    <style>
        html {{
            box-sizing: border-box;
            overflow: -moz-scrollbars-vertical;
            overflow-y: scroll;
        }}
        *, *:before, *:after {{
            box-sizing: inherit;
        }}
        body {{
            margin:0;
            background: #fafafa;
        }}
        .header {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        .generated-info {{
            background: #ecf0f1;
            padding: 10px 20px;
            border-left: 4px solid #3498db;
            margin: 20px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>API Documentation</h1>
        <p>Repository: {repo_id}</p>
    </div>
    
    <div class="generated-info">
        <strong>📚 Auto-generated by Lumière Sémantique Loremaster</strong><br>
        Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}<br>
        Total Endpoints: {len(openapi_spec.get("paths", {}))}<br>
        <em>This documentation is automatically generated from your codebase. For the most up-to-date information, 
        please regenerate after code changes.</em>
    </div>
    
    <div id="swagger-ui"></div>
    
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            const spec = {json.dumps(openapi_spec, indent=2)};
            
            SwaggerUIBundle({{
                url: null,
                spec: spec,
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                tryItOutEnabled: true,
                supportedSubmitMethods: ['get', 'post', 'put', 'delete', 'patch'],
                onComplete: function() {{
                    console.log("Swagger UI loaded successfully");
                }},
                onFailure: function(error) {{
                    console.error("Failed to load Swagger UI:", error);
                }}
            }});
        }};
    </script>
</body>
</html>
        """
    
    def _create_client_snippet_prompt(self, path: str, method: str, parameters: List[Dict], language: str) -> str:
        """Create a prompt for generating client code snippets."""
        params_description = ""
        if parameters:
            params_list = [f"- {p.get('name', 'unknown')}: {p.get('type', 'string')}" for p in parameters]
            params_description = f"Parameters:\n" + "\n".join(params_list)
        
        language_examples = {
            "python": "Use the `requests` library. Include error handling.",
            "javascript": "Use the `fetch` API with async/await. Include error handling.",
            "curl": "Provide a complete curl command with proper headers.",
            "php": "Use cURL functions with proper error handling.",
            "ruby": "Use the Net::HTTP library or HTTParty gem.",
            "go": "Use the standard net/http package.",
            "java": "Use HttpClient from java.net.http package.",
            "csharp": "Use HttpClient class with async/await."
        }
        
        language_instruction = language_examples.get(language.lower(), "Use appropriate HTTP client for the language.")
        
        return f"""Generate a clean, production-ready code snippet in {language} for making an API request.

API Details:
- Method: {method}
- Path: {path}
- {params_description}

Requirements:
- {language_instruction}
- Include proper headers (Content-Type: application/json)
- Add basic error handling
- Make the code readable and well-commented
- Include example values for any required parameters
- Show how to handle the response

Generate only the code snippet without additional explanation."""