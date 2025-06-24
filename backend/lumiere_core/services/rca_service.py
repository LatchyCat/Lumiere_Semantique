# backend/lumiere_core/services/rca_service.py

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from . import llm_service, github
from .ollama import search_index

# Configure logging
logger = logging.getLogger(__name__)

class ComplexityLevel(Enum):
    """Enumeration for complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class FileCategory(Enum):
    """Enumeration for file categories."""
    BACKEND = "backend"
    FRONTEND = "frontend"
    STYLING = "styling"
    CONFIG = "config"
    DOCS = "docs"
    DATABASE = "database"
    INFRASTRUCTURE = "infrastructure"
    BUILD = "build"
    TESTING = "testing"
    OTHER = "other"

@dataclass
class FileRelationships:
    """Data class for file relationship analysis results."""
    total_files: int
    file_types: Dict[str, List[str]]
    type_distribution: Dict[str, int]
    cross_layer_issue: bool
    complexity_indicator: ComplexityLevel
    primary_category: Optional[str] = None
    secondary_categories: List[str] = None

@dataclass
class AnalysisMetadata:
    """Enhanced metadata for analysis results."""
    total_context_chunks: int
    search_time_ms: Optional[float] = None
    analysis_time_ms: Optional[float] = None
    confidence_score: Optional[float] = None
    architectural_context_available: bool = False

def _get_repo_id_from_url(repo_url: str) -> str:
    """
    Helper to derive a filesystem-safe repo_id from a URL.

    Args:
        repo_url: The GitHub repository URL

    Returns:
        Filesystem-safe repository identifier

    Raises:
        ValueError: If the URL format is invalid
    """
    if not repo_url or not isinstance(repo_url, str):
        raise ValueError("Repository URL must be a non-empty string")

    if not repo_url.startswith("https://github.com/"):
        raise ValueError("Repository URL must be a valid GitHub URL")

    try:
        return repo_url.replace("https://github.com/", "").replace("/", "_")
    except Exception as e:
        raise ValueError(f"Failed to parse repository URL: {e}")

def _classify_file_by_extension(file_path: str) -> FileCategory:
    """
    Classify files into broad categories for better analysis context.

    Args:
        file_path: Path to the file

    Returns:
        FileCategory enum value
    """
    if not file_path:
        return FileCategory.OTHER

    # Enhanced extension mapping with more comprehensive coverage
    extension_map = {
        # Backend languages
        '.py': FileCategory.BACKEND, '.rb': FileCategory.BACKEND,
        '.java': FileCategory.BACKEND, '.go': FileCategory.BACKEND,
        '.php': FileCategory.BACKEND, '.cs': FileCategory.BACKEND,
        '.cpp': FileCategory.BACKEND, '.c': FileCategory.BACKEND,
        '.rs': FileCategory.BACKEND, '.kt': FileCategory.BACKEND,

        # Frontend
        '.js': FileCategory.FRONTEND, '.ts': FileCategory.FRONTEND,
        '.jsx': FileCategory.FRONTEND, '.tsx': FileCategory.FRONTEND,
        '.html': FileCategory.FRONTEND, '.htm': FileCategory.FRONTEND,
        '.vue': FileCategory.FRONTEND, '.svelte': FileCategory.FRONTEND,

        # Styling
        '.css': FileCategory.STYLING, '.scss': FileCategory.STYLING,
        '.sass': FileCategory.STYLING, '.less': FileCategory.STYLING,
        '.styl': FileCategory.STYLING,

        # Configuration
        '.json': FileCategory.CONFIG, '.yaml': FileCategory.CONFIG,
        '.yml': FileCategory.CONFIG, '.toml': FileCategory.CONFIG,
        '.ini': FileCategory.CONFIG, '.conf': FileCategory.CONFIG,
        '.env': FileCategory.CONFIG, '.properties': FileCategory.CONFIG,

        # Documentation
        '.md': FileCategory.DOCS, '.rst': FileCategory.DOCS,
        '.txt': FileCategory.DOCS, '.adoc': FileCategory.DOCS,

        # Database
        '.sql': FileCategory.DATABASE, '.graphql': FileCategory.DATABASE,
        '.gql': FileCategory.DATABASE,

        # Infrastructure
        '.tf': FileCategory.INFRASTRUCTURE, '.hcl': FileCategory.INFRASTRUCTURE,
        '.sh': FileCategory.INFRASTRUCTURE, '.bash': FileCategory.INFRASTRUCTURE,
        '.ps1': FileCategory.INFRASTRUCTURE, '.bat': FileCategory.INFRASTRUCTURE,
    }

    # Special filename mappings
    filename_map = {
        'dockerfile': FileCategory.INFRASTRUCTURE,
        'dockerfile.dev': FileCategory.INFRASTRUCTURE,
        'dockerfile.prod': FileCategory.INFRASTRUCTURE,
        'requirements.txt': FileCategory.BUILD,
        'package.json': FileCategory.BUILD,
        'package-lock.json': FileCategory.BUILD,
        'yarn.lock': FileCategory.BUILD,
        'pipfile': FileCategory.BUILD,
        'pipfile.lock': FileCategory.BUILD,
        'gemfile': FileCategory.BUILD,
        'gemfile.lock': FileCategory.BUILD,
        'composer.json': FileCategory.BUILD,
        'composer.lock': FileCategory.BUILD,
        'pom.xml': FileCategory.BUILD,
        'build.gradle': FileCategory.BUILD,
        'cargo.toml': FileCategory.BUILD,
        'cargo.lock': FileCategory.BUILD,
        'makefile': FileCategory.BUILD,
        'cmake.txt': FileCategory.BUILD,
    }

    file_path_lower = file_path.lower()

    # Check for test files first (highest priority)
    test_patterns = ['.test.', '.spec.', '_test.', '/test/', '/tests/', '__test__', '__tests__']
    if any(pattern in file_path_lower for pattern in test_patterns):
        return FileCategory.TESTING

    # Check filename mappings
    filename = Path(file_path_lower).name
    if filename in filename_map:
        return filename_map[filename]

    # Check extension mappings
    suffix = Path(file_path_lower).suffix
    return extension_map.get(suffix, FileCategory.OTHER)

def _determine_complexity_level(file_count: int, type_count: int, cross_layer: bool) -> ComplexityLevel:
    """
    Determine complexity level based on multiple factors.

    Args:
        file_count: Number of files involved
        type_count: Number of different file types
        cross_layer: Whether the issue crosses multiple layers

    Returns:
        ComplexityLevel enum value
    """
    if cross_layer and file_count > 15:
        return ComplexityLevel.VERY_HIGH
    elif file_count > 12 or (cross_layer and type_count > 4):
        return ComplexityLevel.HIGH
    elif file_count > 6 or type_count > 3:
        return ComplexityLevel.MEDIUM
    else:
        return ComplexityLevel.LOW

def _analyze_file_relationships(context_chunks: List[Dict]) -> FileRelationships:
    """
    Analyze relationships between files found in RAG results to provide better context.

    Args:
        context_chunks: List of context chunks from RAG search

    Returns:
        FileRelationships object with comprehensive analysis
    """
    if not context_chunks:
        return FileRelationships(
            total_files=0,
            file_types={},
            type_distribution={},
            cross_layer_issue=False,
            complexity_indicator=ComplexityLevel.LOW,
            primary_category=None,
            secondary_categories=[]
        )

    file_types = defaultdict(list)
    file_paths = set()
    category_counts = defaultdict(int)

    for chunk in context_chunks:
        file_path = chunk.get('file_path', '')
        if file_path:
            file_paths.add(file_path)
            file_category = _classify_file_by_extension(file_path)
            category_str = file_category.value
            file_types[category_str].append(file_path)
            category_counts[category_str] += 1

    # Determine primary and secondary categories
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    primary_category = sorted_categories[0][0] if sorted_categories else None
    secondary_categories = [cat for cat, count in sorted_categories[1:4] if count > 1]

    total_files = len(file_paths)
    type_count = len(file_types)
    cross_layer_issue = type_count > 2
    complexity = _determine_complexity_level(total_files, type_count, cross_layer_issue)

    return FileRelationships(
        total_files=total_files,
        file_types=dict(file_types),
        type_distribution={k: len(set(v)) for k, v in file_types.items()},
        cross_layer_issue=cross_layer_issue,
        complexity_indicator=complexity,
        primary_category=primary_category,
        secondary_categories=secondary_categories
    )

def _load_architectural_context(repo_id: str) -> Tuple[str, bool]:
    """
    Load architectural context from cortex file.

    Args:
        repo_id: Repository identifier

    Returns:
        Tuple of (architectural_context_string, context_available_bool)
    """
    try:
        backend_dir = Path(__file__).resolve().parent.parent.parent
        cortex_path = backend_dir / "cloned_repositories" / repo_id / f"{repo_id}_cortex.json"

        if not cortex_path.exists():
            logger.debug(f"Cortex file not found at {cortex_path}")
            return "No architectural context available for this analysis.", False

        with open(cortex_path, 'r', encoding='utf-8') as f:
            cortex_data = json.load(f)

        graph_data = cortex_data.get('architectural_graph')
        if not graph_data:
            return "Architectural context file found but contains no graph data.", False

        nodes = graph_data.get('nodes', {})  # Default to empty dict
        edges = graph_data.get('edges', [])

        # Correctly get the first 5 nodes from the dictionary's values
        nodes_list = list(nodes.values())

        context = f"""Architectural graph available with {len(nodes)} components and {len(edges)} connections.
Key components identified: {', '.join([node.get('name', 'Unknown') for node in nodes_list[:5]])}
This provides insights into system architecture and component relationships."""

        logger.info(f"Loaded architectural context for {repo_id}: {len(nodes)} nodes, {len(edges)} edges")
        return context, True

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse cortex JSON for {repo_id}: {e}")
        return "Architectural context file found but contains invalid JSON.", False
    except Exception as e:
        logger.error(f"Failed to load architectural context for {repo_id}: {e}")
        return "Error loading architectural context.", False

def _format_context_for_prompt(context_chunks: List[Dict], relationships: FileRelationships) -> str:
    """
    Format context chunks for LLM prompt with improved organization.

    Args:
        context_chunks: List of context chunks
        relationships: File relationship analysis

    Returns:
        Formatted context string
    """
    if not context_chunks:
        return "No relevant code context found."

    # Group chunks by file category for better organization
    categorized_chunks = defaultdict(list)
    for chunk in context_chunks:
        file_path = chunk.get('file_path', '')
        category = _classify_file_by_extension(file_path).value
        categorized_chunks[category].append(chunk)

    context_parts = []

    # Present primary category first
    if relationships.primary_category and relationships.primary_category in categorized_chunks:
        chunks = categorized_chunks[relationships.primary_category]
        context_parts.append(f"=== PRIMARY CATEGORY: {relationships.primary_category.upper()} FILES ===")
        for chunk in chunks:
            context_parts.append(
                f"--- Context from `{chunk['file_path']}` ---\n```\n{chunk['text']}\n```\n"
            )

    # Then secondary categories
    for category in relationships.secondary_categories:
        if category in categorized_chunks:
            chunks = categorized_chunks[category]
            context_parts.append(f"=== SECONDARY CATEGORY: {category.upper()} FILES ===")
            for chunk in chunks:
                context_parts.append(
                    f"--- Context from `{chunk['file_path']}` ---\n```\n{chunk['text']}\n```\n"
                )

    # Finally, remaining categories
    for category, chunks in categorized_chunks.items():
        if category != relationships.primary_category and category not in relationships.secondary_categories:
            context_parts.append(f"=== {category.upper()} FILES ===")
            for chunk in chunks:
                context_parts.append(
                    f"--- Context from `{chunk['file_path']}` ---\n```\n{chunk['text']}\n```\n"
                )

    return "\n\n".join(context_parts)

def generate_briefing(issue_url: str, model_identifier: str) -> Dict[str, Any]:
    """
    Generates a 'Pre-flight Briefing' for a GitHub issue using RAG.

    Args:
        issue_url: GitHub issue URL
        model_identifier: LLM model identifier

    Returns:
        Dictionary containing briefing and metadata
    """
    logger.info(f"Generating briefing for issue: {issue_url}")

    try:
        # Validate inputs
        if not issue_url or not isinstance(issue_url, str):
            return {"error": "Invalid issue URL provided"}

        if not model_identifier:
            return {"error": "Model identifier is required"}

        # Scrape GitHub issue
        issue_data = github.scrape_github_issue(issue_url)
        if not issue_data:
            return {"error": "Could not retrieve issue details from GitHub. Please check the URL and try again."}

        repo_id = _get_repo_id_from_url(issue_data['repo_url'])
        query = issue_data['full_text_query']

        if not query:
            return {"error": "No query text found in issue data"}

        # Perform RAG search
        try:
            context_chunks = search_index(
                query_text=query,
                model_name='snowflake-arctic-embed2:latest',
                repo_id=repo_id,
                k=7
            )
        except FileNotFoundError:
            return {
                "error": f"Vector index for repository '{repo_id}' not found. "
                        "Please ensure the repository has been ingested with clone/embed enabled."
            }
        except Exception as e:
            logger.error(f"RAG search failed for {repo_id}: {e}")
            return {"error": f"Failed to retrieve context from vector index: {str(e)}"}

        # Analyze relationships and format context
        relationships = _analyze_file_relationships(context_chunks)
        context_string = _format_context_for_prompt(context_chunks, relationships)

        # Enhanced prompt with better structure
        prompt = f"""You are Lumière Sémantique, an expert AI programming assistant.
Your mission is to provide a comprehensive "Pre-flight Briefing" for a developer about to work on this GitHub issue.

**CODEBASE ANALYSIS SUMMARY:**
- Files involved: {relationships.total_files} files across {len(relationships.file_types)} different categories
- Primary category: {relationships.primary_category or 'Unknown'}
- File categories: {', '.join(relationships.file_types.keys())}
- Complexity level: {relationships.complexity_indicator.value}
- Cross-layer issue: {'Yes' if relationships.cross_layer_issue else 'No'}

**INSTRUCTIONS:**
Analyze the GitHub issue and the provided codebase context to generate a comprehensive briefing report.

The report must be clear, well-structured, and formatted in Markdown. Include these sections:

1. **🎯 Task Summary**
   - Concise rephrasing of the issue request
   - Key objectives and expected outcomes

2. **🏗️ Codebase Architecture**
   - Relevant system architecture and component relationships
   - How the affected components interact

3. **🔍 Current System Analysis**
   - How the system currently handles the functionality in question
   - Existing patterns and conventions

4. **📁 Key Files & Components**
   - Most important files and functions from the context
   - Their roles and relationships

5. **🚀 Suggested Approach**
   - High-level implementation strategy
   - Recommended order of operations
   - Potential challenges and considerations

6. **⚠️ Important Notes**
   - Dependencies and side effects to consider
   - Testing recommendations

--- CODEBASE CONTEXT ---
{context_string}
--- END CONTEXT ---

**GITHUB ISSUE DETAILS:**
{query}

Generate the comprehensive Pre-flight Briefing now:
"""

        # Generate briefing
        briefing_report = llm_service.generate_text(prompt, model_identifier)

        # Prepare metadata
        relationships_dict = relationships.__dict__
        relationships_dict['complexity_indicator'] = relationships.complexity_indicator.value

        metadata = {
            "file_relationships": relationships_dict,
            "issue_url": issue_url,
            "repo_id": repo_id,
            "context_chunks_count": len(context_chunks),
            "model_used": model_identifier
        }

        logger.info(f"Successfully generated briefing for {issue_url}")
        return {"briefing": briefing_report, "metadata": metadata}

    except ValueError as e:
        logger.error(f"Validation error in generate_briefing: {e}")
        return {"error": f"Input validation failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error in generate_briefing: {e}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

def perform_rca(
    repo_url: str,
    bug_description: str,
    model_identifier: str,
    advanced_analysis: bool = False,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Performs a multi-file, context-aware Root Cause Analysis using RAG.

    Args:
        repo_url: GitHub repository URL
        bug_description: Description of the bug to analyze
        model_identifier: LLM model identifier
        advanced_analysis: Whether to perform advanced analysis with more context
        confidence_threshold: Minimum confidence threshold for results

    Returns:
        Dictionary containing analysis results and metadata
    """
    logger.info(f"Performing Multi-file RCA for bug: '{bug_description[:100]}...'")

    try:
        # Validate inputs
        if not repo_url or not isinstance(repo_url, str):
            return {"error": "Invalid repository URL provided"}

        if not bug_description or not isinstance(bug_description, str):
            return {"error": "Bug description is required"}

        if not model_identifier:
            return {"error": "Model identifier is required"}

        if not 0.0 <= confidence_threshold <= 1.0:
            return {"error": "Confidence threshold must be between 0.0 and 1.0"}

        repo_id = _get_repo_id_from_url(repo_url)

        # Load architectural context
        architectural_context, arch_available = _load_architectural_context(repo_id)

        # Perform RAG search with enhanced parameters
        try:
            logger.debug("Searching for relevant code chunks...")
            initial_k = 25 if advanced_analysis else 15

            context_chunks = search_index(
                query_text=bug_description,
                model_name='snowflake-arctic-embed2:latest',
                repo_id=repo_id,
                k=initial_k
            )
        except FileNotFoundError:
            return {
                "error": f"Vector index for repository '{repo_id}' not found. "
                        "Please ensure the repository has been ingested with clone/embed enabled."
            }
        except Exception as e:
            logger.error(f"RAG search failed during RCA for {repo_id}: {e}")
            return {"error": f"RAG search failed during RCA: {str(e)}"}

        if not context_chunks:
            return {
                "analysis": "Could not find any relevant code context for the bug description. "
                           "Unable to perform RCA. Please try rephrasing the bug description or "
                           "ensure the repository has been properly indexed."
            }

        logger.debug("Analyzing file relationships and filtering context...")
        relationships = _analyze_file_relationships(context_chunks)

        # Format context with improved organization
        logger.debug("Synthesizing context from suspect files...")
        formatted_context = _format_context_for_prompt(context_chunks, relationships)

        # Generate complexity guidance
        complexity_guidance = ""
        if relationships.cross_layer_issue:
            complexity_guidance = """
**COMPLEXITY ALERT:** This issue spans multiple system layers. Pay special attention to:
- Interface boundaries and data contracts
- State management across components
- Error propagation paths
- Dependency chains and side effects"""

        if relationships.complexity_indicator in [ComplexityLevel.HIGH, ComplexityLevel.VERY_HIGH]:
            complexity_guidance += """
- Consider breaking down the analysis into smaller, focused areas
- Look for common patterns or shared dependencies
- Pay attention to configuration and environment differences"""

        # Enhanced RCA prompt
        prompt = f"""You are a world-class debugging expert performing a comprehensive Root Cause Analysis (RCA).

**BUG DESCRIPTION:**
{bug_description}

**SYSTEM CONTEXT:**
{architectural_context}

**CODEBASE ANALYSIS:**
- Relevant files: {relationships.total_files} files analyzed
- Primary category: {relationships.primary_category or 'Mixed'}
- File categories: {', '.join(relationships.file_types.keys())}
- Complexity level: {relationships.complexity_indicator.value}
- Cross-layer issue: {'Yes' if relationships.cross_layer_issue else 'No'}

{complexity_guidance}

**ANALYSIS INSTRUCTIONS:**
You must analyze ALL provided context systematically. Use the code evidence to build a comprehensive understanding of the bug's root cause.

Your analysis must follow this structure:

## 🎯 Executive Summary
A clear, single-sentence explanation of the root cause.

## 🏗️ System Overview
Brief explanation of how the affected components are designed to interact and what the expected behavior should be.

## 🔍 Root Cause Analysis
Detailed breakdown of:
- What is happening vs. what should happen
- The specific mechanism causing the failure
- Why this particular scenario triggers the bug

## 📋 Evidence & Reasoning
Cite specific files, functions, and code snippets that support your analysis:
- Direct evidence from the code
- Logical connections between components
- Data flow analysis

## 💥 Impact Assessment
Explain the cascading effects:
- What breaks when this bug occurs
- Which users/systems are affected
- Performance or security implications

## 🛠️ Recommended Fix Strategy
High-level approach including:
- Which specific files need modification
- Order of operations for the fix
- Testing strategy to verify the fix
- Potential risks and mitigation strategies

## ⚠️ Prevention Recommendations
Suggestions to prevent similar issues in the future.

**RELEVANT CODE CONTEXT:**
{formatted_context}

Now generate your comprehensive Root Cause Analysis:
"""

        logger.debug("Generating comprehensive RCA report...")
        analysis_report = llm_service.generate_text(prompt, model_identifier)

        # Prepare enhanced metadata
        metadata = AnalysisMetadata(
            total_context_chunks=len(context_chunks),
            architectural_context_available=arch_available,
            confidence_score=None  # Could be implemented based on context quality
        )

        relationships_dict = relationships.__dict__
        relationships_dict['complexity_indicator'] = relationships.complexity_indicator.value

        result = {
            "analysis": analysis_report,
            "metadata": {
                **relationships_dict,
                **metadata.__dict__,
                "repo_id": repo_id,
                "advanced_analysis": advanced_analysis,
                "confidence_threshold": confidence_threshold
            }
        }

        logger.info(f"Successfully completed RCA for {repo_id}")
        return result

    except ValueError as e:
        logger.error(f"Validation error in perform_rca: {e}")
        return {"error": f"Input validation failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error in perform_rca: {e}")
        return {"error": f"An unexpected error occurred during RCA: {str(e)}"}

def get_analysis_health_check(repo_id: str) -> Dict[str, Any]:
    """
    Perform a health check for RCA analysis capabilities.

    Args:
        repo_id: Repository identifier

    Returns:
        Dictionary containing health check results
    """
    try:
        backend_dir = Path(__file__).resolve().parent.parent.parent
        cortex_path = backend_dir / "cloned_repositories" / repo_id / f"{repo_id}_cortex.json"

        health_status = {
            "repo_id": repo_id,
            "vector_index_available": False,
            "architectural_context_available": False,
            "cortex_file_exists": cortex_path.exists(),
            "recommendations": []
        }

        # Check vector index (this would need to be implemented based on your vector storage)
        try:
            # Placeholder for vector index check
            test_chunks = search_index(
                query_text="test query",
                model_name='snowflake-arctic-embed2:latest',
                repo_id=repo_id,
                k=1
            )
            health_status["vector_index_available"] = len(test_chunks) > 0
        except FileNotFoundError:
            health_status["recommendations"].append(
                "Vector index not found. Please re-ingest the repository with embedding enabled."
            )
        except Exception as e:
            health_status["recommendations"].append(f"Vector index check failed: {str(e)}")

        # Check architectural context
        if cortex_path.exists():
            try:
                with open(cortex_path, 'r', encoding='utf-8') as f:
                    cortex_data = json.load(f)
                graph_data = cortex_data.get('architectural_graph')
                health_status["architectural_context_available"] = bool(graph_data)

                if not graph_data:
                    health_status["recommendations"].append(
                        "Cortex file exists but contains no architectural graph data."
                    )
            except Exception as e:
                health_status["recommendations"].append(f"Failed to parse cortex file: {str(e)}")
        else:
            health_status["recommendations"].append(
                "No architectural context available. Consider running architectural analysis."
            )

        # Overall health assessment
        if health_status["vector_index_available"] and health_status["architectural_context_available"]:
            health_status["overall_status"] = "excellent"
        elif health_status["vector_index_available"]:
            health_status["overall_status"] = "good"
        else:
            health_status["overall_status"] = "poor"

        return health_status

    except Exception as e:
        logger.error(f"Health check failed for {repo_id}: {e}")
        return {
            "repo_id": repo_id,
            "overall_status": "error",
            "error": str(e)
        }
