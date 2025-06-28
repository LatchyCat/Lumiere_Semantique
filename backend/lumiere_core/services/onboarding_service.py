"""
Onboarding Service

This service orchestrates the generation of personalized onboarding paths for GitHub issues.
It combines impact analysis, expertise detection, and architectural understanding to create
comprehensive guides for new developers. This is the capstone feature of the Onboarding Concierge.
"""

import json
import pathlib
from typing import Dict, List, Any, Optional
import logging

from . import impact_analyzer
from . import expertise_service
from . import oracle_service
from . import ollama
from . import llm_service
from .llm_service import TaskType

logger = logging.getLogger(__name__)


class OnboardingService:
    """
    Service that orchestrates the generation of personalized onboarding paths
    by combining multiple analysis services into a comprehensive guide.
    """
    
    def __init__(self, cortex_directory: Optional[pathlib.Path] = None):
        """
        Initialize the Onboarding Service.
        
        Args:
            cortex_directory: Optional path to the cortex directory.
                            If None, will use default backend/cloned_repositories/
        """
        self.cortex_directory = cortex_directory or pathlib.Path("backend/cloned_repositories")
        self.impact_analyzer = impact_analyzer.ImpactAnalyzer(cortex_directory)
        self.expertise_service = expertise_service.ExpertiseService(cortex_directory)
    
    def generate_onboarding_path(self, repo_id: str, issue_number: int) -> Dict[str, Any]:
        """
        Generate a complete, step-by-step onboarding guide for a given GitHub issue.
        
        Args:
            repo_id: Repository identifier
            issue_number: GitHub issue number
            
        Returns:
            Comprehensive onboarding guide with learning path and expert contacts
        """
        try:
            logger.info(f"Generating onboarding path for issue #{issue_number} in {repo_id}")
            
            # Step 1: Identify Locus of Change using mock issue data
            # In a real implementation, this would fetch from GitHub API
            issue_data = self._get_issue_data(repo_id, issue_number)
            
            locus_files = self._identify_locus_of_change(
                repo_id, 
                issue_data.get('title', ''), 
                issue_data.get('description', '')
            )
            
            if not locus_files:
                return {
                    'error': 'Could not identify relevant files for this issue',
                    'issue_number': issue_number,
                    'repo_id': repo_id
                }
            
            # Step 2: Build Epistemic Subgraph & Learning Path
            learning_path = self._build_learning_path(repo_id, locus_files)
            
            # Step 3: Enrich Each Step with Oracle summaries and Expert contacts
            enriched_steps = []
            
            for step in learning_path:
                file_path = step['file_path']
                
                # Get file summary from Oracle
                file_summary = self._get_file_summary(repo_id, file_path)
                
                # Find experts for this file
                experts = self.expertise_service.find_experts_for_file(repo_id, file_path)
                top_expert = experts[0] if experts else None
                
                enriched_step = {
                    'file_path': file_path,
                    'step_number': step['step_number'],
                    'dependency_level': step['dependency_level'],
                    'summary': file_summary,
                    'top_expert': top_expert,
                    'learning_objective': self._generate_learning_objective(file_path, file_summary)
                }
                
                enriched_steps.append(enriched_step)
            
            # Step 4: Synthesize Final Report
            final_report = self._synthesize_final_report(
                repo_id, 
                issue_number, 
                issue_data, 
                enriched_steps,
                locus_files
            )
            
            return {
                'repository': repo_id,
                'issue_number': issue_number,
                'issue_title': issue_data.get('title', 'Unknown Issue'),
                'locus_files': locus_files,
                'learning_path_steps': len(enriched_steps),
                'onboarding_guide': final_report,
                'enriched_steps': enriched_steps,
                'generation_successful': True
            }
            
        except Exception as e:
            logger.error(f"Error generating onboarding path for issue #{issue_number}: {e}")
            return {
                'repository': repo_id,
                'issue_number': issue_number,
                'error': str(e),
                'generation_successful': False
            }
    
    def _get_issue_data(self, repo_id: str, issue_number: int) -> Dict[str, str]:
        """
        Get issue data. In the future, this would fetch from GitHub API.
        For now, return mock data.
        """
        # TODO: Integrate with GitHub service to fetch real issue data
        return {
            'title': f'Sample Issue #{issue_number}',
            'description': f'This is a sample issue description for issue #{issue_number} in repository {repo_id}. It involves fixing a bug or implementing a feature.'
        }
    
    def _identify_locus_of_change(self, repo_id: str, title: str, description: str) -> List[str]:
        """
        Identify the core files related to an issue using RAG search.
        
        Args:
            repo_id: Repository identifier
            title: Issue title
            description: Issue description
            
        Returns:
            List of file paths that are likely to be affected
        """
        try:
            # Combine title and description for search
            issue_text = f"{title}\n{description}"
            
            # Use RAG search to find relevant code chunks
            search_results = ollama.search_index(
                query_text=issue_text,
                model_name="snowflake-arctic-embed",
                repo_id=repo_id,
                k=5
            )
            
            # Extract unique file paths from search results
            locus_files = []
            for result in search_results:
                chunk_id = result.get('chunk_id', '')
                if chunk_id:
                    # Extract file path from chunk_id (format: repo_id_filepath_chunk_index)
                    parts = chunk_id.replace(f"{repo_id}_", "", 1).rsplit('_', 1)
                    if parts:
                        file_path = parts[0]
                        if file_path not in locus_files:
                            locus_files.append(file_path)
            
            return locus_files[:3]  # Return top 3 relevant files
            
        except Exception as e:
            logger.warning(f"RAG search failed for locus identification: {e}")
            return []
    
    def _build_learning_path(self, repo_id: str, locus_files: List[str]) -> List[Dict[str, Any]]:
        """
        Build a learning path by analyzing dependencies and creating a topological sort.
        
        Args:
            repo_id: Repository identifier
            locus_files: Core files identified for the issue
            
        Returns:
            Ordered list of files to learn, from least-dependent to most-dependent
        """
        try:
            # Load architectural graph
            architectural_graph = self._load_architectural_graph(repo_id)
            
            if not architectural_graph:
                # Fallback: simple ordering based on file paths
                return self._create_simple_learning_path(locus_files)
            
            # Create epistemic subgraph containing locus files and neighbors
            subgraph_files = self._create_epistemic_subgraph(architectural_graph, locus_files)
            
            # Perform topological sort to determine learning order
            learning_order = self._topological_sort(architectural_graph, subgraph_files)
            
            # Create learning path with metadata
            learning_path = []
            for i, file_path in enumerate(learning_order):
                step = {
                    'step_number': i + 1,
                    'file_path': file_path,
                    'dependency_level': self._calculate_dependency_level(architectural_graph, file_path),
                    'is_locus_file': file_path in locus_files
                }
                learning_path.append(step)
            
            return learning_path
            
        except Exception as e:
            logger.warning(f"Failed to build dependency-based learning path: {e}")
            return self._create_simple_learning_path(locus_files)
    
    def _load_architectural_graph(self, repo_id: str) -> Optional[Dict[str, Any]]:
        """Load architectural graph from cortex file."""
        cortex_file = self.cortex_directory / repo_id / "cortex.json"
        
        if not cortex_file.exists():
            return None
            
        try:
            with open(cortex_file, 'r', encoding='utf-8') as f:
                cortex_data = json.load(f)
            return cortex_data.get('architectural_graph')
        except Exception as e:
            logger.error(f"Error loading architectural graph: {e}")
            return None
    
    def _create_epistemic_subgraph(self, graph: Dict[str, Any], locus_files: List[str]) -> List[str]:
        """
        Create a subgraph containing locus files and their immediate neighbors.
        
        Args:
            graph: Full architectural graph
            locus_files: Core files for the issue
            
        Returns:
            List of files in the epistemic subgraph
        """
        subgraph_files = set(locus_files)
        
        # Extract edges from graph
        edges = graph.get('edges', [])
        
        # Add immediate neighbors (1-hop away)
        for edge in edges:
            source = self._extract_file_from_edge(edge, 'source')
            target = self._extract_file_from_edge(edge, 'target')
            
            if source in locus_files:
                subgraph_files.add(target)
            if target in locus_files:
                subgraph_files.add(source)
        
        return list(subgraph_files)
    
    def _extract_file_from_edge(self, edge: Any, direction: str) -> Optional[str]:
        """Extract file path from graph edge."""
        if isinstance(edge, dict):
            node_ref = edge.get(direction)
            if isinstance(node_ref, str):
                return node_ref
            elif isinstance(node_ref, dict):
                return node_ref.get('id') or node_ref.get('file_path')
        return None
    
    def _topological_sort(self, graph: Dict[str, Any], files: List[str]) -> List[str]:
        """
        Perform topological sort on the subgraph to determine learning order.
        
        Args:
            graph: Architectural graph
            files: Files to sort
            
        Returns:
            Files sorted from least-dependent to most-dependent
        """
        # Build adjacency list for the subgraph
        dependencies = {f: [] for f in files}
        
        edges = graph.get('edges', [])
        for edge in edges:
            source = self._extract_file_from_edge(edge, 'source')
            target = self._extract_file_from_edge(edge, 'target')
            
            if source in files and target in files:
                dependencies[source].append(target)
        
        # Kahn's algorithm for topological sorting
        in_degree = {f: 0 for f in files}
        for f in files:
            for dep in dependencies[f]:
                in_degree[dep] += 1
        
        queue = [f for f in files if in_degree[f] == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in dependencies[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def _create_simple_learning_path(self, locus_files: List[str]) -> List[Dict[str, Any]]:
        """Fallback simple learning path when graph analysis fails."""
        learning_path = []
        for i, file_path in enumerate(locus_files):
            step = {
                'step_number': i + 1,
                'file_path': file_path,
                'dependency_level': 'unknown',
                'is_locus_file': True
            }
            learning_path.append(step)
        return learning_path
    
    def _calculate_dependency_level(self, graph: Dict[str, Any], file_path: str) -> str:
        """Calculate dependency level (low, medium, high) for a file."""
        edges = graph.get('edges', [])
        
        # Count incoming dependencies
        incoming = sum(1 for edge in edges if self._extract_file_from_edge(edge, 'target') == file_path)
        
        if incoming == 0:
            return 'low'
        elif incoming <= 3:
            return 'medium'
        else:
            return 'high'
    
    def _get_file_summary(self, repo_id: str, file_path: str) -> str:
        """Get a summary of the file's purpose using Oracle service."""
        try:
            question = f"Provide a concise, one-paragraph summary of the purpose of the file '{file_path}' for a new developer."
            
            response = oracle_service.answer_question(
                repo_id=repo_id,
                question=question,
                context_limit=3  # Limit context for focused summary
            )
            
            # Extract just the answer portion
            if isinstance(response, dict) and 'answer' in response:
                return response['answer']
            elif isinstance(response, str):
                return response
            else:
                return f"File: {file_path} - Purpose unknown (Oracle query failed)"
                
        except Exception as e:
            logger.warning(f"Failed to get Oracle summary for {file_path}: {e}")
            return f"File: {file_path} - Summary not available"
    
    def _generate_learning_objective(self, file_path: str, summary: str) -> str:
        """Generate a learning objective for a file."""
        # Extract key concepts from the summary for the learning objective
        if "service" in summary.lower():
            return f"Understand how the {file_path.split('/')[-1]} service works and its role in the system"
        elif "model" in summary.lower():
            return f"Learn the data structure and relationships defined in {file_path.split('/')[-1]}"
        elif "controller" in summary.lower() or "view" in summary.lower():
            return f"Understand the request handling and user interface logic in {file_path.split('/')[-1]}"
        elif "util" in summary.lower() or "helper" in summary.lower():
            return f"Learn the utility functions and helper methods in {file_path.split('/')[-1]}"
        else:
            return f"Understand the purpose and functionality of {file_path.split('/')[-1]}"
    
    def _synthesize_final_report(
        self, 
        repo_id: str, 
        issue_number: int, 
        issue_data: Dict[str, str], 
        enriched_steps: List[Dict[str, Any]],
        locus_files: List[str]
    ) -> str:
        """
        Synthesize the final onboarding report using LLM.
        
        Args:
            repo_id: Repository identifier
            issue_number: Issue number
            issue_data: Issue title and description
            enriched_steps: Learning path with summaries and experts
            locus_files: Core files for the issue
            
        Returns:
            Formatted Markdown report
        """
        try:
            # Prepare context for LLM
            steps_context = ""
            for step in enriched_steps:
                expert_info = ""
                if step.get('top_expert'):
                    expert = step['top_expert']
                    expert_info = f" (Expert: {expert['author']} - {expert['email']})"
                
                steps_context += f"""
**Step {step['step_number']}: {step['file_path']}**
- Learning Objective: {step['learning_objective']}
- Dependency Level: {step['dependency_level']}
- Summary: {step['summary'][:200]}...{expert_info}

"""
            
            # Create comprehensive prompt for LLM
            prompt = f"""You are an expert technical mentor creating a personalized onboarding guide for a new developer. 

Generate a friendly, encouraging, and well-structured Markdown report to help a newcomer understand how to approach this GitHub issue.

**Context:**
- Repository: {repo_id}
- Issue #{issue_number}: {issue_data.get('title', 'Unknown')}
- Issue Description: {issue_data.get('description', 'No description available')}
- Core files to focus on: {', '.join(locus_files)}

**Learning Path (ordered from foundational to advanced):**
{steps_context}

**Instructions:**
1. Create a warm, welcoming introduction explaining the issue and what they'll learn
2. Present the learning path as a clear, step-by-step guide
3. For each step, explain WHY they should understand this file before moving to the next
4. Include the expert contact information naturally in each step
5. Add encouragement and tips for new developers
6. End with next steps and how to get help

Format the response as clean Markdown with proper headers, bullet points, and emphasis.
Make it feel like a personal mentorship session, not a dry technical document.
"""
            
            # Generate the final report
            final_report = llm_service.generate_text(
                prompt,
                task_type=TaskType.COMPLEX_REASONING
            )
            
            return final_report
            
        except Exception as e:
            logger.error(f"Failed to synthesize final report: {e}")
            # Fallback simple report
            return self._create_fallback_report(repo_id, issue_number, issue_data, enriched_steps)
    
    def _create_fallback_report(self, repo_id: str, issue_number: int, issue_data: Dict[str, str], enriched_steps: List[Dict[str, Any]]) -> str:
        """Create a simple fallback report when LLM synthesis fails."""
        report = f"""# Onboarding Guide: Issue #{issue_number}

## Issue Overview
**Repository:** {repo_id}
**Issue:** {issue_data.get('title', 'Unknown Issue')}

{issue_data.get('description', 'No description available')}

## Learning Path

"""
        
        for step in enriched_steps:
            expert_info = ""
            if step.get('top_expert'):
                expert = step['top_expert']
                expert_info = f"\n**Expert Contact:** {expert['author']} ({expert['email']})"
            
            report += f"""### Step {step['step_number']}: {step['file_path']}

**Learning Objective:** {step['learning_objective']}

**Summary:** {step['summary'][:300]}...{expert_info}

---

"""
        
        report += """
## Next Steps
1. Review each file in the order presented above
2. Reach out to the experts if you have questions
3. Start with small changes and test frequently

Good luck with your contribution!
"""
        
        return report


# Convenience functions for external use
def generate_onboarding_path(repo_id: str, issue_number: int) -> Dict[str, Any]:
    """
    Convenience function to generate an onboarding path.
    
    Args:
        repo_id: Repository identifier
        issue_number: GitHub issue number
        
    Returns:
        Comprehensive onboarding guide
    """
    service = OnboardingService()
    return service.generate_onboarding_path(repo_id, issue_number)