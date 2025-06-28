"""
Expertise Service

This service identifies knowledgeable contributors for specific files or modules
by analyzing git blame data and commit history. This powers the "Find an Expert" feature
of the Onboarding Concierge.
"""

import json
import pathlib
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ExpertiseService:
    """
    Service to calculate and rank experts for given code artifacts based on
    contribution patterns from git blame and commit history.
    """
    
    def __init__(self, cortex_directory: Optional[pathlib.Path] = None):
        """
        Initialize the Expertise Service.
        
        Args:
            cortex_directory: Optional path to the cortex directory.
                            If None, will use default backend/cloned_repositories/
        """
        self.cortex_directory = cortex_directory or pathlib.Path("backend/cloned_repositories")
    
    def find_experts_for_file(self, repo_id: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Find and rank experts for a specific file based on contribution metrics.
        
        Args:
            repo_id: Repository identifier
            file_path: Path to the file relative to repository root
            
        Returns:
            List of experts sorted by expertise score, each containing:
            {
                'author': str,           # Author name
                'email': str,            # Author email
                'score': float,          # Expertise score (0-100)
                'details': {
                    'blame_lines': int,     # Lines attributed via git blame
                    'commits': int,         # Number of commits (if available)
                    'percentage': float     # Percentage of total lines
                }
            }
        """
        try:
            # Load blame cache data
            blame_data = self._load_blame_cache(repo_id)
            
            if not blame_data or file_path not in blame_data:
                logger.warning(f"No blame data found for {file_path} in {repo_id}")
                return []
            
            file_blame = blame_data[file_path]
            
            if not file_blame:
                logger.warning(f"Empty blame data for {file_path}")
                return []
            
            # Calculate total lines for percentage calculation
            total_lines = sum(file_blame.values())
            
            # Score each contributor
            experts = []
            for email, line_count in file_blame.items():
                # Extract name from email if available, otherwise use email
                author_name = self._extract_name_from_email(email)
                
                # Calculate base score from line contribution
                line_percentage = (line_count / total_lines) * 100 if total_lines > 0 else 0
                
                # For now, use blame data as primary metric
                # TODO: In Phase 3.1 enhancement, add commit history data
                expertise_score = self._calculate_expertise_score(
                    blame_lines=line_count,
                    total_lines=total_lines,
                    commits=0  # Placeholder for future GitHub API integration
                )
                
                expert_data = {
                    'author': author_name,
                    'email': email,
                    'score': round(expertise_score, 1),
                    'details': {
                        'blame_lines': line_count,
                        'commits': 0,  # Placeholder
                        'percentage': round(line_percentage, 1)
                    }
                }
                
                experts.append(expert_data)
            
            # Sort by expertise score descending
            experts.sort(key=lambda x: x['score'], reverse=True)
            
            # Limit to top 10 experts
            return experts[:10]
            
        except Exception as e:
            logger.error(f"Error finding experts for {file_path} in {repo_id}: {e}")
            return []
    
    def find_experts_for_module(self, repo_id: str, module_pattern: str) -> List[Dict[str, Any]]:
        """
        Find experts for a module or directory pattern.
        
        Args:
            repo_id: Repository identifier
            module_pattern: Pattern to match files (e.g., "src/services/", "*.py")
            
        Returns:
            Aggregated list of experts across all matching files
        """
        try:
            blame_data = self._load_blame_cache(repo_id)
            
            if not blame_data:
                return []
            
            # Find matching files
            matching_files = []
            for file_path in blame_data.keys():
                if self._matches_pattern(file_path, module_pattern):
                    matching_files.append(file_path)
            
            if not matching_files:
                logger.warning(f"No files matching pattern '{module_pattern}' found in {repo_id}")
                return []
            
            # Aggregate contributions across all matching files
            aggregated_contributions = {}
            total_lines_across_files = 0
            
            for file_path in matching_files:
                file_blame = blame_data[file_path]
                for email, line_count in file_blame.items():
                    aggregated_contributions[email] = aggregated_contributions.get(email, 0) + line_count
                    total_lines_across_files += line_count
            
            # Generate expert rankings
            experts = []
            for email, total_line_count in aggregated_contributions.items():
                author_name = self._extract_name_from_email(email)
                
                expertise_score = self._calculate_expertise_score(
                    blame_lines=total_line_count,
                    total_lines=total_lines_across_files,
                    commits=0  # Placeholder
                )
                
                percentage = (total_line_count / total_lines_across_files) * 100 if total_lines_across_files > 0 else 0
                
                expert_data = {
                    'author': author_name,
                    'email': email,
                    'score': round(expertise_score, 1),
                    'details': {
                        'blame_lines': total_line_count,
                        'commits': 0,  # Placeholder
                        'percentage': round(percentage, 1),
                        'files_involved': len([f for f in matching_files if email in blame_data[f]]),
                        'module_pattern': module_pattern
                    }
                }
                
                experts.append(expert_data)
            
            experts.sort(key=lambda x: x['score'], reverse=True)
            return experts[:10]
            
        except Exception as e:
            logger.error(f"Error finding experts for module '{module_pattern}' in {repo_id}: {e}")
            return []
    
    def _load_blame_cache(self, repo_id: str) -> Optional[Dict[str, Dict[str, int]]]:
        """
        Load the blame cache file for a repository.
        
        Args:
            repo_id: Repository identifier
            
        Returns:
            Blame cache data or None if not found
        """
        blame_cache_file = self.cortex_directory / repo_id / f"{repo_id}_blame_cache.json"
        
        if not blame_cache_file.exists():
            logger.warning(f"Blame cache file not found: {blame_cache_file}")
            return None
        
        try:
            with open(blame_cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading blame cache {blame_cache_file}: {e}")
            return None
    
    def _extract_name_from_email(self, email: str) -> str:
        """
        Extract a display name from an email address.
        
        Args:
            email: Email address
            
        Returns:
            Display name or email if extraction fails
        """
        if '@' in email:
            local_part = email.split('@')[0]
            # Convert common patterns to readable names
            name = local_part.replace('.', ' ').replace('_', ' ').replace('-', ' ')
            # Capitalize words
            return ' '.join(word.capitalize() for word in name.split())
        return email
    
    def _calculate_expertise_score(self, blame_lines: int, total_lines: int, commits: int = 0) -> float:
        """
        Calculate expertise score using weighted formula.
        
        Args:
            blame_lines: Number of lines attributed to this author
            total_lines: Total lines in the file/module
            commits: Number of commits (placeholder for future enhancement)
            
        Returns:
            Expertise score between 0 and 100
        """
        # Current implementation focuses on blame data
        # Future enhancement will incorporate commit history
        
        if total_lines == 0:
            return 0.0
        
        # Base score from line contribution percentage
        line_percentage = (blame_lines / total_lines) * 100
        
        # Apply logarithmic scaling to avoid overly penalizing smaller contributors
        import math
        
        # Score based on contribution percentage with diminishing returns
        if line_percentage <= 0:
            score = 0.0
        elif line_percentage >= 50:
            score = 90.0 + (line_percentage - 50) * 0.2  # Slow growth after 50%
        else:
            score = line_percentage * 1.8  # Linear growth up to 50%
        
        # Future: Add commit frequency bonus
        # commit_bonus = min(commits * 2, 10)  # Up to 10 bonus points
        # score += commit_bonus
        
        return min(score, 100.0)
    
    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """
        Check if a file path matches a given pattern.
        
        Args:
            file_path: File path to check
            pattern: Pattern to match (supports * wildcards and directory matching)
            
        Returns:
            True if file matches pattern
        """
        import fnmatch
        
        # Handle directory patterns
        if pattern.endswith('/'):
            return file_path.startswith(pattern) or f"/{pattern}" in file_path
        
        # Handle file extension patterns
        if pattern.startswith('*.'):
            return file_path.endswith(pattern[1:])
        
        # Handle general glob patterns
        return fnmatch.fnmatch(file_path, pattern)
    
    def get_repository_experts_summary(self, repo_id: str) -> Dict[str, Any]:
        """
        Get an overall summary of expertise distribution in the repository.
        
        Args:
            repo_id: Repository identifier
            
        Returns:
            Summary with top contributors and statistics
        """
        try:
            blame_data = self._load_blame_cache(repo_id)
            
            if not blame_data:
                return {'error': 'No blame data available'}
            
            # Aggregate all contributions
            global_contributions = {}
            total_files = len(blame_data)
            total_lines = 0
            
            for file_path, file_blame in blame_data.items():
                for email, line_count in file_blame.items():
                    if email not in global_contributions:
                        global_contributions[email] = {
                            'total_lines': 0,
                            'files_contributed': 0
                        }
                    
                    global_contributions[email]['total_lines'] += line_count
                    global_contributions[email]['files_contributed'] += 1
                    total_lines += line_count
            
            # Calculate top contributors
            top_contributors = []
            for email, data in global_contributions.items():
                author_name = self._extract_name_from_email(email)
                contribution_percentage = (data['total_lines'] / total_lines) * 100 if total_lines > 0 else 0
                
                contributor = {
                    'author': author_name,
                    'email': email,
                    'total_lines': data['total_lines'],
                    'files_contributed': data['files_contributed'],
                    'contribution_percentage': round(contribution_percentage, 1)
                }
                
                top_contributors.append(contributor)
            
            # Sort by total lines contributed
            top_contributors.sort(key=lambda x: x['total_lines'], reverse=True)
            
            return {
                'repository': repo_id,
                'total_files_analyzed': total_files,
                'total_lines_of_code': total_lines,
                'total_contributors': len(global_contributions),
                'top_contributors': top_contributors[:10],
                'analysis_timestamp': self._get_current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Error generating repository experts summary for {repo_id}: {e}")
            return {'error': str(e)}
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()


# Convenience functions for external use
def find_experts_for_file(repo_id: str, file_path: str) -> List[Dict[str, Any]]:
    """
    Convenience function to find experts for a specific file.
    
    Args:
        repo_id: Repository identifier
        file_path: Path to the file
        
    Returns:
        List of experts ranked by expertise score
    """
    service = ExpertiseService()
    return service.find_experts_for_file(repo_id, file_path)


def find_experts_for_module(repo_id: str, module_pattern: str) -> List[Dict[str, Any]]:
    """
    Convenience function to find experts for a module or pattern.
    
    Args:
        repo_id: Repository identifier
        module_pattern: File/directory pattern to match
        
    Returns:
        List of experts ranked by expertise score
    """
    service = ExpertiseService()
    return service.find_experts_for_module(repo_id, module_pattern)