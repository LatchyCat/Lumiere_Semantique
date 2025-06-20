# In ingestion/crawler.py

import git
import tempfile
import pathlib
import fnmatch
from typing import List

# Default exclusion patterns for files and directories
DEFAULT_EXCLUDE_PATTERNS = [
    '.git', '.idea', '.vscode', '__pycache__', 'dist', 'build', '*.pyc',
    '*.egg-info', '*.so', '*.o', 'node_modules', 'venv', '.env', 'media', 'static'
]

class IntelligentCrawler:
    """
    Clones a Git repository and provides a filtered list of file paths
    based on inclusion/exclusion rules.
    """
    def __init__(self, repo_url: str, exclude_patterns: List[str] = None):
        self.repo_url = repo_url
        self.exclude_patterns = exclude_patterns or DEFAULT_EXCLUDE_PATTERNS
        self.temp_dir_handle = tempfile.TemporaryDirectory()
        self.repo_path = pathlib.Path(self.temp_dir_handle.name)

    def _is_excluded(self, path: pathlib.Path) -> bool:
        """Checks if a file or directory should be excluded."""
        # Use relative_to to correctly check parts of the path within the repo
        try:
            relative_path = path.relative_to(self.repo_path)
            parts = relative_path.parts
            for part in parts:
                for pattern in self.exclude_patterns:
                    if fnmatch.fnmatch(part, pattern):
                        return True
        except ValueError:
            # This can happen if the path is not within the repo_path, which is unexpected
            # but we handle it safely.
            return True
        return False

    def clone_and_process(self) -> List[pathlib.Path]:
        """
        Clones the repository and returns a list of file paths to be processed.

        Returns:
            A list of pathlib.Path objects for each file to process.
        """
        print(f"Cloning repository: {self.repo_url} into {self.repo_path}")
        try:
            git.Repo.clone_from(self.repo_url, self.repo_path)
            print("Repository cloned successfully.")
        except git.GitCommandError as e:
            print(f"Error cloning repository: {e}")
            return []

        all_paths = list(self.repo_path.rglob('*'))
        processed_files = []

        for path in all_paths:
            if self._is_excluded(path):
                continue
            if path.is_file():
                processed_files.append(path)

        return processed_files

    def cleanup(self):
        """Explicitly cleans up the temporary directory."""
        self.temp_dir_handle.cleanup()
        print(f"Cleaned up temporary directory: {self.repo_path}")
