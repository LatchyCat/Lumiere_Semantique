# In ingestion/crawler.py
import subprocess
import tempfile
import pathlib
from typing import List, Optional, Union, Dict

class IntelligentCrawler:
    """
    Clones a Git repository and performs file operations safely.
    Includes path-finding, git blame, and git diff capabilities.
    """

    def __init__(self, repo_url: str):
        """
        Initializes the crawler with the repository URL.
        """
        self.repo_url = repo_url
        self.temp_dir_handle = tempfile.TemporaryDirectory()
        self.repo_path = pathlib.Path(self.temp_dir_handle.name)
        self._file_paths_cache: Optional[List[pathlib.Path]] = None

    def __enter__(self):
        """
        Enters the context manager, cloning the repository.
        """
        print(f"Entering context: Cloning {self.repo_url} into {self.repo_path}")
        self._clone_repo()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the context manager, cleaning up resources.
        """
        print("Exiting context: Cleaning up resources.")
        self.cleanup()

    def _clone_repo(self):
        """
        Clones the full git repository, including all branches and tags.
        """
        try:
            # '--mirror' is too aggressive, '--bare' isn't a working tree.
            # We will clone normally and then fetch all tags and branches.
            subprocess.run(
                ['git', 'clone', self.repo_url, str(self.repo_path)],
                check=True, capture_output=True, text=True
            )
            # After cloning, fetch all tags and remote branches explicitly.
            # 'git fetch origin --tags' and 'git fetch origin' ensures everything is available.
            subprocess.run(['git', 'fetch', 'origin', '--tags'], cwd=self.repo_path, check=True, capture_output=True, text=True)
            subprocess.run(['git', 'remote', 'update'], cwd=self.repo_path, check=True, capture_output=True, text=True)

            print(f"Repository cloned successfully and all refs fetched from {self.repo_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e.stderr.strip()}")
            raise

    def get_blame_for_file(self, target_file: str) -> str:
        """
        Runs `git blame` on a specific file in the repo.
        """
        file_full_path = self.repo_path / target_file
        if not file_full_path.exists():
            return f"Error from crawler: File '{target_file}' does not exist in the repository."

        try:
            print(f"Running 'git blame' on {file_full_path}...")
            result = subprocess.run(
                ['git', 'blame', '--show-email', str(file_full_path)],
                cwd=self.repo_path, check=True, capture_output=True, text=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            error_message = f"Error running 'git blame' on '{target_file}': {e.stderr.strip()}"
            print(error_message)
            return error_message

    def get_diff_for_branch(self, ref_name: str, base_ref: str = 'main') -> str:
        """
        [Final Version] Gets the `git diff` between two refs (branch, tag, or commit).
        This version is robust and relies on the three-dot diff syntax.
        """
        try:
            print(f"Attempting to calculate diff for '{ref_name}' against base '{base_ref}'...")

            # The '...' syntax finds the diff from the common ancestor, which is what a
            # code review for a PR/feature branch usually wants. We prepend 'origin/'
            # to ensure we're comparing against the fetched remote state.
            # For tags, they don't need 'origin/'. Git is smart enough.
            # Let's check if the ref is a tag.

            # To make this truly robust, we'll try to resolve the refs first.
            # Let's stick to the simplest, most powerful git syntax.

            # The refs are specified as 'origin/<branch_name>' for remote branches.
            # Tags are just referred to by their name. Git resolves this automatically
            # if we have fetched all data. The logic here simplifies to trying 'main'
            # then 'master' as a base.

            diff_command = ['git', 'diff', f'origin/{base_ref}...{ref_name}']

            print(f"   -> Running command: {' '.join(diff_command)}")
            result = subprocess.run(
                diff_command,
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            # If the command failed, it might be because the base is 'master' not 'main'.
            if base_ref == 'main':
                print(f"   -> Diff against 'origin/main' failed. Trying 'origin/master' as base...")
                return self.get_diff_for_branch(ref_name, 'master')

            error_message = f"Error running 'git diff' between '{base_ref}' and '{ref_name}': {e.stderr.strip()}"
            print(error_message)
            return f"Error from crawler: {error_message}"


    def find_file_path(self, target_filename: str) -> Union[str, Dict, None]:
        """
        Searches the repository for a file by its name.
        """
        print(f"Searching for file matching '{target_filename}'...")
        all_files = self.get_file_paths()

        possible_matches = []
        for file_path in all_files:
            relative_path = file_path.relative_to(self.repo_path)
            if relative_path.name == target_filename or str(relative_path).endswith('/' + target_filename):
                possible_matches.append(relative_path)

        if not possible_matches:
            print(f"   -> No match found for '{target_filename}'.")
            return None

        if len(possible_matches) == 1:
            match = str(possible_matches[0])
            print(f"   -> Found unique match: {match}")
            return match

        print(f"   -> Found multiple matches: {[str(p) for p in possible_matches]}. Checking for a definitive root-level file.")
        root_matches = [p for p in possible_matches if len(p.parts) == 1]

        if len(root_matches) == 1:
            match = str(root_matches[0])
            print(f"   -> Prioritized unique root match: {match}")
            return match

        print(f"   -> Ambiguity detected. Multiple candidates found. Reporting conflict.")
        return {
            "error": "ambiguous_path",
            "message": f"Multiple files found matching '{target_filename}'. Please specify one.",
            "options": [str(p) for p in possible_matches]
        }


    def get_file_paths(self) -> List[pathlib.Path]:
        """
        Scans the cloned repo and returns a list of relevant files.
        Caches the result for performance.
        """
        if self._file_paths_cache is not None:
            return self._file_paths_cache

        print("Scanning for relevant files...")
        files_to_process = []
        included_extensions = [
            '*.py', '*.md', '*.txt', '*.rst', '*.json', '*.toml', '*.yaml',
            '*.js', 'Dockerfile', 'LICENSE'
        ]
        excluded_dirs = {'.git', '__pycache__', 'venv', 'node_modules', '.vscode', '.idea', 'dist', 'build'}

        for file_path in self.repo_path.rglob('*'):
            if any(part in excluded_dirs for part in file_path.relative_to(self.repo_path).parts):
                continue
            if file_path.is_file() and any(file_path.match(ext) for ext in included_extensions):
                files_to_process.append(file_path)

        self._file_paths_cache = files_to_process
        print(f"Found and cached {len(files_to_process)} files to process.")
        return self._file_paths_cache

    def cleanup(self):
        """
        Removes the temporary directory and all its contents.
        """
        self.temp_dir_handle.cleanup()
        print(f"Cleaned up temporary directory: {self.repo_path}")
