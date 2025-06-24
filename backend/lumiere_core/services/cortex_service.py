# backend/lumiere_core/services/cortex_service.py

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Constants
CORTEX_FILENAME_TEMPLATE = "{repo_id}_cortex.json"
CLONED_REPOS_DIR = "cloned_repositories"


class CortexFileNotFound(Exception):
    """Raised when the Cortex file is missing for a given repository."""


class CortexFileMalformed(Exception):
    """Raised when the Cortex file is unreadable or not valid JSON."""


def _get_cortex_path(repo_id: str) -> Path:
    """
    Constructs the full path to a repository's Cortex file.

    Args:
        repo_id: The unique ID of the repository.

    Returns:
        Path object pointing to the expected Cortex file location.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    return base_dir / CLONED_REPOS_DIR / repo_id / CORTEX_FILENAME_TEMPLATE.format(repo_id=repo_id)


def load_cortex_data(repo_id: str) -> Dict[str, Any]:
    """
    Loads and parses the cortex JSON file for a given repository.

    Args:
        repo_id: The unique ID of the repository.

    Returns:
        Parsed JSON data as a dictionary.

    Raises:
        CortexFileNotFound: If the cortex file does not exist.
        CortexFileMalformed: If the file is not valid JSON.
    """
    cortex_path = _get_cortex_path(repo_id)

    if not cortex_path.exists():
        logger.error(f"Cortex file not found at: {cortex_path}")
        raise CortexFileNotFound(f"Cortex file not found for repo: {repo_id}")

    try:
        with cortex_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.exception(f"Failed to parse cortex file: {cortex_path}")
        raise CortexFileMalformed(f"Failed to load or parse cortex file: {e}") from e


def get_file_content(repo_id: str, file_path: str) -> Optional[str]:
    """
    Retrieves the raw content of a specific file from the repository's Cortex data.

    Args:
        repo_id: The unique ID of the repository.
        file_path: The relative path of the file within the repo.

    Returns:
        The raw file content as a string, or None if the file isn't found.
    """
    try:
        cortex_data = load_cortex_data(repo_id)
        for file_entry in cortex_data.get("files", []):
            if file_entry.get("file_path") == file_path:
                return file_entry.get("raw_content")
    except (CortexFileNotFound, CortexFileMalformed):
        return None

    return None
