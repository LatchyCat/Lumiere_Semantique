# backend/lumiere_core/services/diff_parser.py

import re
import logging
from typing import Dict, List, Set, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class DiffStats:
    """Statistics about a parsed diff."""
    total_files_changed: int
    total_lines_added: int
    total_lines_removed: int
    affected_nodes: int
    files_with_unknown_nodes: List[str]

@dataclass
class FileChange:
    """Represents changes to a single file."""
    file_path: str
    lines_added: int
    lines_removed: int
    changed_lines: Set[int]
    is_new_file: bool = False
    is_deleted_file: bool = False

class DiffParseError(Exception):
    """Custom exception for diff parsing errors."""
    pass

def get_changed_files_from_diff(diff_text: str) -> List[str]:
    """
    Extracts a list of file paths that were changed in the diff.

    Args:
        diff_text: The full text of a git diff

    Returns:
        Sorted list of unique file paths that were modified

    Raises:
        DiffParseError: If the diff text is malformed or empty
    """
    if not diff_text or not diff_text.strip():
        logger.warning("Empty or whitespace-only diff text provided")
        return []

    try:
        # Enhanced pattern to handle various diff formats
        pattern = re.compile(r'^\+\+\+\s(?:b/)?(.+?)(?:\s|$)', re.MULTILINE)
        matches = pattern.findall(diff_text)

        # Filter out /dev/null (used for new/deleted files) and empty matches
        file_paths = [
            match.strip() for match in matches
            if match.strip() and match.strip() != '/dev/null'
        ]

        if not file_paths:
            logger.info("No file changes detected in diff")

        return sorted(list(set(file_paths)))

    except re.error as e:
        raise DiffParseError(f"Regex compilation error: {e}") from e
    except Exception as e:
        raise DiffParseError(f"Unexpected error parsing diff: {e}") from e

def get_changed_files_with_stats(diff_text: str) -> List[FileChange]:
    """
    Enhanced version that returns detailed information about each changed file.

    Args:
        diff_text: The full text of a git diff

    Returns:
        List of FileChange objects with detailed statistics
    """
    if not diff_text or not diff_text.strip():
        return []

    file_changes = {}

    # Split diff by file sections
    file_sections = re.split(r'^diff --git', diff_text, flags=re.MULTILINE)

    for section in file_sections:
        if not section.strip():
            continue

        # Extract file path
        file_match = re.search(r'^\+\+\+\s(?:b/)?(.+?)(?:\s|$)', section, re.MULTILINE)
        if not file_match:
            continue

        file_path = file_match.group(1).strip()
        if file_path == '/dev/null':
            continue

        # Check if it's a new or deleted file
        is_new_file = '/dev/null' in re.search(r'^---\s(.+?)(?:\s|$)', section, re.MULTILINE).group(1) if re.search(r'^---\s(.+?)(?:\s|$)', section, re.MULTILINE) else False
        is_deleted_file = file_path == '/dev/null' or '--- a/' + file_path in section and '+++ /dev/null' in section

        # Count line changes and extract changed line numbers
        lines_added = len(re.findall(r'^\+(?!\+)', section, re.MULTILINE))
        lines_removed = len(re.findall(r'^-(?!-)', section, re.MULTILINE))

        # Extract changed line numbers
        changed_lines = set()
        hunk_headers = re.findall(r'^@@\s-\d+(?:,\d+)?\s\+(\d+)(?:,(\d+))?\s@@', section, re.MULTILINE)

        for start_line_str, length_str in hunk_headers:
            start_line = int(start_line_str)
            length = int(length_str) if length_str else 1
            changed_lines.update(range(start_line, start_line + length))

        file_changes[file_path] = FileChange(
            file_path=file_path,
            lines_added=lines_added,
            lines_removed=lines_removed,
            changed_lines=changed_lines,
            is_new_file=is_new_file,
            is_deleted_file=is_deleted_file
        )

    return sorted(file_changes.values(), key=lambda x: x.file_path)

def parse_diff_to_nodes(diff_text: str, file_to_node_map: Dict[str, List[Dict]]) -> List[str]:
    """
    Parses a git diff and, using a map of files to their nodes (functions, classes),
    identifies which specific nodes were modified.

    Args:
        diff_text: The full text of a `git diff`.
        file_to_node_map: A dictionary mapping file paths to a list of their contained
                          nodes, where each node has a 'name', 'start_line', and 'end_line'.
                          e.g., {'src/main.py': [{'id': '...', 'start_line': 10, 'end_line': 25}]}

    Returns:
        A list of unique node IDs that were affected by the changes in the diff.

    Raises:
        DiffParseError: If the diff text is malformed
        ValueError: If file_to_node_map has invalid structure
    """
    if not diff_text or not diff_text.strip():
        logger.warning("Empty diff text provided to parse_diff_to_nodes")
        return []

    if not isinstance(file_to_node_map, dict):
        raise ValueError("file_to_node_map must be a dictionary")

    # Validate node map structure
    _validate_node_map(file_to_node_map)

    affected_node_ids: Set[str] = set()

    try:
        # Split the diff by file sections - more robust approach
        file_sections = re.split(r'^diff --git', diff_text, flags=re.MULTILINE)

        for section in file_sections:
            if not section.strip():
                continue

            # Extract file path with better error handling
            file_path_match = re.search(r'^\+\+\+\s(?:b/)?(.+?)(?:\s|$)', section, re.MULTILINE)
            if not file_path_match:
                continue

            current_file = file_path_match.group(1).strip()

            # Skip /dev/null (new/deleted files)
            if current_file == '/dev/null':
                continue

            # If we don't have a map for this file, log it and continue
            if current_file not in file_to_node_map:
                logger.debug(f"No node mapping found for file: {current_file}")
                continue

            # Extract changed lines more efficiently
            changed_lines = _extract_changed_lines(section)

            if not changed_lines:
                continue

            # Find affected nodes
            affected_nodes = _find_affected_nodes(
                file_to_node_map[current_file],
                changed_lines
            )
            affected_node_ids.update(affected_nodes)

        result = sorted(list(affected_node_ids))
        logger.info(f"Found {len(result)} affected nodes from diff")
        return result

    except re.error as e:
        raise DiffParseError(f"Regex error while parsing diff: {e}") from e
    except Exception as e:
        raise DiffParseError(f"Unexpected error parsing diff to nodes: {e}") from e

def parse_diff_to_nodes_with_stats(
    diff_text: str,
    file_to_node_map: Dict[str, List[Dict]]
) -> Tuple[List[str], DiffStats]:
    """
    Enhanced version that returns both affected nodes and detailed statistics.

    Args:
        diff_text: The full text of a git diff
        file_to_node_map: Dictionary mapping file paths to their nodes

    Returns:
        Tuple of (affected_node_ids, diff_stats)
    """
    if not diff_text or not diff_text.strip():
        return [], DiffStats(0, 0, 0, 0, [])

    file_changes = get_changed_files_with_stats(diff_text)
    affected_node_ids = parse_diff_to_nodes(diff_text, file_to_node_map)

    files_with_unknown_nodes = [
        fc.file_path for fc in file_changes
        if fc.file_path not in file_to_node_map
    ]

    total_lines_added = sum(fc.lines_added for fc in file_changes)
    total_lines_removed = sum(fc.lines_removed for fc in file_changes)

    stats = DiffStats(
        total_files_changed=len(file_changes),
        total_lines_added=total_lines_added,
        total_lines_removed=total_lines_removed,
        affected_nodes=len(affected_node_ids),
        files_with_unknown_nodes=files_with_unknown_nodes
    )

    return affected_node_ids, stats

def filter_nodes_by_change_type(
    diff_text: str,
    file_to_node_map: Dict[str, List[Dict]],
    change_types: Set[str] = {'added', 'modified', 'deleted'}
) -> Dict[str, List[str]]:
    """
    Categorizes affected nodes by the type of change.

    Args:
        diff_text: The full text of a git diff
        file_to_node_map: Dictionary mapping file paths to their nodes
        change_types: Set of change types to include ('added', 'modified', 'deleted')

    Returns:
        Dictionary mapping change types to lists of affected node IDs
    """
    result = {change_type: [] for change_type in change_types}

    if not diff_text or not diff_text.strip():
        return result

    file_changes = get_changed_files_with_stats(diff_text)

    for file_change in file_changes:
        if file_change.file_path not in file_to_node_map:
            continue

        nodes = file_to_node_map[file_change.file_path]

        if file_change.is_new_file and 'added' in change_types:
            # All nodes in new files are considered added
            result['added'].extend(node['id'] for node in nodes)
        elif file_change.is_deleted_file and 'deleted' in change_types:
            # All nodes in deleted files are considered deleted
            result['deleted'].extend(node['id'] for node in nodes)
        elif 'modified' in change_types:
            # Find nodes that intersect with changed lines
            affected_nodes = _find_affected_nodes(nodes, file_change.changed_lines)
            result['modified'].extend(affected_nodes)

    # Remove duplicates and sort
    for change_type in result:
        result[change_type] = sorted(list(set(result[change_type])))

    return result

def _validate_node_map(file_to_node_map: Dict[str, List[Dict]]) -> None:
    """Validates the structure of the file_to_node_map."""
    for file_path, nodes in file_to_node_map.items():
        if not isinstance(nodes, list):
            raise ValueError(f"Nodes for file {file_path} must be a list")

        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                raise ValueError(f"Node {i} in file {file_path} must be a dictionary")

            required_keys = {'id', 'start_line', 'end_line'}
            if not required_keys.issubset(node.keys()):
                missing = required_keys - node.keys()
                raise ValueError(f"Node {i} in file {file_path} missing keys: {missing}")

            if not isinstance(node['start_line'], int) or not isinstance(node['end_line'], int):
                raise ValueError(f"Node {i} in file {file_path} line numbers must be integers")

            if node['start_line'] > node['end_line']:
                raise ValueError(f"Node {i} in file {file_path} has start_line > end_line")

def _extract_changed_lines(file_diff_section: str) -> Set[int]:
    """Extracts the set of changed line numbers from a file's diff section."""
    changed_lines: Set[int] = set()

    # Find all hunk headers like '@@ -15,7 +15,9 @@'
    hunk_headers = re.findall(
        r'^@@\s-\d+(?:,\d+)?\s\+(\d+)(?:,(\d+))?\s@@',
        file_diff_section,
        re.MULTILINE
    )

    for start_line_str, length_str in hunk_headers:
        start_line = int(start_line_str)
        # Length defaults to 1 if not specified
        length = int(length_str) if length_str else 1

        # Add all lines in this hunk to the changed lines set
        changed_lines.update(range(start_line, start_line + length))

    return changed_lines

def _find_affected_nodes(nodes: List[Dict], changed_lines: Set[int]) -> List[str]:
    """Finds nodes that are affected by the given changed lines."""
    affected_node_ids = []

    for node in nodes:
        node_line_range = range(node['start_line'], node['end_line'] + 1)

        # Check if any changed line falls within this node's range
        if any(line in node_line_range for line in changed_lines):
            affected_node_ids.append(node['id'])

    return affected_node_ids

# Backward compatibility aliases
def get_files_from_diff(diff_text: str) -> List[str]:
    """Deprecated: Use get_changed_files_from_diff instead."""
    logger.warning("get_files_from_diff is deprecated, use get_changed_files_from_diff")
    return get_changed_files_from_diff(diff_text)

# Utility functions for common use cases
def is_valid_diff(diff_text: str) -> bool:
    """
    Checks if the provided text appears to be a valid git diff.

    Args:
        diff_text: Text to validate

    Returns:
        True if the text appears to be a valid diff, False otherwise
    """
    if not diff_text or not diff_text.strip():
        return False

    # Look for common diff indicators
    diff_indicators = [
        r'^diff --git',
        r'^---\s',
        r'^\+\+\+\s',
        r'^@@.*@@'
    ]

    return any(re.search(pattern, diff_text, re.MULTILINE) for pattern in diff_indicators)

def get_diff_summary(diff_text: str) -> str:
    """
    Returns a human-readable summary of the diff.

    Args:
        diff_text: The full text of a git diff

    Returns:
        A summary string describing the changes
    """
    if not is_valid_diff(diff_text):
        return "Invalid or empty diff"

    file_changes = get_changed_files_with_stats(diff_text)

    if not file_changes:
        return "No file changes detected"

    total_files = len(file_changes)
    total_added = sum(fc.lines_added for fc in file_changes)
    total_removed = sum(fc.lines_removed for fc in file_changes)
    new_files = sum(1 for fc in file_changes if fc.is_new_file)
    deleted_files = sum(1 for fc in file_changes if fc.is_deleted_file)

    parts = [f"{total_files} file{'s' if total_files != 1 else ''} changed"]

    if total_added > 0:
        parts.append(f"{total_added} insertion{'s' if total_added != 1 else ''}")

    if total_removed > 0:
        parts.append(f"{total_removed} deletion{'s' if total_removed != 1 else ''}")

    if new_files > 0:
        parts.append(f"{new_files} new file{'s' if new_files != 1 else ''}")

    if deleted_files > 0:
        parts.append(f"{deleted_files} deleted file{'s' if deleted_files != 1 else ''}")

    return ", ".join(parts)
