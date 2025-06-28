# backend/lumiere_core/services/ingestion_service.py

import json
import traceback
import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from contextlib import contextmanager
from datetime import datetime

from ingestion.crawler import IntelligentCrawler, LocalDirectoryCrawler, BaseCrawler
from ingestion.jsonifier import Jsonifier
from ingestion.indexing import EmbeddingIndexer
from . import sentinel_service

# BOM parser
from .bom_parser import parse_all_manifests, EnumEncoder

# Configure logging
logger = logging.getLogger(__name__)

class IngestionError(Exception):
    """Custom exception for ingestion-related errors."""
    pass

class RepositoryValidationError(IngestionError):
    """Raised when repository validation fails."""
    pass

def validate_repo_url(repo_url: str) -> bool:
    """
    Validate that the repository URL is a valid GitHub URL.

    Args:
        repo_url: The repository URL to validate

    Returns:
        bool: True if valid, False otherwise

    Raises:
        RepositoryValidationError: If URL is invalid
    """
    if not repo_url or not isinstance(repo_url, str):
        raise RepositoryValidationError("Repository URL must be a non-empty string")

    github_pattern = r'^https://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/?$'
    if not re.match(github_pattern, repo_url.rstrip('/')):
        raise RepositoryValidationError(f"Invalid GitHub URL format: {repo_url}")

    return True

def generate_repo_id(repo_url: str, max_length: int = 100) -> str:
    """
    Generate a safe repository ID from a GitHub URL with intelligent truncation.

    Args:
        repo_url: The GitHub repository URL
        max_length: Maximum length for the generated ID

    Returns:
        str: Safe repository identifier

    Raises:
        RepositoryValidationError: If URL is invalid
    """
    validate_repo_url(repo_url)

    # Extract the repo path (owner/repo-name)
    repo_path = repo_url.replace("https://github.com/", "").replace("/", "_").rstrip("/")

    # Remove any potentially dangerous characters
    repo_path = re.sub(r'[^a-zA-Z0-9_.-]', '_', repo_path)

    # If it's already within limits, return as-is (backward compatibility)
    if len(repo_path) <= max_length:
        return repo_path

    # Split into owner and repo name for intelligent truncation
    original_path = repo_url.replace("https://github.com/", "").rstrip("/")
    parts = original_path.split("/", 1)

    if len(parts) != 2:
        # Fallback: just truncate the whole thing
        return repo_path[:max_length]

    owner, repo_name = parts
    # Ensure we don't end with an underscore and leave room for separator
    truncated_owner = owner[:40].rstrip('_')
    remaining_length = max_length - len(truncated_owner) - 1  # -1 for separator
    truncated_repo = repo_name[:remaining_length].rstrip('_')

    return f"{truncated_owner}_{truncated_repo}"

def _load_existing_metrics(metrics_path: Path) -> list:
    """
    Load existing metrics from file with error handling.

    Args:
        metrics_path: Path to the metrics file

    Returns:
        list: Existing metrics or empty list if file doesn't exist or is corrupted
    """
    if not metrics_path.exists():
        return []

    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load existing metrics from {metrics_path}: {e}")
        return []

def _save_metrics(metrics_path: Path, metrics_data: list) -> None:
    """
    Save metrics to file with error handling.

    Args:
        metrics_path: Path to save metrics
        metrics_data: Metrics data to save

    Raises:
        IngestionError: If saving fails
    """
    try:
        # Ensure directory exists
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2)
    except IOError as e:
        raise IngestionError(f"Failed to save metrics: {e}")

def _save_cortex_file(output_path: Path, project_cortex: Dict[str, Any]) -> None:
    """
    Save project cortex to file with error handling.

    Args:
        output_path: Path to save the cortex file
        project_cortex: The cortex data to save

    Raises:
        IngestionError: If saving fails
    """
    try:
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(project_cortex, f, indent=2, cls=EnumEncoder)
    except IOError as e:
        raise IngestionError(f"Failed to save cortex file: {e}")

@contextmanager
def _cleanup_on_failure(output_cortex_path: Path):
    """
    Context manager to cleanup cortex file on failure.

    Args:
        output_cortex_path: Path to the cortex file to cleanup on failure
    """
    try:
        yield
    except Exception:
        if output_cortex_path.exists():
            try:
                os.remove(output_cortex_path)
                logger.info(f"Cleaned up cortex file: {output_cortex_path}")
            except OSError as e:
                logger.warning(f"Failed to cleanup cortex file {output_cortex_path}: {e}")
        raise

def _setup_directories(repo_id: str) -> Tuple[Path, Path, Path]:
    """
    Setup required directories for ingestion.

    Args:
        repo_id: Repository identifier

    Returns:
        Tuple of (repo_output_dir, output_cortex_path, metrics_path)
    """
    backend_dir = Path(__file__).resolve().parent.parent.parent
    artifacts_base_dir = backend_dir / "cloned_repositories"
    repo_output_dir = artifacts_base_dir / repo_id
    repo_output_dir.mkdir(parents=True, exist_ok=True)

    output_cortex_path = repo_output_dir / f"{repo_id}_cortex.json"
    metrics_path = repo_output_dir / "metrics.json"

    return repo_output_dir, output_cortex_path, metrics_path

def clone_and_embed_repository(
    repo_url: str,
    embedding_model: str = 'snowflake-arctic-embed2:latest',
    skip_indexing: bool = False
) -> Dict[str, Any]:
    """
    Orchestrates the entire ingestion pipeline, including the Sentinel metrics capture.

    Args:
        repo_url: The GitHub repository URL to process
        embedding_model: The embedding model to use for vector indexing
        skip_indexing: If True, skip the vector indexing step

    Returns:
        Dict containing the result status and metadata

    Raises:
        RepositoryValidationError: If repository URL is invalid
        IngestionError: If ingestion process fails
    """
    start_time = datetime.now()

    try:
        # Validate inputs
        validate_repo_url(repo_url)
        if not embedding_model or not isinstance(embedding_model, str):
            raise IngestionError("Embedding model must be a non-empty string")

        repo_id = generate_repo_id(repo_url)
        repo_output_dir, output_cortex_path, metrics_path = _setup_directories(repo_id)

        logger.info(f"Starting ingestion for {repo_id}")
        logger.info(f"Artifacts will be saved to: {repo_output_dir}")

        with _cleanup_on_failure(output_cortex_path):
            project_cortex = None

            # --- Step 1: Crawl & Jsonify ---
            logger.info("[1/4] Cloning repository and generating Project Cortex file...")

            with IntelligentCrawler(repo_url=repo_url) as crawler:
                files_to_process = crawler.get_file_paths()
                if not files_to_process:
                    raise IngestionError("No files found to process in the repository")

                jsonifier = Jsonifier(
                    file_paths=files_to_process,
                    repo_root=crawler.repo_path,
                    repo_id=repo_id
                )
                # The generate_cortex function now returns a complete dictionary, ready for JSON
                project_cortex = jsonifier.generate_cortex()

                if not project_cortex:
                    raise IngestionError("Failed to generate project cortex")

                # --- Step 2: Generate Bill of Materials (BOM) ---
                logger.info("Generating Tech Stack Bill of Materials...")
                try:
                    tech_stack_bom = parse_all_manifests(crawler.repo_path)
                    if tech_stack_bom:
                        project_cortex["tech_stack_bom"] = tech_stack_bom.to_dict()
                        logger.info(f"✓ BOM Generated. Found {len(tech_stack_bom.dependencies['application'])} application dependencies.")
                    else:
                        logger.warning("BOM Generated. Found 0 total dependencies.")
                except Exception as e:
                    logger.warning(f"Failed to generate BOM: {e}")
                    # Continue with ingestion even if BOM fails

                logger.info(f"Project Cortex created successfully: {output_cortex_path}")

                # --- Step 3: Calculate Sentinel Metrics ---
                logger.info("[2/4] Sentinel: Calculating health metrics...")
                graph_data = project_cortex.get("architectural_graph", {})

                try:
                    latest_metrics = sentinel_service.calculate_snapshot_metrics(
                        crawler.repo_path, graph_data
                    )

                    # Add timestamp to metrics
                    latest_metrics['timestamp'] = start_time.isoformat()
                    latest_metrics['repo_id'] = repo_id

                    # Load existing metrics and append the new snapshot
                    historical_metrics = _load_existing_metrics(metrics_path)
                    historical_metrics.append(latest_metrics)

                    _save_metrics(metrics_path, historical_metrics)
                    logger.info("Sentinel: Health metrics saved successfully")

                except Exception as e:
                    logger.warning(f"Failed to calculate or save metrics: {e}")
                    # Continue with ingestion even if metrics fail

            # Save cortex file
            _save_cortex_file(output_cortex_path, project_cortex)

            # --- Step 3: Vector Indexing (Optional) ---
            if not skip_indexing:
                logger.info(f"[3/4] Starting vector indexing with model '{embedding_model}'...")
                try:
                    indexer = EmbeddingIndexer(model_name=embedding_model)
                    indexer.process_cortex(str(output_cortex_path))
                    logger.info("Vector indexing complete")
                except Exception as e:
                    logger.error(f"Vector indexing failed: {e}")
                    # Don't fail the entire process if indexing fails
                    raise IngestionError(f"Vector indexing failed: {e}")
            else:
                logger.info("[3/4] Skipping vector indexing as requested")

            logger.info("[4/4] Ingestion complete")

            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                "status": "success",
                "message": f"Repository '{repo_id}' was successfully cloned, embedded, and indexed.",
                "repo_id": repo_id,
                "original_url": repo_url,
                "processing_time_seconds": processing_time,
                "files_processed": len(files_to_process) if 'files_to_process' in locals() else 0,
                "cortex_path": str(output_cortex_path),
                "metrics_path": str(metrics_path),
                "indexing_skipped": skip_indexing
            }

    except (RepositoryValidationError, IngestionError) as e:
        logger.error(f"Ingestion failed for {repo_url}: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "repo_id": repo_id if 'repo_id' in locals() else None,
            "processing_time_seconds": (datetime.now() - start_time).total_seconds()
        }
    except Exception as e:
        logger.error(f"Unexpected error during ingestion for {repo_url}: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "failed",
            "error": str(e),
            "details": traceback.format_exc(),
            "repo_id": repo_id if 'repo_id' in locals() else None,
            "processing_time_seconds": (datetime.now() - start_time).total_seconds()
        }

# --- LIBRARIAN'S ARCHIVES: LOCAL DIRECTORY INGESTION ---

def generate_archive_id(directory_path: str) -> str:
    """
    Generate a unique archive identifier from a directory path.
    
    Args:
        directory_path: Path to the local directory
        
    Returns:
        str: Unique archive identifier
    """
    import hashlib
    
    # Normalize path and create hash
    normalized_path = Path(directory_path).resolve()
    path_str = str(normalized_path)
    
    # Create a hash from the path
    hash_obj = hashlib.md5(path_str.encode('utf-8'))
    path_hash = hash_obj.hexdigest()[:8]
    
    # Use last directory name + hash for readability
    dir_name = normalized_path.name
    archive_id = f"archive_{dir_name}_{path_hash}"
    
    return archive_id

def _setup_archive_directories(archive_id: str) -> Tuple[Path, Path, Path]:
    """
    Setup required directories for archive ingestion.
    
    Args:
        archive_id: Archive identifier
        
    Returns:
        Tuple of (archive_output_dir, output_cortex_path, metrics_path)
    """
    backend_dir = Path(__file__).resolve().parent.parent.parent
    archives_base_dir = backend_dir / "local_archives"
    archive_output_dir = archives_base_dir / archive_id
    archive_output_dir.mkdir(parents=True, exist_ok=True)
    
    output_cortex_path = archive_output_dir / f"{archive_id}_cortex.json"
    metrics_path = archive_output_dir / "metrics.json"
    
    return archive_output_dir, output_cortex_path, metrics_path

def ingest_local_directory(
    directory_path: str,
    embedding_model: str = 'snowflake-arctic-embed2:latest',
    skip_indexing: bool = False
) -> Dict[str, Any]:
    """
    Ingest a local directory as an Archive using the Librarian's Archives feature.
    Part of the Archetype Evolution from Seeker -> Librarian.
    
    Args:
        directory_path: Path to the local directory to ingest
        embedding_model: The embedding model to use for vector indexing
        skip_indexing: If True, skip the vector indexing step
        
    Returns:
        Dict containing the result status and metadata
        
    Raises:
        IngestionError: If ingestion process fails
    """
    start_time = datetime.now()
    
    try:
        # Validate inputs
        if not directory_path or not isinstance(directory_path, str):
            raise IngestionError("Directory path must be a non-empty string")
        
        if not embedding_model or not isinstance(embedding_model, str):
            raise IngestionError("Embedding model must be a non-empty string")
        
        directory_path = Path(directory_path).resolve()
        if not directory_path.exists():
            raise IngestionError(f"Directory does not exist: {directory_path}")
        
        if not directory_path.is_dir():
            raise IngestionError(f"Path is not a directory: {directory_path}")
        
        archive_id = generate_archive_id(str(directory_path))
        archive_output_dir, output_cortex_path, metrics_path = _setup_archive_directories(archive_id)
        
        logger.info(f"Starting local directory ingestion for {archive_id}")
        logger.info(f"Source directory: {directory_path}")
        logger.info(f"Artifacts will be saved to: {archive_output_dir}")
        
        with _cleanup_on_failure(output_cortex_path):
            project_cortex = None
            
            # --- Step 1: Crawl & Jsonify Local Directory ---
            logger.info("[1/4] Scanning local directory and generating Archive Cortex file...")
            
            # Use LocalDirectoryCrawler instead of IntelligentCrawler
            crawler = LocalDirectoryCrawler(directory_path)
            try:
                files_to_process = crawler.get_file_paths()
                if not files_to_process:
                    raise IngestionError("No files found to process in the directory")
                
                jsonifier = Jsonifier(
                    file_paths=files_to_process,
                    repo_root=crawler.directory_path,
                    repo_id=archive_id
                )
                
                # Generate cortex for the local directory
                project_cortex = jsonifier.generate_cortex()
                
                if not project_cortex:
                    raise IngestionError("Failed to generate archive cortex")
                
                # Update metadata to indicate this is an archive, not a repository
                project_cortex["archive_metadata"] = {
                    "source_type": "local_directory",
                    "source_path": str(directory_path),
                    "archive_id": archive_id,
                    "ingested_at": start_time.isoformat(),
                    "directory_stats": crawler.get_directory_stats()
                }
                
                # Remove git-specific metadata
                if "repository_metadata" in project_cortex:
                    del project_cortex["repository_metadata"]
                
                logger.info(f"Archive Cortex created successfully: {output_cortex_path}")
                
                # --- Step 2: Calculate Directory Metrics ---
                logger.info("[2/4] Calculating directory health metrics...")
                graph_data = project_cortex.get("architectural_graph", {})
                
                try:
                    # Use a simplified metrics calculation for local directories
                    latest_metrics = {
                        'total_files': len(files_to_process),
                        'directory_size_mb': crawler.get_directory_stats().get('directory_size_mb', 0),
                        'primary_language': crawler.get_directory_stats().get('primary_language', 'Unknown'),
                        'supported_languages': crawler.get_directory_stats().get('supported_languages', 0),
                        'timestamp': start_time.isoformat(),
                        'archive_id': archive_id,
                        'source_type': 'local_directory'
                    }
                    
                    # Load existing metrics and append the new snapshot
                    historical_metrics = _load_existing_metrics(metrics_path)
                    historical_metrics.append(latest_metrics)
                    
                    _save_metrics(metrics_path, historical_metrics)
                    logger.info("Directory metrics saved successfully")
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate or save metrics: {e}")
                    # Continue with ingestion even if metrics fail
                
            finally:
                crawler.cleanup()
            
            # Save cortex file
            _save_cortex_file(output_cortex_path, project_cortex)
            
            # --- Step 3: Vector Indexing (Optional) ---
            if not skip_indexing:
                logger.info(f"[3/4] Starting vector indexing with model '{embedding_model}'...")
                try:
                    indexer = EmbeddingIndexer(model_name=embedding_model)
                    indexer.process_cortex(str(output_cortex_path))
                    logger.info("Vector indexing complete")
                except Exception as e:
                    logger.error(f"Vector indexing failed: {e}")
                    # Don't fail the entire process if indexing fails
                    raise IngestionError(f"Vector indexing failed: {e}")
            else:
                logger.info("[3/4] Skipping vector indexing as requested")
            
            logger.info("[4/4] Archive ingestion complete")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": "success",
                "message": f"Directory '{directory_path}' was successfully ingested as archive '{archive_id}'.",
                "archive_id": archive_id,
                "source_path": str(directory_path),
                "processing_time_seconds": processing_time,
                "files_processed": len(files_to_process) if 'files_to_process' in locals() else 0,
                "cortex_path": str(output_cortex_path),
                "metrics_path": str(metrics_path),
                "indexing_skipped": skip_indexing,
                "archive_type": "local_directory"
            }
    
    except IngestionError as e:
        logger.error(f"Archive ingestion failed for {directory_path}: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "archive_id": archive_id if 'archive_id' in locals() else None,
            "processing_time_seconds": (datetime.now() - start_time).total_seconds()
        }
    except Exception as e:
        logger.error(f"Unexpected error during archive ingestion for {directory_path}: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "failed",
            "error": str(e),
            "details": traceback.format_exc(),
            "archive_id": archive_id if 'archive_id' in locals() else None,
            "processing_time_seconds": (datetime.now() - start_time).total_seconds()
        }

# Backward compatibility alias
def clone_and_embed_repository_legacy(repo_url: str, embedding_model: str = 'snowflake-arctic-embed2:latest') -> Dict[str, Any]:
    """
    Legacy function signature for backward compatibility.
    """
    return clone_and_embed_repository(repo_url, embedding_model, skip_indexing=False)
