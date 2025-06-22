# In backend/lumiere_core/services/crucible.py

import os
import uuid
from pathlib import Path
from typing import Dict, Tuple, Optional

import docker
from docker.errors import BuildError, ContainerError, APIError, DockerException

from .utils import clean_llm_code_output
from ingestion.crawler import IntelligentCrawler

# --- Environment Detection Helpers ---
def _detect_dependencies(repo_path: Path) -> Tuple[str, str]:
    print("   -> Detecting dependency file...")
    if (repo_path / "requirements.txt").exists():
        print("      - Found requirements.txt")
        return "pip install -r requirements.txt", "requirements.txt"
    if (repo_path / "pyproject.toml").exists():
        print("      - Found pyproject.toml (assuming standard PEP 517 build)")
        return "pip install .", "pyproject.toml"
    print("      - No dependency file found. Assuming no special dependencies.")
    return "echo 'No dependencies to install'", ""

def _detect_test_runner(repo_path: Path) -> str:
    print("   -> Detecting test runner...")
    if (repo_path / "pytest.ini").exists() or (repo_path / "tox.ini").exists():
        print("      - Found pytest.ini or tox.ini. Using 'pytest'.")
        return "pytest"
    if (repo_path / "tests").is_dir() or (repo_path / "test").is_dir():
        print("      - Found 'tests' directory. Assuming 'pytest'.")
        return "pytest"
    print("      - No specific test runner found. Falling back to 'python -m unittest discover'.")
    return "python -m unittest discover"

def _generate_dockerfile(install_command: str, dependency_file_path: str, test_command: str) -> str:
    print("   -> Generating dynamic Dockerfile...")
    dockerfile_content = f"""
FROM python:3.10-slim
WORKDIR /app
COPY . /app
"""
    if dependency_file_path:
        dockerfile_content += f"RUN {install_command}\n"
    dockerfile_content += f"CMD {test_command}\n"
    print("      - Dockerfile generated successfully.")
    return dockerfile_content

# --- Main Service Orchestrator ---
def validate_fix(repo_url: str, target_file: str, modified_code: str) -> Dict[str, str]:
    """
    The main logic for The Crucible agent.
    """
    print("\n--- CRUCIBLE AGENT ACTIVATED ---")
    print(f"Validating fix for '{target_file}' in '{repo_url}'")

    # ** THE FIX **: Add a top-level try/except to catch the Docker connection error.
    try:
        client = docker.from_env()
        # Ping the docker daemon to ensure it's running before we do anything else
        client.ping()
        print("✓ Successfully connected to Docker daemon.")
    except DockerException:
        error_msg = "Crucible Error: Could not connect to the Docker daemon. Please ensure Docker Desktop is running."
        print(f"✗ {error_msg}")
        return {"status": "error", "logs": error_msg}

    with IntelligentCrawler(repo_url=repo_url) as crawler:
        repo_path = crawler.repo_path
        print("[Step 1/5] Patching file in local clone...")
        full_target_path = repo_path / target_file
        if not full_target_path.exists():
            return {"status": "error", "logs": f"Crucible Error: Target file '{target_file}' not found."}
        full_target_path.write_text(modified_code, encoding='utf-8')
        print("✓ File patched.")

        print("[Step 2/5] Analyzing project environment...")
        install_cmd, dep_file = _detect_dependencies(repo_path)
        test_cmd = _detect_test_runner(repo_path)
        dockerfile_str = _generate_dockerfile(install_cmd, dep_file, test_cmd)
        (repo_path / "Dockerfile.lumiere").write_text(dockerfile_str, encoding='utf-8')
        print("✓ Environment analysis complete.")

        print("[Step 3/5] Building validation image...")
        image_tag = f"lumiere-crucible/{uuid.uuid4()}"
        image = None
        try:
            image, build_logs = client.images.build(path=str(repo_path), dockerfile="Dockerfile.lumiere", tag=image_tag, rm=True, forcerm=True)
            print(f"✓ Image '{image_tag}' built.")
        except BuildError as e:
            print(f"✗ Build Failed: {e}")
            logs = "\n".join([log.get('stream', '').strip() for log in e.build_log])
            return {"status": "failed", "logs": f"Docker image build failed:\n{logs}"}

        try:
            print(f"[Step 4/5] Running tests in container from image '{image_tag}'...")
            container_output = client.containers.run(image_tag, remove=True)
            print("✓ Tests PASSED.")
            return {"status": "passed", "logs": container_output.decode('utf-8')}
        except ContainerError as e:
            print(f"✗ Tests FAILED. Exit code: {e.exit_status}")
            logs = e.stderr.decode('utf-8') if e.stderr else e.stdout.decode('utf-8')
            return {"status": "failed", "logs": logs}
        finally:
            print("[Step 5/5] Cleaning up validation image...")
            if image:
                try: client.images.remove(image.id, force=True); print(f"✓ Image '{image_tag}' removed.")
                except APIError as e: print(f"Warning: Could not remove image '{image_tag}'. Error: {e}")

    print("--- CRUCIBLE AGENT MISSION COMPLETE ---")
    return {"status": "error", "logs": "Crucible process completed without a definitive pass/fail result."}
