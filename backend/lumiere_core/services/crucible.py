# In backend/lumiere_core/services/crucible.py

import os
import uuid
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

import docker
from docker.errors import BuildError, ContainerError, APIError, DockerException

from .utils import clean_llm_code_output
from ingestion.crawler import IntelligentCrawler

# --- Enhanced Data Structures ---
@dataclass
class ValidationResult:
    """Enhanced result structure with detailed metrics"""
    status: str
    logs: str
    execution_time: float = 0.0
    build_time: float = 0.0
    test_time: float = 0.0
    project_type: str = "unknown"
    image_size: Optional[int] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for backward compatibility"""
        result = {"status": self.status, "logs": self.logs}
        # Add enhanced fields only if they have meaningful values
        if self.execution_time > 0:
            result["execution_time"] = self.execution_time
        if self.build_time > 0:
            result["build_time"] = self.build_time
        if self.test_time > 0:
            result["test_time"] = self.test_time
        if self.project_type != "unknown":
            result["project_type"] = self.project_type
        if self.image_size:
            result["image_size_mb"] = round(self.image_size / (1024 * 1024), 2)
        if self.warnings:
            result["warnings"] = self.warnings
        return result

# --- Environment Detection Helpers (Enhanced) ---
def _detect_project_type(repo_path: Path) -> str:
    """Detects the project type based on key files with enhanced detection."""
    # Rust detection
    if (repo_path / "Cargo.toml").exists():
        return "rust"

    # Python detection (enhanced)
    python_indicators = [
        "requirements.txt", "pyproject.toml", "setup.py", "Pipfile",
        "poetry.lock", "conda.yml", "environment.yml"
    ]
    if any((repo_path / indicator).exists() for indicator in python_indicators):
        return "python"

    # Node.js detection (new)
    if (repo_path / "package.json").exists():
        return "node"

    # Java detection (new)
    java_indicators = ["pom.xml", "build.gradle", "build.gradle.kts"]
    if any((repo_path / indicator).exists() for indicator in java_indicators):
        return "java"

    # Go detection (new)
    if (repo_path / "go.mod").exists():
        return "go"

    # Generic detection based on file extensions
    code_files = list(repo_path.rglob("*.py"))
    if code_files:
        return "python"

    code_files = list(repo_path.rglob("*.js")) + list(repo_path.rglob("*.ts"))
    if code_files:
        return "node"

    code_files = list(repo_path.rglob("*.java"))
    if code_files:
        return "java"

    code_files = list(repo_path.rglob("*.go"))
    if code_files:
        return "go"

    return "unknown"

def _get_project_commands(project_type: str, repo_path: Path) -> Tuple[str, str, str, Optional[str]]:
    """Returns install command, test command, base image, and dependency file path."""
    print(f"   -> Detected project type: {project_type}")

    if project_type == "rust":
        return "cargo build --release", "cargo test --release", "rust:latest", "Cargo.toml"

    # --- Enhanced Python Logic ---
    if project_type == "python":
        install_cmd, dep_file = "echo 'No dependencies to install'", None

        # Priority order for Python dependency management
        if (repo_path / "poetry.lock").exists():
            install_cmd, dep_file = "pip install poetry && poetry install", "pyproject.toml"
        elif (repo_path / "Pipfile").exists():
            install_cmd, dep_file = "pip install pipenv && pipenv install", "Pipfile"
        elif (repo_path / "requirements.txt").exists():
            install_cmd, dep_file = "pip install -r requirements.txt", "requirements.txt"
        elif (repo_path / "pyproject.toml").exists():
            install_cmd, dep_file = "pip install .", "pyproject.toml"
        elif (repo_path / "setup.py").exists():
            install_cmd, dep_file = "pip install -e .", "setup.py"

        # Enhanced test command detection
        test_cmd = "python -m unittest discover"
        if (repo_path / "pytest.ini").exists() or (repo_path / "pyproject.toml").exists():
            test_cmd = "pytest"
        elif (repo_path / "tox.ini").exists():
            test_cmd = "tox"
        elif (repo_path / "tests").is_dir() or (repo_path / "test").is_dir():
            test_cmd = "pytest" if any(repo_path.rglob("test_*.py")) else "python -m unittest discover"

        return install_cmd, test_cmd, "python:3.11-slim", dep_file

    # --- New Node.js Support ---
    if project_type == "node":
        install_cmd = "npm install"
        if (repo_path / "yarn.lock").exists():
            install_cmd = "yarn install"
        elif (repo_path / "pnpm-lock.yaml").exists():
            install_cmd = "pnpm install"

        test_cmd = "npm test"
        if (repo_path / "yarn.lock").exists():
            test_cmd = "yarn test"
        elif (repo_path / "pnpm-lock.yaml").exists():
            test_cmd = "pnpm test"

        return install_cmd, test_cmd, "node:18-alpine", "package.json"

    # --- New Java Support ---
    if project_type == "java":
        if (repo_path / "pom.xml").exists():
            return "mvn compile", "mvn test", "openjdk:17-jdk-slim", "pom.xml"
        elif (repo_path / "build.gradle").exists() or (repo_path / "build.gradle.kts").exists():
            return "./gradlew build", "./gradlew test", "openjdk:17-jdk-slim", "build.gradle"

    # --- New Go Support ---
    if project_type == "go":
        return "go mod download", "go test ./...", "golang:1.21-alpine", "go.mod"

    # Fallback for unknown project types
    return "echo 'Unknown project type, cannot install dependencies'", "echo 'No test runner found'", "alpine:latest", None

def _generate_dockerfile(install_command: str, test_command: str, base_image: str,
                        project_type: str = "unknown") -> str:
    """
    Generate optimized Dockerfile with caching and security improvements.
    """
    print("   -> Generating dynamic Dockerfile...")

    # Base dockerfile with common optimizations
    dockerfile_content = f"FROM {base_image}\n"
    dockerfile_content += "WORKDIR /app\n"
    # Security: Create a non-root user. The specific command depends on the base image (alpine vs debian)
    # Using a generic adduser should work for most.
    dockerfile_content += "RUN adduser -D -h /app appuser && chown -R appuser /app\n\n"

    # --- THIS IS THE CRITICAL CHANGE: COPY CODE FIRST ---
    # Copy the entire project context into the build environment.
    dockerfile_content += "# Copy source code\n"
    dockerfile_content += "COPY . /app/\n\n"

    # Now, run the install command with the code present.
    # We add 'chown' for cases where dependencies create root-owned files (like node_modules)
    dockerfile_content += f"# Install dependencies\n"
    dockerfile_content += f"RUN {install_command} && chown -R appuser:appuser /app\n\n"

    # Switch to the non-root user for running tests
    dockerfile_content += "# Switch to non-root user for security\n"
    dockerfile_content += "USER appuser\n\n"

    # The final command to run is the test command
    # Use exec form to handle signals correctly
    cmd_parts = test_command.split()
    cmd_json = ', '.join([f'"{part}"' for part in cmd_parts])
    dockerfile_content += f"# Run tests\n"
    dockerfile_content += f"CMD [{cmd_json}]\n"

    print("      - Dockerfile generated successfully with corrected build order.")
    return dockerfile_content

def _analyze_dockerfile_warnings(repo_path: Path, project_type: str) -> List[str]:
    """Analyze potential issues and generate warnings."""
    warnings = []

    # Check for common security issues
    if project_type == "python":
        req_file = repo_path / "requirements.txt"
        if req_file.exists():
            content = req_file.read_text()
            if "==" not in content and ">=" not in content:
                warnings.append("Consider pinning Python package versions for reproducible builds")

    # Check for large files that might slow down builds
    large_files = []
    for file_path in repo_path.rglob("*"):
        if file_path.is_file() and file_path.stat().st_size > 50 * 1024 * 1024:  # 50MB
            large_files.append(file_path.name)

    if large_files:
        warnings.append(f"Large files detected that may slow builds: {', '.join(large_files[:3])}")

    # Check for missing .dockerignore
    if not (repo_path / ".dockerignore").exists():
        warnings.append("Consider adding .dockerignore to exclude unnecessary files from Docker context")

    return warnings

# --- Enhanced Main Service Orchestrator ---
def validate_fix(repo_url: str, target_file: str, modified_code: str,
                 enhanced_output: bool = False) -> Dict[str, str]:
    """
    The main logic for The Crucible agent with enhanced features.

    Args:
        repo_url: Repository URL to validate
        target_file: Target file path to modify
        modified_code: Modified code content
        enhanced_output: Whether to return enhanced output with metrics (backward compatible)

    Returns:
        Dictionary with validation results (backward compatible format by default)
    """
    print("\n--- CRUCIBLE AGENT ACTIVATED ---")
    print(f"Validating fix for '{target_file}' in '{repo_url}'")

    start_time = time.time()
    result = ValidationResult(status="error", logs="Initialization failed")

    try:
        client = docker.from_env()
        client.ping()
        print("✓ Successfully connected to Docker daemon.")
    except DockerException:
        error_msg = "Crucible Error: Could not connect to the Docker daemon. Please ensure Docker Desktop is running."
        print(f"✗ {error_msg}")
        result.logs = error_msg
        return result.to_dict() if enhanced_output else {"status": result.status, "logs": result.logs}

    with IntelligentCrawler(repo_url=repo_url) as crawler:
        repo_path = crawler.repo_path
        print("[Step 1/6] Patching file in local clone...")
        full_target_path = repo_path / target_file
        if not full_target_path.exists():
            full_target_path.parent.mkdir(parents=True, exist_ok=True)
        full_target_path.write_text(modified_code, encoding='utf-8')
        print("✓ File patched.")

        print("[Step 2/6] Analyzing project environment...")
        project_type = _detect_project_type(repo_path)
        result.project_type = project_type
        install_cmd, test_cmd, base_image, _ = _get_project_commands(project_type, repo_path)

        # Generate warnings
        result.warnings = _analyze_dockerfile_warnings(repo_path, project_type)

        dockerfile_str = _generate_dockerfile(install_cmd, test_cmd, base_image, project_type)
        (repo_path / "Dockerfile.lumiere").write_text(dockerfile_str, encoding='utf-8')
        print("✓ Environment analysis complete.")

        print("[Step 3/6] Building validation image...")
        build_start = time.time()
        image_tag = f"lumiere-crucible/{uuid.uuid4()}"
        image = None
        try:
            image, build_logs = client.images.build(
                path=str(repo_path),
                dockerfile="Dockerfile.lumiere",
                tag=image_tag,
                rm=True,
                forcerm=True,
                pull=True  # Ensure base image is up to date
            )
            result.build_time = time.time() - build_start
            result.image_size = image.attrs.get('Size', 0)
            print(f"✓ Image '{image_tag}' built in {result.build_time:.2f}s")
        except BuildError as e:
            print(f"✗ Build Failed: {e}")
            logs = "\n".join([log.get('stream', '').strip() for log in e.build_log if 'stream' in log])
            result.status = "failed"
            result.logs = f"Docker image build failed:\n{logs}"
            result.execution_time = time.time() - start_time
            return result.to_dict() if enhanced_output else {"status": result.status, "logs": result.logs}

        try:
            print(f"[Step 4/6] Running tests in container from image '{image_tag}'...")
            test_start = time.time()

            # Enhanced container run with resource limits
            container_output = client.containers.run(
                image_tag,
                remove=True,
                mem_limit="1g",  # Limit memory usage
                cpu_count=2,     # Limit CPU usage
                network_disabled=True  # Disable network for security
            )

            result.test_time = time.time() - test_start
            print(f"✓ Tests PASSED in {result.test_time:.2f}s")
            result.status = "passed"
            result.logs = container_output.decode('utf-8')

        except ContainerError as e:
            result.test_time = time.time() - test_start if 'test_start' in locals() else 0
            print(f"✗ Tests FAILED. Exit code: {e.exit_status}")
            logs = e.stderr.decode('utf-8') if e.stderr else e.stdout.decode('utf-8')
            result.status = "failed"
            result.logs = logs

        finally:
            print("[Step 5/6] Cleaning up validation image...")
            if image:
                try:
                    client.images.remove(image.id, force=True)
                    print(f"✓ Image '{image_tag}' removed.")
                except APIError as e:
                    print(f"Warning: Could not remove image '{image_tag}'. Error: {e}")
                    result.warnings.append(f"Failed to cleanup image: {image_tag}")

    result.execution_time = time.time() - start_time
    print(f"[Step 6/6] Validation completed in {result.execution_time:.2f}s")
    print("--- CRUCIBLE AGENT MISSION COMPLETE ---")

    # Return backward compatible format by default
    if enhanced_output:
        return result.to_dict()
    else:
        return {"status": result.status, "logs": result.logs}

# --- Backward Compatibility Aliases ---
def crucible_validate(repo_url: str, target_file: str, modified_code: str) -> Dict[str, str]:
    """Backward compatibility alias for validate_fix"""
    return validate_fix(repo_url, target_file, modified_code)

# --- Enhanced API Functions ---
def validate_fix_enhanced(repo_url: str, target_file: str, modified_code: str) -> Dict:
    """Enhanced version that returns detailed metrics and warnings"""
    return validate_fix(repo_url, target_file, modified_code, enhanced_output=True)

def get_supported_project_types() -> List[str]:
    """Returns list of supported project types"""
    return ["python", "rust", "node", "java", "go"]

def analyze_project(repo_url: str) -> Dict[str, str]:
    """Analyze project type and structure without running tests"""
    with IntelligentCrawler(repo_url=repo_url) as crawler:
        repo_path = crawler.repo_path
        project_type = _detect_project_type(repo_path)
        install_cmd, test_cmd, base_image, dep_file = _get_project_commands(project_type, repo_path)

        return {
            "project_type": project_type,
            "install_command": install_cmd,
            "test_command": test_cmd,
            "base_image": base_image,
            "dependency_file": dep_file,
            "supported": project_type in get_supported_project_types()
        }
