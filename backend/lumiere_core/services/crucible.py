# In backend/lumiere_core/services/crucible.py

import os
import uuid
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
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
    detected_languages: List[str] = None
    build_tools: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.detected_languages is None:
            self.detected_languages = []
        if self.build_tools is None:
            self.build_tools = []

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
        if self.detected_languages:
            result["detected_languages"] = self.detected_languages
        if self.build_tools:
            result["build_tools"] = self.build_tools
        return result

# --- Enhanced Environment Detection (Polyglot Support) ---
def _detect_project_ecosystem(repo_path: Path) -> Tuple[str, List[str], List[str]]:
    """
    Enhanced project detection supporting multiple languages and ecosystems.
    Returns: (primary_project_type, detected_languages, build_tools)
    """
    detected_languages = []
    build_tools = []
    project_types = []

    # File-based detection patterns
    detection_patterns = {
        # Python ecosystem
        "python": {
            "files": ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile",
                     "poetry.lock", "conda.yml", "environment.yml", "setup.cfg"],
            "extensions": [".py"],
            "build_tools": ["pip", "poetry", "pipenv", "conda"]
        },
        # Node.js ecosystem
        "node": {
            "files": ["package.json"],
            "extensions": [".js", ".ts", ".jsx", ".tsx"],
            "build_tools": ["npm", "yarn", "pnpm"]
        },
        # Java ecosystem
        "java": {
            "files": ["pom.xml", "build.gradle", "build.gradle.kts", "gradle.properties"],
            "extensions": [".java"],
            "build_tools": ["maven", "gradle"]
        },
        # .NET ecosystem
        "dotnet": {
            "files": ["*.csproj", "*.vbproj", "*.fsproj", "*.sln", "global.json"],
            "extensions": [".cs", ".vb", ".fs"],
            "build_tools": ["dotnet"]
        },
        # Go ecosystem
        "go": {
            "files": ["go.mod", "go.sum"],
            "extensions": [".go"],
            "build_tools": ["go"]
        },
        # Rust ecosystem
        "rust": {
            "files": ["Cargo.toml", "Cargo.lock"],
            "extensions": [".rs"],
            "build_tools": ["cargo"]
        },
        # C/C++ ecosystem
        "cpp": {
            "files": ["CMakeLists.txt", "Makefile", "makefile", "meson.build", "conanfile.txt"],
            "extensions": [".c", ".cpp", ".cxx", ".cc", ".h", ".hpp"],
            "build_tools": ["cmake", "make", "meson", "conan"]
        },
        # Swift ecosystem
        "swift": {
            "files": ["Package.swift", "*.xcodeproj", "*.xcworkspace"],
            "extensions": [".swift"],
            "build_tools": ["swift", "xcodebuild"]
        },
        # Ruby ecosystem
        "ruby": {
            "files": ["Gemfile", "Gemfile.lock", "Rakefile", "*.gemspec"],
            "extensions": [".rb"],
            "build_tools": ["bundler", "rake", "gem"]
        },
        # PHP ecosystem
        "php": {
            "files": ["composer.json", "composer.lock"],
            "extensions": [".php"],
            "build_tools": ["composer"]
        },
        # Dart/Flutter ecosystem
        "dart": {
            "files": ["pubspec.yaml", "pubspec.lock"],
            "extensions": [".dart"],
            "build_tools": ["pub", "flutter"]
        },
        # Kotlin ecosystem
        "kotlin": {
            "files": ["build.gradle.kts"],
            "extensions": [".kt", ".kts"],
            "build_tools": ["gradle"]
        },
        # Scala ecosystem
        "scala": {
            "files": ["build.sbt", "project/build.properties"],
            "extensions": [".scala"],
            "build_tools": ["sbt"]
        },
        # Clojure ecosystem
        "clojure": {
            "files": ["project.clj", "deps.edn", "build.boot"],
            "extensions": [".clj", ".cljs", ".cljc"],
            "build_tools": ["leiningen", "clojure"]
        },
        # Haskell ecosystem
        "haskell": {
            "files": ["*.cabal", "stack.yaml", "cabal.project"],
            "extensions": [".hs", ".lhs"],
            "build_tools": ["cabal", "stack"]
        },
        # Elixir ecosystem
        "elixir": {
            "files": ["mix.exs", "mix.lock"],
            "extensions": [".ex", ".exs"],
            "build_tools": ["mix"]
        },
        # R ecosystem
        "r": {
            "files": ["DESCRIPTION", "renv.lock", ".Rprofile"],
            "extensions": [".r", ".R"],
            "build_tools": ["R"]
        }
    }

    # Check for file indicators
    for ecosystem, config in detection_patterns.items():
        found_files = []
        for pattern in config["files"]:
            if "*" in pattern:
                found_files.extend(repo_path.glob(pattern))
            else:
                if (repo_path / pattern).exists():
                    found_files.append(pattern)

        if found_files:
            detected_languages.append(ecosystem)
            build_tools.extend(config["build_tools"])
            project_types.append(ecosystem)

    # Check for source code files if no config files found
    if not detected_languages:
        for ecosystem, config in detection_patterns.items():
            for ext in config["extensions"]:
                if list(repo_path.rglob(f"*{ext}")):
                    detected_languages.append(ecosystem)
                    build_tools.extend(config["build_tools"])
                    project_types.append(ecosystem)
                    break

    # Determine primary project type
    if project_types:
        # Priority order for mixed projects
        priority_order = ["python", "node", "java", "dotnet", "go", "rust", "cpp", "swift"]
        for priority_type in priority_order:
            if priority_type in project_types:
                primary_type = priority_type
                break
        else:
            primary_type = project_types[0]
    else:
        primary_type = "unknown"

    print(f"   -> Detected ecosystem: {primary_type}")
    print(f"   -> Languages found: {detected_languages}")
    print(f"   -> Build tools: {build_tools}")

    return primary_type, detected_languages, build_tools

def _get_enhanced_project_commands(project_type: str, repo_path: Path, detected_languages: List[str]) -> Tuple[str, str, str, Optional[str]]:
    """
    Enhanced command generation supporting polyglot projects.
    Returns: (install_command, test_command, base_image, dependency_file)
    """
    print(f"   -> Configuring build for project type: {project_type}")

    # Multi-stage build support for polyglot projects
    base_images = {
        "python": "python:3.11-slim",
        "node": "node:18-alpine",
        "java": "openjdk:17-jdk-slim",
        "dotnet": "mcr.microsoft.com/dotnet/sdk:7.0",
        "go": "golang:1.21-alpine",
        "rust": "rust:1.75-slim",
        "cpp": "gcc:12",
        "swift": "swift:5.9",
        "ruby": "ruby:3.2-alpine",
        "php": "php:8.2-cli",
        "dart": "dart:stable",
        "kotlin": "openjdk:17-jdk-slim",
        "scala": "openjdk:17-jdk-slim",
        "clojure": "clojure:lein-alpine",
        "haskell": "haskell:9.4",
        "elixir": "elixir:1.15-alpine",
        "r": "r-base:4.3.0"
    }

    # Primary language configuration
    if project_type == "python":
        install_cmd, dep_file = _configure_python_build(repo_path)
        test_cmd = _detect_python_test_runner(repo_path)
        base_image = base_images["python"]

    elif project_type == "node":
        install_cmd, dep_file = _configure_node_build(repo_path)
        test_cmd = _detect_node_test_runner(repo_path)
        base_image = base_images["node"]

    elif project_type == "java":
        install_cmd, dep_file = _configure_java_build(repo_path)
        test_cmd = _detect_java_test_runner(repo_path)
        base_image = base_images["java"]

    elif project_type == "dotnet":
        install_cmd, dep_file = _configure_dotnet_build(repo_path)
        test_cmd = _detect_dotnet_test_runner(repo_path)
        base_image = base_images["dotnet"]

    elif project_type == "go":
        install_cmd, dep_file = _configure_go_build(repo_path)
        test_cmd = _detect_go_test_runner(repo_path)
        base_image = base_images["go"]

    elif project_type == "rust":
        install_cmd, dep_file = _configure_rust_build(repo_path)
        test_cmd = _detect_rust_test_runner(repo_path)
        base_image = base_images["rust"]

    elif project_type == "cpp":
        install_cmd, dep_file = _configure_cpp_build(repo_path)
        test_cmd = _detect_cpp_test_runner(repo_path)
        base_image = base_images["cpp"]

    elif project_type == "swift":
        install_cmd, dep_file = _configure_swift_build(repo_path)
        test_cmd = _detect_swift_test_runner(repo_path)
        base_image = base_images["swift"]

    elif project_type == "ruby":
        install_cmd, dep_file = _configure_ruby_build(repo_path)
        test_cmd = _detect_ruby_test_runner(repo_path)
        base_image = base_images["ruby"]

    elif project_type == "php":
        install_cmd, dep_file = _configure_php_build(repo_path)
        test_cmd = _detect_php_test_runner(repo_path)
        base_image = base_images["php"]

    elif project_type == "dart":
        install_cmd, dep_file = _configure_dart_build(repo_path)
        test_cmd = _detect_dart_test_runner(repo_path)
        base_image = base_images["dart"]

    else:
        # Fallback for unknown or unsupported languages
        install_cmd = "echo 'Unknown project type, attempting generic build'"
        test_cmd = "echo 'No test runner detected for this project type'"
        base_image = "alpine:latest"
        dep_file = None

    return install_cmd, test_cmd, base_image, dep_file

# Language-specific build configuration functions
def _configure_python_build(repo_path: Path) -> Tuple[str, Optional[str]]:
    """Configure Python build commands and detect dependency file."""
    if (repo_path / "poetry.lock").exists():
        return "pip install poetry && poetry install", "pyproject.toml"
    elif (repo_path / "Pipfile").exists():
        return "pip install pipenv && pipenv install --system", "Pipfile"
    elif (repo_path / "requirements.txt").exists():
        return "pip install -r requirements.txt", "requirements.txt"
    elif (repo_path / "pyproject.toml").exists():
        return "pip install .", "pyproject.toml"
    elif (repo_path / "setup.py").exists():
        return "pip install -e .", "setup.py"
    elif (repo_path / "conda.yml").exists() or (repo_path / "environment.yml").exists():
        env_file = "conda.yml" if (repo_path / "conda.yml").exists() else "environment.yml"
        return f"conda env create -f {env_file} && conda activate $(head -1 {env_file} | cut -d' ' -f2)", env_file
    else:
        return "echo 'No Python dependencies to install'", None

def _detect_python_test_runner(repo_path: Path) -> str:
    """Detect appropriate Python test runner."""
    if (repo_path / "pytest.ini").exists() or (repo_path / "pyproject.toml").exists():
        content = ""
        if (repo_path / "pyproject.toml").exists():
            content = (repo_path / "pyproject.toml").read_text()
        if "pytest" in content or (repo_path / "pytest.ini").exists():
            return "python -m pytest"

    if (repo_path / "tox.ini").exists():
        return "tox"

    if any(repo_path.rglob("test_*.py")) or any(repo_path.rglob("*_test.py")):
        return "python -m pytest"

    if (repo_path / "tests").is_dir() or (repo_path / "test").is_dir():
        return "python -m unittest discover"

    return "python -m unittest discover"

def _configure_node_build(repo_path: Path) -> Tuple[str, Optional[str]]:
    """Configure Node.js build commands."""
    if (repo_path / "pnpm-lock.yaml").exists():
        return "pnpm install", "package.json"
    elif (repo_path / "yarn.lock").exists():
        return "yarn install --frozen-lockfile", "package.json"
    elif (repo_path / "package.json").exists():
        return "npm ci", "package.json"
    else:
        return "echo 'No Node.js dependencies found'", None

def _detect_node_test_runner(repo_path: Path) -> str:
    """Detect appropriate Node.js test runner."""
    if (repo_path / "package.json").exists():
        try:
            package_json = json.loads((repo_path / "package.json").read_text())
            scripts = package_json.get("scripts", {})

            if "test" in scripts:
                if (repo_path / "pnpm-lock.yaml").exists():
                    return "pnpm test"
                elif (repo_path / "yarn.lock").exists():
                    return "yarn test"
                else:
                    return "npm test"
        except:
            pass

    # Check for specific test framework files
    if (repo_path / "jest.config.js").exists() or (repo_path / "jest.config.ts").exists():
        return "npx jest"
    elif (repo_path / "vitest.config.js").exists() or (repo_path / "vitest.config.ts").exists():
        return "npx vitest run"
    elif (repo_path / "cypress.config.js").exists():
        return "npx cypress run"
    elif (repo_path / "playwright.config.js").exists():
        return "npx playwright test"

    return "npm test"

def _configure_java_build(repo_path: Path) -> Tuple[str, Optional[str]]:
    """Configure Java build commands."""
    if (repo_path / "pom.xml").exists():
        return "mvn compile", "pom.xml"
    elif (repo_path / "build.gradle").exists() or (repo_path / "build.gradle.kts").exists():
        gradle_file = "build.gradle.kts" if (repo_path / "build.gradle.kts").exists() else "build.gradle"
        return "./gradlew build", gradle_file
    else:
        return "echo 'No Java build file found'", None

def _detect_java_test_runner(repo_path: Path) -> str:
    """Detect appropriate Java test runner."""
    if (repo_path / "pom.xml").exists():
        return "mvn test"
    elif (repo_path / "build.gradle").exists() or (repo_path / "build.gradle.kts").exists():
        return "./gradlew test"
    else:
        return "echo 'No Java test runner found'"

def _configure_dotnet_build(repo_path: Path) -> Tuple[str, Optional[str]]:
    """Configure .NET build commands."""
    sln_files = list(repo_path.glob("*.sln"))
    csproj_files = list(repo_path.glob("*.csproj"))

    if sln_files:
        return "dotnet restore && dotnet build", sln_files[0].name
    elif csproj_files:
        return "dotnet restore && dotnet build", csproj_files[0].name
    else:
        return "dotnet build", None

def _detect_dotnet_test_runner(repo_path: Path) -> str:
    """Detect appropriate .NET test runner."""
    return "dotnet test"

def _configure_go_build(repo_path: Path) -> Tuple[str, Optional[str]]:
    """Configure Go build commands."""
    if (repo_path / "go.mod").exists():
        return "go mod download && go build ./...", "go.mod"
    else:
        return "go build ./...", None

def _detect_go_test_runner(repo_path: Path) -> str:
    """Detect appropriate Go test runner."""
    return "go test ./..."

def _configure_rust_build(repo_path: Path) -> Tuple[str, Optional[str]]:
    """Configure Rust build commands."""
    if (repo_path / "Cargo.toml").exists():
        return "cargo build --release", "Cargo.toml"
    else:
        return "echo 'No Cargo.toml found'", None

def _detect_rust_test_runner(repo_path: Path) -> str:
    """Detect appropriate Rust test runner."""
    return "cargo test --release"

def _configure_cpp_build(repo_path: Path) -> Tuple[str, Optional[str]]:
    """Configure C++ build commands."""
    if (repo_path / "CMakeLists.txt").exists():
        return "mkdir -p build && cd build && cmake .. && make", "CMakeLists.txt"
    elif (repo_path / "Makefile").exists() or (repo_path / "makefile").exists():
        makefile = "Makefile" if (repo_path / "Makefile").exists() else "makefile"
        return "make", makefile
    elif (repo_path / "meson.build").exists():
        return "meson setup builddir && meson compile -C builddir", "meson.build"
    else:
        return "echo 'No C++ build system found'", None

def _detect_cpp_test_runner(repo_path: Path) -> str:
    """Detect appropriate C++ test runner."""
    if (repo_path / "CMakeLists.txt").exists():
        return "cd build && ctest"
    elif (repo_path / "Makefile").exists():
        return "make test"
    else:
        return "echo 'No C++ test runner found'"

def _configure_swift_build(repo_path: Path) -> Tuple[str, Optional[str]]:
    """Configure Swift build commands."""
    if (repo_path / "Package.swift").exists():
        return "swift build", "Package.swift"
    else:
        return "echo 'No Package.swift found'", None

def _detect_swift_test_runner(repo_path: Path) -> str:
    """Detect appropriate Swift test runner."""
    return "swift test"

def _configure_ruby_build(repo_path: Path) -> Tuple[str, Optional[str]]:
    """Configure Ruby build commands."""
    if (repo_path / "Gemfile").exists():
        return "bundle install", "Gemfile"
    else:
        return "echo 'No Gemfile found'", None

def _detect_ruby_test_runner(repo_path: Path) -> str:
    """Detect appropriate Ruby test runner."""
    if (repo_path / "Rakefile").exists():
        return "bundle exec rake test"
    elif any(repo_path.rglob("*_spec.rb")):
        return "bundle exec rspec"
    elif any(repo_path.rglob("test_*.rb")):
        return "bundle exec ruby -Itest test/test_*.rb"
    else:
        return "bundle exec rake test"

def _configure_php_build(repo_path: Path) -> Tuple[str, Optional[str]]:
    """Configure PHP build commands."""
    if (repo_path / "composer.json").exists():
        return "composer install", "composer.json"
    else:
        return "echo 'No composer.json found'", None

def _detect_php_test_runner(repo_path: Path) -> str:
    """Detect appropriate PHP test runner."""
    if (repo_path / "phpunit.xml").exists() or (repo_path / "phpunit.xml.dist").exists():
        return "vendor/bin/phpunit"
    elif (repo_path / "composer.json").exists():
        try:
            composer_json = json.loads((repo_path / "composer.json").read_text())
            scripts = composer_json.get("scripts", {})
            if "test" in scripts:
                return "composer test"
        except:
            pass
    return "vendor/bin/phpunit"

def _configure_dart_build(repo_path: Path) -> Tuple[str, Optional[str]]:
    """Configure Dart build commands."""
    if (repo_path / "pubspec.yaml").exists():
        # Check if it's a Flutter project
        pubspec_content = (repo_path / "pubspec.yaml").read_text()
        if "flutter:" in pubspec_content:
            return "flutter pub get && flutter build apk --debug", "pubspec.yaml"
        else:
            return "dart pub get", "pubspec.yaml"
    else:
        return "echo 'No pubspec.yaml found'", None

def _detect_dart_test_runner(repo_path: Path) -> str:
    """Detect appropriate Dart test runner."""
    if (repo_path / "pubspec.yaml").exists():
        pubspec_content = (repo_path / "pubspec.yaml").read_text()
        if "flutter:" in pubspec_content:
            return "flutter test"
        else:
            return "dart test"
    return "dart test"

def _generate_polyglot_dockerfile(install_command: str, test_command: str, base_image: str,
                                project_type: str = "unknown", detected_languages: List[str] = None) -> str:
    """
    Generate optimized Dockerfile with polyglot support and multi-stage builds when needed.
    """
    print("   -> Generating enhanced polyglot Dockerfile...")

    dockerfile_content = f"FROM {base_image}\n"
    dockerfile_content += "WORKDIR /app\n\n"

    # Enhanced security and optimization
    dockerfile_content += "# Security: Create non-root user\n"
    dockerfile_content += "RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser\n\n"

    # Language-specific optimizations
    if project_type == "node":
        dockerfile_content += "# Node.js optimizations\n"
        dockerfile_content += "ENV NODE_ENV=production\n"
        dockerfile_content += "ENV CI=true\n"
        dockerfile_content += "COPY package*.json yarn.lock* pnpm-lock.yaml* ./\n"
        dockerfile_content += f"RUN {install_command}\n\n"
        dockerfile_content += "# Copy source code after dependency installation for better caching\n"
        dockerfile_content += "COPY . .\n\n"

    elif project_type == "python":
        dockerfile_content += "# Python optimizations\n"
        dockerfile_content += "ENV PYTHONUNBUFFERED=1\n"
        dockerfile_content += "ENV PYTHONDONTWRITEBYTECODE=1\n"
        dockerfile_content += "ENV PIP_NO_CACHE_DIR=1\n"
        dockerfile_content += "ENV PIP_DISABLE_PIP_VERSION_CHECK=1\n"

        # Copy dependency files first for better caching
        dependency_files = ["requirements*.txt", "pyproject.toml", "setup.py", "setup.cfg",
                          "Pipfile", "Pipfile.lock", "poetry.lock", "conda.yml", "environment.yml"]
        dockerfile_content += "# Copy dependency files\n"
        for dep_file in dependency_files:
            dockerfile_content += f"COPY {dep_file} ./\n"
        dockerfile_content += f"RUN {install_command}\n\n"
        dockerfile_content += "# Copy source code\n"
        dockerfile_content += "COPY . .\n\n"

    elif project_type == "java":
        dockerfile_content += "# Java optimizations\n"
        dockerfile_content += "ENV JAVA_OPTS=\"-Xmx512m -XX:+UseContainerSupport\"\n"
        dockerfile_content += "COPY pom.xml build.gradle* build.gradle.kts* gradle.properties* gradlew* ./\n"
        dockerfile_content += "COPY gradle/ gradle/ 2>/dev/null || true\n"
        dockerfile_content += f"RUN {install_command}\n\n"
        dockerfile_content += "COPY . .\n\n"

    elif project_type == "go":
        dockerfile_content += "# Go optimizations\n"
        dockerfile_content += "ENV CGO_ENABLED=0\n"
        dockerfile_content += "ENV GOOS=linux\n"
        dockerfile_content += "COPY go.mod go.sum ./\n"
        dockerfile_content += f"RUN {install_command}\n\n"
        dockerfile_content += "COPY . .\n\n"

    elif project_type == "rust":
        dockerfile_content += "# Rust optimizations\n"
        dockerfile_content += "ENV CARGO_NET_GIT_FETCH_WITH_CLI=true\n"
        dockerfile_content += "COPY Cargo.toml Cargo.lock ./\n"
        dockerfile_content += "# Pre-build dependencies for better caching\n"
        dockerfile_content += "RUN mkdir src && echo 'fn main() {}' > src/main.rs\n"
        dockerfile_content += "RUN cargo build --release && rm -rf src\n"
        dockerfile_content += "COPY . .\n"
        dockerfile_content += "RUN touch src/main.rs\n"
        dockerfile_content += "RUN cargo build --release\n\n"

    else:
        # Generic approach for other languages
        dockerfile_content += "# Copy source code\n"
        dockerfile_content += "COPY . .\n\n"
        dockerfile_content += f"# Install dependencies\n"
        dockerfile_content += f"RUN {install_command}\n\n"

    # Set proper ownership and switch to non-root user
    dockerfile_content += "# Set ownership and switch to non-root user\n"
    dockerfile_content += "RUN chown -R appuser:appuser /app\n"
    dockerfile_content += "USER appuser\n\n"

    # Health check for long-running tests
    dockerfile_content += "# Health check\n"
    dockerfile_content += "HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\\n"
    dockerfile_content += "  CMD echo 'Health check: Tests running'\n\n"

    # Final command
    cmd_parts = test_command.split()
    cmd_json = ', '.join([f'"{part}"' for part in cmd_parts])
    dockerfile_content += f"# Run tests\n"
    dockerfile_content += f"CMD [{cmd_json}]\n"

    print(f"      - Enhanced polyglot Dockerfile generated for {project_type}")
    return dockerfile_content

def _analyze_polyglot_warnings(repo_path: Path, project_type: str, detected_languages: List[str]) -> List[str]:
    """Enhanced warning analysis for polyglot projects."""
    warnings = []

    # Multi-language project warnings
    if len(detected_languages) > 1:
        warnings.append(f"Multi-language project detected: {', '.join(detected_languages)}. Build complexity may be higher.")

    # Language-specific warnings
    for language in detected_languages:
        if language == "python":
            req_file = repo_path / "requirements.txt"
            if req_file.exists():
                content = req_file.read_text()
                if "==" not in content and ">=" not in content:
                    warnings.append("Python: Consider pinning package versions for reproducible builds")

        elif language == "node":
            package_json = repo_path / "package.json"
            if package_json.exists() and not (repo_path / "package-lock.json").exists() and not (repo_path / "yarn.lock").exists():
                warnings.append("Node.js: No lock file found. Consider using npm ci or yarn for reproducible builds")

        elif language == "java":
            if not any((repo_path / f).exists() for f in ["pom.xml", "build.gradle", "build.gradle.kts"]):
                warnings.append("Java: No standard build file found. Manual compilation may be required")

    # Security warnings
    security_files = [".env", "config.json", "secrets.yml"]
    for sec_file in security_files:
        if (repo_path / sec_file).exists():
            warnings.append(f"Security: {sec_file} found. Ensure sensitive data is not included in container")

    # Performance warnings
    large_files = []
    for file_path in repo_path.rglob("*"):
        if file_path.is_file() and file_path.stat().st_size > 50 * 1024 * 1024:  # 50MB
            large_files.append(file_path.name)

    if large_files:
        warnings.append(f"Performance: Large files detected: {', '.join(large_files[:3])}. Consider .dockerignore")

    # Dockerfile optimization warnings
    if not (repo_path / ".dockerignore").exists():
        warnings.append("Optimization: Consider adding .dockerignore to exclude unnecessary files")

    return warnings

# --- Enhanced Main Service Orchestrator ---
def validate_fix(repo_url: str, target_file: str, modified_code: str,
                 enhanced_output: bool = False) -> Dict[str, str]:
    """
    Enhanced Crucible agent with comprehensive polyglot support.

    Args:
        repo_url: Repository URL to validate
        target_file: Target file path to modify
        modified_code: Modified code content
        enhanced_output: Whether to return enhanced output with metrics

    Returns:
        Dictionary with validation results
    """
    print("\n--- ENHANCED POLYGLOT CRUCIBLE ACTIVATED ---")
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
        print("[Step 1/7] Patching file in local clone...")
        full_target_path = repo_path / target_file
        if not full_target_path.exists():
            full_target_path.parent.mkdir(parents=True, exist_ok=True)
        full_target_path.write_text(modified_code, encoding='utf-8')
        print("✓ File patched.")

        print("[Step 2/7] Analyzing polyglot project environment...")
        project_type, detected_languages, build_tools = _detect_project_ecosystem(repo_path)
        result.project_type = project_type
        result.detected_languages = detected_languages
        result.build_tools = build_tools

        install_cmd, test_cmd, base_image, dep_file = _get_enhanced_project_commands(
            project_type, repo_path, detected_languages
        )

        # Generate enhanced warnings
        result.warnings = _analyze_polyglot_warnings(repo_path, project_type, detected_languages)

        dockerfile_str = _generate_polyglot_dockerfile(
            install_cmd, test_cmd, base_image, project_type, detected_languages
        )
        (repo_path / "Dockerfile.lumiere").write_text(dockerfile_str, encoding='utf-8')
        print("✓ Enhanced polyglot environment analysis complete.")

        print("[Step 3/7] Building polyglot validation image...")
        build_start = time.time()
        image_tag = f"lumiere-crucible-polyglot/{uuid.uuid4()}"
        image = None
        try:
            image, build_logs = client.images.build(
                path=str(repo_path),
                dockerfile="Dockerfile.lumiere",
                tag=image_tag,
                rm=True,
                forcerm=True,
                pull=True,
                platform="linux/amd64"  # Ensure consistent platform
            )
            result.build_time = time.time() - build_start
            result.image_size = image.attrs.get('Size', 0)
            print(f"✓ Polyglot image '{image_tag}' built in {result.build_time:.2f}s")
        except BuildError as e:
            print(f"✗ Build Failed: {e}")
            logs = "\n".join([log.get('stream', '').strip() for log in e.build_log if 'stream' in log])
            result.status = "failed"
            result.logs = f"Docker image build failed:\n{logs}"
            result.execution_time = time.time() - start_time
            return result.to_dict() if enhanced_output else {"status": result.status, "logs": result.logs}

        try:
            print(f"[Step 4/7] Running polyglot tests in container '{image_tag}'...")
            test_start = time.time()

            # Enhanced container run with better resource limits
            container_output = client.containers.run(
                image_tag,
                remove=True,
                mem_limit="2g",  # Increased memory for polyglot builds
                cpu_count=4,     # More CPU for complex builds
                network_disabled=True,
                security_opt=["no-new-privileges:true"],  # Enhanced security
                read_only=False,  # Some builds need write access
                tmpfs={"/tmp": "size=512m,noexec"}  # Secure tmp
            )

            result.test_time = time.time() - test_start
            print(f"✓ Polyglot tests PASSED in {result.test_time:.2f}s")
            result.status = "passed"
            result.logs = container_output.decode('utf-8')

        except ContainerError as e:
            result.test_time = time.time() - test_start if 'test_start' in locals() else 0
            print(f"✗ Polyglot tests FAILED. Exit code: {e.exit_status}")
            logs = e.stderr.decode('utf-8') if e.stderr else e.stdout.decode('utf-8')
            result.status = "failed"
            result.logs = logs

        finally:
            print("[Step 5/7] Cleaning up polyglot validation image...")
            if image:
                try:
                    client.images.remove(image.id, force=True)
                    print(f"✓ Image '{image_tag}' removed.")
                except APIError as e:
                    print(f"Warning: Could not remove image '{image_tag}'. Error: {e}")
                    result.warnings.append(f"Failed to cleanup image: {image_tag}")

    result.execution_time = time.time() - start_time
    print(f"[Step 6/7] Polyglot validation completed in {result.execution_time:.2f}s")
    print(f"[Step 7/7] Languages processed: {', '.join(result.detected_languages)}")
    print("--- ENHANCED POLYGLOT CRUCIBLE MISSION COMPLETE ---")

    # Return backward compatible format by default
    if enhanced_output:
        return result.to_dict()
    else:
        return {"status": result.status, "logs": result.logs}

# --- Enhanced API Functions ---
def validate_fix_enhanced(repo_url: str, target_file: str, modified_code: str) -> Dict:
    """Enhanced version that returns detailed polyglot metrics and warnings"""
    return validate_fix(repo_url, target_file, modified_code, enhanced_output=True)

def get_supported_project_types() -> List[str]:
    """Returns list of supported project types with enhanced polyglot support"""
    return [
        "python", "node", "java", "dotnet", "go", "rust", "cpp", "swift",
        "ruby", "php", "dart", "kotlin", "scala", "clojure", "haskell",
        "elixir", "r"
    ]

def analyze_project_ecosystem(repo_url: str) -> Dict[str, Any]:
    """Enhanced project analysis with comprehensive ecosystem detection"""
    with IntelligentCrawler(repo_url=repo_url) as crawler:
        repo_path = crawler.repo_path
        project_type, detected_languages, build_tools = _detect_project_ecosystem(repo_path)
        install_cmd, test_cmd, base_image, dep_file = _get_enhanced_project_commands(
            project_type, repo_path, detected_languages
        )
        warnings = _analyze_polyglot_warnings(repo_path, project_type, detected_languages)

        return {
            "primary_project_type": project_type,
            "detected_languages": detected_languages,
            "build_tools": build_tools,
            "install_command": install_cmd,
            "test_command": test_cmd,
            "base_image": base_image,
            "dependency_file": dep_file,
            "warnings": warnings,
            "polyglot_support": True,
            "supported": project_type in get_supported_project_types()
        }

# --- Backward Compatibility Aliases ---
def crucible_validate(repo_url: str, target_file: str, modified_code: str) -> Dict[str, str]:
    """Backward compatibility alias for validate_fix"""
    return validate_fix(repo_url, target_file, modified_code)

def analyze_project(repo_url: str) -> Dict[str, str]:
    """Backward compatibility alias for analyze_project_ecosystem"""
    result = analyze_project_ecosystem(repo_url)
    # Convert to old format for backward compatibility
    return {
        "project_type": result["primary_project_type"],
        "install_command": result["install_command"],
        "test_command": result["test_command"],
        "base_image": result["base_image"],
        "dependency_file": result["dependency_file"],
        "supported": result["supported"]
    }
