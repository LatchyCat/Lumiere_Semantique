# backend/lumiere_core/services/bom_parser.py
import json
import re
import toml
import xml.etree.ElementTree as ET
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DependencyType(Enum):
    APPLICATION = "application"
    DEVELOPMENT = "development"
    BUILD_TOOL = "build_tool"
    TESTING = "testing"
    PEER = "peer"
    OPTIONAL = "optional"

class SecurityRisk(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EnumEncoder(json.JSONEncoder):
    """ Custom JSON encoder to handle Enum objects. """
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value # Convert Enum to its string value
        return super().default(obj)

@dataclass
class Dependency:
    name: str
    version: str
    source: str
    dependency_type: DependencyType
    ecosystem: str
    license: Optional[str] = None
    security_risk: Optional[SecurityRisk] = None
    deprecated: bool = False
    description: Optional[str] = None
    homepage: Optional[str] = None
    repository: Optional[str] = None
    vulnerability_count: int = 0
    last_updated: Optional[str] = None

@dataclass
class Service:
    name: str
    version: str
    source: str
    service_type: str  # database, cache, message_queue, etc.
    ports: List[int] = None
    environment: Dict[str, str] = None
    volumes: List[str] = None
    networks: List[str] = None

@dataclass
class BuildTool:
    name: str
    version: str
    source: str
    purpose: str  # bundler, compiler, test_runner, etc.
    configuration: Dict[str, Any] = None

@dataclass
class TechStackBOM:
    summary: Dict[str, Any]
    dependencies: Dict[str, List[Dependency]]
    services: List[Service]
    build_tools: List[BuildTool]
    languages: Dict[str, Dict[str, Any]]
    infrastructure: Dict[str, Any]
    security_analysis: Dict[str, Any]
    metadata: Dict[str, Any]

    # THE FIX IS HERE: This method converts the object to a JSON-serializable dictionary.
    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass instance to a dictionary, correctly handling Enums."""
            # asdict converts the dataclass to a dict, but enums are still objects.
            # json.dumps with our custom encoder turns enums into strings.
            # json.loads turns the resulting JSON string back into a final, clean dictionary.
        return json.loads(json.dumps(asdict(self), cls=EnumEncoder))



def _parse_requirements_txt(file_path: Path, dep_type: DependencyType = DependencyType.APPLICATION) -> List[Dependency]:
    """Enhanced requirements.txt parser with better version handling."""
    dependencies = []
    if not file_path.exists():
        return dependencies

    # Enhanced pattern to capture package name, version specifiers, and extras
    pattern = re.compile(r"^\s*([a-zA-Z0-9\-_.]+)(\[.*\])?\s*([<>=!~^].*)?(?:\s*#.*)?$")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    # Handle -e editable installs
                    if line.startswith('-e '):
                        continue

                    match = pattern.match(line)
                    if match:
                        name, extras, version = match.groups()
                        dependencies.append(Dependency(
                            name=name,
                            version=version.strip() if version else "any",
                            source=file_path.name,
                            dependency_type=dep_type,
                            ecosystem="python"
                        ))
    except Exception as e:
        print(f"Warning: Could not parse {file_path.name}: {e}")

    return dependencies

def _parse_python_deps(repo_root: Path) -> Tuple[List[Dependency], List[BuildTool], Dict[str, Any]]:
    """Enhanced Python dependency parser with better Poetry/PDM support."""
    dependencies = []
    build_tools = []
    lang_info = {
        'version': None,
        'build_system': None,
        'package_manager': None,
        'virtual_env': None
    }

    # Parse pyproject.toml (Poetry, PDM, setuptools)
    pyproject_paths = list(repo_root.rglob("pyproject.toml"))
    for pyproject_path in pyproject_paths:
        try:
            data = toml.load(pyproject_path)

            # Detect build system
            build_system = data.get("build-system", {})
            if build_system:
                lang_info['build_system'] = build_system.get("build-backend", "unknown")
                requires = build_system.get("requires", [])
                for req in requires:
                    # --- THE FIX: Make this parsing more robust ---
                    # The original line caused `list index out of range` on simple requirements like 'hatchling'
                    # The corrected version safely handles various formats.
                    name_part = req.split(">=")[0].split("==")[0].split("~=")[0].split("<=")[0].strip()
                    if name_part:
                        build_tools.append(BuildTool(
                            name=name_part,
                            version="any",
                            source="pyproject.toml",
                            purpose="build_system"
                        ))
                    # --- END OF FIX ---

            # Poetry dependencies
            poetry_config = data.get("tool", {}).get("poetry", {})
            if poetry_config:
                lang_info['package_manager'] = 'poetry'

                # Parse Python version requirement
                python_version = poetry_config.get("dependencies", {}).get("python")
                if python_version:
                    lang_info['version'] = python_version

                # Application dependencies
                app_deps = poetry_config.get("dependencies", {})
                for name, version_info in app_deps.items():
                    if name.lower() == "python":
                        continue

                    version = version_info if isinstance(version_info, str) else version_info.get("version", "any")
                    optional = version_info.get("optional", False) if isinstance(version_info, dict) else False

                    dependencies.append(Dependency(
                        name=name,
                        version=version,
                        source="pyproject.toml",
                        dependency_type=DependencyType.OPTIONAL if optional else DependencyType.APPLICATION,
                        ecosystem="python"
                    ))

                # Development dependencies
                dev_deps = poetry_config.get("group", {}).get("dev", {}).get("dependencies", {})
                for name, version_info in dev_deps.items():
                    version = version_info if isinstance(version_info, str) else version_info.get("version", "any")
                    dependencies.append(Dependency(
                        name=name,
                        version=version,
                        source="pyproject.toml",
                        dependency_type=DependencyType.DEVELOPMENT,
                        ecosystem="python"
                    ))

                build_tools.append(BuildTool(
                    name="poetry",
                    version="any",
                    source="pyproject.toml",
                    purpose="package_manager"
                ))

            # PDM dependencies
            pdm_config = data.get("tool", {}).get("pdm", {})
            if pdm_config:
                lang_info['package_manager'] = 'pdm'
                build_tools.append(BuildTool(
                    name="pdm",
                    version="any",
                    source="pyproject.toml",
                    purpose="package_manager"
                ))

        except Exception as e:
            print(f"Warning: Could not parse pyproject.toml: {e}")

    # Parse requirements files
    requirements_files_patterns = [
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-test.txt",
        "dev-requirements.txt",
        "test-requirements.txt",
    ]

    for pattern in requirements_files_patterns:
        for req_file_path in repo_root.rglob(pattern):
            dependencies.extend(_parse_requirements_txt(req_file_path, DependencyType.APPLICATION)) # Default to APPLICATION, can be refined later

    # Check for virtual environment indicators
    if (repo_root / "Pipfile").exists():
        lang_info['package_manager'] = 'pipenv'
        lang_info['virtual_env'] = 'pipenv'
    elif (repo_root / "poetry.lock").exists():
        lang_info['virtual_env'] = 'poetry'
    elif (repo_root / "requirements.txt").exists():
        lang_info['virtual_env'] = 'pip'

    return dependencies, build_tools, lang_info

def _parse_npm_deps(repo_root: Path) -> Tuple[List[Dependency], List[BuildTool], Dict[str, Any]]:
    """Enhanced Node.js dependency parser with better framework detection."""
    dependencies = []
    build_tools = []
    lang_info = {
        'version': None,
        'package_manager': None,
        'frameworks': [],
        'bundler': None
    }

    package_json_paths = list(repo_root.rglob("package.json"))
    if not package_json_paths:
        return dependencies, build_tools, lang_info

    for package_json_path in package_json_paths:
        try:
            with open(package_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Detect Node.js version
            engines = data.get("engines", {})
            if "node" in engines:
                lang_info['version'] = engines["node"]

            # Detect package manager
            if (package_json_path.parent / "yarn.lock").exists():
                lang_info['package_manager'] = 'yarn'
            elif (package_json_path.parent / "pnpm-lock.yaml").exists():
                lang_info['package_manager'] = 'pnpm'
            elif (package_json_path.parent / "package-lock.json").exists():
                lang_info['package_manager'] = 'npm'
            else:
                lang_info['package_manager'] = 'npm'

            # Parse dependencies
            dep_sections = [
                ("dependencies", DependencyType.APPLICATION),
                ("devDependencies", DependencyType.DEVELOPMENT),
                ("peerDependencies", DependencyType.PEER),
                ("optionalDependencies", DependencyType.OPTIONAL)
            ]

            framework_indicators = {
                'react': 'React',
                '@angular/core': 'Angular',
                'vue': 'Vue.js',
                'svelte': 'Svelte',
                'next': 'Next.js',
                'nuxt': 'Nuxt.js',
                'express': 'Express.js',
                'fastify': 'Fastify',
                'koa': 'Koa.js',
                'nestjs': 'NestJS'
            }

            for section, dep_type in dep_sections:
                for name, version in data.get(section, {}).items():
                    dependencies.append(Dependency(
                        name=name,
                        version=version,
                        source=package_json_path.name,
                        dependency_type=dep_type,
                        ecosystem="javascript"
                    ))

                    # Detect frameworks
                    for indicator, framework in framework_indicators.items():
                        if indicator in name.lower():
                            if framework not in lang_info['frameworks']:
                                lang_info['frameworks'].append(framework)

            # Parse scripts for build tools
            scripts = data.get("scripts", {})
            build_tool_indicators = {
                'webpack': 'webpack',
                'vite': 'vite',
                'rollup': 'rollup',
                'esbuild': 'esbuild',
                'parcel': 'parcel',
                'babel': 'babel',
                'tsc': 'typescript',
                'eslint': 'eslint',
                'prettier': 'prettier',
                'jest': 'jest',
                'vitest': 'vitest',
                'cypress': 'cypress',
                'playwright': 'playwright'
            }

            detected_tools = set()
            for script_name, script_content in scripts.items():
                for tool_indicator, tool_name in build_tool_indicators.items():
                    if tool_indicator in script_content.lower():
                        detected_tools.add(tool_name)

            for tool in detected_tools:
                purpose = "bundler" if tool in ['webpack', 'vite', 'rollup', 'esbuild', 'parcel'] else "build_tool"
                if purpose == "bundler":
                    lang_info['bundler'] = tool

                build_tools.append(BuildTool(
                    name=tool,
                    version="any",
                    source=package_json_path.name,
                    purpose=purpose
                ))

        except Exception as e:
            print(f"Warning: Could not parse {package_json_path.name}: {e}")

    return dependencies, build_tools, lang_info

def parse_all_manifests(repo_root: Path) -> Optional[TechStackBOM]:
    """
    Enhanced main orchestrator function that creates a comprehensive BOM.
    """
    logger.info(f"BOM Parser: Starting analysis for repository root: {repo_root}")
    all_dependencies = []
    all_services = []
    all_build_tools = []
    language_info = {}

    # Parse different ecosystems
    parsers = {
        "Python": _parse_python_deps,
        "Node.js": _parse_npm_deps,
    }

    for ecosystem, parser_func in parsers.items():
        logger.info(f"BOM Parser: Running {ecosystem} parser...")
        try:
            deps, tools, lang_info = parser_func(repo_root)
            all_dependencies.extend(deps)
            all_build_tools.extend(tools)
            if lang_info:
                language_info[ecosystem.lower()] = lang_info
        except Exception as e:
            logger.error(f"BOM Parser: Error running '{ecosystem}' parser: {e}")

    # Parse Docker services
    logger.info("BOM Parser: Running Docker services parser...")
    try:
        services, base_images = _parse_docker_services(repo_root)
        all_services.extend(services)
        all_dependencies.extend(base_images)
    except Exception as e:
        logger.error(f"BOM Parser: Error parsing Docker services: {e}")

    detected_languages = _detect_languages(repo_root)

    primary_language = "Unknown"
    if detected_languages:
        primary_language = max(detected_languages.keys(),
                             key=lambda x: detected_languages.get(x, {}).get('lines', 0))

    categorized_deps = {
        'application': [d for d in all_dependencies if d.dependency_type == DependencyType.APPLICATION],
        'development': [d for d in all_dependencies if d.dependency_type == DependencyType.DEVELOPMENT],
        'testing': [d for d in all_dependencies if d.dependency_type == DependencyType.TESTING],
        'build': [d for d in all_dependencies if d.dependency_type == DependencyType.BUILD_TOOL],
        'peer': [d for d in all_dependencies if d.dependency_type == DependencyType.PEER],
        'optional': [d for d in all_dependencies if d.dependency_type == DependencyType.OPTIONAL]
    }

    security_analysis = _analyze_security_risks(all_dependencies)

    infrastructure = {
        'containerized': len([s for s in all_services if 'docker' in s.source.lower()]) > 0,
        'databases': [s for s in all_services if s.service_type == 'database'],
        'caches': [s for s in all_services if s.service_type == 'cache'],
        'web_servers': [s for s in all_services if s.service_type == 'web_server'],
        'message_queues': [s for s in all_services if s.service_type == 'message_queue']
    }

    bom = TechStackBOM(
        summary={
            'primary_language': primary_language,
            'total_dependencies': len(all_dependencies),
            'total_services': len(all_services),
            'total_build_tools': len(all_build_tools),
            'languages_detected': len(detected_languages),
            'ecosystems': list(set(d.ecosystem for d in all_dependencies)),
            'last_updated': None
        },
        dependencies={k: [asdict(d) for d in v] for k, v in categorized_deps.items()},
        services=[asdict(s) for s in all_services],
        build_tools=[asdict(t) for t in all_build_tools],
        languages=detected_languages,
        infrastructure=infrastructure,
        security_analysis=security_analysis,
        metadata={
            'parser_version': '2.1.0', # Version bump for robustness fix
            'parsing_errors': [],
            'repository_size': sum(1 for _ in repo_root.rglob('*') if _.is_file()),
            'config_files_found': _count_config_files(repo_root)
        }
    )

    logger.info(f"BOM Parser: Finished analysis. Found {len(all_dependencies)} total dependencies.")
    return bom

def _count_config_files(repo_root: Path) -> Dict[str, int]:
    """Count different types of configuration files."""
    config_patterns = {
        'package_managers': ['package.json', 'pyproject.toml', 'Cargo.toml', 'pom.xml'],
        'docker': ['Dockerfile*', 'docker-compose*.yml', 'docker-compose*.yaml'],
        'ci_cd': ['.github/workflows/*.yml', '.gitlab-ci.yml', 'Jenkinsfile'],
        'environment': ['.env*', 'config/*.yml', 'config/*.yaml'],
        'build_tools': ['webpack.config.js', 'vite.config.js', 'rollup.config.js']
    }

    counts = {}
    for category, patterns in config_patterns.items():
        count = 0
        for pattern in patterns:
            try:
                # Use rglob for patterns that might be in subdirectories
                if '*' in pattern or '?' in pattern:
                     count += len(list(repo_root.rglob(pattern)))
                else: # Use glob for top-level files
                     count += len(list(repo_root.glob(pattern)))
            except re.error:
                 logger.warning(f"BOM Parser: Invalid pattern in BOM parser config: {pattern}")
                 continue
        counts[category] = count

    return counts

def _parse_requirements_txt(file_path: Path, dep_type: DependencyType = DependencyType.APPLICATION) -> List[Dependency]:
    """Enhanced requirements.txt parser with better version handling."""
    dependencies = []
    logger.debug(f"BOM Parser: Parsing requirements.txt: {file_path}")
    if not file_path.exists():
        logger.debug(f"BOM Parser: requirements.txt not found: {file_path}")
        return dependencies

    # Enhanced pattern to capture package name, version specifiers, and extras
    pattern = re.compile(r"^\s*([a-zA-Z0-9\-_.]+)(\[.*\])?\s*([<>=!~^].*)?(?:\s*#.*)?$")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    # Handle -e editable installs
                    if line.startswith('-e '):
                        continue

                    match = pattern.match(line)
                    if match:
                        name, extras, version = match.groups()
                        dependencies.append(Dependency(
                            name=name,
                            version=version.strip() if version else "any",
                            source=file_path.name,
                            dependency_type=dep_type,
                            ecosystem="python"
                        ))
                        logger.debug(f"BOM Parser: Found Python dependency: {name} {version} in {file_path.name}")
    except Exception as e:
        logger.warning(f"BOM Parser: Could not parse {file_path.name}: {e}")

    return dependencies

def _parse_python_deps(repo_root: Path) -> Tuple[List[Dependency], List[BuildTool], Dict[str, Any]]:
    """Enhanced Python dependency parser with better Poetry/PDM support."""
    dependencies = []
    build_tools = []
    lang_info = {
        'version': None,
        'build_system': None,
        'package_manager': None,
        'virtual_env': None
    }

    # Parse pyproject.toml (Poetry, PDM, setuptools)
    pyproject_paths = list(repo_root.rglob("pyproject.toml"))
    logger.debug(f"BOM Parser: Found pyproject.toml files: {pyproject_paths}")
    for pyproject_path in pyproject_paths:
        logger.debug(f"BOM Parser: Parsing pyproject.toml: {pyproject_path}")
        try:
            data = toml.load(pyproject_path)

            # Detect build system
            build_system = data.get("build-system", {})
            if build_system:
                lang_info['build_system'] = build_system.get("build-backend", "unknown")
                requires = build_system.get("requires", [])
                for req in requires:
                    # --- THE FIX: Make this parsing more robust ---
                    # The original line caused `list index out of range` on simple requirements like 'hatchling'
                    # The corrected version safely handles various formats.
                    name_part = req.split(">=")[0].split("==")[0].split("~=")[0].split("<=")[0].strip()
                    if name_part:
                        build_tools.append(BuildTool(
                            name=name_part,
                            version="any",
                            source="pyproject.toml",
                            purpose="build_system"
                        ))
                        logger.debug(f"BOM Parser: Found Python build tool: {name_part} in {pyproject_path.name}")

            # Poetry dependencies
            poetry_config = data.get("tool", {}).get("poetry", {})
            if poetry_config:
                lang_info['package_manager'] = 'poetry'

                # Parse Python version requirement
                python_version = poetry_config.get("dependencies", {}).get("python")
                if python_version:
                    lang_info['version'] = python_version

                # Application dependencies
                app_deps = poetry_config.get("dependencies", {})
                for name, version_info in app_deps.items():
                    if name.lower() == "python":
                        continue

                    version = version_info if isinstance(version_info, str) else version_info.get("version", "any")
                    optional = version_info.get("optional", False) if isinstance(version_info, dict) else False

                    dependencies.append(Dependency(
                        name=name,
                        version=version,
                        source="pyproject.toml",
                        dependency_type=DependencyType.OPTIONAL if optional else DependencyType.APPLICATION,
                        ecosystem="python"
                    ))
                    logger.debug(f"BOM Parser: Found Poetry dependency: {name} {version} in {pyproject_path.name}")

                # Development dependencies
                dev_deps = poetry_config.get("group", {}).get("dev", {}).get("dependencies", {})
                for name, version_info in dev_deps.items():
                    version = version_info if isinstance(version_info, str) else version_info.get("version", "any")
                    dependencies.append(Dependency(
                        name=name,
                        version=version,
                        source="pyproject.toml",
                        dependency_type=DependencyType.DEVELOPMENT,
                        ecosystem="python"
                    ))
                    logger.debug(f"BOM Parser: Found Poetry dev dependency: {name} {version} in {pyproject_path.name}")

                build_tools.append(BuildTool(
                    name="poetry",
                    version="any",
                    source="pyproject.toml",
                    purpose="package_manager"
                ))

            # PDM dependencies
            pdm_config = data.get("tool", {}).get("pdm", {})
            if pdm_config:
                lang_info['package_manager'] = 'pdm'
                build_tools.append(BuildTool(
                    name="pdm",
                    version="any",
                    source="pyproject.toml",
                    purpose="package_manager"
                ))
                logger.debug(f"BOM Parser: Found PDM config in {pyproject_path.name}")

        except Exception as e:
            logger.warning(f"BOM Parser: Could not parse pyproject.toml: {e}")

    # Parse requirements files
    requirements_files_patterns = [
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-test.txt",
        "dev-requirements.txt",
        "test-requirements.txt",
    ]

    for pattern in requirements_files_patterns:
        req_file_paths = list(repo_root.rglob(pattern))
        logger.debug(f"BOM Parser: Found {pattern} files: {req_file_paths}")
        for req_file_path in req_file_paths:
            dependencies.extend(_parse_requirements_txt(req_file_path, DependencyType.APPLICATION)) # Default to APPLICATION, can be refined later

    # Check for virtual environment indicators
    if (repo_root / "Pipfile").exists():
        lang_info['package_manager'] = 'pipenv'
        lang_info['virtual_env'] = 'pipenv'
        logger.debug(f"BOM Parser: Found Pipfile in {repo_root}")
    elif (repo_root / "poetry.lock").exists():
        lang_info['virtual_env'] = 'poetry'
        logger.debug(f"BOM Parser: Found poetry.lock in {repo_root}")
    elif (repo_root / "requirements.txt").exists():
        lang_info['virtual_env'] = 'pip'
        logger.debug(f"BOM Parser: Found requirements.txt in {repo_root}")

    return dependencies, build_tools, lang_info

def _parse_npm_deps(repo_root: Path) -> Tuple[List[Dependency], List[BuildTool], Dict[str, Any]]:
    """Enhanced Node.js dependency parser with better framework detection."""
    dependencies = []
    build_tools = []
    lang_info = {
        'version': None,
        'package_manager': None,
        'frameworks': [],
        'bundler': None
    }

    package_json_paths = list(repo_root.rglob("package.json"))
    logger.debug(f"BOM Parser: Found package.json files: {package_json_paths}")
    if not package_json_paths:
        return dependencies, build_tools, lang_info

    for package_json_path in package_json_paths:
        logger.debug(f"BOM Parser: Parsing package.json: {package_json_path}")
        try:
            with open(package_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Detect Node.js version
            engines = data.get("engines", {})
            if "node" in engines:
                lang_info['version'] = engines["node"]
                logger.debug(f"BOM Parser: Found Node.js version: {engines['node']} in {package_json_path.name}")

            # Detect package manager
            if (package_json_path.parent / "yarn.lock").exists():
                lang_info['package_manager'] = 'yarn'
                logger.debug(f"BOM Parser: Found yarn.lock in {package_json_path.parent}")
            elif (package_json_path.parent / "pnpm-lock.yaml").exists():
                lang_info['package_manager'] = 'pnpm'
                logger.debug(f"BOM Parser: Found pnpm-lock.yaml in {package_json_path.parent}")
            elif (package_json_path.parent / "package-lock.json").exists():
                lang_info['package_manager'] = 'npm'
                logger.debug(f"BOM Parser: Found package-lock.json in {package_json_path.parent}")
            else:
                lang_info['package_manager'] = 'npm'
                logger.debug(f"BOM Parser: Defaulting to npm package manager for {package_json_path.name}")

            # Parse dependencies
            dep_sections = [
                ("dependencies", DependencyType.APPLICATION),
                ("devDependencies", DependencyType.DEVELOPMENT),
                ("peerDependencies", DependencyType.PEER),
                ("optionalDependencies", DependencyType.OPTIONAL)
            ]

            framework_indicators = {
                'react': 'React',
                '@angular/core': 'Angular',
                'vue': 'Vue.js',
                'svelte': 'Svelte',
                'next': 'Next.js',
                'nuxt': 'Nuxt.js',
                'express': 'Express.js',
                'fastify': 'Fastify',
                'koa': 'Koa.js',
                'nestjs': 'NestJS'
            }

            for section, dep_type in dep_sections:
                for name, version in data.get(section, {}).items():
                    dependencies.append(Dependency(
                        name=name,
                        version=version,
                        source=package_json_path.name,
                        dependency_type=dep_type,
                        ecosystem="javascript"
                    ))
                    logger.debug(f"BOM Parser: Found JS dependency: {name} {version} in {package_json_path.name}")

                    # Detect frameworks
                    for indicator, framework in framework_indicators.items():
                        if indicator in name.lower():
                            if framework not in lang_info['frameworks']:
                                lang_info['frameworks'].append(framework)
                                logger.debug(f"BOM Parser: Detected framework: {framework} in {package_json_path.name}")

            # Parse scripts for build tools
            scripts = data.get("scripts", {})
            build_tool_indicators = {
                'webpack': 'webpack',
                'vite': 'vite',
                'rollup': 'rollup',
                'esbuild': 'esbuild',
                'parcel': 'parcel',
                'babel': 'babel',
                'tsc': 'typescript',
                'eslint': 'eslint',
                'prettier': 'prettier',
                'jest': 'jest',
                'vitest': 'vitest',
                'cypress': 'cypress',
                'playwright': 'playwright'
            }

            detected_tools = set()
            for script_name, script_content in scripts.items():
                for tool_indicator, tool_name in build_tool_indicators.items():
                    if tool_indicator in script_content.lower():
                        detected_tools.add(tool_name)

            for tool in detected_tools:
                purpose = "bundler" if tool in ['webpack', 'vite', 'rollup', 'esbuild', 'parcel'] else "build_tool"
                if purpose == "bundler":
                    lang_info['bundler'] = tool

                build_tools.append(BuildTool(
                    name=tool,
                    version="any",
                    source=package_json_path.name,
                    purpose=purpose
                ))
                logger.debug(f"BOM Parser: Found JS build tool: {tool} in {package_json_path.name}")

        except Exception as e:
            logger.warning(f"BOM Parser: Could not parse {package_json_path.name}: {e}")

    return dependencies, build_tools, lang_info

def _parse_docker_services(repo_root: Path) -> Tuple[List[Service], List[Dependency]]:
    """Enhanced Docker parser with service analysis."""
    services = []
    base_images = []

    # Parse docker-compose files
    compose_files = list(repo_root.rglob('**/docker-compose*.yml')) + list(repo_root.rglob('**/docker-compose*.yaml'))
    logger.debug(f"BOM Parser: Found docker-compose files: {compose_files}")

    for compose_file in compose_files:
        logger.debug(f"BOM Parser: Parsing docker-compose file: {compose_file}")
        try:
            with open(compose_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            compose_services = data.get('services', {})
            for service_name, service_config in compose_services.items():
                image = service_config.get('image', '')
                if image:
                    name, version = (image.split(':') + ['latest'])[:2]

                    # Determine service type
                    service_type = _determine_service_type(name)

                    services.append(Service(
                        name=name,
                        version=version,
                        source=compose_file.name,
                        service_type=service_type,
                        ports=service_config.get('ports', []),
                        environment=service_config.get('environment', {}),
                        volumes=service_config.get('volumes', []),
                        networks=list(service_config.get('networks', {}).keys()) if isinstance(service_config.get('networks'), dict) else service_config.get('networks', [])
                    ))
                    logger.debug(f"BOM Parser: Found Docker service: {name} {version} in {compose_file.name}")

        except Exception as e:
            logger.warning(f"BOM Parser: Could not parse {compose_file.name}: {e}")

    # Parse Dockerfiles
    dockerfiles = list(repo_root.rglob('**/Dockerfile*'))
    logger.debug(f"BOM Parser: Found Dockerfiles: {dockerfiles}")
    for dockerfile in dockerfiles:
        logger.debug(f"BOM Parser: Parsing Dockerfile: {dockerfile}")
        try:
            content = dockerfile.read_text(encoding='utf-8')
            from_matches = re.findall(r'^\s*FROM\s+([^\s]+)', content, re.MULTILINE | re.IGNORECASE)

            for match in from_matches:
                # Skip build stages
                if ' as ' in match.lower():
                    match = match.split(' as ')[0]

                if ':' in match:
                    name, version = match.split(':', 1)
                else:
                    name, version = match, 'latest'

                # Skip obvious build stages
                if name.isalpha() and '.' not in name and '/' not in name:
                    continue

                base_images.append(Dependency(
                    name=name,
                    version=version,
                    source=dockerfile.name,
                    dependency_type=DependencyType.APPLICATION,
                    ecosystem="docker"
                ))
                logger.debug(f"BOM Parser: Found Docker base image: {name} {version} in {dockerfile.name}")

        except Exception as e:
            logger.warning(f"BOM Parser: Could not parse {dockerfile.name}: {e}")

    return services, base_images

def _determine_service_type(image_name: str) -> str:
    """Determine service type based on image name."""
    service_types = {
        'postgres': 'database',
        'mysql': 'database',
        'mongodb': 'database',
        'redis': 'cache',
        'memcached': 'cache',
        'nginx': 'web_server',
        'apache': 'web_server',
        'rabbitmq': 'message_queue',
        'kafka': 'message_queue',
        'elasticsearch': 'search',
        'solr': 'search',
        'prometheus': 'monitoring',
        'grafana': 'monitoring',
        'jaeger': 'tracing',
        'zipkin': 'tracing'
    }

    for key, service_type in service_types.items():
        if key in image_name.lower():
            return service_type

    return 'application'

def _analyze_security_risks(dependencies: List[Dependency]) -> Dict[str, Any]:
    """Analyze security risks (placeholder for future vulnerability scanning)."""
    total_deps = len(dependencies)
    high_risk_count = 0  # Placeholder

    return {
        'total_dependencies': total_deps,
        'high_risk_dependencies': high_risk_count,
        'last_scan': None,
        'recommendations': []
    }

def _detect_languages(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    """Detect programming languages used in the repository."""
    from collections import defaultdict

    # This mapping can be expanded as needed
    LANGUAGE_MAP = {
        '.py': {'name': 'Python'},
        '.js': {'name': 'JavaScript'},
        '.ts': {'name': 'TypeScript'},
        '.java': {'name': 'Java'},
        '.go': {'name': 'Go'},
        '.rs': {'name': 'Rust'},
    }

    language_stats = defaultdict(lambda: {'files': 0, 'lines': 0})

    for file_path in repo_root.rglob('*'):
        if file_path.is_file() and file_path.suffix in LANGUAGE_MAP:
            lang_info = LANGUAGE_MAP[file_path.suffix]
            lang_name = lang_info['name']

            try:
                line_count = len(file_path.read_text(encoding='utf-8', errors='ignore').splitlines())
                language_stats[lang_name]['files'] += 1
                language_stats[lang_name]['lines'] += line_count
                language_stats[lang_name]['config'] = lang_info
            except Exception:
                continue

    return dict(language_stats)

def parse_all_manifests(repo_root: Path) -> Optional[TechStackBOM]:
    """
    Enhanced main orchestrator function that creates a comprehensive BOM.
    """
    all_dependencies = []
    all_services = []
    all_build_tools = []
    language_info = {}

    # Parse different ecosystems
    parsers = {
        "Python": _parse_python_deps,
        "Node.js": _parse_npm_deps,
    }

    for ecosystem, parser_func in parsers.items():
        try:
            deps, tools, lang_info = parser_func(repo_root)
            all_dependencies.extend(deps)
            all_build_tools.extend(tools)
            if lang_info:
                language_info[ecosystem.lower()] = lang_info
        except Exception as e:
            print(f"Error running '{ecosystem}' parser: {e}")

    # Parse Docker services
    try:
        services, base_images = _parse_docker_services(repo_root)
        all_services.extend(services)
        all_dependencies.extend(base_images)
    except Exception as e:
        print(f"Error parsing Docker services: {e}")

    detected_languages = _detect_languages(repo_root)

    primary_language = "Unknown"
    if detected_languages:
        primary_language = max(detected_languages.keys(),
                             key=lambda x: detected_languages.get(x, {}).get('lines', 0))

    categorized_deps = {
        'application': [d for d in all_dependencies if d.dependency_type == DependencyType.APPLICATION],
        'development': [d for d in all_dependencies if d.dependency_type == DependencyType.DEVELOPMENT],
        'testing': [d for d in all_dependencies if d.dependency_type == DependencyType.TESTING],
        'build': [d for d in all_dependencies if d.dependency_type == DependencyType.BUILD_TOOL],
        'peer': [d for d in all_dependencies if d.dependency_type == DependencyType.PEER],
        'optional': [d for d in all_dependencies if d.dependency_type == DependencyType.OPTIONAL]
    }

    security_analysis = _analyze_security_risks(all_dependencies)

    infrastructure = {
        'containerized': len([s for s in all_services if 'docker' in s.source.lower()]) > 0,
        'databases': [s for s in all_services if s.service_type == 'database'],
        'caches': [s for s in all_services if s.service_type == 'cache'],
        'web_servers': [s for s in all_services if s.service_type == 'web_server'],
        'message_queues': [s for s in all_services if s.service_type == 'message_queue']
    }

    bom = TechStackBOM(
        summary={
            'primary_language': primary_language,
            'total_dependencies': len(all_dependencies),
            'total_services': len(all_services),
            'total_build_tools': len(all_build_tools),
            'languages_detected': len(detected_languages),
            'ecosystems': list(set(d.ecosystem for d in all_dependencies)),
            'last_updated': None
        },
        dependencies={k: [asdict(d) for d in v] for k, v in categorized_deps.items()},
        services=[asdict(s) for s in all_services],
        build_tools=[asdict(t) for t in all_build_tools],
        languages=detected_languages,
        infrastructure=infrastructure,
        security_analysis=security_analysis,
        metadata={
            'parser_version': '2.1.0', # Version bump for robustness fix
            'parsing_errors': [],
            'repository_size': sum(1 for _ in repo_root.rglob('*') if _.is_file()),
            'config_files_found': _count_config_files(repo_root)
        }
    )

    return bom

def _count_config_files(repo_root: Path) -> Dict[str, int]:
    """Count different types of configuration files."""
    config_patterns = {
        'package_managers': ['package.json', 'pyproject.toml', 'Cargo.toml', 'pom.xml'],
        'docker': ['Dockerfile*', 'docker-compose*.yml', 'docker-compose*.yaml'],
        'ci_cd': ['.github/workflows/*.yml', '.gitlab-ci.yml', 'Jenkinsfile'],
        'environment': ['.env*', 'config/*.yml', 'config/*.yaml'],
        'build_tools': ['webpack.config.js', 'vite.config.js', 'rollup.config.js']
    }

    counts = {}
    for category, patterns in config_patterns.items():
        count = 0
        for pattern in patterns:
            try:
                # Use rglob for patterns that might be in subdirectories
                if '*' in pattern or '?' in pattern:
                     count += len(list(repo_root.rglob(pattern)))
                else: # Use glob for top-level files
                     count += len(list(repo_root.glob(pattern)))
            except re.error:
                 print(f"Warning: Invalid pattern in BOM parser config: {pattern}")
                 continue
        counts[category] = count

    return counts

def get_bom_data(repo_root: Path) -> Optional[TechStackBOM]:
    """Get BOM data for a repository."""
    return parse_all_manifests(repo_root)

def has_bom_data(repo_root: Path) -> bool:
    """Check if BOM data exists for a repository."""
    try:
        bom = parse_all_manifests(repo_root)
        return bom is not None and bom.summary.get('total_dependencies', 0) > 0
    except Exception:
        return False
