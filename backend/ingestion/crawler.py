# In ingestion/crawler.py
import subprocess
import tempfile
import pathlib
from typing import List, Optional, Union, Dict, Set
import logging
import os
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger(__name__)

# --- THE FIX: Silence noisy loggers ---
# Set the log level for httpx (used by ollama) and faiss to WARNING to reduce noise.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)

from abc import ABC, abstractmethod
from typing import Union

class BaseCrawler(ABC):
    """
    Abstract base class for all crawlers following the Archetype Evolution principle.
    Defines the common interface for crawling different data sources.
    """
    
    @abstractmethod
    def get_file_paths(self, custom_extensions: Optional[List[str]] = None,
                      custom_excluded_dirs: Optional[Set[str]] = None,
                      include_hidden: bool = False) -> List[pathlib.Path]:
        """
        Scan and return a list of relevant files.
        Must be implemented by all crawler subclasses.
        """
        pass
    
    @abstractmethod
    def get_file_content(self, file_path: str, encoding: str = 'utf-8') -> Optional[str]:
        """
        Read the content of a file.
        Must be implemented by all crawler subclasses.
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """
        Clean up resources.
        Must be implemented by all crawler subclasses.
        """
        pass

class GitCrawler(BaseCrawler):
    """
    Clones a Git repository and performs file operations safely.
    Includes path-finding, git blame, and git diff capabilities.
    Enhanced with comprehensive polyglot language support.
    """

    # Class-level constants for better maintainability
    DEFAULT_EXCLUDED_DIRS = {
        '.git', '__pycache__', 'venv', 'node_modules', '.vscode', '.idea',
        'dist', 'build', '.pytest_cache', '.mypy_cache', '.tox', 'target',
        'bin', 'obj', 'out', '.gradle', '.mvn', 'vendor', 'deps', '_build',
        '.stack-work', '.cabal-sandbox', 'elm-stuff', '.pub-cache', '.dart_tool'
    }

    # Comprehensive language support - organized by ecosystem
    DEFAULT_INCLUDED_EXTENSIONS = [
        # === WEB TECHNOLOGIES ===
        # Frontend JavaScript/TypeScript
        '*.js', '*.mjs', '*.cjs', '*.jsx', '*.ts', '*.tsx', '*.vue', '*.svelte',
        '*.astro', '*.lit', '*.stencil', '*.qwik',
        # Web markup and styling
        '*.html', '*.htm', '*.xhtml', '*.xml', '*.css', '*.scss', '*.sass',
        '*.less', '*.styl', '*.stylus', '*.postcss',
        # Web frameworks and configs
        '*.ejs', '*.pug', '*.handlebars', '*.hbs', '*.mustache', '*.twig',
        '*.blade.php', '*.erb', '*.haml', '*.slim',

        # === BACKEND LANGUAGES ===
        # Python ecosystem
        '*.py', '*.pyx', '*.pyi', '*.pyw', '*.py3',
        # JavaScript/Node.js
        '*.gs',  # Google Apps Script
        # Java ecosystem
        '*.java', '*.kt', '*.kts', '*.scala', '*.groovy', '*.clj', '*.cljs', '*.cljc',
        # .NET ecosystem
        '*.cs', '*.vb', '*.fs', '*.fsx', '*.csx',
        # Systems programming
        '*.c', '*.cpp', '*.cxx', '*.cc', '*.c++', '*.h', '*.hpp', '*.hxx', '*.h++',
        '*.rs', '*.go', '*.zig', '*.odin', '*.v', '*.nim', '*.crystal',
        # Apple ecosystem
        '*.swift', '*.m', '*.mm',
        # Other compiled languages
        '*.d', '*.ada', '*.adb', '*.ads',

        # === SCRIPTING LANGUAGES ===
        '*.rb', '*.rake', '*.gemspec', '*.rbw',  # Ruby
        '*.php', '*.phtml', '*.php3', '*.php4', '*.php5', '*.php7', '*.php8',  # PHP
        '*.pl', '*.pm', '*.t', '*.pod',  # Perl
        '*.lua', '*.luac',  # Lua
        '*.tcl', '*.tk',  # Tcl/Tk
        '*.ps1', '*.psm1', '*.psd1',  # PowerShell

        # === SHELL SCRIPTING ===
        '*.sh', '*.bash', '*.zsh', '*.fish', '*.csh', '*.tcsh', '*.ksh',
        '*.bat', '*.cmd', '*.command',

        # === FUNCTIONAL LANGUAGES ===
        '*.hs', '*.lhs',  # Haskell
        '*.ml', '*.mli', '*.mll', '*.mly',  # OCaml
        '*.elm',  # Elm
        '*.ex', '*.exs',  # Elixir
        '*.erl', '*.hrl',  # Erlang
        '*.lisp', '*.lsp', '*.cl', '*.el',  # Lisp family
        '*.scm', '*.ss', '*.rkt',  # Scheme/Racket
        '*.f', '*.for', '*.f90', '*.f95', '*.f03', '*.f08',  # Fortran

        # === DATA SCIENCE & MATH ===
        '*.r', '*.R', '*.rmd', '*.rnw',  # R
        '*.jl',  # Julia
        '*.m', '*.mat',  # MATLAB/Octave
        '*.nb', '*.wl',  # Mathematica
        '*.sas', '*.stata', '*.do',  # Statistics
        '*.ipynb',  # Jupyter notebooks

        # === MOBILE DEVELOPMENT ===
        '*.dart',  # Dart/Flutter
        '*.kt', '*.kts',  # Kotlin (Android)
        '*.java',  # Java (Android)

        # === GAME DEVELOPMENT ===
        '*.cs',  # Unity C#
        '*.gd', '*.tres', '*.tscn',  # Godot
        '*.lua',  # Love2D, World of Warcraft addons
        '*.as', '*.mxml',  # ActionScript/Flex
        '*.hlsl', '*.glsl', '*.cg', '*.shader',  # Shaders

        # === DATABASE ===
        '*.sql', '*.sqlite', '*.db', '*.mysql', '*.pgsql', '*.plsql',
        '*.cypher', '*.cql', '*.sparql', '*.graphql', '*.gql',

        # === CONFIGURATION FORMATS ===
        '*.json', '*.json5', '*.jsonc', '*.jsonl', '*.ndjson',
        '*.toml', '*.yaml', '*.yml', '*.ini', '*.cfg', '*.conf', '*.config',
        '*.properties', '*.env', '*.dotenv', '*.editorconfig',
        '*.hcl', '*.tf', '*.tfvars',  # Terraform
        '*.dhall', '*.nix',  # Nix
        '*.hocon',  # HOCON (Typesafe Config)

        # === MARKUP & DOCUMENTATION ===
        '*.md', '*.markdown', '*.mdown', '*.mkd', '*.mdx',
        '*.rst', '*.adoc', '*.asciidoc', '*.txt', '*.text',
        '*.tex', '*.latex', '*.ltx', '*.cls', '*.sty',
        '*.org', '*.wiki', '*.textile',

        # === DATA FORMATS ===
        '*.csv', '*.tsv', '*.psv', '*.ssv',
        '*.parquet', '*.avro', '*.orc', '*.arrow',
        '*.proto', '*.protobuf',  # Protocol Buffers
        '*.thrift',  # Apache Thrift
        '*.capnp',  # Cap'n Proto

        # === SCIENTIFIC COMPUTING ===
        '*.cu', '*.cuh',  # CUDA
        '*.opencl', '*.cl',  # OpenCL
        '*.sage', '*.magma',  # Mathematical software

        # === EMERGING LANGUAGES ===
        '*.move',  # Move (Diem/Aptos)
        '*.sol',  # Solidity (Ethereum)
        '*.cairo',  # Cairo (StarkNet)
        '*.fe',  # Fe (Ethereum)
        '*.gleam',  # Gleam
        '*.roc',  # Roc
        '*.grain',  # Grain
        '*.red',  # Red
        '*.io',  # Io
        '*.pony',  # Pony
        '*.chapel', '*.chpl',  # Chapel

        # === DOMAIN-SPECIFIC LANGUAGES ===
        '*.vhdl', '*.vhd',  # VHDL
        '*.v', '*.sv',  # Verilog/SystemVerilog
        '*.asl', '*.dsl',  # Domain-specific languages
        '*.feature',  # Gherkin/Cucumber
        '*.story',  # JBehave
        '*.bdd',  # Behavior-driven development

        # === LEGACY & SPECIALTY ===
        '*.pas', '*.pp',  # Pascal
        '*.cob', '*.cbl', '*.cpy',  # COBOL
        '*.for', '*.f77',  # FORTRAN 77
        '*.asm', '*.s', '*.S',  # Assembly
        '*.awk',  # AWK
        '*.sed',  # Sed scripts
        '*.regex', '*.re',  # Regex files

        # === AUTOMATION & TESTING ===
        '*.robot',  # Robot Framework
        '*.feature',  # Cucumber/Gherkin
        '*.spec', '*.test',  # Test specifications
        '*.e2e', '*.integration',  # Test files

        # === BUILD & INFRASTRUCTURE ===
        '*.bazel', '*.bzl',  # Bazel
        '*.buck',  # Buck
        '*.ninja',  # Ninja
        '*.gyp', '*.gypi',  # GYP

        # === VIRTUALIZATION & CONTAINERS ===
        '*.dockerfile',  # Dockerfile variants
        '*.containerfile',
        '*.vagrantfile',
    ]

    DEFAULT_SPECIAL_FILES = [
        # === BUILD SYSTEMS ===
        'Dockerfile', 'Containerfile', 'docker-compose.yml', 'docker-compose.yaml',
        'Makefile', 'makefile', 'GNUmakefile', 'Makefile.am', 'Makefile.in',
        'CMakeLists.txt', 'cmake.txt', 'meson.build', 'meson_options.txt',
        'SConstruct', 'SConscript', 'wscript', 'waf',
        'BUILD', 'BUILD.bazel', 'WORKSPACE', 'WORKSPACE.bazel',
        'buck', 'BUCK', 'TARGETS',
        'ninja.build', 'build.ninja',

        # === JAVASCRIPT/NODE.JS ECOSYSTEM ===
        'package.json', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
        'bower.json', 'component.json', 'npm-shrinkwrap.json',
        'webpack.config.js', 'webpack.config.ts', 'webpack.common.js',
        'vite.config.js', 'vite.config.ts', 'rollup.config.js', 'rollup.config.ts',
        'parcel.config.js', 'snowpack.config.js', 'esbuild.config.js',
        'tsconfig.json', 'jsconfig.json', 'tsconfig.build.json',
        'babel.config.js', 'babel.config.json', '.babelrc', '.babelrc.js',
        'postcss.config.js', 'tailwind.config.js', 'tailwind.config.ts',
        '.eslintrc.js', '.eslintrc.json', '.eslintrc.yml', '.eslintrc.yaml',
        '.prettierrc', '.prettierrc.js', '.prettierrc.json', '.prettierignore',
        'jest.config.js', 'jest.config.ts', 'vitest.config.js', 'vitest.config.ts',
        'playwright.config.js', 'playwright.config.ts', 'cypress.config.js',

        # === PYTHON ECOSYSTEM ===
        'requirements.txt', 'requirements-dev.txt', 'requirements-test.txt',
        'Pipfile', 'Pipfile.lock', 'poetry.lock', 'pdm.lock',
        'pyproject.toml', 'setup.py', 'setup.cfg', 'manifest.in',
        'tox.ini', 'pytest.ini', 'conftest.py', '.coveragerc', 'coverage.ini',
        'mypy.ini', '.mypy.ini', 'pyrightconfig.json',
        'flake8.cfg', '.flake8', 'pylintrc', '.pylintrc',
        'black.toml', 'isort.cfg', '.isort.cfg', 'bandit.yaml',
        'environment.yml', 'conda.yml', 'environment.yaml',

        # === RUST ECOSYSTEM ===
        'Cargo.toml', 'Cargo.lock', 'rust-toolchain', 'rust-toolchain.toml',
        'clippy.toml', 'rustfmt.toml', '.rustfmt.toml',

        # === GO ECOSYSTEM ===
        'go.mod', 'go.sum', 'go.work', 'go.work.sum',

        # === JAVA ECOSYSTEM ===
        'pom.xml', 'build.gradle', 'build.gradle.kts', 'settings.gradle',
        'gradle.properties', 'gradlew', 'gradlew.bat',
        'build.xml', 'ivy.xml', 'build.sbt', 'project.clj',

        # === .NET ECOSYSTEM ===
        '*.csproj', '*.vbproj', '*.fsproj', '*.sln', '*.proj',
        'packages.config', 'nuget.config', 'global.json',
        'Directory.Build.props', 'Directory.Build.targets',

        # === RUBY ECOSYSTEM ===
        'Gemfile', 'Gemfile.lock', 'Rakefile', '.ruby-version',
        'config.ru', '.rspec', '.rubocop.yml',

        # === PHP ECOSYSTEM ===
        'composer.json', 'composer.lock', 'phpunit.xml', 'phpunit.xml.dist',
        '.php_cs', '.php_cs.dist', 'phpstan.neon', 'psalm.xml',

        # === INFRASTRUCTURE AS CODE ===
        'terraform.tf', 'main.tf', 'variables.tf', 'outputs.tf',
        'terraform.tfvars', 'terraform.tfvars.json',
        'ansible.cfg', 'playbook.yml', 'hosts', 'inventory',
        'Vagrantfile', 'Berksfile', 'Policyfile.rb',
        'docker-stack.yml', 'docker-swarm.yml',

        # === KUBERNETES ===
        'kustomization.yaml', 'kustomization.yml',
        'deployment.yaml', 'service.yaml', 'ingress.yaml',
        'configmap.yaml', 'secret.yaml', 'namespace.yaml',

        # === CI/CD ===
        '.gitlab-ci.yml', '.gitlab-ci.yaml',
        'Jenkinsfile', 'jenkins.yml', 'jenkins.yaml',
        '.circleci/config.yml', '.circle/config.yml',
        '.travis.yml', 'appveyor.yml', '.appveyor.yml',
        'azure-pipelines.yml', 'azure-pipelines.yaml',
        'bitbucket-pipelines.yml', 'drone.yml', '.drone.yml',
        'wercker.yml', 'shippable.yml', 'codefresh.yml',

        # === GITHUB ACTIONS ===
        '.github/workflows/*.yml', '.github/workflows/*.yaml',
        '.github/dependabot.yml', '.github/renovate.json',

        # === DOCUMENTATION ===
        'README', 'README.txt', 'README.md', 'README.rst',
        'CHANGELOG', 'CHANGELOG.md', 'CHANGELOG.txt', 'CHANGES',
        'CONTRIBUTING.md', 'CONTRIBUTING.rst', 'CONTRIBUTING.txt',
        'CODE_OF_CONDUCT.md', 'SECURITY.md', 'SUPPORT.md',
        'LICENSE', 'LICENSE.txt', 'LICENSE.md', 'COPYING',
        'AUTHORS', 'AUTHORS.txt', 'AUTHORS.md', 'MAINTAINERS',
        'NOTICE', 'ACKNOWLEDGMENTS', 'CREDITS',

        # === VERSION CONTROL ===
        '.gitignore', '.gitattributes', '.gitmodules', '.gitmessage',
        '.hgignore', '.svnignore', '.bzrignore',

        # === EDITOR CONFIGURATIONS ===
        '.editorconfig', '.dir-locals.el', '.projectile',
        '.vimrc', '.nvimrc', '.emacs', '.spacemacs',

        # === MOBILE DEVELOPMENT ===
        'pubspec.yaml', 'pubspec.lock',  # Dart/Flutter
        'android_app.yaml', 'ios_app.yaml',
        'Info.plist', 'AndroidManifest.xml',
        'build.gradle', 'proguard-rules.pro',

        # === GAME DEVELOPMENT ===
        'project.godot', 'export_presets.cfg',  # Godot
        'game.project', 'main.tscn',
        'love.exe', 'main.lua',  # Love2D

        # === DATA SCIENCE ===
        'requirements-dev.txt', 'environment.yml',
        'notebook.ipynb', '*.ipynb',
        '.RData', '.Rprofile', 'DESCRIPTION',

        # === DATABASE ===
        'schema.sql', 'migrations.sql', 'seeds.sql',
        'alembic.ini', 'flyway.conf',
        '.sequelizerc', 'knexfile.js',

        # === WEB FRAMEWORKS ===
        'next.config.js', 'nuxt.config.js', 'svelte.config.js',
        'gatsby-config.js', 'gridsome.config.js',
        'quasar.conf.js', 'vue.config.js', 'angular.json',
        'ember-cli-build.js', '.ember-cli',

        # === TESTING ===
        'karma.conf.js', 'protractor.conf.js', 'wdio.conf.js',
        'codecept.conf.js', 'nightwatch.conf.js',
        'testcafe.json', 'cucumber.js',

        # === MONITORING & OBSERVABILITY ===
        'prometheus.yml', 'grafana.json', 'jaeger.yml',
        'newrelic.yml', 'datadog.yaml', 'sentry.properties',

        # === MISC CONFIGURATION ===
        'nodemon.json', 'browserslist', '.nvmrc', '.node-version',
        'lerna.json', 'rush.json', 'nx.json', 'workspace.json',
        'bit.json', '.bitmap', 'now.json', 'vercel.json',
        'netlify.toml', '_redirects', '_headers',
        'firebase.json', '.firebaserc', 'app.yaml', 'cron.yaml',
    ]

    def __init__(self, repo_url: str, shallow: bool = False, depth: Optional[int] = None):
        """
        Initializes the crawler with the repository URL.

        Args:
            repo_url: The Git repository URL to clone
            shallow: Whether to perform a shallow clone (faster, less history)
            depth: Depth for shallow clone (only used if shallow=True)
        """
        self.repo_url = repo_url
        self.shallow = shallow
        self.depth = depth or 1 if shallow else None
        self.temp_dir_handle = tempfile.TemporaryDirectory()
        self.repo_path = pathlib.Path(self.temp_dir_handle.name)
        self._file_paths_cache: Optional[List[pathlib.Path]] = None
        self._current_ref: Optional[str] = None

    def __enter__(self):
        """
        Enters the context manager, cloning the repository.
        """
        logger.info(f"Cloning {self.repo_url} into {self.repo_path}")
        self._clone_repo()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the context manager, cleaning up resources.
        """
        logger.info("Cleaning up resources")
        self.cleanup()

    def _run_git_command(self, cmd: List[str], check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
        """
        Helper method to run git commands with consistent error handling.

        Args:
            cmd: Git command as list of strings
            check: Whether to raise CalledProcessError on non-zero exit
            capture_output: Whether to capture stdout/stderr

        Returns:
            CompletedProcess result
        """
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                check=check,
                capture_output=capture_output,
                text=True,
                timeout=300  # 5 minute timeout for git operations
            )
            return result
        except subprocess.TimeoutExpired:
            logger.error(f"Git command timed out: {' '.join(cmd)}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {' '.join(cmd)}, Error: {e.stderr}")
            raise

    def _clone_repo(self):
        """
        Clones the git repository with optimized settings.
        """
        try:
            clone_cmd = ['git', 'clone']

            if self.shallow:
                clone_cmd.extend(['--depth', str(self.depth)])
                clone_cmd.append('--single-branch')

            clone_cmd.extend([self.repo_url, str(self.repo_path)])

            self._run_git_command(clone_cmd)

            if not self.shallow:
                # Fetch all tags and remote branches for full clones
                self._run_git_command(['git', 'fetch', 'origin', '--tags'])
                self._run_git_command(['git', 'remote', 'update'])

            # Get current branch/ref
            result = self._run_git_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
            self._current_ref = result.stdout.strip()

            logger.info(f"Repository cloned successfully on branch: {self._current_ref}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e.stderr}")
            raise

    def get_current_ref(self) -> str:
        """
        Gets the currently checked out reference.

        Returns:
            Current branch/tag/commit hash
        """
        if self._current_ref is None:
            result = self._run_git_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
            self._current_ref = result.stdout.strip()
        return self._current_ref

    def list_branches(self, remote: bool = True) -> List[str]:
        """
        Lists available branches.

        Args:
            remote: Whether to include remote branches

        Returns:
            List of branch names
        """
        cmd = ['git', 'branch']
        if remote:
            cmd.append('-a')

        result = self._run_git_command(cmd)
        branches = []
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if line.startswith('*'):
                line = line[1:].strip()
            if line and not line.startswith('('):
                branches.append(line)

        return branches

    def list_tags(self) -> List[str]:
        """
        Lists available tags.

        Returns:
            List of tag names
        """
        result = self._run_git_command(['git', 'tag', '-l'])
        return [tag.strip() for tag in result.stdout.strip().split('\n') if tag.strip()]

    def get_blame_for_file(self, target_file: str, line_range: Optional[tuple] = None) -> str:
        """
        Runs `git blame` on a specific file in the repo.

        Args:
            target_file: Path to the file relative to repo root
            line_range: Optional tuple of (start_line, end_line) for partial blame

        Returns:
            Git blame output or error message
        """
        file_full_path = self.repo_path / target_file
        if not file_full_path.exists():
            error_msg = f"Error from crawler: File '{target_file}' does not exist in the repository."
            logger.warning(error_msg)
            return error_msg

        try:
            logger.info(f"Running 'git blame' on {file_full_path}")

            cmd = ['git', 'blame', '--show-email']
            if line_range:
                start, end = line_range
                cmd.extend(['-L', f'{start},{end}'])
            cmd.append(str(file_full_path))

            result = self._run_git_command(cmd)
            return result.stdout

        except subprocess.CalledProcessError as e:
            error_message = f"Error running 'git blame' on '{target_file}': {e.stderr}"
            logger.error(error_message)
            return f"Error from crawler: {error_message}"

    def get_blame_data_for_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        Runs `git blame --line-porcelain` on a specific file and parses the output 
        for machine processing. Returns structured blame data per line.

        Args:
            file_path: Path to the file relative to repo root

        Returns:
            List of dictionaries with blame data for each line, e.g.,
            [{'commit': '...', 'author': 'J. Doe', 'email': 'j.doe@example.com', 'line_num': 1}, ...]
        """
        file_full_path = self.repo_path / file_path
        if not file_full_path.exists():
            logger.warning(f"File '{file_path}' does not exist in the repository.")
            return []

        try:
            logger.debug(f"Running 'git blame --line-porcelain' on {file_full_path}")

            cmd = ['git', 'blame', '--line-porcelain', str(file_full_path)]
            result = self._run_git_command(cmd)
            
            return self._parse_porcelain_blame(result.stdout)

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running 'git blame --line-porcelain' on '{file_path}': {e.stderr}")
            return []

    def _parse_porcelain_blame(self, porcelain_output: str) -> List[Dict[str, str]]:
        """
        Parses the --line-porcelain output from git blame into structured data.
        
        Args:
            porcelain_output: Raw output from git blame --line-porcelain
            
        Returns:
            List of dictionaries with blame data for each line
        """
        lines = porcelain_output.strip().split('\n')
        blame_data = []
        current_entry = {}
        line_number = 1
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if not line:
                i += 1
                continue
                
            # First line of each entry: commit hash, original line number, final line number, group size
            if re.match(r'^[0-9a-f]{40}', line):
                # Parse the header line: commit_hash orig_line final_line [group_size]
                parts = line.split()
                current_entry = {
                    'commit': parts[0],
                    'line_num': line_number
                }
                line_number += 1
                
            elif line.startswith('author '):
                current_entry['author'] = line[7:]  # Remove 'author ' prefix
                
            elif line.startswith('author-mail '):
                # Extract email, removing angle brackets
                email = line[12:].strip()
                if email.startswith('<') and email.endswith('>'):
                    email = email[1:-1]
                current_entry['email'] = email
                
            elif line.startswith('\t'):
                # This is the actual source line, end of this entry
                blame_data.append(current_entry.copy())
                current_entry = {}
                
            i += 1
            
        return blame_data

    def get_diff_for_branch(self, ref_name: str, base_ref: str = 'main', stat_only: bool = False) -> str:
        """
        Gets the `git diff` between two refs (branch, tag, or commit).

        Args:
            ref_name: The reference to compare
            base_ref: The base reference to compare against
            stat_only: Whether to return only diff statistics

        Returns:
            Git diff output or error message
        """
        try:
            logger.info(f"Calculating diff for '{ref_name}' against base '{base_ref}'")

            cmd = ['git', 'diff']
            if stat_only:
                cmd.append('--stat')
            cmd.append(f'{base_ref}...{ref_name}')

            logger.debug(f"Running command: {' '.join(cmd)}")
            result = self._run_git_command(cmd)
            return result.stdout

        except subprocess.CalledProcessError:
            # Try with 'master' if 'main' fails
            if base_ref == 'main':
                logger.info("Diff against 'main' failed, trying 'master' as base")
                return self.get_diff_for_branch(ref_name, 'master', stat_only)

            error_message = f"Error running 'git diff' between '{base_ref}' and '{ref_name}'"
            logger.error(error_message)
            return f"Error from crawler: {error_message}"

    def checkout_ref(self, ref_name: str, create_branch: bool = False) -> bool:
        """
        Checks out a specific git reference (branch, tag, or commit hash).

        Args:
            ref_name: The name of the reference to check out
            create_branch: Whether to create a new branch if it doesn't exist

        Returns:
            True if checkout was successful, False otherwise
        """
        logger.info(f"Attempting to checkout ref '{ref_name}'")
        try:
            # Invalidate caches since the files will change
            self._file_paths_cache = None

            cmd = ['git', 'checkout']
            if create_branch:
                cmd.extend(['-b', ref_name])
            else:
                cmd.append(ref_name)

            self._run_git_command(cmd)
            self._current_ref = ref_name
            logger.info(f"Successfully checked out '{ref_name}'")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to checkout '{ref_name}': {e.stderr}")
            return False

    def get_file_content(self, file_path: str, encoding: str = 'utf-8') -> Optional[str]:
        """
        Reads the content of a file in the repository.

        Args:
            file_path: Path to the file relative to repo root
            encoding: File encoding (default: utf-8)

        Returns:
            File content or None if file doesn't exist/can't be read
        """
        full_path = self.repo_path / file_path
        try:
            return full_path.read_text(encoding=encoding)
        except (FileNotFoundError, UnicodeDecodeError, PermissionError) as e:
            logger.warning(f"Could not read file '{file_path}': {e}")
            return None

    def find_file_path(self, target_filename: str, case_sensitive: bool = True) -> Union[str, Dict, None]:
        """
        Searches the repository for a file by its name.

        Args:
            target_filename: Name of the file to search for
            case_sensitive: Whether to perform case-sensitive search

        Returns:
            File path, conflict dict with options, or None if not found
        """
        logger.info(f"Searching for file matching '{target_filename}'")
        all_files = self.get_file_paths()

        possible_matches = []
        search_name = target_filename if case_sensitive else target_filename.lower()

        for file_path in all_files:
            relative_path = file_path.relative_to(self.repo_path)
            file_name = relative_path.name if case_sensitive else relative_path.name.lower()
            path_str = str(relative_path) if case_sensitive else str(relative_path).lower()

            if file_name == search_name or path_str.endswith('/' + search_name):
                possible_matches.append(relative_path)

        if not possible_matches:
            logger.info(f"No match found for '{target_filename}'")
            return None

        if len(possible_matches) == 1:
            match = str(possible_matches[0])
            logger.info(f"Found unique match: {match}")
            return match

        logger.info(f"Found multiple matches, checking for root-level file")
        root_matches = [p for p in possible_matches if len(p.parts) == 1]

        if len(root_matches) == 1:
            match = str(root_matches[0])
            logger.info(f"Prioritized unique root match: {match}")
            return match

        logger.warning(f"Ambiguous path detected for '{target_filename}'")
        return {
            "error": "ambiguous_path",
            "message": f"Multiple files found matching '{target_filename}'. Please specify one.",
            "options": [str(p) for p in possible_matches]
        }

    def get_file_paths(self, custom_extensions: Optional[List[str]] = None,
                      custom_excluded_dirs: Optional[Set[str]] = None,
                      include_hidden: bool = False) -> List[pathlib.Path]:
        """
        Scans the cloned repo and returns a list of relevant files.
        Caches the result for performance.

        Args:
            custom_extensions: Override default file extensions to include
            custom_excluded_dirs: Override default directories to exclude
            include_hidden: Whether to include hidden files/directories

        Returns:
            List of file paths
        """
        # Use custom parameters if provided, otherwise use defaults
        if custom_extensions is not None or custom_excluded_dirs is not None or include_hidden:
            # Don't use cache for custom parameters
            return self._scan_files(custom_extensions, custom_excluded_dirs, include_hidden)

        if self._file_paths_cache is not None:
            return self._file_paths_cache

        logger.info("Scanning for relevant files...")
        self._file_paths_cache = self._scan_files()
        logger.info(f"Found and cached {len(self._file_paths_cache)} files to process")
        return self._file_paths_cache

    def _scan_files(self, custom_extensions: Optional[List[str]] = None,
                   custom_excluded_dirs: Optional[Set[str]] = None,
                   include_hidden: bool = False) -> List[pathlib.Path]:
        """
        Internal method to scan files with given parameters.
        """
        extensions = custom_extensions if custom_extensions is not None else (
            self.DEFAULT_INCLUDED_EXTENSIONS + self.DEFAULT_SPECIAL_FILES
        )
        excluded_dirs = custom_excluded_dirs if custom_excluded_dirs is not None else self.DEFAULT_EXCLUDED_DIRS

        files_to_process = []

        for file_path in self.repo_path.rglob('*'):
            relative_path = file_path.relative_to(self.repo_path)

            # Skip hidden files/directories unless explicitly included
            if not include_hidden and any(part.startswith('.') for part in relative_path.parts):
                # Allow certain dotfiles that are in our special files list
                if file_path.name not in self.DEFAULT_SPECIAL_FILES:
                    continue

            # Skip excluded directories
            if any(part in excluded_dirs for part in relative_path.parts):
                continue

            if file_path.is_file():
                # Check against extensions (patterns) and special filenames
                if any(file_path.match(ext) for ext in extensions):
                    files_to_process.append(file_path)

        return files_to_process

    def get_language_statistics(self) -> Dict[str, Dict[str, Union[int, List[str]]]]:
        """
        Enhanced method to get comprehensive language statistics.

        Returns:
            Dictionary with language statistics including file counts and examples
        """
        files = self.get_file_paths()

        # Comprehensive language mapping
        LANGUAGE_MAP = {
            # Web Technologies
            '.js': 'JavaScript', '.mjs': 'JavaScript', '.cjs': 'JavaScript',
            '.jsx': 'JavaScript (React)', '.ts': 'TypeScript', '.tsx': 'TypeScript (React)',
            '.vue': 'Vue.js', '.svelte': 'Svelte', '.astro': 'Astro',
            '.html': 'HTML', '.htm': 'HTML', '.css': 'CSS', '.scss': 'SASS',
            '.sass': 'SASS', '.less': 'LESS', '.styl': 'Stylus',

            # Backend Languages
            '.py': 'Python', '.pyx': 'Cython', '.pyi': 'Python Interface',
            '.java': 'Java', '.kt': 'Kotlin', '.scala': 'Scala', '.groovy': 'Groovy',
            '.cs': 'C#', '.vb': 'Visual Basic .NET', '.fs': 'F#',
            '.c': 'C', '.cpp': 'C++', '.cxx': 'C++', '.cc': 'C++', '.h': 'C/C++ Header',
            '.rs': 'Rust', '.go': 'Go', '.swift': 'Swift',
            '.rb': 'Ruby', '.php': 'PHP', '.pl': 'Perl', '.lua': 'Lua',

            # Functional Languages
            '.hs': 'Haskell', '.ml': 'OCaml', '.elm': 'Elm', '.ex': 'Elixir',
            '.erl': 'Erlang', '.clj': 'Clojure', '.scm': 'Scheme', '.lisp': 'Common Lisp',

            # Data Science
            '.r': 'R', '.jl': 'Julia', '.m': 'MATLAB', '.ipynb': 'Jupyter Notebook',

            # Mobile
            '.dart': 'Dart', '.gs': 'Google Apps Script',

            # Systems
            '.zig': 'Zig', '.nim': 'Nim', '.crystal': 'Crystal', '.d': 'D',

            # Markup and Config
            '.md': 'Markdown', '.rst': 'reStructuredText', '.tex': 'LaTeX',
            '.json': 'JSON', '.yaml': 'YAML', '.yml': 'YAML', '.toml': 'TOML',
            '.xml': 'XML', '.ini': 'INI', '.cfg': 'Config',

            # Database
            '.sql': 'SQL', '.graphql': 'GraphQL', '.gql': 'GraphQL',

            # Shell
            '.sh': 'Shell Script', '.bash': 'Bash', '.zsh': 'Zsh', '.fish': 'Fish',
            '.ps1': 'PowerShell', '.bat': 'Batch', '.cmd': 'Command Script',

            # Infrastructure
            '.tf': 'Terraform', '.hcl': 'HCL', '.dockerfile': 'Dockerfile',

            # Game Development
            '.gd': 'GDScript', '.hlsl': 'HLSL', '.glsl': 'GLSL', '.shader': 'Shader',

            # Emerging/Blockchain
            '.sol': 'Solidity', '.move': 'Move', '.cairo': 'Cairo',

            # Legacy
            '.pas': 'Pascal', '.cob': 'COBOL', '.for': 'FORTRAN', '.asm': 'Assembly',

            # Special cases
            'no_extension': 'No Extension'
        }

        language_stats = {}

        for file_path in files:
            ext = file_path.suffix.lower() or 'no_extension'
            language = LANGUAGE_MAP.get(ext, f'Unknown ({ext})')

            if language not in language_stats:
                language_stats[language] = {
                    'count': 0,
                    'extensions': set(),
                    'examples': []
                }

            language_stats[language]['count'] += 1
            language_stats[language]['extensions'].add(ext)

            # Add up to 3 example files
            if len(language_stats[language]['examples']) < 3:
                relative_path = str(file_path.relative_to(self.repo_path))
                language_stats[language]['examples'].append(relative_path)

        # Convert sets to lists for JSON serialization
        for lang_data in language_stats.values():
            lang_data['extensions'] = list(lang_data['extensions'])

        return language_stats

    def get_repo_stats(self) -> Dict[str, Union[int, str, List[str], Dict]]:
        """
        Gets comprehensive statistics about the repository including language breakdown.

        Returns:
            Dictionary with enhanced repo statistics
        """
        try:
            files = self.get_file_paths()
            language_stats = self.get_language_statistics()

            # Count files by extension (legacy compatibility)
            extensions = {}
            for file_path in files:
                ext = file_path.suffix.lower() or 'no_extension'
                extensions[ext] = extensions.get(ext, 0) + 1

            # Get commit count (if not shallow)
            commit_count = "N/A (shallow clone)"
            if not self.shallow:
                try:
                    result = self._run_git_command(['git', 'rev-list', '--count', 'HEAD'])
                    commit_count = int(result.stdout.strip())
                except:
                    pass

            # Detect primary language
            if language_stats:
                primary_language = max(language_stats.items(), key=lambda x: x[1]['count'])
                primary_lang_name = primary_language[0]
                primary_lang_count = primary_language[1]['count']
            else:
                primary_lang_name = "Unknown"
                primary_lang_count = 0

            return {
                'total_files': len(files),
                'primary_language': primary_lang_name,
                'primary_language_files': primary_lang_count,
                'language_breakdown': language_stats,
                'file_extensions': extensions,  # Legacy compatibility
                'current_ref': self.get_current_ref(),
                'commit_count': commit_count,
                'branches': self.list_branches() if not self.shallow else "N/A (shallow clone)",
                'tags': self.list_tags() if not self.shallow else "N/A (shallow clone)",
                'polyglot_support': True,
                'supported_languages': len(language_stats)
            }
        except Exception as e:
            logger.error(f"Error getting repo stats: {e}")
            return {'error': str(e)}

    def cleanup(self):
        """
        Removes the temporary directory and all its contents.
        """
        try:
            self.temp_dir_handle.cleanup()
            logger.info(f"Cleaned up temporary directory: {self.repo_path}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    # Enhanced utility methods
    def find_files_by_language(self, language: str) -> List[pathlib.Path]:
        """
        Find files by programming language name.

        Args:
            language: Language name (e.g., 'Python', 'JavaScript', 'Rust')

        Returns:
            List of file paths for that language
        """
        language_stats = self.get_language_statistics()
        matching_files = []

        if language in language_stats:
            extensions = language_stats[language]['extensions']
            all_files = self.get_file_paths()

            for file_path in all_files:
                ext = file_path.suffix.lower() or 'no_extension'
                if ext in extensions:
                    matching_files.append(file_path)

        return matching_files

    def get_project_type_hints(self) -> Dict[str, Union[str, List[str]]]:
        """
        Analyze the repository to determine likely project types and frameworks.

        Returns:
            Dictionary with project type information
        """
        files = self.get_file_paths()
        file_names = {f.name for f in files}
        language_stats = self.get_language_statistics()

        project_hints = {
            'primary_language': None,
            'frameworks': [],
            'build_systems': [],
            'package_managers': [],
            'project_types': []
        }

        # Determine primary language
        if language_stats:
            primary_lang = max(language_stats.items(), key=lambda x: x[1]['count'])
            project_hints['primary_language'] = primary_lang[0]

        # Framework detection
        framework_indicators = {
            'React': ['package.json', '.jsx', '.tsx'] + [f for f in file_names if 'react' in f.lower()],
            'Vue.js': ['vue.config.js', '.vue'] + [f for f in file_names if 'vue' in f.lower()],
            'Angular': ['angular.json', '.component.ts'] + [f for f in file_names if 'angular' in f.lower()],
            'Next.js': ['next.config.js'] + [f for f in file_names if 'next' in f.lower()],
            'Svelte': ['svelte.config.js', '.svelte'],
            'Django': ['manage.py', 'wsgi.py'] + [f for f in file_names if 'django' in f.lower()],
            'Flask': [f for f in file_names if 'flask' in f.lower()],
            'FastAPI': [f for f in file_names if 'fastapi' in f.lower()],
            'Spring': ['pom.xml'] + [f for f in file_names if 'spring' in f.lower()],
            'Express': [f for f in file_names if 'express' in f.lower()],
            'Ruby on Rails': ['Gemfile', 'config.ru'] + [f for f in file_names if 'rails' in f.lower()],
            'Laravel': ['composer.json'] + [f for f in file_names if 'laravel' in f.lower()],
            'Unity': [f for f in file_names if f.endswith('.unity')],
            'Godot': ['project.godot', '.gd'],
            'Flutter': ['pubspec.yaml', '.dart'],
            'React Native': ['metro.config.js'] + [f for f in file_names if 'react-native' in f.lower()],
        }

        for framework, indicators in framework_indicators.items():
            if any(indicator in file_names or any(f.endswith(indicator) for f in file_names) for indicator in indicators):
                project_hints['frameworks'].append(framework)

        # Build system detection
        build_systems = {
            'Make': ['Makefile', 'makefile', 'GNUmakefile'],
            'CMake': ['CMakeLists.txt'],
            'Gradle': ['build.gradle', 'gradlew'],
            'Maven': ['pom.xml'],
            'Cargo': ['Cargo.toml'],
            'npm': ['package.json'],
            'Webpack': ['webpack.config.js'],
            'Vite': ['vite.config.js'],
            'Bazel': ['WORKSPACE', 'BUILD'],
        }

        for build_system, files_list in build_systems.items():
            if any(f in file_names for f in files_list):
                project_hints['build_systems'].append(build_system)

        # Package manager detection
        package_managers = {
            'npm': ['package-lock.json'],
            'yarn': ['yarn.lock'],
            'pnpm': ['pnpm-lock.yaml'],
            'pip': ['requirements.txt'],
            'poetry': ['poetry.lock'],
            'pipenv': ['Pipfile'],
            'conda': ['environment.yml'],
            'cargo': ['Cargo.lock'],
            'composer': ['composer.lock'],
            'bundler': ['Gemfile.lock'],
        }

        for pm, files_list in package_managers.items():
            if any(f in file_names for f in files_list):
                project_hints['package_managers'].append(pm)

        # Project type classification
        if 'package.json' in file_names:
            project_hints['project_types'].append('Node.js Application')
        if any(f.endswith('.py') for f in file_names):
            project_hints['project_types'].append('Python Application')
        if 'Cargo.toml' in file_names:
            project_hints['project_types'].append('Rust Application')
        if any(f.endswith('.java') for f in file_names):
            project_hints['project_types'].append('Java Application')
        if 'go.mod' in file_names:
            project_hints['project_types'].append('Go Application')
        if any(f.endswith('.cs') for f in file_names):
            project_hints['project_types'].append('.NET Application')
        if 'Dockerfile' in file_names:
            project_hints['project_types'].append('Containerized Application')
        if any(f.endswith('.tf') for f in file_names):
            project_hints['project_types'].append('Infrastructure as Code')

        return project_hints

    # Backward compatibility aliases
    def find_files_by_extension(self, extension: str) -> List[pathlib.Path]:
        """
        Backward compatibility method to find files by extension.
        """
        all_files = self.get_file_paths()
        return [f for f in all_files if f.suffix.lower() == extension.lower()]


class LocalDirectoryCrawler(BaseCrawler):
    """
    Crawls a local directory structure for document analysis.
    Part of the Librarian's Archives feature for universal knowledge management.
    """
    
    # Inherit the same comprehensive file support from GitCrawler
    DEFAULT_EXCLUDED_DIRS = GitCrawler.DEFAULT_EXCLUDED_DIRS
    DEFAULT_INCLUDED_EXTENSIONS = GitCrawler.DEFAULT_INCLUDED_EXTENSIONS
    DEFAULT_SPECIAL_FILES = GitCrawler.DEFAULT_SPECIAL_FILES
    
    def __init__(self, directory_path: Union[str, pathlib.Path]):
        """
        Initialize the local directory crawler.
        
        Args:
            directory_path: Path to the local directory to crawl
        """
        self.directory_path = pathlib.Path(directory_path).resolve()
        self._file_paths_cache: Optional[List[pathlib.Path]] = None
        
        if not self.directory_path.exists():
            raise ValueError(f"Directory does not exist: {self.directory_path}")
        
        if not self.directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {self.directory_path}")
        
        logger.info(f"LocalDirectoryCrawler initialized for: {self.directory_path}")
    
    def get_file_paths(self, custom_extensions: Optional[List[str]] = None,
                      custom_excluded_dirs: Optional[Set[str]] = None,
                      include_hidden: bool = False) -> List[pathlib.Path]:
        """
        Scan the local directory and return a list of relevant files.
        
        Args:
            custom_extensions: Override default file extensions to include
            custom_excluded_dirs: Override default directories to exclude
            include_hidden: Whether to include hidden files/directories
            
        Returns:
            List of file paths
        """
        # Use custom parameters if provided, otherwise use defaults
        if custom_extensions is not None or custom_excluded_dirs is not None or include_hidden:
            # Don't use cache for custom parameters
            return self._scan_files(custom_extensions, custom_excluded_dirs, include_hidden)
        
        if self._file_paths_cache is not None:
            return self._file_paths_cache
        
        logger.info("Scanning local directory for relevant files...")
        self._file_paths_cache = self._scan_files()
        logger.info(f"Found and cached {len(self._file_paths_cache)} files to process")
        return self._file_paths_cache
    
    def _scan_files(self, custom_extensions: Optional[List[str]] = None,
                   custom_excluded_dirs: Optional[Set[str]] = None,
                   include_hidden: bool = False) -> List[pathlib.Path]:
        """
        Internal method to scan files with given parameters.
        """
        extensions = custom_extensions if custom_extensions is not None else (
            self.DEFAULT_INCLUDED_EXTENSIONS + self.DEFAULT_SPECIAL_FILES
        )
        excluded_dirs = custom_excluded_dirs if custom_excluded_dirs is not None else self.DEFAULT_EXCLUDED_DIRS
        
        files_to_process = []
        
        for file_path in self.directory_path.rglob('*'):
            if not file_path.is_file():
                continue
                
            relative_path = file_path.relative_to(self.directory_path)
            
            # Skip hidden files/directories unless explicitly included
            if not include_hidden and any(part.startswith('.') for part in relative_path.parts):
                # Allow certain dotfiles that are in our special files list
                if file_path.name not in self.DEFAULT_SPECIAL_FILES:
                    continue
            
            # Skip excluded directories
            if any(part in excluded_dirs for part in relative_path.parts):
                continue
            
            # Check against extensions (patterns) and special filenames
            if any(file_path.match(ext) for ext in extensions):
                files_to_process.append(file_path)
        
        return files_to_process
    
    def get_file_content(self, file_path: str, encoding: str = 'utf-8') -> Optional[str]:
        """
        Read the content of a file in the directory.
        
        Args:
            file_path: Path to the file relative to directory root
            encoding: File encoding (default: utf-8)
            
        Returns:
            File content or None if file doesn't exist/can't be read
        """
        # Handle both absolute and relative paths
        if pathlib.Path(file_path).is_absolute():
            full_path = pathlib.Path(file_path)
        else:
            full_path = self.directory_path / file_path
        
        try:
            return full_path.read_text(encoding=encoding)
        except (FileNotFoundError, UnicodeDecodeError, PermissionError) as e:
            logger.warning(f"Could not read file '{file_path}': {e}")
            return None
    
    def get_directory_stats(self) -> Dict[str, Union[int, str, List[str], Dict]]:
        """
        Get comprehensive statistics about the local directory.
        
        Returns:
            Dictionary with directory statistics
        """
        try:
            files = self.get_file_paths()
            language_stats = self._get_language_statistics(files)
            
            # Count files by extension
            extensions = {}
            for file_path in files:
                ext = file_path.suffix.lower() or 'no_extension'
                extensions[ext] = extensions.get(ext, 0) + 1
            
            # Detect primary language
            if language_stats:
                primary_language = max(language_stats.items(), key=lambda x: x[1]['count'])
                primary_lang_name = primary_language[0]
                primary_lang_count = primary_language[1]['count']
            else:
                primary_lang_name = "Unknown"
                primary_lang_count = 0
            
            return {
                'total_files': len(files),
                'primary_language': primary_lang_name,
                'primary_language_files': primary_lang_count,
                'language_breakdown': language_stats,
                'file_extensions': extensions,
                'directory_path': str(self.directory_path),
                'supported_languages': len(language_stats),
                'directory_size_mb': self._get_directory_size() / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Error getting directory stats: {e}")
            return {'error': str(e)}
    
    def _get_language_statistics(self, files: List[pathlib.Path]) -> Dict[str, Dict[str, Union[int, List[str]]]]:
        """
        Get language statistics for the directory (reusing logic from GitCrawler).
        """
        # Language mapping (same as GitCrawler)
        LANGUAGE_MAP = {
            '.js': 'JavaScript', '.mjs': 'JavaScript', '.cjs': 'JavaScript',
            '.jsx': 'JavaScript (React)', '.ts': 'TypeScript', '.tsx': 'TypeScript (React)',
            '.vue': 'Vue.js', '.svelte': 'Svelte', '.astro': 'Astro',
            '.html': 'HTML', '.htm': 'HTML', '.css': 'CSS', '.scss': 'SASS',
            '.sass': 'SASS', '.less': 'LESS', '.styl': 'Stylus',
            '.py': 'Python', '.pyx': 'Cython', '.pyi': 'Python Interface',
            '.java': 'Java', '.kt': 'Kotlin', '.scala': 'Scala', '.groovy': 'Groovy',
            '.cs': 'C#', '.vb': 'Visual Basic .NET', '.fs': 'F#',
            '.c': 'C', '.cpp': 'C++', '.cxx': 'C++', '.cc': 'C++', '.h': 'C/C++ Header',
            '.rs': 'Rust', '.go': 'Go', '.swift': 'Swift',
            '.rb': 'Ruby', '.php': 'PHP', '.pl': 'Perl', '.lua': 'Lua',
            '.hs': 'Haskell', '.ml': 'OCaml', '.elm': 'Elm', '.ex': 'Elixir',
            '.erl': 'Erlang', '.clj': 'Clojure', '.scm': 'Scheme', '.lisp': 'Common Lisp',
            '.r': 'R', '.jl': 'Julia', '.m': 'MATLAB', '.ipynb': 'Jupyter Notebook',
            '.dart': 'Dart', '.gs': 'Google Apps Script',
            '.zig': 'Zig', '.nim': 'Nim', '.crystal': 'Crystal', '.d': 'D',
            '.md': 'Markdown', '.rst': 'reStructuredText', '.tex': 'LaTeX',
            '.json': 'JSON', '.yaml': 'YAML', '.yml': 'YAML', '.toml': 'TOML',
            '.xml': 'XML', '.ini': 'INI', '.cfg': 'Config',
            '.sql': 'SQL', '.graphql': 'GraphQL', '.gql': 'GraphQL',
            '.sh': 'Shell Script', '.bash': 'Bash', '.zsh': 'Zsh', '.fish': 'Fish',
            '.ps1': 'PowerShell', '.bat': 'Batch', '.cmd': 'Command Script',
            '.tf': 'Terraform', '.hcl': 'HCL', '.dockerfile': 'Dockerfile',
            '.gd': 'GDScript', '.hlsl': 'HLSL', '.glsl': 'GLSL', '.shader': 'Shader',
            '.sol': 'Solidity', '.move': 'Move', '.cairo': 'Cairo',
            '.pas': 'Pascal', '.cob': 'COBOL', '.for': 'FORTRAN', '.asm': 'Assembly',
            'no_extension': 'No Extension'
        }
        
        language_stats = {}
        
        for file_path in files:
            ext = file_path.suffix.lower() or 'no_extension'
            language = LANGUAGE_MAP.get(ext, f'Unknown ({ext})')
            
            if language not in language_stats:
                language_stats[language] = {
                    'count': 0,
                    'extensions': set(),
                    'examples': []
                }
            
            language_stats[language]['count'] += 1
            language_stats[language]['extensions'].add(ext)
            
            # Add up to 3 example files
            if len(language_stats[language]['examples']) < 3:
                relative_path = str(file_path.relative_to(self.directory_path))
                language_stats[language]['examples'].append(relative_path)
        
        # Convert sets to lists for JSON serialization
        for lang_data in language_stats.values():
            lang_data['extensions'] = list(lang_data['extensions'])
        
        return language_stats
    
    def _get_directory_size(self) -> int:
        """Get total size of the directory in bytes."""
        total_size = 0
        try:
            for file_path in self.directory_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Error calculating directory size: {e}")
        return total_size
    
    def find_file_path(self, target_filename: str, case_sensitive: bool = True) -> Union[str, Dict, None]:
        """
        Search the directory for a file by its name.
        
        Args:
            target_filename: Name of the file to search for
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            File path, conflict dict with options, or None if not found
        """
        logger.info(f"Searching for file matching '{target_filename}'")
        all_files = self.get_file_paths()
        
        possible_matches = []
        search_name = target_filename if case_sensitive else target_filename.lower()
        
        for file_path in all_files:
            relative_path = file_path.relative_to(self.directory_path)
            file_name = relative_path.name if case_sensitive else relative_path.name.lower()
            path_str = str(relative_path) if case_sensitive else str(relative_path).lower()
            
            if file_name == search_name or path_str.endswith('/' + search_name):
                possible_matches.append(relative_path)
        
        if not possible_matches:
            logger.info(f"No match found for '{target_filename}'")
            return None
        
        if len(possible_matches) == 1:
            match = str(possible_matches[0])
            logger.info(f"Found unique match: {match}")
            return match
        
        logger.info(f"Found multiple matches, checking for root-level file")
        root_matches = [p for p in possible_matches if len(p.parts) == 1]
        
        if len(root_matches) == 1:
            match = str(root_matches[0])
            logger.info(f"Prioritized unique root match: {match}")
            return match
        
        logger.warning(f"Ambiguous path detected for '{target_filename}'")
        return {
            "error": "ambiguous_path",
            "message": f"Multiple files found matching '{target_filename}'. Please specify one.",
            "options": [str(p) for p in possible_matches]
        }
    
    def cleanup(self):
        """
        Clean up resources (no-op for local directory crawler).
        """
        logger.info("LocalDirectoryCrawler cleanup completed (no resources to clean)")
        self._file_paths_cache = None


# Backward compatibility - keep the old name as an alias
IntelligentCrawler = GitCrawler
