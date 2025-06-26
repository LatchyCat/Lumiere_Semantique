#!/usr/bin/env python3
"""
Enhanced Tree-sitter Parser Builder for Lumiere Core
Builds parsers for all supported languages in your polyglot application.
Now handles grammars that require generation step.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

try:
    from tree_sitter import Language
except ImportError:
    print("✗ 'tree_sitter' library not found. Install with:")
    print("  pip install tree-sitter==0.20.1")
    sys.exit(1)

# --- Enhanced Configuration ---
GRAMMARS_DIR = Path("lumiere_core/services/grammars")
BUILD_DIR = Path("lumiere_core/services/build")
LIBRARY_PATH = BUILD_DIR / "my-languages.so"

# Comprehensive language mappings based on your polyglot config
LANGUAGE_GRAMMARS = {
    # Web Technologies
    'javascript': {
        'url': 'https://github.com/tree-sitter/tree-sitter-javascript',
        'extensions': ['.js', '.jsx', '.mjs'],
        'priority': 'high'
    },
    # 'typescript': {
    #     'url': 'https://github.com/tree-sitter/tree-sitter-typescript',
    #     'extensions': ['.ts', '.tsx'],
    #     'priority': 'high',
    #     'subdirs': ['typescript', 'tsx']  # TypeScript repo has multiple grammars
    # },
    'html': {
        'url': 'https://github.com/tree-sitter/tree-sitter-html',
        'extensions': ['.html', '.htm'],
        'priority': 'medium'
    },
    'css': {
        'url': 'https://github.com/tree-sitter/tree-sitter-css',
        'extensions': ['.css', '.scss', '.sass'],
        'priority': 'medium'
    },
    # 'vue': {
    #     'url': 'https://github.com/ikatyang/tree-sitter-vue',
    #     'extensions': ['.vue'],
    #     'priority': 'medium'
    # },

    # Backend Languages
    'python': {
        'url': 'https://github.com/tree-sitter/tree-sitter-python',
        'extensions': ['.py', '.pyx', '.pyi'],
        'priority': 'high'
    },
    'java': {
        'url': 'https://github.com/tree-sitter/tree-sitter-java',
        'extensions': ['.java'],
        'priority': 'high'
    },
    'c_sharp': {
        'url': 'https://github.com/tree-sitter/tree-sitter-c-sharp',
        'extensions': ['.cs'],
        'priority': 'high'
    },
    'go': {
        'url': 'https://github.com/tree-sitter/tree-sitter-go',
        'extensions': ['.go'],
        'priority': 'high'
    },
    'rust': {
        'url': 'https://github.com/tree-sitter/tree-sitter-rust',
        'extensions': ['.rs'],
        'priority': 'high',
        'requires_generate': True  
    },
    'swift': {
        'url': 'https://github.com/tree-sitter/tree-sitter-swift',
        'extensions': ['.swift'],
        'priority': 'medium',
        'requires_generate': True
    },

    # Functional Languages
    'haskell': {
        'url': 'https://github.com/tree-sitter/tree-sitter-haskell',
        'extensions': ['.hs', '.lhs'],
        'priority': 'medium'
    },
    'elixir': {
        'url': 'https://github.com/elixir-lang/tree-sitter-elixir',
        'extensions': ['.ex', '.exs'],
        'priority': 'medium',
        'requires_generate': True
    },
    'clojure': {
        'url': 'https://github.com/sogaiu/tree-sitter-clojure',
        'extensions': ['.clj', '.cljs', '.cljc'],
        'priority': 'medium'
    },

    # Systems Languages
    'c': {
        'url': 'https://github.com/tree-sitter/tree-sitter-c',
        'extensions': ['.c', '.h'],
        'priority': 'high'
    },
    # 'cpp': {
    #     'url': 'https://github.com/tree-sitter/tree-sitter-cpp',
    #     'extensions': ['.cpp', '.cxx', '.cc', '.hpp', '.hxx'],
    #     'priority': 'high'
    # },

    # Scripting Languages
    'ruby': {
        'url': 'https://github.com/tree-sitter/tree-sitter-ruby',
        'extensions': ['.rb', '.rake'],
        'priority': 'medium'
    },
    # 'php': {
    #     'url': 'https://github.com/tree-sitter/tree-sitter-php',
    #     'extensions': ['.php'],
    #     'priority': 'medium',
    #     'requires_generate': True  # This grammar needs generation step
    # },

    # Mobile/Cross-platform
    'kotlin': {
        'url': 'https://github.com/fwcd/tree-sitter-kotlin',
        'extensions': ['.kt', '.kts'],
        'priority': 'medium',
        'requires_generate': True
    },
    'dart': {
        'url': 'https://github.com/UserNobody14/tree-sitter-dart',
        'extensions': ['.dart'],
        'priority': 'medium',
        'requires_generate': True
    },

    # Data Science
    'r': {
        'url': 'https://github.com/r-lib/tree-sitter-r',
        'extensions': ['.r', '.R'],
        'priority': 'low'
    },
    'julia': {
        'url': 'https://github.com/tree-sitter/tree-sitter-julia',
        'extensions': ['.jl'],
        'priority': 'low'
    },

    # Additional useful languages
    'bash': {
        'url': 'https://github.com/tree-sitter/tree-sitter-bash',
        'extensions': ['.sh', '.bash'],
        'priority': 'medium'
    },
    'json': {
        'url': 'https://github.com/tree-sitter/tree-sitter-json',
        'extensions': ['.json'],
        'priority': 'high'
    },
    # 'yaml': {
    #     'url': 'https://github.com/ikatyang/tree-sitter-yaml',
    #     'extensions': ['.yml', '.yaml'],
    #     'priority': 'medium'
    # },
    'toml': {
        'url': 'https://github.com/ikatyang/tree-sitter-toml',
        'extensions': ['.toml'],
        'priority': 'low'
    },
    # 'sql': {
    #     'url': 'https://github.com/derekstride/tree-sitter-sql',
    #     'extensions': ['.sql'],
    #     'priority': 'medium',
    #     'requires_generate': True
    # }
}

# Languages known to require generation or have build issues
REQUIRES_GENERATION = {'php', 'haskell', 'ocaml', 'agda', 'sql', 'swift', 'kotlin', 'dart', 'elixir', 'vue'}

class ParserBuilder:
    def __init__(self, languages_subset: Optional[List[str]] = None,
                 priority_filter: Optional[str] = None):
        self.languages_subset = languages_subset
        self.priority_filter = priority_filter
        self.failed_languages = []
        self.successful_languages = []
        self.skipped_languages = []

    def check_prerequisites(self) -> bool:
        """Check if all required tools are available."""
        logger.info("Checking prerequisites...")

        # Check if we're in the right directory
        if not Path("manage.py").exists() or not Path("lumiere_core").is_dir():
            logger.error("This script must be run from the 'backend' directory")
            logger.error("Please run: cd backend && python build_parsers.py")
            return False

        # Check Git
        try:
            subprocess.run(["git", "--version"], check=True,
                         capture_output=True, text=True)
            logger.info("✓ Git is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("✗ Git is required but not found in PATH")
            return False

        # Check for C/C++ compiler
        compilers = ['gcc', 'clang', 'cl']  # Windows, Unix
        compiler_found = False
        for compiler in compilers:
            try:
                subprocess.run([compiler, '--version'], check=True,
                             capture_output=True, text=True)
                logger.info(f"✓ C/C++ compiler found: {compiler}")
                compiler_found = True
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

        if not compiler_found:
            logger.error("✗ No C/C++ compiler found")
            logger.error("Install build tools:")
            logger.error("  macOS: xcode-select --install")
            logger.error("  Ubuntu/Debian: sudo apt install build-essential")
            logger.error("  Fedora/CentOS: sudo dnf groupinstall 'Development Tools'")
            return False

        # Check Node.js/npm (needed for grammar generation)
        try:
            subprocess.run(["node", "--version"], check=True,
                         capture_output=True, text=True)
            subprocess.run(["npm", "--version"], check=True,
                         capture_output=True, text=True)
            logger.info("✓ Node.js and npm are available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("⚠ Node.js/npm not found - some grammars may fail to build")
            logger.warning("  Install from: https://nodejs.org/")

        return True

    def get_languages_to_build(self) -> Dict[str, dict]:
        """Get the filtered list of languages to build."""
        languages = LANGUAGE_GRAMMARS.copy()

        # Filter by subset if specified
        if self.languages_subset:
            languages = {k: v for k, v in languages.items()
                        if k in self.languages_subset}

        # Filter by priority if specified
        if self.priority_filter:
            languages = {k: v for k, v in languages.items()
                        if v.get('priority', 'medium') == self.priority_filter}

        return languages

    def clone_or_update_grammar(self, name: str, config: dict) -> bool:
        """Clone or update a single grammar repository."""
        lang_path = GRAMMARS_DIR / name
        url = config['url']

        try:
            if lang_path.is_dir():
                logger.info(f"Updating {name}...")
                result = subprocess.run(
                    ["git", "pull", "--quiet"],
                    cwd=lang_path,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode != 0:
                    logger.warning(f"Failed to update {name}: {result.stderr}")
                    return False
            else:
                logger.info(f"Cloning {name}...")
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", "--quiet", url, str(lang_path)],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode != 0:
                    logger.error(f"Failed to clone {name}: {result.stderr}")
                    return False

            return True

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout while processing {name}")
            return False
        except Exception as e:
            logger.error(f"Error processing {name}: {e}")
            return False

    def generate_parser_if_needed(self, name: str, lang_path: Path) -> bool:
        """Generate parser.c if the grammar requires it."""
        # Check all common source locations for an existing parser file
        src_locations = ["src", "php/src", "php_only/src", "."]
        for loc in src_locations:
            if list((lang_path / loc).glob("parser.c")):
                # Found it, no need to generate
                return True

        # If not found, and it's a language that requires generation, then run the build steps.
        if name in REQUIRES_GENERATION or LANGUAGE_GRAMMARS[name].get('requires_generate'):
            logger.info(f"Generating parser for {name}...")

            if not (lang_path / "package.json").exists():
                logger.warning(f"Generation required for {name} but no package.json found. Skipping generation.")
                return False

            try:
                # Use npm ci for faster, more reliable installs if a lock file exists
                install_command = ["npm", "ci"] if (lang_path / "package-lock.json").exists() else ["npm", "install"]

                logger.info(f"Installing dependencies for {name} using '{' '.join(install_command)}'...")
                # We need to run with shell=True on Windows for npm.cmd, and it's generally safer for npm scripts.
                result = subprocess.run(
                    install_command,
                    cwd=lang_path,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=300, # Increased timeout for slow installs
                    shell=sys.platform == 'win32'
                )
                if result.stdout: logger.debug(f"npm install stdout for {name}: {result.stdout}")
                if result.stderr: logger.warning(f"npm install stderr for {name}: {result.stderr}")

                # After installation, re-check for the generated parser file. Some packages generate on install.
                for loc in src_locations:
                    if list((lang_path / loc).glob("parser.c")):
                        logger.info(f"✓ Parser for {name} was generated during npm install.")
                        return True

                # If it wasn't generated during install, try the `generate` command explicitly.
                logger.info(f"Attempting explicit 'tree-sitter generate' for {name}...")
                subprocess.run(
                    ["npx", "tree-sitter", "generate"],
                    cwd=lang_path,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    shell=sys.platform == 'win32'
                )

                # Final check
                for loc in src_locations:
                    if list((lang_path / loc).glob("parser.c")):
                        logger.info(f"✓ Successfully generated parser for {name}")
                        return True

                logger.error(f"Failed to generate parser.c for {name} after all steps.")
                return False

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to generate parser for {name}. Command: '{e.cmd}'. Stderr: {e.stderr}. Stdout: {e.stdout}")
                return False
            except subprocess.TimeoutExpired:
                logger.error(f"Timeout generating parser for {name}")
                return False

        # Grammar doesn't require generation, but we didn't find the file.
        logger.warning(f"No parser.c found for {name}, and it is not marked for generation.")
        return False

    def clone_grammars_parallel(self, languages: Dict[str, dict]) -> List[str]:
        """Clone/update all grammars in parallel for speed."""
        logger.info(f"Processing {len(languages)} language grammars...")
        GRAMMARS_DIR.mkdir(parents=True, exist_ok=True)

        successful_langs = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_lang = {
                executor.submit(self.clone_or_update_grammar, name, config): name
                for name, config in languages.items()
            }

            for future in as_completed(future_to_lang):
                lang_name = future_to_lang[future]
                try:
                    if future.result():
                        # Check if parser generation is needed
                        lang_path = GRAMMARS_DIR / lang_name
                        if self.generate_parser_if_needed(lang_name, lang_path):
                            successful_langs.append(lang_name)
                        else:
                            logger.warning(f"Skipping {lang_name} - parser generation failed")
                            self.skipped_languages.append(lang_name)
                    else:
                        self.failed_languages.append(lang_name)
                except Exception as e:
                    logger.error(f"Unexpected error with {lang_name}: {e}")
                    self.failed_languages.append(lang_name)

        return successful_langs

    def build_library(self, successful_languages: List[str]) -> bool:
        """Build the shared library from successfully cloned grammars."""
        if not successful_languages:
            logger.error("No languages available to build")
            return False

        logger.info(f"Building library with {len(successful_languages)} languages...")
        BUILD_DIR.mkdir(parents=True, exist_ok=True)

        # Prepare grammar paths, handling special cases
        grammar_paths = []
        for lang in successful_languages:
            lang_path = GRAMMARS_DIR / lang
            config = LANGUAGE_GRAMMARS[lang]

            # Handle languages with subdirectories (like TypeScript)
            if 'subdirs' in config:
                for subdir in config['subdirs']:
                    subdir_path = lang_path / subdir
                    if subdir_path.exists():
                        grammar_paths.append(str(subdir_path))
            else:
                grammar_paths.append(str(lang_path))

        # Remove any paths that don't exist
        grammar_paths = [p for p in grammar_paths if Path(p).exists()]

        if not grammar_paths:
            logger.error("No valid grammar paths found")
            return False

        # Try to build with all grammars first
        logger.info(f"Compiling {len(grammar_paths)} grammars into {LIBRARY_PATH}")

        try:
            Language.build_library(str(LIBRARY_PATH), grammar_paths)
            logger.info("✓ Successfully built language library")

            # Verify the library was created and has reasonable size
            if LIBRARY_PATH.exists():
                size_mb = LIBRARY_PATH.stat().st_size / (1024 * 1024)
                logger.info(f"Library size: {size_mb:.1f} MB")
                return True
            else:
                logger.error("Library file was not created")
                return False

        except Exception as e:
            logger.error(f"Failed to build library: {e}")

            # If it's a C++ compilation error and Vue is in the mix, try without Vue
            if "scanner.cc" in str(e) and any("vue" in p for p in grammar_paths):
                logger.warning("Retrying build without Vue grammar due to C++ compilation issue...")

                # Remove Vue from successful languages and grammar paths
                if 'vue' in successful_languages:
                    successful_languages.remove('vue')
                    self.skipped_languages.append('vue')

                grammar_paths = [p for p in grammar_paths if "vue" not in p]

                if grammar_paths:
                    try:
                        Language.build_library(str(LIBRARY_PATH), grammar_paths)
                        logger.info("✓ Successfully built language library (without Vue)")

                        if LIBRARY_PATH.exists():
                            size_mb = LIBRARY_PATH.stat().st_size / (1024 * 1024)
                            logger.info(f"Library size: {size_mb:.1f} MB")
                            return True
                    except Exception as e2:
                        logger.error(f"Second build attempt failed: {e2}")
                        return False

            logger.error("This usually indicates a missing C/C++ compiler or development headers")
            return False

    def generate_language_mapping(self, successful_languages: List[str]) -> None:
        """Generate a language mapping file for the application."""
        mapping_file = BUILD_DIR / "language_mapping.json"

        mapping = {}
        for lang in successful_languages:
            config = LANGUAGE_GRAMMARS[lang]
            for ext in config.get('extensions', []):
                mapping[ext] = lang

        try:
            with open(mapping_file, 'w') as f:
                json.dump(mapping, f, indent=2, sort_keys=True)
            logger.info(f"Generated language mapping: {mapping_file}")
        except Exception as e:
            logger.warning(f"Could not generate language mapping: {e}")

    def print_summary(self, successful_languages: List[str]) -> None:
        """Print build summary."""
        logger.info("=" * 60)
        logger.info("BUILD SUMMARY")
        logger.info("=" * 60)

        if successful_languages:
            logger.info(f"✓ Successfully built {len(successful_languages)} languages:")
            for lang in sorted(successful_languages):
                extensions = LANGUAGE_GRAMMARS[lang].get('extensions', [])
                logger.info(f"  {lang:<12} -> {', '.join(extensions)}")

        if self.skipped_languages:
            logger.warning(f"⚠ Skipped {len(self.skipped_languages)} languages (generation failed):")
            for lang in sorted(self.skipped_languages):
                logger.warning(f"  {lang}")

        if self.failed_languages:
            logger.warning(f"✗ Failed to build {len(self.failed_languages)} languages:")
            for lang in sorted(self.failed_languages):
                logger.warning(f"  {lang}")

        logger.info("=" * 60)
        logger.info("Ready to start your server with: ./run_server.sh")

    def build(self) -> bool:
        """Main build process."""
        if not self.check_prerequisites():
            return False

        languages = self.get_languages_to_build()
        if not languages:
            logger.error("No languages selected for building")
            return False

        successful_languages = self.clone_grammars_parallel(languages)

        if not successful_languages:
            logger.error("No languages were successfully downloaded")
            return False

        if not self.build_library(successful_languages):
            return False

        self.generate_language_mapping(successful_languages)
        self.print_summary(successful_languages)

        return True

def main():
    """Main entry point with CLI argument support."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build Tree-sitter parsers for Lumiere Core"
    )
    parser.add_argument(
        '--languages',
        nargs='+',
        help='Specific languages to build (default: all)'
    )
    parser.add_argument(
        '--priority',
        choices=['high', 'medium', 'low'],
        help='Build only languages with specific priority'
    )
    parser.add_argument(
        '--list-languages',
        action='store_true',
        help='List all available languages and exit'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean existing grammars and build directories'
    )
    parser.add_argument(
        '--skip-problematic',
        action='store_true',
        help='Skip languages that commonly fail (PHP, Haskell, etc.)'
    )

    args = parser.parse_args()

    if args.list_languages:
        print("Available languages:")
        for name, config in sorted(LANGUAGE_GRAMMARS.items()):
            priority = config.get('priority', 'medium')
            extensions = ', '.join(config.get('extensions', []))
            needs_gen = " [requires generation]" if config.get('requires_generate') or name in REQUIRES_GENERATION else ""
            print(f"  {name:<12} [{priority:<6}] -> {extensions}{needs_gen}")
        return

    if args.clean:
        logger.info("Cleaning existing directories...")
        for path in [GRAMMARS_DIR, BUILD_DIR]:
            if path.exists():
                shutil.rmtree(path)
                logger.info(f"Removed {path}")

    # Handle skip-problematic flag
    languages_to_skip = set()
    if args.skip_problematic:
        languages_to_skip = REQUIRES_GENERATION.copy()
        logger.info(f"Skipping problematic languages: {', '.join(sorted(languages_to_skip))}")

    # Filter out skipped languages
    if languages_to_skip and not args.languages:
        for lang in languages_to_skip:
            if lang in LANGUAGE_GRAMMARS:
                del LANGUAGE_GRAMMARS[lang]

    builder = ParserBuilder(
        languages_subset=args.languages,
        priority_filter=args.priority
    )

    success = builder.build()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
