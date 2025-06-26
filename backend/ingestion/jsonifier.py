import json
import pathlib
import datetime
import ast
import re
from typing import List, Dict, TypedDict, Optional, Any, Tuple
from tqdm import tqdm

from lumiere_core.services import cartographer


class TextChunk(TypedDict):
    chunk_id: str
    chunk_text: str
    token_count: int
    chunk_type: str
    language: Optional[str]
    start_line: Optional[int]
    end_line: Optional[int]


class FileCortex(TypedDict):
    file_path: str
    file_size_kb: float
    raw_content: str
    code_smells: List[str]
    ast_summary: str
    text_chunks: List[TextChunk]
    detected_language: str
    framework_hints: List[str]


class ProjectCortex(TypedDict):
    repo_id: str
    last_crawled_utc: str
    project_health_score: float
    project_structure_tree: str
    github_metadata: Dict
    files: List[FileCortex]
    architectural_graph: Optional[Dict[str, Any]]
    language_statistics: Dict[str, Any]
    polyglot_summary: Dict[str, Any]


class PolyglotChunker:
    """
    Universal code chunker that can intelligently parse and chunk code from multiple languages.
    """

    # Language-specific patterns for different constructs
    LANGUAGE_PATTERNS = {
        'python': {
            'function': r'^\s*def\s+(\w+)\s*\(',
            'class': r'^\s*class\s+(\w+)\s*[\(:]',
            'import': r'^\s*(?:from\s+\S+\s+)?import\s+',
            'comment': r'^\s*#',
            'docstring': r'^\s*["\']{{3}',
        },
        'javascript': {
            'function': r'^\s*(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:function|\(.*\)\s*=>))',
            'class': r'^\s*class\s+(\w+)',
            'import': r'^\s*(?:import|export)',
            'comment': r'^\s*//',
            'block_comment': r'/\*.*?\*/',
        },
        'typescript': {
            'function': r'^\s*(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:function|\(.*\)\s*=>))',
            'class': r'^\s*(?:export\s+)?class\s+(\w+)',
            'interface': r'^\s*(?:export\s+)?interface\s+(\w+)',
            'type': r'^\s*(?:export\s+)?type\s+(\w+)',
            'import': r'^\s*(?:import|export)',
            'comment': r'^\s*//',
        },
        'java': {
            'function': r'^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)*(\w+)\s*\(',
            'class': r'^\s*(?:public\s+)?class\s+(\w+)',
            'interface': r'^\s*(?:public\s+)?interface\s+(\w+)',
            'import': r'^\s*import\s+',
            'comment': r'^\s*//',
        },
        'csharp': {
            'function': r'^\s*(?:public|private|protected|internal)?\s*(?:static\s+)?(?:\w+\s+)*(\w+)\s*\(',
            'class': r'^\s*(?:public\s+)?class\s+(\w+)',
            'interface': r'^\s*(?:public\s+)?interface\s+(\w+)',
            'using': r'^\s*using\s+',
            'comment': r'^\s*//',
        },
        'go': {
            'function': r'^\s*func\s+(?:\(.*\)\s+)?(\w+)\s*\(',
            'struct': r'^\s*type\s+(\w+)\s+struct',
            'interface': r'^\s*type\s+(\w+)\s+interface',
            'import': r'^\s*import\s+',
            'comment': r'^\s*//',
        },
        'rust': {
            'function': r'^\s*(?:pub\s+)?fn\s+(\w+)\s*\(',
            'struct': r'^\s*(?:pub\s+)?struct\s+(\w+)',
            'enum': r'^\s*(?:pub\s+)?enum\s+(\w+)',
            'trait': r'^\s*(?:pub\s+)?trait\s+(\w+)',
            'impl': r'^\s*impl\s+(?:<.*>\s+)?(\w+)',
            'use': r'^\s*use\s+',
            'comment': r'^\s*//',
        },
        'ruby': {
            'function': r'^\s*def\s+(\w+)',
            'class': r'^\s*class\s+(\w+)',
            'module': r'^\s*module\s+(\w+)',
            'require': r'^\s*require',
            'comment': r'^\s*#',
        },
        'php': {
            'function': r'^\s*(?:public|private|protected)?\s*function\s+(\w+)\s*\(',
            'class': r'^\s*class\s+(\w+)',
            'interface': r'^\s*interface\s+(\w+)',
            'namespace': r'^\s*namespace\s+',
            'use': r'^\s*use\s+',
            'comment': r'^\s*//',
        },
        'cpp': {
            'function': r'^\s*(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*{',
            'class': r'^\s*class\s+(\w+)',
            'struct': r'^\s*struct\s+(\w+)',
            'namespace': r'^\s*namespace\s+(\w+)',
            'include': r'^\s*#include',
            'comment': r'^\s*//',
        },
        'swift': {
            'function': r'^\s*(?:public|private|internal)?\s*func\s+(\w+)\s*\(',
            'class': r'^\s*(?:public|private|internal)?\s*class\s+(\w+)',
            'struct': r'^\s*(?:public|private|internal)?\s*struct\s+(\w+)',
            'protocol': r'^\s*(?:public|private|internal)?\s*protocol\s+(\w+)',
            'import': r'^\s*import\s+',
            'comment': r'^\s*//',
        }
    }

    @staticmethod
    def detect_language_from_extension(file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = pathlib.Path(file_path).suffix.lower()

        language_map = {
            '.py': 'python', '.pyx': 'python', '.pyi': 'python',
            '.js': 'javascript', '.mjs': 'javascript', '.cjs': 'javascript',
            '.jsx': 'javascript', '.gs': 'javascript',
            '.ts': 'typescript', '.tsx': 'typescript',
            '.java': 'java',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.cpp': 'cpp', '.cxx': 'cpp', '.cc': 'cpp', '.c': 'cpp',
            '.h': 'cpp', '.hpp': 'cpp',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.hs': 'haskell',
            '.ml': 'ocaml',
            '.ex': 'elixir',
            '.erl': 'erlang',
            '.clj': 'clojure',
            '.lua': 'lua',
            '.pl': 'perl',
            '.r': 'r',
            '.jl': 'julia',
            '.dart': 'dart',
            '.zig': 'zig',
            '.nim': 'nim',
            '.crystal': 'crystal',
        }

        return language_map.get(ext, 'unknown')

    @staticmethod
    def detect_framework_hints(content: str, language: str, file_path: str) -> List[str]:
        """Detect framework or library usage from content."""
        hints = []
        content_lower = content.lower()

        # Framework detection patterns
        framework_patterns = {
            'react': ['import react', 'from \'react\'', 'from "react"', 'usestate', 'useeffect'],
            'vue': ['vue.component', 'new vue', '@vue/', 'vue-'],
            'angular': ['@angular/', '@component', '@injectable', 'ngmodule'],
            'express': ['express()', 'app.get(', 'app.post(', 'require(\'express\')'],
            'fastapi': ['from fastapi', 'fastapi()', '@app.get', '@app.post'],
            'django': ['from django', 'django.', 'models.model', 'django.conf'],
            'flask': ['from flask', 'flask()', '@app.route'],
            'spring': ['@controller', '@service', '@repository', '@autowired'],
            'laravel': ['illuminate\\', 'artisan', 'eloquent'],
            'rails': ['activerecord', 'actioncontroller', 'rails.application'],
            'jquery': ['$(', 'jquery', '.ready('],
            'bootstrap': ['bootstrap', 'btn-', 'col-', 'row'],
            'tailwind': ['tailwind', 'tw-', 'bg-', 'text-'],
            'material-ui': ['@mui/', '@material-ui/', 'makeStyles'],
            'styled-components': ['styled-components', 'styled.'],
            'redux': ['redux', 'createstore', 'useselector', 'usedispatch'],
            'tensorflow': ['tensorflow', 'tf.', 'keras'],
            'pytorch': ['torch', 'pytorch', 'nn.module'],
            'numpy': ['import numpy', 'np.'],
            'pandas': ['import pandas', 'pd.'],
            'unittest': ['import unittest', 'testcase'],
            'pytest': ['import pytest', '@pytest.'],
            'jest': ['describe(', 'it(', 'expect('],
        }

        for framework, patterns in framework_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                hints.append(framework)

        return hints

    @classmethod
    def chunk_by_language(cls, content: str, language: str, file_path: str) -> List[Dict[str, Any]]:
        """Intelligently chunk content based on detected programming language."""
        if language == 'python':
            return cls._chunk_python(content)
        elif language in ['javascript', 'typescript']:
            return cls._chunk_javascript_typescript(content, language)
        elif language in ['java', 'csharp', 'cpp', 'swift']:
            return cls._chunk_c_style(content, language)
        elif language == 'go':
            return cls._chunk_go(content)
        elif language == 'rust':
            return cls._chunk_rust(content)
        elif language == 'ruby':
            return cls._chunk_ruby(content)
        elif language == 'php':
            return cls._chunk_php(content)
        else:
            return cls._chunk_generic(content, language)

    @classmethod
    def _chunk_python(cls, content: str) -> List[Dict[str, Any]]:
        """Enhanced Python chunking using AST when possible."""
        chunks = []

        try:
            tree = ast.parse(content)
            lines = content.splitlines()

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunk_text = ast.get_source_segment(content, node)
                    if chunk_text:
                        chunks.append({
                            "text": chunk_text,
                            "type": "function_definition",
                            "name": node.name,
                            "start_line": node.lineno,
                            "end_line": node.end_lineno
                        })
                elif isinstance(node, ast.ClassDef):
                    chunk_text = ast.get_source_segment(content, node)
                    if chunk_text:
                        chunks.append({
                            "text": chunk_text,
                            "type": "class_definition",
                            "name": node.name,
                            "start_line": node.lineno,
                            "end_line": node.end_lineno
                        })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    chunk_text = ast.get_source_segment(content, node)
                    if chunk_text:
                        chunks.append({
                            "text": chunk_text,
                            "type": "import_statement",
                            "start_line": node.lineno,
                            "end_line": node.end_lineno
                        })

            # Fill gaps with generic chunks
            if chunks:
                chunks = cls._fill_gaps_with_generic_chunks(content, chunks)

        except SyntaxError:
            # Fallback to pattern-based chunking
            chunks = cls._chunk_by_patterns(content, 'python')

        return chunks or cls._chunk_generic(content, 'python')

    @classmethod
    def _chunk_javascript_typescript(cls, content: str, language: str) -> List[Dict[str, Any]]:
        """Chunk JavaScript/TypeScript using pattern matching."""
        return cls._chunk_by_patterns(content, language)

    @classmethod
    def _chunk_c_style(cls, content: str, language: str) -> List[Dict[str, Any]]:
        """Chunk C-style languages (Java, C#, C++, Swift)."""
        return cls._chunk_by_patterns(content, language)

    @classmethod
    def _chunk_go(cls, content: str) -> List[Dict[str, Any]]:
        """Chunk Go code."""
        return cls._chunk_by_patterns(content, 'go')

    @classmethod
    def _chunk_rust(cls, content: str) -> List[Dict[str, Any]]:
        """Chunk Rust code."""
        return cls._chunk_by_patterns(content, 'rust')

    @classmethod
    def _chunk_ruby(cls, content: str) -> List[Dict[str, Any]]:
        """Chunk Ruby code."""
        return cls._chunk_by_patterns(content, 'ruby')

    @classmethod
    def _chunk_php(cls, content: str) -> List[Dict[str, Any]]:
        """Chunk PHP code."""
        return cls._chunk_by_patterns(content, 'php')

    @classmethod
    def _chunk_by_patterns(cls, content: str, language: str) -> List[Dict[str, Any]]:
        """Generic pattern-based chunking for various languages."""
        if language not in cls.LANGUAGE_PATTERNS:
            return cls._chunk_generic(content, language)

        patterns = cls.LANGUAGE_PATTERNS[language]
        lines = content.splitlines()
        chunks = []
        current_chunk = []
        current_type = "code_block"
        current_name = None
        start_line = 1

        for i, line in enumerate(lines, 1):
            chunk_detected = False

            # Check for different construct patterns
            for construct_type, pattern in patterns.items():
                if construct_type in ['comment', 'block_comment']:
                    continue

                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    # Save previous chunk if it has meaningful content
                    if current_chunk and "\n".join(current_chunk).strip():
                        chunks.append({
                            "text": "\n".join(current_chunk),
                            "type": current_type,
                            "name": current_name,
                            "start_line": start_line,
                            "end_line": i - 1
                        })

                    # Start new chunk
                    current_chunk = [line]
                    current_type = f"{construct_type}_definition"
                    current_name = match.group(1) if match.groups() else None
                    start_line = i
                    chunk_detected = True
                    break

            if not chunk_detected:
                current_chunk.append(line)

            # For languages with braces, try to detect end of blocks
            if language in ['javascript', 'typescript', 'java', 'csharp', 'cpp', 'swift', 'go', 'rust']:
                if line.strip() == '}' and current_type.endswith('_definition'):
                    # End of current block
                    chunks.append({
                        "text": "\n".join(current_chunk),
                        "type": current_type,
                        "name": current_name,
                        "start_line": start_line,
                        "end_line": i
                    })
                    current_chunk = []
                    current_type = "code_block"
                    current_name = None
                    start_line = i + 1

        # Add remaining content as final chunk if it has meaningful content
        if current_chunk and "\n".join(current_chunk).strip():
            chunks.append({
                "text": "\n".join(current_chunk),
                "type": current_type,
                "name": current_name,
                "start_line": start_line,
                "end_line": len(lines)
            })

        return chunks

    @classmethod
    def _chunk_generic(cls, content: str, language: str) -> List[Dict[str, Any]]:
        """Fallback generic chunking for unsupported languages or when parsing fails."""
        chunks = []

        # Split by logical paragraphs (double newlines)
        paragraphs = content.split('\n\n')
        current_line = 1

        for paragraph in paragraphs:
            if paragraph.strip():
                line_count = paragraph.count('\n') + 1
                chunks.append({
                    "text": paragraph,
                    "type": "paragraph",
                    "start_line": current_line,
                    "end_line": current_line + line_count - 1
                })
                current_line += line_count + 1  # +1 for the empty line
            else:
                current_line += 1

        # If no paragraphs found, split by logical line groups
        if not chunks:
            lines = content.splitlines()
            chunk_size = min(50, max(10, len(lines) // 10))  # Adaptive chunk size

            for i in range(0, len(lines), chunk_size):
                chunk_lines = lines[i:i + chunk_size]
                chunks.append({
                    "text": "\n".join(chunk_lines),
                    "type": "line_group",
                    "start_line": i + 1,
                    "end_line": i + len(chunk_lines)
                })

        return chunks

    @classmethod
    def _fill_gaps_with_generic_chunks(cls, content: str, existing_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fill gaps between structured chunks with generic content chunks."""
        if not existing_chunks:
            return existing_chunks

        lines = content.splitlines()
        all_chunks = []
        last_end = 0

        # Sort chunks by start line
        existing_chunks.sort(key=lambda x: x.get('start_line', 0))

        for chunk in existing_chunks:
            start_line = chunk.get('start_line', 1) - 1  # Convert to 0-based

            # Add gap content if exists
            if start_line > last_end:
                gap_content = "\n".join(lines[last_end:start_line])
                if gap_content.strip():
                    all_chunks.append({
                        "text": gap_content,
                        "type": "code_block",
                        "start_line": last_end + 1,
                        "end_line": start_line
                    })

            all_chunks.append(chunk)
            last_end = chunk.get('end_line', start_line + 1)

        # Add remaining content
        if last_end < len(lines):
            remaining_content = "\n".join(lines[last_end:])
            if remaining_content.strip():
                all_chunks.append({
                    "text": remaining_content,
                    "type": "code_block",
                    "start_line": last_end + 1,
                    "end_line": len(lines)
                })

        return all_chunks


class Jsonifier:
    """
    Enhanced Jsonifier with comprehensive polyglot support.
    Reads a list of files, intelligently chunks their content based on language,
    runs the Cartographer, and builds the Project Cortex JSON object.
    """

    def __init__(self, file_paths: List[pathlib.Path], repo_root: pathlib.Path, repo_id: str):
        self.file_paths = file_paths
        self.repo_root = repo_root
        self.repo_id = repo_id
        self.chunker = PolyglotChunker()

    def _read_file_content(self, file_path: pathlib.Path) -> str:
        """Read file content with multiple encoding fallbacks."""
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']

        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
            except Exception:
                break

        # Final fallback - read as binary and decode with errors='replace'
        try:
            return file_path.read_text(encoding='utf-8', errors='replace')
        except Exception:
            return ""

    def _analyze_code_quality(self, content: str, language: str, file_path: str) -> List[str]:
        """Basic code quality analysis to detect potential issues."""
        smells = []
        lines = content.splitlines()

        # Generic code smells
        if len(lines) > 1000:
            smells.append("very_large_file")

        if len(content) > 100000:  # 100KB
            smells.append("large_file_size")

        # Language-specific analysis
        if language == 'python':
            # Python-specific smells
            if 'import *' in content:
                smells.append("wildcard_import")
            if content.count('except:') > content.count('except '):
                smells.append("bare_except")
            if len([line for line in lines if len(line.strip()) > 100]) > len(lines) * 0.1:
                smells.append("long_lines")

        elif language in ['javascript', 'typescript']:
            # JS/TS-specific smells
            if 'eval(' in content:
                smells.append("eval_usage")
            if content.count('var ') > content.count('let ') + content.count('const '):
                smells.append("var_over_let_const")

        elif language == 'java':
            # Java-specific smells
            if content.count('public class') > 1:
                smells.append("multiple_public_classes")
            if 'System.out.print' in content:
                smells.append("system_out_usage")

        # Security-related patterns
        security_patterns = [
            'password', 'secret', 'api_key', 'private_key', 'token',
            'TODO', 'FIXME', 'HACK', 'XXX'
        ]

        for pattern in security_patterns:
            if pattern.lower() in content.lower():
                smells.append(f"potential_{pattern.lower()}_exposure")

        return smells

    def _create_ast_summary(self, chunks: List[Dict[str, Any]], language: str) -> str:
        """Create a summary of the AST/structure information."""
        summary = {
            "language": language,
            "total_chunks": len(chunks),
            "chunk_types": {},
            "named_constructs": []
        }

        for chunk in chunks:
            chunk_type = chunk.get("type", "unknown")
            summary["chunk_types"][chunk_type] = summary["chunk_types"].get(chunk_type, 0) + 1

            if chunk.get("name"):
                summary["named_constructs"].append({
                    "name": chunk["name"],
                    "type": chunk_type,
                    "line": chunk.get("start_line")
                })

        return json.dumps(summary, indent=2)

    def generate_cortex(self) -> ProjectCortex:
        """
        Enhanced cortex generation with comprehensive polyglot support.

        This method now acts as a sophisticated pre-processor for the Polyglot Cartographer:
        - Detects language for each file automatically
        - Generates intelligent chunks based on language-specific patterns
        - Provides framework detection and code quality analysis
        - Supports 50+ programming languages and frameworks
        - Maintains backward compatibility with existing systems
        """
        print("🔬 Starting enhanced polyglot analysis...")

        all_files_cortex: List[FileCortex] = []
        language_stats = {}

        # Enhanced data collection for the Polyglot Cartographer
        files_for_cartographer: Dict[str, Dict[str, Any]] = {}

        for file_path in tqdm(self.file_paths, desc="Analyzing files", unit="file"):

            content = self._read_file_content(file_path)
            relative_path_str = str(file_path.relative_to(self.repo_root))

            # Enhanced language detection
            detected_language = self.chunker.detect_language_from_extension(relative_path_str)

            # Framework detection
            framework_hints = self.chunker.detect_framework_hints(content, detected_language, relative_path_str)

            # Language-aware intelligent chunking
            raw_chunks = self.chunker.chunk_by_language(content, detected_language, relative_path_str)

            # Code quality analysis
            code_smells = self._analyze_code_quality(content, detected_language, relative_path_str)

            # Prepare data for Cartographer based on language
            if detected_language == 'python':
                try:
                    tree = ast.parse(content)
                    files_for_cartographer[relative_path_str] = {
                        "language": "python",
                        "ast": tree,
                        "content": content
                    }
                except SyntaxError:
                    files_for_cartographer[relative_path_str] = {
                        "language": "python",
                        "content": content,
                        "parse_error": True
                    }

            elif detected_language in ['javascript', 'typescript']:
                files_for_cartographer[relative_path_str] = {
                    "language": detected_language,
                    "content": content,
                    "framework_hints": framework_hints
                }

            elif detected_language in ['java', 'csharp', 'go', 'rust', 'swift', 'kotlin']:
                files_for_cartographer[relative_path_str] = {
                    "language": detected_language,
                    "content": content,
                    "chunks": raw_chunks
                }

            # Convert chunks to TextChunk format for RAG/Indexing
            text_chunks: List[TextChunk] = []
            for i, chunk_data in enumerate(raw_chunks):
                chunk_id = f"{self.repo_id}_{relative_path_str}_{i}"
                chunk_text = chunk_data["text"]

                text_chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text,
                    "token_count": len(chunk_text.split()),
                    "chunk_type": chunk_data.get("type", "unknown"),
                    "language": detected_language,
                    "start_line": chunk_data.get("start_line"),
                    "end_line": chunk_data.get("end_line")
                })

            # Update language statistics
            if detected_language not in language_stats:
                language_stats[detected_language] = {
                    "file_count": 0,
                    "total_lines": 0,
                    "total_chunks": 0,
                    "frameworks": set()
                }

            language_stats[detected_language]["file_count"] += 1
            language_stats[detected_language]["total_lines"] += len(content.splitlines())
            language_stats[detected_language]["total_chunks"] += len(text_chunks)
            language_stats[detected_language]["frameworks"].update(framework_hints)

            # Create enhanced FileCortex
            file_cortex: FileCortex = {
                "file_path": relative_path_str,
                "file_size_kb": round(file_path.stat().st_size / 1024, 2),
                "raw_content": content,
                "code_smells": code_smells,
                "ast_summary": self._create_ast_summary(raw_chunks, detected_language),
                "text_chunks": text_chunks,
                "detected_language": detected_language,
                "framework_hints": framework_hints
            }

            all_files_cortex.append(file_cortex)

        # Convert sets to lists for JSON serialization
        for lang_data in language_stats.values():
            lang_data["frameworks"] = list(lang_data["frameworks"])

        print("🗺️  Calling Polyglot Cartographer...")

        # Call the enhanced Polyglot Cartographer
        architectural_graph = None
        if files_for_cartographer:
            try:
                architectural_graph = cartographer.generate_graph(files_for_cartographer)
            except Exception as e:
                print(f"  ⚠️  Cartographer warning: {e}")
                architectural_graph = {"error": str(e), "supported_files": len(files_for_cartographer)}

        # Create polyglot summary
        polyglot_summary = {
            "total_languages": len(language_stats),
            "primary_language": max(language_stats.items(), key=lambda x: x[1]["file_count"])[0] if language_stats else "unknown",
            "language_distribution": {lang: data["file_count"] for lang, data in language_stats.items()},
            "detected_frameworks": list(set().union(*[data["frameworks"] for data in language_stats.values()])),
            "total_files_analyzed": len(all_files_cortex),
            "total_chunks_generated": sum(len(f["text_chunks"]) for f in all_files_cortex),
            "analysis_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        print(f"✅ Analysis complete! Detected {len(language_stats)} languages across {len(all_files_cortex)} files")

        # Assemble the enhanced Project Cortex object
        return {
            "repo_id": self.repo_id,
            "last_crawled_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "project_health_score": self._calculate_health_score(all_files_cortex, language_stats),
            "project_structure_tree": self._generate_structure_tree(),
            "github_metadata": {},
            "files": all_files_cortex,
            "architectural_graph": architectural_graph,
            "language_statistics": language_stats,
            "polyglot_summary": polyglot_summary,
        }

    def _calculate_health_score(self, files: List[FileCortex], language_stats: Dict) -> float:
        """Calculate a basic project health score based on various metrics."""
        if not files:
            return 0.0

        total_score = 0.0
        factors = 0

        # Factor 1: Code smell density (lower is better)
        total_smells = sum(len(f["code_smells"]) for f in files)
        smell_density = total_smells / len(files)
        smell_score = max(0.0, 1.0 - (smell_density / 10))  # Normalize to 0-1
        total_score += smell_score
        factors += 1

        # Factor 2: Language diversity (moderate diversity is good)
        language_count = len(language_stats)
        if language_count == 1:
            diversity_score = 0.8  # Single language is good
        elif language_count <= 5:
            diversity_score = 1.0  # Moderate diversity is excellent
        else:
            diversity_score = max(0.5, 1.0 - (language_count - 5) * 0.1)  # Too many languages might indicate complexity
        total_score += diversity_score
        factors += 1

        # Factor 3: Documentation presence
        doc_files = [f for f in files if any(keyword in f["file_path"].lower()
                     for keyword in ["readme", "doc", "changelog", "contributing"])]
        doc_score = min(1.0, len(doc_files) / 3)  # Up to 3 doc files gives full score
        total_score += doc_score
        factors += 1

        # Factor 4: Test presence
        test_files = [f for f in files if any(keyword in f["file_path"].lower()
                      for keyword in ["test", "spec", "__test__", ".test.", ".spec."])]
        test_ratio = len(test_files) / len(files)
        test_score = min(1.0, test_ratio * 5)  # 20% test files gives full score
        total_score += test_score
        factors += 1

        return round(total_score / factors, 2)

    def _generate_structure_tree(self) -> str:
        """Generate a simple project structure tree."""
        # This is a simplified version - could be enhanced to show actual directory structure
        paths = [str(fp.relative_to(self.repo_root)) for fp in self.file_paths]
        paths.sort()

        tree_lines = []
        for path in paths[:20]:  # Limit to first 20 files for brevity
            depth = path.count('/')
            indent = "  " * depth
            filename = path.split('/')[-1]
            tree_lines.append(f"{indent}├── {filename}")

        if len(paths) > 20:
            tree_lines.append(f"  ... and {len(paths) - 20} more files")

        return "\n".join(tree_lines)
