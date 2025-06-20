# In ingestion/jsonifier.py

import json
import pathlib
import datetime
import ast  # <-- Import Python's built-in AST module
from typing import List, Dict, TypedDict

# --- Define the Project Cortex Data Structure (no changes here) ---
class TextChunk(TypedDict):
    chunk_id: str
    chunk_text: str
    token_count: int
    chunk_type: str

class FileCortex(TypedDict):
    file_path: str
    file_size_kb: int
    raw_content: str
    code_smells: List[str]
    ast_summary: str
    text_chunks: List[TextChunk]

class ProjectCortex(TypedDict):
    repo_id: str
    last_crawled_utc: str
    project_health_score: float
    project_structure_tree: str
    github_metadata: Dict
    files: List[FileCortex]

# --- New AST-based Chunker ---
# This "Visitor" pattern is the standard way to walk an AST.
class CodeChunker(ast.NodeVisitor):
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.chunks = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """This method is called for every function definition."""
        chunk_text = ast.get_source_segment(self.source_code, node)
        if chunk_text:
            self.chunks.append({"text": chunk_text, "type": "function_definition"})
        # We stop descending here to treat the whole function as one chunk
        # To chunk recursively, call self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """This method is called for every class definition."""
        chunk_text = ast.get_source_segment(self.source_code, node)
        if chunk_text:
            self.chunks.append({"text": chunk_text, "type": "class_definition"})

# --- The Jsonifier Class ---
class Jsonifier:
    """
    Reads a list of files, chunks their content using Python's 'ast' module,
    and builds the Project Cortex JSON object.
    """
    def __init__(self, file_paths: List[pathlib.Path], repo_root: pathlib.Path, repo_id: str):
        self.file_paths = file_paths
        self.repo_root = repo_root
        self.repo_id = repo_id
        # No parser setup needed!

    def _read_file_content(self, file_path: pathlib.Path) -> str:
        try:
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            return file_path.read_text(encoding='latin-1', errors='replace')

    def _chunk_python_file(self, content: str) -> List[dict]:
        """Intelligently chunks Python code using the AST."""
        try:
            tree = ast.parse(content)
            chunker = CodeChunker(content)
            chunker.visit(tree)
            return chunker.chunks
        except SyntaxError:
            # If the file isn't valid Python, fallback to line-by-line chunking
            return [{"text": line, "type": "line"} for line in content.splitlines() if line.strip()]

    def generate_cortex(self) -> ProjectCortex:
        all_files_cortex: List[FileCortex] = []

        for file_path in self.file_paths:
            content = self._read_file_content(file_path)

            raw_chunks = []
            # Only use the AST parser for Python files
            if file_path.suffix == '.py':
                raw_chunks = self._chunk_python_file(content)
            else:
                # For non-Python files (.md, .txt, etc.), just split by paragraph
                raw_chunks = [{"text": chunk, "type": "paragraph"} for chunk in content.split('\n\n') if chunk.strip()]

            text_chunks: List[TextChunk] = []
            relative_path_str = str(file_path.relative_to(self.repo_root))
            for i, chunk_data in enumerate(raw_chunks):
                chunk_id = f"{self.repo_id}_{relative_path_str}_{i}"
                chunk_text = chunk_data['text']
                text_chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text,
                    "token_count": len(chunk_text.split()),
                    "chunk_type": chunk_data['type'],
                })

            file_cortex: FileCortex = {
                "file_path": relative_path_str,
                "file_size_kb": round(file_path.stat().st_size / 1024, 2),
                "raw_content": content,
                "code_smells": [],
                "ast_summary": "{}",
                "text_chunks": text_chunks
            }
            all_files_cortex.append(file_cortex)

        project_cortex: ProjectCortex = {
            "repo_id": self.repo_id,
            "last_crawled_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "project_health_score": 0.0,
            "project_structure_tree": "...",
            "github_metadata": {},
            "files": all_files_cortex
        }

        return project_cortex
