import json
import pathlib
import datetime
import ast
from typing import List, Dict, TypedDict, Optional, Any

from lumiere_core.services import cartographer


class TextChunk(TypedDict):
    chunk_id: str
    chunk_text: str
    token_count: int
    chunk_type: str


class FileCortex(TypedDict):
    file_path: str
    file_size_kb: float
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
    architectural_graph: Optional[Dict[str, Any]]


class CodeChunker(ast.NodeVisitor):
    """
    AST visitor that extracts function and class definitions as code chunks.
    """
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.chunks = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        chunk_text = ast.get_source_segment(self.source_code, node)
        if chunk_text:
            self.chunks.append({"text": chunk_text, "type": "function_definition"})

    def visit_ClassDef(self, node: ast.ClassDef):
        chunk_text = ast.get_source_segment(self.source_code, node)
        if chunk_text:
            self.chunks.append({"text": chunk_text, "type": "class_definition"})


class Jsonifier:
    """
    Reads a list of files, chunks their content, runs the Cartographer,
    and builds the Project Cortex JSON object.
    """
    def __init__(self, file_paths: List[pathlib.Path], repo_root: pathlib.Path, repo_id: str):
        self.file_paths = file_paths
        self.repo_root = repo_root
        self.repo_id = repo_id

    def _read_file_content(self, file_path: pathlib.Path) -> str:
        try:
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            return file_path.read_text(encoding='latin-1', errors='replace')

    def _chunk_python_file(self, content: str) -> List[Dict[str, str]]:
        try:
            tree = ast.parse(content)
            chunker = CodeChunker(content)
            chunker.visit(tree)
            return chunker.chunks
        except SyntaxError:
            return [{"text": line, "type": "line"} for line in content.splitlines() if line.strip()]


    def generate_cortex(self) -> ProjectCortex:
        """
        Processes all repository files to generate the Project Cortex JSON.

        This method now acts as a pre-processor for the Polyglot Cartographer:
        - For Python files (.py), it generates a standard Python AST.
        - For JavaScript files (.js, .gs), it notes them for JS parsing.
        - For all files, it generates text chunks for vector indexing (RAG).
        - It then calls the Cartographer with the collected ASTs and file content.
        """
        all_files_cortex: List[FileCortex] = []

        # This dictionary will hold analysis data (ASTs, content) for ALL supported languages,
        # ready to be passed to the Polyglot Cartographer.
        files_for_cartographer: Dict[str, Dict[str, Any]] = {}

        for file_path in self.file_paths:
            content = self._read_file_content(file_path)
            relative_path_str = str(file_path.relative_to(self.repo_root))
            file_ext = file_path.suffix

            raw_chunks = []
            if file_ext == '.py':
                try:
                    tree = ast.parse(content)
                    # 1. Prepare data for the Cartographer (Python AST)
                    files_for_cartographer[relative_path_str] = {"language": "python", "ast": tree}
                    # 2. Prepare data for RAG/Indexing (detailed chunks)
                    chunker = CodeChunker(content)
                    chunker.visit(tree)
                    raw_chunks = chunker.chunks
                except SyntaxError:
                    # Fallback for RAG if Python parsing fails
                    raw_chunks = [{"text": line, "type": "line"} for line in content.splitlines() if line.strip()]

            # NEW: Handle JavaScript files for the Polyglot Cartographer
            elif file_ext in ['.js', '.gs', '.mjs']:
                # 1. Prepare data for the Cartographer (raw JS content)
                files_for_cartographer[relative_path_str] = {"language": "javascript", "content": content}
                # 2. Prepare data for RAG/Indexing (simple paragraph chunks)
                raw_chunks = [{"text": chunk, "type": "paragraph"} for chunk in content.split('\n\n') if chunk.strip()]

            else:
                # For all other file types, just do simple paragraph chunking for RAG.
                # The Cartographer will ignore these.
                raw_chunks = [{"text": chunk, "type": "paragraph"} for chunk in content.split('\n\n') if chunk.strip()]

            # This part remains the same: process whatever chunks were generated above.
            text_chunks: List[TextChunk] = []
            for i, chunk_data in enumerate(raw_chunks):
                chunk_id = f"{self.repo_id}_{relative_path_str}_{i}"
                chunk_text = chunk_data["text"]
                text_chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text,
                    "token_count": len(chunk_text.split()),
                    "chunk_type": chunk_data["type"],
                })

            file_cortex: FileCortex = {
                "file_path": relative_path_str,
                "file_size_kb": round(file_path.stat().st_size / 1024, 2),
                "raw_content": content,
                "code_smells": [],
                "ast_summary": json.dumps({}),
                "text_chunks": text_chunks,
            }

            all_files_cortex.append(file_cortex)

        # Call the Polyglot Cartographer only if there are supported files to analyze.
        architectural_graph = None
        if files_for_cartographer:
            architectural_graph = cartographer.generate_graph(files_for_cartographer)

        # Assemble the final Project Cortex object.
        return {
            "repo_id": self.repo_id,
            "last_crawled_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "project_health_score": 0.0,
            "project_structure_tree": "...",
            "github_metadata": {},
            "files": all_files_cortex,
            "architectural_graph": architectural_graph,
        }
