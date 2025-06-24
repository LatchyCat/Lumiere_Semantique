# backend/ingestion/indexing.py

import json
import numpy as np
import faiss
from pathlib import Path
from lumiere_core.services.ollama import get_ollama_embeddings
from tqdm import tqdm

class EmbeddingIndexer:
    """
    Loads a Project Cortex JSON, generates embeddings via Ollama,
    and saves the Faiss index and the ID-to-chunk mapping into the SAME
    directory as the source cortex file.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.dimension = None

    def _extract_repo_id_from_cortex(self, project_cortex: dict, cortex_path_obj: Path) -> str:
        """
        Extract repo_id from cortex with fallback strategies for backward compatibility.

        Args:
            project_cortex: The loaded cortex JSON data
            cortex_path_obj: Path object of the cortex file

        Returns:
            A valid repo_id string
        """
        # Primary: Try to get repo_id from the cortex data
        if 'repo_id' in project_cortex and project_cortex['repo_id']:
            return project_cortex['repo_id']

        # Fallback 1: Extract from cortex filename (assumes format: {repo_id}_cortex.json)
        cortex_filename = cortex_path_obj.stem  # Gets filename without extension
        if cortex_filename.endswith('_cortex'):
            potential_repo_id = cortex_filename[:-7]  # Remove '_cortex' suffix
            if potential_repo_id:
                print(f"Warning: repo_id not found in cortex data, using filename-derived ID: {potential_repo_id}")
                return potential_repo_id

        # Fallback 2: Use the parent directory name
        parent_dir_name = cortex_path_obj.parent.name
        if parent_dir_name and parent_dir_name != 'cloned_repositories':
            print(f"Warning: repo_id not found, using parent directory name: {parent_dir_name}")
            return parent_dir_name

        # Fallback 3: Generate from cortex filename
        fallback_id = cortex_filename.replace('_cortex', '') if '_cortex' in cortex_filename else cortex_filename
        print(f"Warning: Using fallback repo_id derived from filename: {fallback_id}")
        return fallback_id or 'unknown_repo'

    def process_cortex(self, cortex_file_path: str):
        """
        Main method to load cortex, create embeddings, and build the index.
        """
        cortex_path_obj = Path(cortex_file_path)

        # Validate that the cortex file exists
        if not cortex_path_obj.exists():
            raise FileNotFoundError(f"Cortex file not found: {cortex_path_obj}")

        print(f"Loading Project Cortex from: {cortex_path_obj}")

        # --- THIS IS THE KEY FIX ---
        # Derive the output directory from the location of the cortex file.
        output_dir = cortex_path_obj.parent

        try:
            with open(cortex_path_obj, 'r', encoding='utf-8') as f:
                project_cortex = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in cortex file {cortex_path_obj}: {e}")
        except Exception as e:
            raise IOError(f"Error reading cortex file {cortex_path_obj}: {e}")

        # 1. Collect all text chunks and their IDs
        all_chunks_text = []
        all_chunk_ids = []
        id_to_chunk_map = {}

        files_data = project_cortex.get('files', [])
        if not files_data:
            print("Warning: No 'files' array found in cortex data.")
            return

        for file_data in files_data:
            if not isinstance(file_data, dict):
                print(f"Warning: Skipping invalid file data entry: {file_data}")
                continue

            file_path = file_data.get('file_path', 'unknown_file')
            text_chunks = file_data.get('text_chunks', [])

            for chunk in text_chunks:
                if not isinstance(chunk, dict):
                    print(f"Warning: Skipping invalid chunk in {file_path}: {chunk}")
                    continue

                chunk_text = chunk.get('chunk_text', '')
                chunk_id = chunk.get('chunk_id')

                if not chunk_text or not chunk_id:
                    print(f"Warning: Skipping chunk with missing text or ID in {file_path}")
                    continue

                all_chunks_text.append(chunk_text)
                all_chunk_ids.append(chunk_id)
                id_to_chunk_map[chunk_id] = {
                    "text": chunk_text,
                    "file_path": file_path
                }

        if not all_chunks_text:
            print("No valid text chunks found. Exiting.")
            return

        print(f"Found {len(all_chunks_text)} text chunks to embed using Ollama model '{self.model_name}'.")

        # 2. Generate embeddings using our Ollama service
        try:
            embeddings_list = get_ollama_embeddings(all_chunks_text, model_name=self.model_name)
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return

        if not embeddings_list:
            print("Embedding generation failed. Exiting.")
            return

        # Validate embeddings
        if len(embeddings_list) != len(all_chunks_text):
            print(f"Error: Mismatch between chunks ({len(all_chunks_text)}) and embeddings ({len(embeddings_list)})")
            return

        self.dimension = len(embeddings_list[0])
        print(f"Ollama model '{self.model_name}' produced embeddings with dimension: {self.dimension}")

        embeddings = np.array(embeddings_list).astype('float32')

        # 3. Create and populate the Faiss index
        print("Creating Faiss index...")
        try:
            index = faiss.IndexFlatL2(self.dimension)
            index.add(embeddings)
            print(f"Faiss index created. Total vectors in index: {index.ntotal}")
        except Exception as e:
            print(f"Error creating Faiss index: {e}")
            return

        # 4. Save the artifacts
        # Use the enhanced repo_id extraction method
        repo_id = self._extract_repo_id_from_cortex(project_cortex, cortex_path_obj)

        # Use the 'output_dir' to save files in the correct location
        index_filename = output_dir / f"{repo_id}_faiss.index"
        map_filename = output_dir / f"{repo_id}_id_map.json"

        print(f"Saving Faiss index to: {index_filename}")
        try:
            faiss.write_index(index, str(index_filename))
        except Exception as e:
            print(f"Error saving Faiss index: {e}")
            return

        print(f"Saving ID-to-Chunk mapping to: {map_filename}")
        save_data = {
            "faiss_id_to_chunk_id": all_chunk_ids,
            "chunk_id_to_data": id_to_chunk_map
        }

        try:
            with open(map_filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2)
        except Exception as e:
            print(f"Error saving ID mapping: {e}")
            return

        print("Indexing complete.")
