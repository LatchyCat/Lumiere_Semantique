# In ingestion/indexing.py

import json
import numpy as np
import faiss
from lumiere_core.services.ollama import get_ollama_embeddings # <-- Import our new service

class EmbeddingIndexer:
    """
    Loads a Project Cortex JSON, generates embeddings via Ollama,
    and saves the Faiss index and the ID-to-chunk mapping.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.dimension = None # We will determine this from the first embedding

    def process_cortex(self, cortex_file_path: str):
        """
        Main method to load cortex, create embeddings, and build the index.
        """
        print(f"Loading Project Cortex from: {cortex_file_path}")
        with open(cortex_file_path, 'r', encoding='utf-8') as f:
            project_cortex = json.load(f)

        # 1. Collect all text chunks and their IDs
        all_chunks_text = []
        all_chunk_ids = []
        id_to_chunk_map = {}

        for file_data in project_cortex['files']:
            for chunk in file_data['text_chunks']:
                all_chunks_text.append(chunk['chunk_text'])
                chunk_id = chunk['chunk_id']
                all_chunk_ids.append(chunk_id)
                id_to_chunk_map[chunk_id] = {
                    "text": chunk['chunk_text'],
                    "file_path": file_data['file_path']
                }

        if not all_chunks_text:
            print("No text chunks found. Exiting.")
            return

        print(f"Found {len(all_chunks_text)} text chunks to embed using Ollama model '{self.model_name}'.")

        # 2. Generate embeddings using our Ollama service
        embeddings_list = get_ollama_embeddings(all_chunks_text, model_name=self.model_name)

        # Determine the embedding dimension from the first result
        self.dimension = len(embeddings_list[0])
        print(f"Ollama model '{self.model_name}' produced embeddings with dimension: {self.dimension}")

        embeddings = np.array(embeddings_list).astype('float32')

        # 3. Create and populate the Faiss index
        print("Creating Faiss index...")
        index = faiss.IndexFlatL2(self.dimension)
        faiss_id_to_chunk_id = all_chunk_ids

        index.add(embeddings)
        print(f"Faiss index created. Total vectors in index: {index.ntotal}")

        # 4. Save the artifacts
        repo_id = project_cortex['repo_id']
        index_filename = f"{repo_id}_faiss.index"
        map_filename = f"{repo_id}_id_map.json"

        print(f"Saving Faiss index to: {index_filename}")
        faiss.write_index(index, index_filename)

        print(f"Saving ID-to-Chunk mapping to: {map_filename}")
        save_data = {
            "faiss_id_to_chunk_id": faiss_id_to_chunk_id,
            "chunk_id_to_data": id_to_chunk_map
        }
        with open(map_filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2)

        print("Indexing complete.")
