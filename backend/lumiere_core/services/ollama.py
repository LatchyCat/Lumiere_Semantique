# In lumiere_core/services/ollama.py

import ollama
from tqdm import tqdm
from typing import List
import faiss
import numpy as np
import json
from pathlib import Path # <--- ADD THIS

def get_ollama_embeddings(chunks: List[str], model_name: str) -> List[List[float]]:
    """
    Generates embeddings for a list of text chunks using a local Ollama model.

    Args:
        chunks: A list of strings to be embedded.
        model_name: The name of the Ollama model to use (e.g., 'snowflake-arctic-embed').

    Returns:
        A list of embeddings, where each embedding is a list of floats.
    """
    embeddings = []
    # The ollama client automatically connects to http://localhost:11434
    client = ollama.Client()

    # Show a progress bar because this can take time
    for text in tqdm(chunks, desc="Generating Ollama Embeddings"):
        response = client.embeddings(model=model_name, prompt=text)
        embeddings.append(response['embedding'])

    return embeddings

def search_index(
    query_text: str,
    model_name: str,
    repo_id: str, # <--- MODIFIED: Take repo_id directly
    k: int = 10,
    **kwargs # <--- MODIFIED: Accept and ignore old path args for compatibility
) -> List[dict]:
    """
    Searches the Faiss index for the top k most similar chunks to a query for a given repo_id.

    Args:
        query_text: The user's search query.
        model_name: The name of the Ollama model used to create the index.
        repo_id: The unique ID of the repository whose index should be searched.
        k: The number of results to return.

    Returns:
        A list of dictionaries, where each dictionary contains the chunk_id,
        file_path, and the original text of the matching chunk.
    """
    # --- THIS IS THE FIX ---
    # Centralize path construction based on repo_id.
    backend_dir = Path(__file__).resolve().parent.parent.parent
    artifacts_dir = backend_dir / "cloned_repositories" / repo_id
    index_path = artifacts_dir / f"{repo_id}_faiss.index"
    map_path = artifacts_dir / f"{repo_id}_id_map.json"

    print(f"Loading index '{index_path}' and map '{map_path}'...")
    # Load the Faiss index
    index = faiss.read_index(str(index_path))

    # Load the ID mapping files
    with open(map_path, 'r', encoding='utf-8') as f:
        id_maps = json.load(f)
    faiss_id_to_chunk_id = id_maps['faiss_id_to_chunk_id']
    chunk_id_to_data = id_maps['chunk_id_to_data']

    print(f"Generating embedding for query: '{query_text}'...")
    # 1. Embed the query using the same Ollama model
    client = ollama.Client()
    response = client.embeddings(model=model_name, prompt=query_text)
    query_vector = np.array([response['embedding']]).astype('float32')

    print(f"Searching index for top {k} results...")
    # 2. Search the Faiss index
    distances, indices = index.search(query_vector, k)

    # 3. Retrieve the results
    results = []
    for i in range(min(k, len(indices[0]))): # Ensure we don't go out of bounds
        faiss_id = indices[0][i]
        chunk_id = faiss_id_to_chunk_id[faiss_id]
        chunk_data = chunk_id_to_data[chunk_id]

        results.append({
            "chunk_id": chunk_id,
            "file_path": chunk_data['file_path'],
            "text": chunk_data['text'],
            "distance": float(distances[0][i])
        })

    return results
