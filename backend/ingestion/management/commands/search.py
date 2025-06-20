# In ingestion/management/commands/search.py

from django.core.management.base import BaseCommand
from lumiere_core.services.ollama import search_index # <-- Import our new function

class Command(BaseCommand):
    help = 'Searches a Faiss index for a given query string.'

    def add_arguments(self, parser):
        parser.add_argument('repo_id', type=str, help="The ID of the repo (e.g., 'pallets_flask').")
        parser.add_argument('query', type=str, help='The search query string.')
        parser.add_argument('--model', type=str, default='snowflake-arctic-embed2:latest', help='The Ollama model to use.')
        parser.add_argument('--k', type=int, default=5, help='The number of results to return.')

    def handle(self, *args, **options):
        repo_id = options['repo_id']
        query = options['query']
        model = options['model']
        k = options['k']

        index_path = f"{repo_id}_faiss.index"
        map_path = f"{repo_id}_id_map.json"

        self.stdout.write(self.style.NOTICE(f"Searching for '{query}'..."))

        try:
            results = search_index(
                query_text=query,
                model_name=model,
                index_path=index_path,
                map_path=map_path,
                k=k
            )

            self.stdout.write(self.style.SUCCESS(f"\n--- Top {len(results)} search results ---"))
            for i, res in enumerate(results):
                self.stdout.write(self.style.HTTP_INFO(f"\n{i+1}. File: {res['file_path']} (Distance: {res['distance']:.4f})"))
                self.stdout.write(f"Chunk ID: {res['chunk_id']}")
                self.stdout.write("---")
                # Print the first few lines of the text chunk
                content_preview = "\n".join(res['text'].splitlines()[:5])
                self.stdout.write(content_preview)
                self.stdout.write("...")

        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f"Could not find index files for '{repo_id}'. Please run the indexer first."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"An error occurred: {e}"))
