# In ingestion/management/commands/run_indexer.py

from django.core.management.base import BaseCommand
from ingestion.indexing import EmbeddingIndexer
import os

class Command(BaseCommand):
    help = 'Loads a Project Cortex JSON file and creates a Faiss index from its text chunks using Ollama.'

    def add_arguments(self, parser):
        parser.add_argument('cortex_file', type=str, help='The path to the Project Cortex JSON file.')
        parser.add_argument(
            '--model',
            type=str,
            default='snowflake-arctic-embed2:latest', # <-- Defaults to your preferred model
            help='The name of the Ollama embedding model to use.'
        )

    def handle(self, *args, **options):
        cortex_file_path = options['cortex_file']
        model_name = options['model']

        if not os.path.exists(cortex_file_path):
            self.stdout.write(self.style.ERROR(f"Error: File not found at '{cortex_file_path}'"))
            return

        self.stdout.write(self.style.NOTICE(f"Starting Ollama indexing for {cortex_file_path} using model '{model_name}'..."))

        try:
            # Pass the model name to the indexer
            indexer = EmbeddingIndexer(model_name=model_name)
            indexer.process_cortex(cortex_file_path)
            self.stdout.write(self.style.SUCCESS('✓ Ollama indexing process completed successfully.'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'An unexpected error occurred during indexing: {e}'))
