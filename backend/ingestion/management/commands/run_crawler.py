# In backend/ingestion/management/commands/run_crawler.py

import json
import traceback
from django.core.management.base import BaseCommand
from ingestion.crawler import IntelligentCrawler
from ingestion.jsonifier import Jsonifier

class Command(BaseCommand):
    help = 'Clones a Git repository, creates the Project Cortex JSON, and saves it.'

    def add_arguments(self, parser):
        parser.add_argument('repo_url', type=str, help='The URL of the Git repository to clone.')

    def handle(self, *args, **options):
        repo_url = options['repo_url']
        # Generate the repo_id just like the API does.
        repo_id = repo_url.replace("https://github.com/", "").replace("/", "_")

        self.stdout.write(self.style.NOTICE(f'Starting process for {repo_id} ({repo_url})...'))

        try:
            # --- FIX: Use the IntelligentCrawler as a context manager ---
            # The `with` statement correctly handles the setup (cloning) and
            # teardown (cleanup) of the temporary repository directory.
            with IntelligentCrawler(repo_url=repo_url) as crawler:
                # The cloning is now handled automatically when the 'with' block is entered.
                # We simply need to get the list of files to process.
                files_to_process = crawler.get_file_paths()

                if files_to_process:
                    self.stdout.write(self.style.SUCCESS(f'\nFound {len(files_to_process)} files. Starting JSON-ification...'))

                    # We now correctly pass the crawler's repo_path attribute.
                    jsonifier = Jsonifier(
                        file_paths=files_to_process,
                        repo_root=crawler.repo_path,
                        repo_id=repo_id
                    )
                    project_cortex = jsonifier.generate_cortex()

                    output_filename = f"{repo_id}_cortex.json"
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        json.dump(project_cortex, f, indent=2)

                    self.stdout.write(self.style.SUCCESS(f'✓ Project Cortex created successfully: {output_filename}'))
                    self.stdout.write(self.style.NOTICE(f"\nNext Step: Run the indexer command:"))
                    self.stdout.write(self.style.SUCCESS(f"python manage.py run_indexer {output_filename}"))


                else:
                    self.stdout.write(self.style.WARNING('No files found to process or an error occurred.'))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'\nAn unexpected error occurred: {e}'))
            self.stdout.write(self.style.ERROR('--- Full Traceback ---'))
            traceback.print_exc()
            self.stdout.write(self.style.ERROR('--- End Traceback ---'))
        # NOTE: No explicit crawler.cleanup() is needed here because the
        # `with` statement guarantees cleanup even if errors occur.
