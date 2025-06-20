# In ingestion/management/commands/run_crawler.py

import json
import traceback # <-- Import the traceback module
from django.core.management.base import BaseCommand
from ingestion.crawler import IntelligentCrawler
from ingestion.jsonifier import Jsonifier

class Command(BaseCommand):
    help = 'Clones a Git repository, creates the Project Cortex JSON, and saves it.'

    def add_arguments(self, parser):
        parser.add_argument('repo_url', type=str, help='The URL of the Git repository to clone.')

    def handle(self, *args, **options):
        repo_url = options['repo_url']
        repo_id = repo_url.replace("https://github.com/", "").replace("/", "_")

        self.stdout.write(self.style.NOTICE(f'Starting process for {repo_id} ({repo_url})...'))

        crawler = IntelligentCrawler(repo_url=repo_url)
        try:
            files_to_process = crawler.clone_and_process()

            if files_to_process:
                self.stdout.write(self.style.SUCCESS(f'\nFound {len(files_to_process)} files. Starting JSON-ification...'))

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

            else:
                self.stdout.write(self.style.WARNING('No files found to process or an error occurred.'))

        except Exception as e:
            # --- THIS PART IS NOW BETTER ---
            self.stdout.write(self.style.ERROR(f'\nAn unexpected error occurred: {e}'))
            self.stdout.write(self.style.ERROR('--- Full Traceback ---'))
            # Print the full traceback to the console
            traceback.print_exc()
            self.stdout.write(self.style.ERROR('--- End Traceback ---'))
        finally:
            crawler.cleanup()
