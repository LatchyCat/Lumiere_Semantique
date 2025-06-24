# backend/ingestion/management/commands/ingest_repo.py

from django.core.management.base import BaseCommand
from rich.console import Console
from rich.panel import Panel
import traceback

from lumiere_core.services import ingestion_service

class Command(BaseCommand):
    help = 'Runs the full ingestion pipeline (clone, embed, index) for a single repository URL.'

    def add_arguments(self, parser):
        parser.add_argument('repo_url', type=str, help='The full URL of the GitHub repository to ingest.')
        parser.add_argument('--embedding_model', type=str, default='snowflake-arctic-embed2:latest', help='The Ollama model to use for embeddings.')

    def handle(self, *args, **options):
        console = Console()
        repo_url = options['repo_url']
        embedding_model = options['embedding_model']

        console.print(
            Panel(
                f"[bold]Starting full ingestion for:[/] [cyan]{repo_url}[/cyan]\n"
                f"[bold]Using embedding model:[/] [yellow]{embedding_model}[/yellow]",
                title="🚀 Lumière Ingestion Service",
                border_style="blue"
            )
        )

        try:
            result = ingestion_service.clone_and_embed_repository(
                repo_url=repo_url,
                embedding_model=embedding_model
            )

            if result.get("status") == "success":
                console.print(
                    Panel(
                        f"[bold green]✓ Success![/bold green]\n{result.get('message', 'Ingestion complete.')}",
                        title="✅ Mission Complete",
                        border_style="green"
                    )
                )
                repo_id = repo_url.replace("https://github.com/", "").replace("/", "_")

                # --- THE FIX: Update the output path to match the new structure ---
                cortex_file_path = f"backend/cloned_repositories/{repo_id}/{repo_id}_cortex.json"
                console.print(f"\n[dim]To inspect the graph, run:[/dim]\n[bold cyan]python backend/manage.py inspect_graph {cortex_file_path}[/bold cyan]")

            else:
                error_details = result.get('details', result.get('error', 'An unknown error occurred.'))
                console.print(
                    Panel(
                        f"[bold red]✗ Ingestion Failed[/bold red]\n\n[yellow]Reason:[/yellow] {error_details}",
                        title="🚨 Error",
                        border_style="red"
                    )
                )

        except Exception as e:
            console.print(
                Panel(
                    f"[bold red]An unexpected critical error occurred:[/bold red]\n\n{traceback.format_exc()}",
                    title="💥 Critical Failure",
                    border_style="red"
                )
            )
