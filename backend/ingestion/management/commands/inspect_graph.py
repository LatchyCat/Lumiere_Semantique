# backend/ingestion/management/commands/inspect_graph.py

import json
from pathlib import Path
from collections import defaultdict
from django.core.management.base import BaseCommand
from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel


class Command(BaseCommand):
    help = 'Loads a Project Cortex JSON file and displays its architectural graph in a human-readable format.'

    def add_arguments(self, parser):
        parser.add_argument('cortex_file', type=str, help='The path to the Project Cortex JSON file.')

    def handle(self, *args, **options):
        console = Console()
        cortex_file_path = Path(options['cortex_file'])

        if not cortex_file_path.exists():
            console.print(f"[bold red]Error: File not found at '{cortex_file_path}'[/bold red]")
            return

        data = self._load_json(console, cortex_file_path)
        if data is None:
            return

        graph_data = data.get('architectural_graph')
        if not graph_data:
            self._print_graph_not_found(console)
            return

        self._display_graph(console, data['repo_id'], graph_data)

    def _load_json(self, console: Console, path: Path):
        try:
            console.print(f"🔎 Reading Cortex file: [cyan]{path}[/cyan]")
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            console.print(f"[bold red]Error: Invalid JSON in file '{path}'[/bold red]")
            return None

    def _print_graph_not_found(self, console: Console):
        console.print(
            Panel(
                "This Project Cortex file was generated before 'The Cartographer' was implemented or the project contains no Python files.\nNo architectural graph is available to display.",
                title="[yellow]Architectural Graph Not Found[/yellow]",
                border_style="yellow",
                expand=False
            )
        )

    def _display_graph(self, console: Console, repo_id: str, graph_data: dict):
        console.print("\n[bold magenta]--- 🗺️ Cartographer's Architectural Graph ---[/bold magenta]")

        nodes = graph_data.get('nodes', {})
        edges = graph_data.get('edges', [])
        edges_by_source = defaultdict(list)
        for edge in edges:
            edges_by_source[edge['source']].append(edge)

        tree = Tree(f"[bold blue]Project: {repo_id}[/bold blue]")
        file_tree_nodes = {}

        # First pass: Build file, class, and function structure
        for node_id, node_data in sorted(nodes.items()):
            if node_data.get('type') == 'file':
                file_branch = tree.add(f"📄 [bold green]{node_id}[/bold green]")
                file_tree_nodes[node_id] = file_branch

                for class_name in sorted(node_data.get('classes', [])):
                    class_id = f"{node_id}::{class_name}"
                    class_branch = file_branch.add(f"📦 [cyan]class[/cyan] {class_name}")
                    for method_name in sorted(nodes.get(class_id, {}).get('methods', [])):
                        class_branch.add(f"  - 🐍 [dim]def[/dim] {method_name}()")

                for func_name in sorted(node_data.get('functions', [])):
                    file_branch.add(f"🐍 [dim]def[/dim] {func_name}()")

        # Second pass: Add import/call edges
        for source_id, edge_list in edges_by_source.items():
            if source_id in file_tree_nodes:
                parent_branch = file_tree_nodes[source_id]
                for edge in edge_list:
                    target = edge.get('target', 'Unknown')
                    edge_type = edge.get('type', 'RELATES_TO')
                    if edge_type == 'IMPORTS':
                        parent_branch.add(f"📥 [dim]imports[/dim] [yellow]{target}[/yellow]")
                    elif edge_type == 'CALLS':
                        parent_branch.add(f"📞 [dim]calls[/dim] [magenta]{target}[/magenta]")

        console.print(tree)
        console.print("\n[bold magenta]------------------------------------[/bold magenta]")
