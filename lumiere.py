# In /Users/latchy/lumiere_semantique/lumiere.py

import typer
import requests
import sys
import re
import shlex
import difflib
import traceback
from typing import Optional, List, Dict, Tuple
from collections import defaultdict  # <--- ADDED: For easier grouping of models
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.markdown import Markdown
from rich.text import Text
from rich.status import Status
from rich.live import Live
from rich.align import Align
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.shortcuts import confirm
from prompt_toolkit.styles import Style
from pathlib import Path
import json
import time
from datetime import datetime

# --- Global Objects & Configuration ---
console = Console()
history_path = Path.home() / ".lumiere" / "history.txt"
config_path = Path.home() / ".lumiere" / "config.json"
history_path.parent.mkdir(parents=True, exist_ok=True)

# --- NEW: Centralized API URL ---
API_BASE_URL = "http://127.0.0.1:8002/api/v1"

# Create command completers for better UX
main_commands = ['analyze', 'a', 'profile', 'p', 'config', 'c', 'help', 'h', 'exit', 'x', 'quit']
analysis_commands = ['list', 'l', 'fix', 'f', 'briefing', 'b', 'rca', 'r', 'details', 'd', 'help', 'h', 'back', 'exit', 'quit']

main_completer = WordCompleter(main_commands, ignore_case=True)
analysis_completer = WordCompleter(analysis_commands, ignore_case=True)

# --- ADDED: Style for prompt_toolkit prompt to match rich colors ---
prompt_style = Style.from_dict({
    'lumiere': 'bold #00ffff',  # bold cyan
    'provider': 'yellow',
    'separator': 'white'
})

prompt_session = PromptSession(
    history=FileHistory(str(history_path)),
    completer=main_completer,
    style=prompt_style
)

# --- Global CLI State ---
cli_state = {
    "model": None,  # Will be populated from config
    "available_models": [],
    "last_repo_url": None,
    "debug_mode": False,
}

def load_config():
    """Load configuration from file if it exists."""
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                cli_state.update(config)
                console.print(f"[dim]✓ Configuration loaded from {config_path}[/dim]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load config: {e}[/yellow]")

def save_config():
    """Save current configuration to file."""
    try:
        with open(config_path, 'w') as f:
            json.dump({
                "model": cli_state["model"],
                "last_repo_url": cli_state["last_repo_url"],
                "debug_mode": cli_state["debug_mode"]
            }, f, indent=2)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not save config: {e}[/yellow]")

def validate_github_url(url: str) -> bool:
    """Validate that the URL is a proper GitHub repository URL."""
    github_pattern = r'^https://github\.com/[\w\-\.]+/[\w\-\.]+/?$'
    return bool(re.match(github_pattern, url))

def format_url(url: str) -> str:
    """Normalize GitHub URL format."""
    url = url.strip().rstrip('/')
    if not url.startswith('https://'):
        if url.startswith('github.com/'):
            url = 'https://' + url
        elif '/' in url and not url.startswith('http'):
            url = 'https://github.com/' + url
    return url

# --- Enhanced API Client ---
class LumiereAPIClient:
    def __init__(self, base_url: str = API_BASE_URL, timeout: int = 600):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()  # Reuse connections

    def _request(self, method: str, endpoint: str, **kwargs):
        try:
            if method.upper() in ["POST"]:
                data = kwargs.get("json", {})
                if "model" not in data:
                    # Ensure a model is selected before making a request
                    if not cli_state.get("model"):
                        console.print("\n[bold red]Error: No LLM model selected.[/bold red]")
                        console.print("Please use the [bold cyan]config[/bold cyan] command to choose a model first.")
                        return None # Abort the request
                    data["model"] = cli_state["model"]
                kwargs["json"] = data

            url = f"{self.base_url}/{endpoint}"

            if cli_state["debug_mode"]:
                console.print(f"[dim]DEBUG: {method} {url}[/dim]")
                if "json" in kwargs:
                    console.print(f"[dim]DEBUG: Payload: {kwargs['json']}[/dim]")

            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError as e:
            console.print(Panel(
                f"[bold red]Cannot connect to Lumière backend[/bold red]\n"
                f"[yellow]Expected URL:[/yellow] {self.base_url}\n"
                f"[yellow]Error:[/yellow] {str(e)}\n\n"
                f"[dim]💡 Make sure the backend server is running:\n"
                f"   • Execute `cd backend && ./run_server.sh`\n"
                f"   • Verify the URL is correct\n"
                f"   • Check firewall settings[/dim]",
                title="[red]Connection Error[/red]",
                border_style="red"
            ))
            return None

        except requests.exceptions.HTTPError as e:
            try:
                error_json = e.response.json()
                error_details = error_json.get('details', str(error_json))
                if "traceback" in error_json and cli_state["debug_mode"]:
                    console.print(Panel(error_json['traceback'], title="[bold red]Server Traceback[/bold red]", border_style="red"))
                console.print(Panel(
                    f"[bold red]API Request Failed[/bold red]\n"
                    f"[yellow]URL:[/yellow] {e.request.url}\n"
                    f"[yellow]Status:[/yellow] {e.response.status_code}\n"
                    f"[yellow]Error:[/yellow] {error_details}",
                    title="[red]API Error[/red]"
                ))
            except:
                console.print(Panel(
                    f"[bold red]HTTP Error {e.response.status_code}[/bold red]\n"
                    f"[yellow]URL:[/yellow] {e.request.url}\n"
                    f"[yellow]Response:[/yellow] {e.response.text[:200]}...",
                    title="[red]API Error[/red]"
                ))
            return None

        except requests.exceptions.Timeout:
            console.print(Panel(
                f"[bold red]Request Timeout[/bold red]\n"
                f"The request took longer than {self.timeout} seconds.\n"
                f"[dim]The backend might be processing a large repository.[/dim]",
                title="[red]Timeout Error[/red]"
            ))
            return None

        except requests.exceptions.RequestException as e:
            console.print(Panel(
                f"[bold red]Request Error[/bold red]\n"
                f"[yellow]Error:[/yellow] {str(e)}",
                title="[red]Request Error[/red]"
            ))
            return None

    def health_check(self) -> bool:
        """Check if the backend is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def list_models(self): return self._request("GET", "models/list/")
    def get_analysis(self, repo_url: str): return self._request("POST", "strategist/prioritize/", json={"repo_url": repo_url})
    def get_briefing(self, issue_url: str): return self._request("POST", "briefing/", json={"issue_url": issue_url})
    def get_rca(self, repo_url: str, bug_description: str, target_file: str): return self._request("POST", "rca/", json={"repo_url": repo_url, "bug_description": bug_description, "target_file": target_file})
    def get_profile(self, username: str): return self._request("POST", "profile/review/", json={"username": username})
    def generate_fix(self, repo_id: str, target_file: str, instruction: str, refinement_history: Optional[List[Dict]] = None): return self._request("POST", "scaffold/", json={"repo_id": repo_id, "target_file": target_file, "instruction": instruction, "refinement_history": refinement_history or []})
    def create_pr(self, issue_url: str, target_file: str, fixed_code: str): return self._request("POST", "ambassador/dispatch/", json={"issue_url": issue_url, "target_file": target_file, "fixed_code": fixed_code})
    def get_diplomat_report(self, issue_title: str, issue_body: str): return self._request("POST", "diplomat/find-similar-issues/", json={"issue_title": issue_title, "issue_body": issue_body})
    def validate_in_crucible(self, repo_url: str, target_file: str, modified_code: str): return self._request("POST", "crucible/validate/", json={"repo_url": repo_url, "target_file": target_file, "modified_code": modified_code})


# --- MODIFIED: Implemented a two-step provider/model selection process. ---
def handle_model_selection(api: "LumiereAPIClient"):
    """
    Guides the user through a two-step process: first selecting an LLM provider,
    then selecting a model from that provider.
    """
    console.print("\n[bold cyan]🤖 LLM Provider & Model Selection[/bold cyan]")
    with Status("[cyan]Fetching available models from backend...[/cyan]"):
        available_models_data = api.list_models()

    if not available_models_data:
        # Error is printed by the API client
        return

    cli_state["available_models"] = available_models_data

    # Step 1: Group models by provider
    providers = defaultdict(list)
    for model in available_models_data:
        provider = model.get('provider', 'unknown').capitalize()
        providers[provider].append(model)

    if not providers:
        console.print("[red]No providers with available models found.[/red]")
        return

    provider_names = list(providers.keys())

    try:
        # Step 2: Prompt for the provider
        console.print("\n[bold]First, select a provider:[/bold]")
        provider_table = Table(show_header=False, box=None, padding=(0, 2))
        for i, name in enumerate(provider_names, 1):
            provider_table.add_row(f"[cyan]({i})[/cyan]", name)
        console.print(provider_table)

        # Determine default provider choice
        current_provider = ""
        if cli_state.get("model"):
            current_provider = cli_state.get("model").split('/')[0].capitalize()

        default_provider_idx = "1"
        if current_provider in provider_names:
            default_provider_idx = str(provider_names.index(current_provider) + 1)

        provider_choice_str = Prompt.ask(
            "Enter your provider choice",
            choices=[str(i) for i in range(1, len(provider_names) + 1)],
            show_choices=False,
            default=default_provider_idx
        )
        selected_provider_name = provider_names[int(provider_choice_str) - 1]

        # Step 3: Prompt for a model from the selected provider
        models_for_provider = providers[selected_provider_name]

        model_table = Table(title=f"Available Models from [yellow]{selected_provider_name}[/yellow]", border_style="blue")
        model_table.add_column("Choice #", style="dim", justify="center")
        model_table.add_column("Model Name", style="white")
        model_table.add_column("Model ID", style="cyan")

        model_choices_for_prompt = []
        for i, model in enumerate(models_for_provider, 1):
            model_choices_for_prompt.append(str(i))
            model_table.add_row(
                str(i),
                model.get('name', 'N/A'),
                model.get('id', 'N/A')
            )
        console.print(model_table)

        # Find default selection if a model from this provider is already selected
        current_model_index_str = "1"
        current_model_id = cli_state.get("model")
        if current_model_id and current_model_id.startswith(selected_provider_name.lower()):
            for i, model in enumerate(models_for_provider):
                if model['id'] == current_model_id:
                    current_model_index_str = str(i + 1)
                    break

        model_choice_str = Prompt.ask(
            "[bold]Select a model number to use[/bold]",
            choices=model_choices_for_prompt,
            show_choices=False,
            default=current_model_index_str
        )
        selected_model_index = int(model_choice_str) - 1
        selected_model_id = models_for_provider[selected_model_index]['id']

        cli_state["model"] = selected_model_id
        save_config()
        console.print(f"✅ Model set to [bold green]{cli_state['model']}[/bold green]. This will be saved for future sessions.")

    except (ValueError, IndexError):
        console.print("[red]❌ Invalid selection.[/red]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Model selection cancelled.[/yellow]")

# --- MODIFIED: Dynamic prompt text based on selected model/provider ---
def get_prompt_text() -> List[Tuple[str, str]]:
    """
    Builds a prompt_toolkit-compatible formatted text list for the prompt.
    Dynamically displays the provider or model name.
    """
    display_name = "Choose Provider"  # Default text when no model is selected
    model_id = cli_state.get("model")

    if model_id:
        parts = model_id.split('/', 1)
        if len(parts) == 2:
            provider, model_name = parts
            if provider == "ollama":
                display_name = model_name  # Show specific model name for ollama
            else:
                display_name = provider.capitalize()  # Show provider name for others (e.g., Gemini)
        else:
            display_name = model_id  # Fallback for malformed ID

    return [
        ('class:lumiere', 'Lumière'),
        ('class:provider', f' ({display_name})'),
        ('class:separator', ' > '),
    ]

# --- Enhanced Analysis Session Manager ---
class AnalysisSession:
    def __init__(self, repo_url: str):
        self.repo_url = format_url(repo_url)
        if not validate_github_url(self.repo_url):
            raise ValueError(f"Invalid GitHub URL: {self.repo_url}")

        self.repo_id = self.repo_url.replace("https://github.com/", "").replace("/", "_")
        self.issues = []
        self.api = LumiereAPIClient()
        cli_state["last_repo_url"] = self.repo_url
        save_config()

    def start(self) -> bool:
        """Initialize the analysis session."""
        console.print(Panel(
            f"[bold cyan]🔍 Analysis Session Starting[/bold cyan]\n"
            f"[yellow]Repository:[/yellow] {self.repo_url}\n"
            f"[yellow]Model:[/yellow] {cli_state['model']}",
            border_style="cyan"
        ))

        with Status("[cyan]Checking backend connection...") as status:
            if not self.api.health_check():
                return False # Error already printed by client
            status.update("[green]✓ Backend connection established")
            time.sleep(0.5)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            transient=True
        ) as progress:
            task = progress.add_task("[green]🤖 Contacting The Strategist...", total=100)
            strategist_data = self.api.get_analysis(self.repo_url)
            if not strategist_data:
                return False
            progress.update(task, completed=100)

        self.issues = strategist_data.get("prioritized_issues", [])
        if not self.issues:
            console.print("[yellow]📭 No open issues found in this repository.[/yellow]")
            return False

        console.print(f"✨ The Strategist identified [bold green]{len(self.issues)}[/bold green] open issues for analysis.")
        self.display_issue_table()
        return True

    def display_issue_table(self):
        """Display a formatted table of prioritized issues."""
        table = Table(
            title="[bold blue]🎯 Prioritized Issue Triage[/bold blue]",
            show_header=True,
            header_style="bold magenta",
            border_style="blue"
        )
        table.add_column("Rank", style="dim", justify="center", width=6)
        table.add_column("Score", style="bold", justify="center", width=8)
        table.add_column("Issue #", style="green", justify="center", width=10)
        table.add_column("Title", style="white", no_wrap=False)

        for issue in self.issues:
            score = issue.get('score', 0)
            score_style = "red" if score >= 90 else "yellow" if score >= 70 else "white"
            score_emoji = "🔥" if score >= 90 else "⚡" if score >= 70 else "📝"

            table.add_row(
                f"#{issue['rank']}",
                f"{score_emoji} [{score_style}]{score}[/{score_style}]",
                f"#{issue['number']}",
                issue['title'][:80] + "..." if len(issue['title']) > 80 else issue['title']
            )
        console.print(table)

    def loop(self):
        """Main interactive loop for the analysis session."""
        display_interactive_help('analyze')

        global prompt_session
        prompt_session = PromptSession(
            history=FileHistory(str(history_path)),
            completer=analysis_completer,
            style=prompt_style
        )

        while True:
            try:
                prompt_text = get_prompt_text()
                command_str = prompt_session.prompt(prompt_text).strip()

                if not command_str:
                    continue

                if command_str.lower() in ("q", "quit", "exit", "back"):
                    break

                self.handle_analysis_command(command_str)

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' or 'back' to return to main menu.[/yellow]")
                continue
            except EOFError:
                break

        console.print("[cyan]📊 Analysis session ended.[/cyan]")

        # Restore main prompt session
        prompt_session = PromptSession(
            history=FileHistory(str(history_path)),
            completer=main_completer,
            style=prompt_style
        )

    def handle_analysis_command(self, command_str: str):
        """Handle commands within the analysis session."""
        try:
            parts = shlex.split(command_str.lower())
        except ValueError:
            console.print("[red]❌ Invalid command syntax.[/red]")
            return

        command = parts[0] if parts else ""
        args = parts[1:] if len(parts) > 1 else []

        if command in ("h", "help"):
            display_interactive_help('analyze')
            return

        if command in ("l", "list"):
            self.display_issue_table()
            return

        if command not in ('f', 'fix', 'b', 'briefing', 'r', 'rca', 'd', 'details'):
            console.print("[red]❌ Unknown command. Type 'help' for available commands.[/red]")
            return

        issue_num_str = None
        if args and args[0].isdigit():
            issue_num_str = args[0]
        else:
            try:
                issue_num_str = Prompt.ask(f"Which issue # for '[cyan]{command}[/cyan]'?").strip()
            except KeyboardInterrupt:
                console.print("\n[yellow]Command cancelled.[/yellow]")
                return

        if not issue_num_str or not issue_num_str.isdigit():
            console.print("[red]❌ Please enter a valid issue number.[/red]")
            return

        target_issue = next((iss for iss in self.issues if iss.get('number') == int(issue_num_str)), None)
        if not target_issue:
            console.print(f"[red]❌ Issue #{issue_num_str} not found in the prioritized list.[/red]")
            console.print("[dim]💡 Use 'list' to see available issues.[/dim]")
            return

        self.execute_action(command, target_issue)
        console.print("\n[dim]💡 Type [bold]list[/bold] to see issues, or [bold]help[/bold] for commands.[/dim]")

    def execute_action(self, command: str, issue: Dict):
        """Execute the specified action on an issue."""
        if command in ("f", "fix"):
            self.handle_fix_dialogue(issue)
            return

        if command in ("r", "rca"):
            self.handle_rca_command(issue)
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            if command in ("b", "briefing"):
                task = progress.add_task(f"[cyan]📋 Getting briefing for issue #{issue['number']}...", total=None)
                briefing_data = self.api.get_briefing(f"{self.repo_url}/issues/{issue['number']}")
                progress.remove_task(task)

                if briefing_data and briefing_data.get("briefing"):
                    console.print(Panel(
                        Markdown(briefing_data["briefing"]),
                        title=f"[bold blue]📋 Issue Briefing #{issue['number']}[/bold blue]",
                        border_style="blue"
                    ))
                else:
                    console.print("[red]❌ Could not retrieve briefing.[/red]")

            elif command in ("d", "details"):
                issue_url = f"{self.repo_url}/issues/{issue['number']}"
                console.print(Panel(
                    f"[bold]Issue #{issue['number']}[/bold]\n"
                    f"[yellow]Title:[/yellow] {issue['title']}\n"
                    f"[yellow]Priority Score:[/yellow] {issue['score']}/100\n"
                    f"[yellow]URL:[/yellow] [link={issue_url}]{issue_url}[/link]\n"
                    f"[yellow]Description:[/yellow] {issue.get('description', 'No description available')[:200]}...",
                    title="[bold green]📝 Issue Details[/bold green]",
                    border_style="green"
                ))

    def handle_rca_command(self, issue: Dict):
        """Handle root cause analysis command."""
        console.print(f"[cyan]🔍 Starting Root Cause Analysis for issue #{issue['number']}[/cyan]")

        issue_desc = issue.get('description', '')
        file_match = re.search(r'in `([\w./\\-]+\.py)`', issue_desc)

        if file_match:
            target_file = file_match.group(1)
            console.print(f"[dim]✓ Auto-detected file: [yellow]{target_file}[/yellow][/dim]")
        else:
            try:
                target_file = Prompt.ask("Enter the target file path for analysis")
                if not target_file.strip():
                    console.print("[yellow]❌ Root cause analysis cancelled.[/yellow]")
                    return
            except KeyboardInterrupt:
                console.print("\n[yellow]Root cause analysis cancelled.[/yellow]")
                return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]🕵️ Performing root cause analysis...", total=None)
            rca_data = self.api.get_rca(self.repo_url, issue.get('description', ''), target_file)
            progress.remove_task(task)

        if rca_data and rca_data.get("analysis"):
            console.print(Panel(
                Markdown(rca_data["analysis"]),
                title=f"[bold red]🕵️ Root Cause Analysis - Issue #{issue['number']}[/bold red]",
                border_style="red"
            ))
        else:
            console.print("[red]❌ Could not perform root cause analysis.[/red]")

    def _display_diff(self, original_code: str, new_code: str):
        """Display a formatted diff of code changes."""
        diff = difflib.unified_diff(
            original_code.splitlines(keepends=True),
            new_code.splitlines(keepends=True),
            fromfile='🔴 Original',
            tofile='🟢 Proposed'
        )

        diff_panel_content = Text()
        for line in diff:
            if line.startswith('+++') or line.startswith('---'):
                diff_panel_content.append(line, style="bold blue")
            elif line.startswith('+'):
                diff_panel_content.append(line, style="green")
            elif line.startswith('-'):
                diff_panel_content.append(line, style="red")
            elif line.startswith('@'):
                diff_panel_content.append(line, style="bold yellow")
            else:
                diff_panel_content.append(line, style="dim")

        console.print(Panel(
            diff_panel_content,
            title="[bold yellow]📝 Proposed Code Changes[/bold yellow]",
            expand=True,
            border_style="yellow"
        ))

    def handle_fix_dialogue(self, issue: Dict):
        """Handle the complete fix dialogue workflow."""
        console.print(Panel(
            f"[bold cyan]🤝 Socratic Dialogue[/bold cyan] starting for:\n"
            f"[bold green]Issue #{issue['number']}: {issue['title']}[/bold green]",
            border_style="cyan"
        ))

        issue_desc = issue.get('description', '')
        issue_title = issue.get('title', '')

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]🕵️ Engaging The Diplomat...", total=None)
            diplomat_report = self.api.get_diplomat_report(issue_title, issue_desc)
            progress.remove_task(task)

        if diplomat_report and diplomat_report.get("summary"):
            console.print(Panel(
                Markdown(diplomat_report["summary"]),
                title="[bold blue]🕵️ Diplomat Intelligence Briefing[/bold blue]",
                border_style="blue"
            ))

            try:
                if not Confirm.ask("\n[bold]🚀 Proceed with generating a fix?[/bold]", default=True):
                    console.print("[yellow]🛑 Operation cancelled.[/yellow]")
                    return
            except KeyboardInterrupt:
                console.print("\n[yellow]🛑 Operation cancelled.[/yellow]")
                return

        file_match = re.search(r'in `([\w./\\-]+\.py)`', issue_desc)
        if not file_match:
            try:
                target_file = Prompt.ask("Could not auto-detect target file. Please enter the file path")
                if not target_file.strip():
                    console.print("[red]❌ Cannot proceed without target file.[/red]")
                    return
            except KeyboardInterrupt:
                console.print("\n[yellow]🛑 Operation cancelled.[/yellow]")
                return
        else:
            target_file = file_match.group(1)
            console.print(f"[dim]✓ Auto-detected file: [yellow]{target_file}[/yellow][/dim]")

        instruction = f"Fix this bug in '{target_file}': {issue_title}\n\n{issue_desc}"
        current_code = ""
        original_code = ""
        refinement_history = []
        iteration_count = 0
        max_iterations = 5

        while iteration_count < max_iterations:
            iteration_count += 1
            console.print(f"\n[dim]🔄 Iteration {iteration_count}/{max_iterations}[/dim]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True
            ) as progress:
                task = progress.add_task(f"[cyan]⚡ Generating code fix...", total=None)
                fix_data = self.api.generate_fix(self.repo_id, target_file, instruction, refinement_history)
                progress.remove_task(task)

            if not fix_data or "generated_code" not in fix_data:
                console.print("[red]❌ Failed to generate fix.[/red]")
                return

            current_code = fix_data["generated_code"]
            if not original_code:
                original_code = fix_data.get("original_content", "")

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan][progress.description]{task.description}"),
                transient=True
            ) as progress:
                task = progress.add_task(f"🔥 Entering The Crucible...", total=None)
                validation_result = self.api.validate_in_crucible(self.repo_url, target_file, current_code)
                progress.remove_task(task)

            if not validation_result:
                console.print(Panel(
                    "🔥 [bold red]Crucible Service Error[/bold red]\n"
                    "The validation service is not available.\n"
                    "[dim]💡 Please check that Docker is running and the service is healthy.[/dim]",
                    title="[red]🔥 Crucible Report[/red]",
                    border_style="red"
                ))
                crucible_passed = False
            else:
                crucible_passed = validation_result.get("status") == "passed"
                if crucible_passed:
                    console.print(Panel(
                        "✅ [bold green]All tests passed![/bold green]\n"
                        "The proposed changes are validated and ready.",
                        title="[green]🔥 Crucible Report[/green]",
                        border_style="green"
                    ))
                else:
                    logs = validation_result.get('logs', 'No logs available')
                    console.print(Panel(
                        f"❌ [bold red]Validation Failed[/bold red]\n"
                        f"[bold]Test Results:[/bold]\n{logs}",
                        title="[red]🔥 Crucible Report[/red]",
                        border_style="red"
                    ))

            console.rule("[bold]📝 Review Proposed Changes[/bold]")
            self._display_diff(original_code, current_code)

            try:
                if crucible_passed:
                    choice = Prompt.ask(
                        "\n[bold]✅ Tests passed! Choose action:[/bold]\n"
                        "[bold green](a)[/bold green] Approve & create PR\n"
                        "[bold yellow](r)[/bold yellow] Refine with feedback\n"
                        "[bold red](c)[/bold red] Cancel",
                        choices=['a', 'r', 'c'],
                        default='a'
                    ).lower()
                else:
                    choice = Prompt.ask(
                        "\n[bold]❌ Tests failed! Choose action:[/bold]\n"
                        "[bold yellow](r)[/bold yellow] Refine with feedback\n"
                        "[bold orange3](a)[/bold orange3] Approve anyway & create PR\n"
                        "[bold red](c)[/bold red] Cancel",
                        choices=['a', 'r', 'c'],
                        default='r'
                    ).lower()
            except KeyboardInterrupt:
                console.print("\n[yellow]🛑 Operation cancelled.[/yellow]")
                break

            if choice == 'c':
                console.print("[yellow]🛑 Operation cancelled.[/yellow]")
                break

            if choice == 'a':
                if not crucible_passed:
                    try:
                        confirmed = Confirm.ask(
                            "[bold yellow]⚠️  Tests failed. Create PR anyway?[/bold yellow]",
                            default=False
                        )
                        if not confirmed:
                            console.print("[yellow]🛑 PR creation cancelled.[/yellow]")
                            continue
                    except KeyboardInterrupt:
                        console.print("\n[yellow]🛑 Operation cancelled.[/yellow]")
                        break

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True
                ) as progress:
                    task = progress.add_task("[cyan]🚀 Dispatching Ambassador...", total=None)
                    pr_data = self.api.create_pr(f"{self.repo_url}/issues/{issue['number']}", target_file, current_code)
                    progress.remove_task(task)

                if pr_data and pr_data.get("pull_request_url"):
                    pr_url = pr_data["pull_request_url"]
                    console.print(Panel(
                        f"✅ [bold green]Success![/bold green]\n"
                        f"Pull request created: [link={pr_url}]{pr_url}[/link]\n\n"
                        f"[dim]🎉 The Ambassador has successfully delivered your fix![/dim]",
                        title="[green]🚀 Mission Complete[/green]",
                        border_style="green"
                    ))
                else:
                    console.print("[red]❌ Failed to create pull request.[/red]")
                break

            if choice == 'r':
                if iteration_count >= max_iterations:
                    console.print(f"[yellow]⚠️  Maximum iterations ({max_iterations}) reached.[/yellow]")
                    break

                try:
                    feedback = Prompt.ask("\n[bold]💭 Your feedback for improvement[/bold]")
                    if not feedback.strip():
                        console.print("[yellow]⚠️  Empty feedback provided, skipping refinement.[/yellow]")
                        continue
                    refinement_history.append({
                        "feedback": feedback,
                        "code": current_code
                    })
                except KeyboardInterrupt:
                    console.print("\n[yellow]🛑 Refinement cancelled by user.[/yellow]")
                    break

# --- Utility Function for Help Display ---
def display_interactive_help(context: str = 'main'):
    """Display help instructions based on the current CLI context."""
    title = f"🆘 Lumière Help — {context.capitalize()} Context"
    help_table = Table(title=f"[bold magenta]{title}[/bold magenta]", border_style="magenta")
    help_table.add_column("Command", style="bold cyan")
    help_table.add_column("Description", style="white")

    if context == 'main':
        help_table.add_row("analyze / a", "Start analysis on a GitHub repo")
        help_table.add_row("profile / p", "Get GitHub user profile analysis")
        # --- DYNAMIC HELP TEXT ---
        if cli_state.get("model"):
            help_table.add_row("config / c", "Change LLM model or view settings")
        else:
            help_table.add_row("config / c", "Choose LLM provider & model") # <--- MODIFIED
        help_table.add_row("help / h", "Show this help menu")
        help_table.add_row("exit / quit", "Exit the application")
    elif context == 'analyze':
        help_table.add_row("list / l", "Show prioritized issues")
        help_table.add_row("briefing / b", "Show issue briefing")
        help_table.add_row("details / d", "Show issue metadata")
        help_table.add_row("rca / r", "Root cause analysis")
        help_table.add_row("fix / f", "Launch fix dialogue")
        help_table.add_row("help / h", "Show this help menu")
        help_table.add_row("back / exit / quit", "Return to main menu")

    console.print(help_table)

# --- Main Entry Point ---
app = typer.Typer()

@app.command()
def run():
    """Launch Lumière interactive shell."""
    api_client = LumiereAPIClient()
    if not api_client.health_check():
        sys.exit(1) # Error already printed by client

    load_config()

    # --- ENHANCED WELCOME PANEL ---
    welcome_text = (
        f"[bold cyan]✨ Welcome to Lumière ✨[/bold cyan]\n"
        f"AI Dev Assistant for Open Source Projects\n\n"
        f"[dim]Backend Status: [green]Online[/green] at [underline]{API_BASE_URL}[/underline]\n"
        f"Date: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}[/dim]"
    )
    console.print(Panel(welcome_text, border_style="cyan"))

    display_interactive_help('main')

    while True:
        try:
            # --- CORRECTED PROMPT HANDLING ---
            prompt_text = get_prompt_text()
            command = prompt_session.prompt(prompt_text).strip()

            if not command:
                continue

            if command.lower() in ("exit", "quit", "x"):
                console.print("[dim]👋 Goodbye![/dim]")
                break

            if command.lower() in ("help", "h"):
                display_interactive_help('main')
                continue

            if command.lower() in ("config", "c"):
                console.print(Panel(
                    f"[bold]Current Settings[/bold]\n"
                    f"  [cyan]LLM Model:[/cyan] [yellow]{cli_state.get('model', 'Not set')}[/yellow]\n"
                    f"  [cyan]Last Repo:[/cyan] {cli_state.get('last_repo_url', 'Not set')}\n"
                    f"  [cyan]Debug Mode:[/cyan] {'On' if cli_state.get('debug_mode') else 'Off'}",
                    title="⚙️ Configuration", border_style="magenta"
                ))
                handle_model_selection(api_client)
                continue

            if command.lower() in ("profile", "p"):
                if not cli_state.get("model"):
                    console.print("[bold red]Please select a model first using the 'config' command.[/bold red]")
                    continue
                username = Prompt.ask("Enter GitHub username")
                if not username.strip():
                    continue
                with Status("[cyan]Generating profile analysis...[/cyan]"):
                    profile = api_client.get_profile(username)
                if profile and profile.get("profile_summary"):
                    console.print(Panel(Markdown(profile["profile_summary"]), title=f"👤 Profile Analysis for {username}"))
                else:
                    console.print("[red]❌ Could not retrieve profile.[/red]")
                continue

            if command.lower() in ("analyze", "a"):
                if not cli_state.get("model"):
                    console.print("[bold red]Please select a model first using the 'config' command.[/bold red]")
                    continue
                repo_url = Prompt.ask("Enter GitHub repository URL").strip()
                if not repo_url: # Gracefully handle empty input
                    continue
                try:
                    session = AnalysisSession(repo_url)
                    if session.start():
                        session.loop()
                except ValueError as e:
                    console.print(f"[red]{e}[/red]")
                continue

            console.print("[red]❌ Unknown command. Type 'help' for options.[/red]")

        except KeyboardInterrupt:
            console.print("\n[dim]💤 Interrupted. Type 'exit' to quit.[/dim]")
            continue
        except EOFError:
            break

if __name__ == "__main__":
    app()
