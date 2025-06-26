# In /Users/latchy/lumiere_semantique/lumiere.py

import typer
import sys
sys.path.append('backend')
import requests
import sys
import re
import shlex
import difflib
import traceback
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
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
import textwrap
from rich.tree import Tree

# --- Global Objects & Configuration ---
console = Console()
history_path = Path.home() / ".lumiere" / "history.txt"
config_path = Path.home() / ".lumiere" / "config.json"
history_path.parent.mkdir(parents=True, exist_ok=True)

# --- Centralized API URL ---
API_BASE_URL = "http://127.0.0.1:8002/api/v1"

# Create command completers for better UX
main_commands = ['analyze', 'a', 'ask', 'oracle', 'review', 'dashboard', 'd', 'profile', 'p', 'config', 'c', 'help', 'h', 'exit', 'x', 'quit', 'list-repos', 'lr']
analysis_commands = ['list', 'l', 'fix', 'f', 'briefing', 'b', 'rca', 'r', 'details', 'd', 'graph', 'g', 'help', 'h', 'back', 'exit', 'quit']
oracle_commands = ['help', 'h', 'back', 'exit', 'quit']

main_completer = WordCompleter(main_commands, ignore_case=True)
analysis_completer = WordCompleter(analysis_commands, ignore_case=True)
oracle_completer = WordCompleter(oracle_commands, ignore_case=True)

# --- Style for prompt_toolkit prompt to match rich colors ---
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

# --- NEW UTILITY FUNCTIONS for managing analyzed repos ---

def check_if_repo_is_analyzed(repo_id: str) -> bool:
    """Checks if a specific repo_id directory has all the necessary analysis files."""
    cloned_repos_dir = Path("backend/cloned_repositories")
    repo_dir = cloned_repos_dir / repo_id
    if not repo_dir.is_dir():
        return False

    cortex_file = repo_dir / f"{repo_id}_cortex.json"
    faiss_file = repo_dir / f"{repo_id}_faiss.index"
    map_file = repo_dir / f"{repo_id}_id_map.json"

    return all([cortex_file.exists(), faiss_file.exists(), map_file.exists()])

def find_analyzed_repos() -> List[Dict[str, str]]:
    """Scans for previously analyzed repositories and validates their artifacts."""
    analyzed_repos = []
    cloned_repos_dir = Path("backend/cloned_repositories")
    if not cloned_repos_dir.is_dir():
        return []

    for repo_dir in cloned_repos_dir.iterdir():
        if repo_dir.is_dir():
            repo_id = repo_dir.name
            if check_if_repo_is_analyzed(repo_id):
                # Attempt to reconstruct a user-friendly name and the full URL
                display_name = repo_id.replace("_", "/", 1)
                full_url = f"https://github.com/{display_name}"
                analyzed_repos.append({"repo_id": repo_id, "display_name": display_name, "url": full_url})

    return sorted(analyzed_repos, key=lambda x: x['repo_id'])


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

def _insert_docstring_into_code(code: str, docstring: str) -> str:
    """Intelligently inserts a docstring into a Python code snippet."""
    # Find the first function or class definition
    match = re.search(r"^(?P<indent>\s*)(def|class)\s+\w+", code, re.MULTILINE)
    if not match:
        return code # Cannot find where to insert, return original

    indentation = match.group('indent')
    insertion_point = match.end()

    # Prepare the docstring with the correct indentation
    indented_docstring = textwrap.indent(f'"""{docstring}"""', indentation + '    ')

    # Insert the docstring
    return f"{code[:insertion_point]}\n{indented_docstring}{code[insertion_point:]}"

# --- Enhanced API Client ---
class LumiereAPIClient:
    def __init__(self, base_url: str = API_BASE_URL, timeout: int = 600):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()

    def _request(self, method: str, endpoint: str, **kwargs):
        try:
            if method.upper() in ["POST"]:
                data = kwargs.get("json", {})
                # The Task Router now handles model selection, so we don't add it here.
                # Only check for existence if it's a legacy endpoint that needs it.
                kwargs["json"] = data
            url = f"{self.base_url}/{endpoint}"
            if cli_state["debug_mode"]:
                console.print(f"[dim]DEBUG: {method} {url}[/dim]")
                if "json" in kwargs:
                    console.print(f"[dim]DEBUG: Payload: {kwargs['json']}[/dim]")
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            # Handle potential empty responses from server
            if response.status_code == 204 or not response.content:
                return {}
            return response.json()
        except requests.exceptions.ConnectionError as e:
            console.print(Panel(f"[bold red]Cannot connect to Lumière backend[/bold red]\n[yellow]Expected URL:[/yellow] {self.base_url}\n[yellow]Error:[/yellow] {str(e)}\n\n[dim]💡 Make sure the backend server is running.[/dim]", title="[red]Connection Error[/red]", border_style="red"))
            return None
        except requests.exceptions.HTTPError as e:
            try:
                error_json = e.response.json()
                error_details = error_json.get('error', str(error_json))
                console.print(Panel(f"[bold red]API Request Failed[/bold red]\n[yellow]URL:[/yellow] {e.request.url}\n[yellow]Status:[/yellow] {e.response.status_code}\n[yellow]Error:[/yellow] {error_details}", title="[red]API Error[/red]", border_style="red"))
                llm_response_for_debug = error_json.get('llm_response')
                if llm_response_for_debug:
                    console.print(Panel(Text(llm_response_for_debug, overflow="fold"), title="[bold yellow]🔍 LLM Raw Response (for debugging)[/bold yellow]", border_style="yellow", expand=False))
            except json.JSONDecodeError:
                console.print(Panel(f"[bold red]HTTP Error {e.response.status_code}[/bold red]\n[yellow]URL:[/yellow] {e.request.url}\n\n[bold]Response Text:[/bold]\n{e.response.text[:500]}...", title="[red]Non-JSON API Error[/red]", border_style="red"))
            return None
        except requests.exceptions.Timeout:
            console.print(Panel(f"The request took longer than {self.timeout} seconds.", title="[red]Timeout Error[/red]"))
            return None
        except requests.exceptions.RequestException as e:
            console.print(Panel(f"[bold red]Request Error[/bold red]\n[yellow]Error:[/yellow] {str(e)}", title="[red]Request Error[/red]"))
            return None

    def health_check(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def list_models(self): return self._request("GET", "models/list/")
    def get_analysis(self, repo_url: str): return self._request("POST", "strategist/prioritize/", json={"repo_url": repo_url})
    def get_briefing(self, issue_url: str): return self._request("POST", "briefing/", json={"issue_url": issue_url})
    def get_rca(self, repo_url: str, bug_description: str): return self._request("POST", "rca/", json={"repo_url": repo_url, "bug_description": bug_description})
    def get_profile(self, username: str): return self._request("POST", "profile/review/", json={"username": username})
    def get_graph(self, repo_id: str): return self._request("GET", f"graph/?repo_id={repo_id}")
    def generate_docstring(self, repo_id: str, code: str, instruction: str): return self._request("POST", "generate-docstring/", json={"repo_id": repo_id, "new_code": code, "instruction": instruction})
    def generate_tests(self, repo_id: str, code_to_test: str, instruction: str): return self._request("POST", "generate-tests/", json={"repo_id": repo_id, "new_code": code_to_test, "instruction": instruction})
    def generate_scaffold(self, repo_id: str, target_files: List[str], instruction: str, rca_report: str, refinement_history: Optional[List[Dict]] = None):
        payload = {"repo_id": repo_id, "target_files": target_files, "instruction": instruction, "rca_report": rca_report, "refinement_history": refinement_history or []}
        return self._request("POST", "scaffold/", json=payload)
    def create_pr(self, issue_url: str, modified_files: Dict[str, str]): return self._request("POST", "ambassador/dispatch/", json={"issue_url": issue_url, "modified_files": modified_files})
    def get_diplomat_report(self, issue_title: str, issue_body: str): return self._request("POST", "diplomat/find-similar-issues/", json={"issue_title": issue_title, "issue_body": issue_body})
    def validate_in_crucible(self, repo_url: str, target_file: str, modified_code: str): return self._request("POST", "crucible/validate/", json={"repo_url": repo_url, "target_file": target_file, "modified_code": modified_code})
    def ingest_repository(self, repo_url: str): return self._request("POST", "ingest/", json={"repo_url": repo_url})
    def ask_oracle(self, repo_id: str, question: str): return self._request("POST", "oracle/ask/", json={"repo_id": repo_id, "question": question})
    def adjudicate_pr(self, pr_url: str): return self._request("POST", "review/adjudicate/", json={"pr_url": pr_url})
    def harmonize_fix(self, pr_url: str, review_text: str): return self._request("POST", "review/harmonize/", json={"pr_url": pr_url, "review_text": review_text})
    def get_sentinel_briefing(self, repo_id: str): return self._request("GET", f"sentinel/briefing/?repo_id={repo_id}")
    # --- NEW: Method for the Mission Controller ---
    def get_next_actions(self, last_action: str, result_data: dict): return self._request("POST", "suggest-actions/", json={"last_action": last_action, "result_data": result_data})


def _present_next_actions(api_client, last_action: str, context: dict) -> Tuple[Optional[str], dict]:
    """
    Gets suggestions from the backend and presents a dynamic menu to the user.
    This is the core of the Conversational Mission Controller.

    Args:
        api_client: The instance of LumiereAPIClient.
        last_action: The command that was just executed.
        context: A dictionary containing data from the last action's result.

    Returns:
        A tuple of (command_string, context_dict) for the next action, or (None, {})
    """
    response = api_client.get_next_actions(last_action, context)
    if not response or "suggestions" not in response:
        return None, {}  # No suggestions, fall back to main loop

    suggestions = response["suggestions"]
    recommended_choice = response["recommended_choice"]

    if not suggestions:
        return None, {}

    # Build the rich prompt
    table = Table(title="[bold yellow]🚀 What's Next?[/bold yellow]", show_header=False, box=None, padding=(0, 2))
    choices = []
    choice_map = {}
    for item in suggestions:
        key = item["key"]
        text = item["text"]
        command = item["command"]
        table.add_row(f"([bold cyan]{key}[/bold cyan])", text)
        choices.append(key)
        choice_map[key] = command

    console.print(table)

    try:
        user_choice_key = Prompt.ask("Select an action", choices=choices, default=recommended_choice)
        selected_command = choice_map[user_choice_key]

        if selected_command == "back":
            return "back", {}

        # The context dictionary is passed through to the next command handler.
        return selected_command, context

    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Action cancelled.[/yellow]")
        return "back", {} # Treat cancel as going back to main menu
    except (ValueError, KeyError):
        console.print("[red]Invalid selection.[/red]")
        return "back", {}

# --- NEW: Oracle Session Manager ---
class OracleSession:
    def __init__(self, repo_url: str):
        self.repo_url = repo_url
        self.repo_id = self.repo_url.replace("https://github.com/", "").replace("/", "_")
        self.api = LumiereAPIClient()
        console.print(Panel(
            f"[bold magenta]🔮 Oracle Session Activated[/bold magenta]\n"
            f"[yellow]Repository:[/yellow] {self.repo_id}\n"
            f"[yellow]Model:[/yellow] {cli_state['model']}\n\n"
            "[dim]Ask any architectural question about the codebase. Type 'back' or 'exit' to finish.[/dim]",
            border_style="magenta"
        ))

    def loop(self):
        """Main interactive Q&A loop for The Oracle."""
        global prompt_session
        prompt_session = PromptSession(
            history=FileHistory(str(history_path)),
            completer=oracle_completer,
            style=prompt_style
        )

        while True:
            try:
                # Custom prompt for the Oracle
                oracle_prompt_text = [
                    ('class:lumiere', 'Lumière'),
                    ('class:provider', f' (Oracle/{self.repo_id})'),
                    ('class:separator', ' > '),
                ]

                question = prompt_session.prompt(oracle_prompt_text).strip()

                if not question:
                    continue
                if question.lower() in ("q", "quit", "exit", "back"):
                    break
                if question.lower() in ("h", "help"):
                     console.print("\n[dim]Enter your question or type 'back' to exit The Oracle.[/dim]")
                     continue

                with Status("[cyan]The Oracle is consulting the archives...[/cyan]", spinner="dots"):
                    response = self.api.ask_oracle(self.repo_id, question)

                if response and response.get("answer"):
                    console.print(Panel(
                        Markdown(response["answer"]),
                        title="[bold magenta]🔮 The Oracle's Answer[/bold magenta]",
                        border_style="magenta"
                    ))
                elif response and response.get("error"):
                    console.print(Panel(response["error"], title="[yellow]Oracle Warning[/yellow]", border_style="yellow"))
                else:
                    console.print("[red]❌ The Oracle did not provide an answer.[/red]")

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' or 'back' to return to the main menu.[/yellow]")
                continue
            except EOFError:
                break

        console.print("[magenta]🔮 Oracle session ended.[/magenta]")
        # Restore main completer
        prompt_session = PromptSession(
            history=FileHistory(str(history_path)),
            completer=main_completer,
            style=prompt_style
        )


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
        # --- State for RCA-to-Fix Pipeline ---
        self.last_rca_report = None
        self.last_rca_issue_num = None
        # ---
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

        is_already_analyzed = check_if_repo_is_analyzed(self.repo_id)
        do_embed = False

        if is_already_analyzed:
            console.print(f"\n[bold green]✓ Found existing analyzed repository for '{self.repo_id}'.[/bold green] [dim]Skipping ingestion.[/dim]")
        else:
            try:
                do_embed = Confirm.ask(
                    "\n[bold]Do you want to clone and embed this repo for full analysis (briefing, rca, fix)?[/bold]\n"
                    "[dim](This can take a few minutes for large repos. Choose 'N' for issue listing only.)[/dim]",
                    default=True
                )
            except KeyboardInterrupt:
                console.print("\n[yellow]Analysis cancelled.[/yellow]")
                return False

        if do_embed:
            with Status("[cyan]🚀 Beginning ingestion...[/cyan]", spinner="earth") as status:
                status.update("[cyan]Cloning repository and analyzing files...[/cyan]")
                ingest_result = self.api.ingest_repository(self.repo_url)
                if ingest_result and ingest_result.get("status") == "success":
                    status.update("[green]✓ Repository cloned and embedded successfully.[/green]")
                    time.sleep(1)
                else:
                    error_details = ingest_result.get('error', 'Unknown error during ingestion.') if ingest_result else "No response from server."
                    console.print(f"\n[bold red]❌ Ingestion failed:[/bold red] {error_details}")
                    try:
                        if not Confirm.ask("[yellow]Would you like to continue with limited (issue list only) analysis?[/yellow]", default=True):
                            return False
                    except KeyboardInterrupt:
                        return False

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("[green]🤖 Contacting The Strategist...", total=None)
            strategist_data = self.api.get_analysis(self.repo_url)
            if not strategist_data:
                return False

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

    def display_graph(self, graph_data: dict, repo_id: str):
        """
        [NEW & IMPROVED] Displays the architectural graph in a summarized, readable format.
        It now groups and counts calls to avoid overwhelming the user.
        """
        console.print("\n[bold magenta]--- 🗺️ Cartographer's Architectural Graph (Summary) ---[/bold magenta]")

        nodes = graph_data.get('nodes', {})
        edges = graph_data.get('edges', [])

        if not nodes:
            console.print("[yellow]No architectural nodes were mapped for this project.[/yellow]")
            return

        edges_by_source = defaultdict(lambda: {'imports': [], 'calls': defaultdict(int)})
        for edge in edges:
            source_id = edge['source']
            edge_type = edge['type']
            target_id = edge['target']

            if edge_type == 'IMPORTS':
                edges_by_source[source_id]['imports'].append(target_id)
            elif edge_type == 'CALLS':
                edges_by_source[source_id]['calls'][target_id] += 1

        tree = Tree(f"[bold blue]Project: {repo_id}[/bold blue]", guide_style="cyan")
        file_tree_nodes = {}

        for node_id, node_data in sorted(nodes.items()):
            if node_data.get('type') == 'file':
                lang = node_data.get('language', 'unknown')
                icon = "📄"
                if lang == 'python': icon = "🐍"
                if lang == 'javascript': icon = "🟨"

                file_branch = tree.add(f"{icon} [bold green]{node_id}[/bold green] [dim]({lang})[/dim]")
                file_tree_nodes[node_id] = file_branch

                for class_name in sorted(node_data.get('classes', [])):
                    class_node_id = f"{node_id}::{class_name}"
                    class_branch = file_branch.add(f"📦 [cyan]class[/cyan] {class_name}")

                    for method_name in sorted(nodes.get(class_node_id, {}).get('methods', [])):
                        class_branch.add(f"  -  M [dim]{method_name}()[/dim]")

                for func_name in sorted(node_data.get('functions', [])):
                    file_branch.add(f"  - F [dim]{func_name}()[/dim]")

        for source_id, relationships in edges_by_source.items():
            if source_id in file_tree_nodes:
                parent_branch = file_tree_nodes[source_id]

                if relationships['imports']:
                    import_branch = parent_branch.add("📥 [bold]Imports[/bold]")
                    for target in sorted(list(set(relationships['imports']))):
                        import_branch.add(f"[yellow]{target}[/yellow]")

                if relationships['calls']:
                    calls_branch = parent_branch.add("📞 [bold]Calls[/bold]")
                    sorted_calls = sorted(relationships['calls'].items(), key=lambda item: item[1], reverse=True)

                    max_calls_to_show = 15
                    for i, (target, count) in enumerate(sorted_calls):
                        if i >= max_calls_to_show:
                            calls_branch.add(f"[dim]... and {len(sorted_calls) - max_calls_to_show} more.[/dim]")
                            break

                        count_str = f" [dim](x{count})[/dim]" if count > 1 else ""
                        calls_branch.add(f"[magenta]{target}[/magenta]{count_str}")


        console.print(tree)
        console.print("\n[bold magenta]--------------------------------------------------------[/bold magenta]")

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

        if command in ('g', 'graph'):
            self.execute_action(command, {}) # Pass empty dict, issue not needed
            console.print("\n[dim]💡 Type [bold]list[/bold] to see issues, or [bold]help[/bold] for commands.[/dim]")
            return

        if command in ('f', 'fix') and self.last_rca_report:
             issue = next((iss for iss in self.issues if iss.get('number') == self.last_rca_issue_num), None)
             if issue:
                 self.execute_action(command, issue)
                 console.print("\n[dim]💡 Type [bold]list[/bold] to see issues, or [bold]help[/bold] for commands.[/dim]")
                 return

        if command not in ('f', 'fix', 'b', 'briefing', 'r', 'rca', 'd', 'details'):
            console.print("[red]❌ Unknown command. Type 'help' for available commands.[/red]")
            return

        issue_num_str = None
        if args and args[0].isdigit():
            issue_num_str = args[0]
        else:
            try:
                prompt_ask_text = f"Which issue # for '[cyan]{command}[/cyan]'?"
                if command in ('f', 'fix') and self.last_rca_report:
                     prompt_ask_text += f"\n[dim](Press Enter to fix issue #{self.last_rca_issue_num} from the last RCA)[/dim]"

                issue_num_str = Prompt.ask(prompt_ask_text).strip()
            except KeyboardInterrupt:
                console.print("\n[yellow]Command cancelled.[/yellow]")
                return

        if not issue_num_str:
            if command in ('f', 'fix') and self.last_rca_report:
                issue = next((iss for iss in self.issues if iss.get('number') == self.last_rca_issue_num), None)
                if issue:
                    self.execute_action(command, issue)
                else:
                    console.print("[red]❌ Could not find issue from last RCA. Please specify an issue number.[/red]")
            else:
                console.print("[red]❌ Please enter a valid issue number.[/red]")
            return

        if not issue_num_str.isdigit():
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

        if command in ('g', 'graph'):
            with Status("[cyan]🗺️ Contacting Cartographer's Architectural Graph...[/cyan]", spinner="earth") as status:
                graph_data = self.api.get_graph(self.repo_id)
                status.update("[green]✓ Graph retrieved.[/green]")
                time.sleep(0.5)

            if graph_data and graph_data.get("graph"):
                self.display_graph(graph_data["graph"], graph_data["repo_id"])
            elif graph_data and graph_data.get("message"):
                console.print(Panel(graph_data["message"], title="[yellow]Graph Not Available[/yellow]", border_style="yellow"))
            else:
                 console.print("[red]❌ Could not retrieve architectural graph.[/red]")
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
        """Handle the new context-aware root cause analysis command."""
        console.print(f"[cyan]🔍 Starting Root Cause Analysis for issue #{issue['number']}[/cyan]")
        issue_desc = f"Title: {issue.get('title', '')}\n\nDescription: {issue.get('description', '')}"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]🕵️ Performing multi-file root cause analysis...", total=None)
            rca_data = self.api.get_rca(self.repo_url, issue_desc)
            progress.remove_task(task)

        if rca_data and rca_data.get("analysis"):
            analysis_text = rca_data["analysis"]

            is_error = "Error from Gemini API" in analysis_text or "API Request Failed" in analysis_text

            if is_error:
                self.last_rca_report = None
                self.last_rca_issue_num = None
                console.print(Panel(
                    Markdown(analysis_text),
                    title=f"[bold red]🕵️ Root Cause Analysis - Issue #{issue['number']} (Failed)[/bold red]",
                    border_style="red"
                ))
            else:
                self.last_rca_report = analysis_text
                self.last_rca_issue_num = issue['number']
                console.print(Panel(
                    Markdown(self.last_rca_report),
                    title=f"[bold red]🕵️ Root Cause Analysis - Issue #{issue['number']}[/bold red]",
                    border_style="red"
                ))
                console.print("\n[bold yellow]💡 Pro-tip:[/bold yellow] [dim]You can now type '[/dim][bold]f[/bold][dim]' to start fixing this issue.[/dim]")
        else:
            self.last_rca_report = None
            self.last_rca_issue_num = None
            console.print("[red]❌ Could not perform root cause analysis.[/red]")

    def _display_diff(self, original_code: str, new_code: str, filename: str):
        """Display a formatted diff of code changes for a single file."""
        diff = difflib.unified_diff(
            original_code.splitlines(keepends=True),
            new_code.splitlines(keepends=True),
            fromfile=f'🔴 {filename} (Original)',
            tofile=f'🟢 {filename} (Proposed)'
        )
        diff_panel_content = Text()
        has_changes = False
        for line in diff:
            has_changes = True
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

        if not has_changes:
            return

        console.print(Panel(
            diff_panel_content,
            title=f"[bold yellow]📝 Proposed Changes for {filename}[/bold yellow]",
            expand=True,
            border_style="yellow"
        ))

    def _extract_filenames_from_rca(self, rca_report: str) -> List[str]:
        """Extracts filenames from markdown code fences, inline backticks, and lists."""
        pattern = r'(?:\s|-|\*|`)([\w./-]+\.(?:py|js|ts|gs|json|md|html|css|yaml|yml|toml|txt))\b'
        matches = re.findall(pattern, rca_report)

        backtick_pattern = r'`([\w./\\-]+)`'
        matches.extend(re.findall(backtick_pattern, rca_report))

        filenames = sorted(list(set(matches)))
        return [f for f in filenames if '.' in f and f.lower() not in ['true', 'false']]

    def handle_fix_dialogue(self, issue: Dict):
        """Handle the complete fix dialogue, now driven by RCA and with documentation and automated test generation."""
        console.print(Panel(
            f"[bold cyan]🤝 Socratic Dialogue[/bold cyan] starting for:\n"
            f"[bold green]Issue #{issue['number']}: {issue['title']}[/bold green]",
            border_style="cyan"
        ))

        if not self.last_rca_report or self.last_rca_issue_num != issue['number']:
             console.print("[yellow]⚠️  Warning: No Root Cause Analysis has been run for this issue.[/yellow]")
             try:
                 if Confirm.ask("[bold]Would you like to run RCA first to provide context for the fix?[/bold]", default=True):
                     self.handle_rca_command(issue)
                     if not self.last_rca_report:
                         console.print("[red]❌ Cannot proceed with fix without a successful RCA.[/red]")
                         return
                 else:
                     console.print("[red]❌ Fix command cancelled. Please run RCA first.[/red]")
                     return
             except KeyboardInterrupt:
                 console.print("\n[yellow]Fix command cancelled.[/yellow]")
                 return

        issue_desc = f"Title: {issue.get('title', '')}\n\nDescription: {issue.get('description', '')}"

        target_files = self._extract_filenames_from_rca(self.last_rca_report)
        if not target_files:
            console.print("[red]❌ Could not automatically determine target files from the RCA report.[/red]")
            return

        console.print(f"[dim]✓ Identified suspect files from RCA: {', '.join(target_files)}[/dim]")

        refinement_history = []
        iteration_count = 0
        max_iterations = 5
        modified_files = {}
        original_contents = {}
        is_documented = False
        are_tests_generated = False

        while iteration_count < max_iterations:
            iteration_count += 1

            if not modified_files or refinement_history:
                console.print(f"\n[dim]🔄 Iteration {iteration_count}/{max_iterations}[/dim]")
                with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
                    task = progress.add_task(f"[cyan]⚡ Generating multi-file code fix...", total=None)
                    fix_data = self.api.generate_scaffold(self.repo_id, target_files, issue_desc, self.last_rca_report, refinement_history)
                    progress.remove_task(task)

                if not fix_data:
                    console.print("[red]❌ Failed to generate fix or no files were modified.[/red]")
                    return

                if "modified_files" not in fix_data or not fix_data["modified_files"]:
                    console.print("[red]❌ Failed to generate fix or no files were modified.[/red]")
                    if fix_data and fix_data.get("llm_response"): console.print(Panel(Text(fix_data["llm_response"], overflow="fold"), title="[yellow]🔍 LLM Raw Response (for debugging)[/yellow]", border_style="yellow"))
                    return

                modified_files = fix_data["modified_files"]
                original_contents = fix_data["original_contents"]
                is_documented = False
                are_tests_generated = False

            console.rule("[bold]📝 Review Proposed Changes[/bold]")
            for filename, new_code in modified_files.items():
                if original_contents.get(filename, "") != new_code:
                    self._display_diff(original_contents.get(filename, ""), new_code, filename)

            all_validations_passed = True
            files_to_validate = [
                f for f in modified_files.keys()
                if not f.lower().endswith(('.md', '.txt', '.json', '.toml', '.yaml', '.yml', '.ron'))
            ]

            if not files_to_validate:
                 console.print(Panel("✅ [bold green]No runnable code files to validate. Skipping Crucible.[/bold green]", title="[green]🔥 Crucible Report[/green]", border_style="green"))
            else:
                for filename in files_to_validate:
                    new_code = modified_files[filename]
                    with Progress(SpinnerColumn(), TextColumn("[bold cyan][progress.description]{task.description}"), transient=True) as progress:
                        task = progress.add_task(f"🔥 Entering The Crucible for {filename}...", total=None)
                        validation_result = self.api.validate_in_crucible(self.repo_url, filename, new_code)
                        progress.remove_task(task)

                    if not validation_result or validation_result.get("status") != "passed":
                        all_validations_passed = False
                        console.print(Panel(f"❌ [bold red]Validation Failed for {filename}[/bold red]\n[bold]Test Results:[/bold]\n{validation_result.get('logs', 'No logs') if validation_result else 'Crucible service error'}", title=f"[red]🔥 Crucible Report: {filename}[/red]", border_style="red"))
                        break

            if all_validations_passed and files_to_validate:
                 console.print(Panel("✅ [bold green]All tests passed for all modified files![/bold green]", title="[green]🔥 Crucible Report[/green]", border_style="green"))

            while True:
                try:
                    action_choices = ['r', 'c']
                    prompt_text = ""

                    if all_validations_passed:
                        action_choices.append('a')
                        prompt_text += "\n[bold]✅ All tests passed! Choose action:[/bold]\n[bold green](a)[/bold green] Approve & create PR\n"
                        if not is_documented and any(f.endswith((".py", ".js", ".ts")) for f in modified_files.keys()):
                           action_choices.append('d')
                           prompt_text += "[bold blue](d)[/bold blue] Document the changes\n"
                        if not are_tests_generated:
                            action_choices.append('t')
                            prompt_text += "[bold yellow](t)[/bold yellow] Generate tests for the fix\n"
                    else:
                        prompt_text += "\n[bold red]❌ Validation failed. Choose action:[/bold red]\n"

                    prompt_text += "[bold yellow](r)[/bold yellow] Refine with feedback\n"
                    prompt_text += "[bold red](c)[/bold red] Cancel"

                    default_choice = 'a' if all_validations_passed else 'r'
                    choice = Prompt.ask(prompt_text, choices=action_choices, default=default_choice).lower()

                except KeyboardInterrupt:
                    choice = 'c'

                if choice == 'd':
                    if is_documented:
                        console.print("[yellow]Code is already documented.[/yellow]")
                        continue
                    if not all_validations_passed:
                        console.print("[red]Cannot document code that has failed validation.[/red]")
                        continue

                    documented_files = {}
                    with Status("[bold blue]✒️  Calling The Chronicler agent to document changes...[/bold blue]") as status:
                        for filename, code in modified_files.items():
                            if not any(filename.endswith(ext) for ext in [".py", ".js", ".ts"]): continue
                            status.update(f"[bold blue]✒️  Documenting {filename}...[/bold blue]")
                            doc_result = self.api.generate_docstring(self.repo_id, code, issue_desc)
                            if doc_result and doc_result.get("docstring"):
                                documented_code = _insert_docstring_into_code(code, doc_result["docstring"])
                                documented_files[filename] = documented_code
                            else:
                                documented_files[filename] = code
                    modified_files.update(documented_files)
                    is_documented = True
                    console.print("[green]✓ Documentation complete.[/green]")
                    console.rule("[bold]📝 Review Updated Changes with Documentation[/bold]")
                    for filename, new_code in modified_files.items():
                         if original_contents.get(filename, "") != new_code:
                            self._display_diff(original_contents.get(filename, ""), new_code, filename)
                    continue

                if choice == 't':
                    if are_tests_generated:
                        console.print("[yellow]Tests have already been generated for this fix.[/yellow]")
                        continue
                    if not all_validations_passed:
                        console.print("[red]Cannot generate tests for code that has failed validation.[/red]")
                        continue

                    generated_test_files = {}
                    with Status("[bold yellow]🔬 Calling Test Generation Agent...[/bold yellow]") as status:
                        for filename, code in modified_files.items():
                            if any(filename.endswith(ext) for ext in [".py", ".js", ".ts"]) and 'test' not in filename:
                                status.update(f"[bold yellow]🔬 Generating tests for {filename}...[/bold yellow]")
                                test_result = self.api.generate_tests(self.repo_id, code, issue_desc)

                                if test_result and test_result.get("generated_tests"):
                                    test_file_path = f"tests/test_{Path(filename).stem}.py"
                                    generated_test_files[test_file_path] = test_result["generated_tests"]
                                    console.print(f"  [green]✓[/green] Generated test file: [cyan]{test_file_path}[/cyan]")
                                else:
                                    console.print(f"  [red]✗[/red] Failed to generate tests for [cyan]{filename}[/cyan].")

                    if generated_test_files:
                        modified_files.update(generated_test_files)
                        are_tests_generated = True
                        console.print("\n[green]✓ Test generation complete.[/green]")
                        console.rule("[bold]📝 Review New Test Files[/bold]")
                        for filename, new_code in generated_test_files.items():
                            self._display_diff("", new_code, filename)
                    else:
                        console.print("[yellow]No new tests were generated.[/yellow]")

                    continue

                if choice == 'c': break
                if choice == 'a' and all_validations_passed: break
                if choice == 'r': break

            if choice == 'c':
                console.print("[yellow]🛑 Operation cancelled.[/yellow]")
                break

            if choice == 'r':
                if iteration_count >= max_iterations:
                    console.print(f"[yellow]⚠️ Maximum iterations ({max_iterations}) reached.[/yellow]")
                    break
                try:
                    feedback = Prompt.ask("\n[bold]💭 Your feedback for improvement[/bold]")
                    if not feedback.strip():
                        console.print("[yellow]⚠️ Empty feedback, skipping refinement.[/yellow]")
                        continue
                    refinement_history.append({"feedback": feedback, "code_generated": modified_files})
                    modified_files.clear()
                    continue
                except KeyboardInterrupt:
                    break

            if choice == 'a' and all_validations_passed:
                with Progress(SpinnerColumn(),TextColumn("[progress.description]{task.description}"),transient=True) as progress:
                    task = progress.add_task("[cyan]🚀 Dispatching Ambassador for multi-file PR...", total=None)
                    pr_data = self.api.create_pr(f"{self.repo_url}/issues/{issue['number']}", modified_files)
                    progress.remove_task(task)
                if pr_data and pr_data.get("pull_request_url"):
                    console.print(Panel(f"✅ [bold green]Success![/bold green]\nPull request created: [link={pr_data['pull_request_url']}]{pr_data['pull_request_url']}[/link]", title="[green]🚀 Mission Complete[/green]", border_style="green"))
                else:
                    console.print("[red]❌ Failed to create pull request.[/red]")
                break

        self.last_rca_report = None
        self.last_rca_issue_num = None


# --- Utility Function for Help Display ---
def display_interactive_help(context: str = 'main'):
    """Display help instructions based on the current CLI context."""
    title = f"🆘 Lumière Help — {context.capitalize()} Context"
    help_table = Table(title=f"[bold magenta]{title}[/bold magenta]", border_style="magenta")
    help_table.add_column("Command", style="bold cyan")
    help_table.add_column("Description", style="white")

    if context == 'main':
        help_table.add_row("analyze / a", "Ingest or re-ingest a repo for analysis")
        help_table.add_row("ask / oracle", "Ask architectural questions about a repo")
        help_table.add_row("review", "Perform an AI-powered review of a Pull Request")
        help_table.add_row("dashboard / d", "View the project health dashboard")
        help_table.add_row("profile / p", "Get GitHub user profile analysis")

        # --- DYNAMIC HELP TEXT ---
        if cli_state.get("model"):
            help_table.add_row("config / c", "Change LLM model or view settings")
        else:
            help_table.add_row("config / c", "Choose LLM provider & model")
        help_table.add_row("help / h", "Show this help menu")
        help_table.add_row("exit / quit", "Exit the application")
    elif context == 'analyze':
        help_table.add_row("list / l", "Show prioritized issues")
        help_table.add_row("graph / g", "Display the repository's architectural graph")
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

    health_status = api_client._request("GET", "health/")
    if health_status is None:
        console.print("[bold red]Lumière CLI cannot start without a backend connection.[/bold red]")
        sys.exit(1)

    load_config()

    welcome_text = (f"[bold cyan]✨ Welcome to Lumière Sémantique ✨[/bold cyan]\n"
                    f"The Conversational Mission Controller is active.\n\n"
                    f"[dim]Backend Status: [green]Online[/green] at [underline]{API_BASE_URL}[/underline][/dim]")
    console.print(Panel(welcome_text, border_style="cyan"))
    display_interactive_help('main')

    next_command = None
    context = {}

    while True:
        try:
            if next_command is None:
                prompt_text = get_prompt_text()
                command = prompt_session.prompt(prompt_text).strip()
            else:
                command = next_command
                console.print(f"\n[dim]Executing suggested action: [bold]{command}[/bold]...[/dim]")

            next_command = None

            if not command:
                continue

            if command.lower() in ("exit", "quit", "x"):
                console.print("[dim]👋 Goodbye![/dim]")
                break

            elif command.lower() == "back":
                console.print()
                continue

            elif command.lower() in ("help", "h"):
                display_interactive_help('main')
                continue

            elif command.lower() in ("config", "c"):
                console.print(Panel(
                    f"[bold]Current Settings[/bold]\n"
                    f"  [cyan]LLM Model:[/cyan] [yellow]{cli_state.get('model', 'Not set')}[/yellow]\n"
                    f"  [cyan]Last Repo:[/cyan] {cli_state.get('last_repo_url', 'Not set')}\n"
                    f"  [cyan]Debug Mode:[/cyan] {'On' if cli_state.get('debug_mode') else 'Off'}",
                    title="⚙️ Configuration", border_style="magenta"
                ))
                handle_model_selection(api_client)
                continue

            elif command.lower() in ("profile", "p"):
                if not cli_state.get("model"):
                    console.print("[bold red]Please select a model first using the 'config' command.[/bold red]")
                    continue
                username = Prompt.ask("Enter GitHub username")
                if not username.strip(): continue
                with Status("[cyan]Generating profile analysis...[/cyan]"):
                    profile = api_client.get_profile(username)
                if profile and profile.get("profile_summary"):
                    console.print(Panel(Markdown(profile["profile_summary"]), title=f"👤 Profile Analysis for {username}"))
                else: console.print("[red]❌ Could not retrieve profile.[/red]")
                continue

            elif command.lower() in ("ask", "oracle"):
                if not cli_state.get("model"):
                    console.print("[bold red]Please select a model first using the 'config' command.[/bold red]")
                    continue

                repo_url = context.get("repo_url") or context.get("pr_url", "").split("/pull/")[0]
                if not repo_url:
                    analyzed_repos = find_analyzed_repos()
                    if not analyzed_repos:
                        console.print("[yellow]No analyzed repositories found. Use 'analyze' to ingest a repo first.[/yellow]")
                        continue
                    console.print(Panel("Select a repository to ask questions about.", title="[magenta]🔮 The Oracle[/magenta]", border_style="magenta"))
                    table = Table(show_header=False, box=None, padding=(0, 2))
                    for i, repo in enumerate(analyzed_repos, 1): table.add_row(f"([bold cyan]{i}[/bold cyan])", repo['display_name'])
                    console.print(table)
                    try:
                        choice = Prompt.ask("Enter choice", choices=[str(i) for i in range(1, len(analyzed_repos) + 1)], show_choices=False, default='1')
                        repo_url = analyzed_repos[int(choice) - 1]['url']
                    except (ValueError, IndexError, KeyboardInterrupt): continue

                oracle_session = OracleSession(repo_url)
                oracle_session.loop()
                context = {}
                continue

            elif command.lower() in ("review",):
                if not cli_state.get("model"):
                    console.print("[bold red]Please select a model first using the 'config' command.[/bold red]")
                    continue

                pr_url = Prompt.ask("Enter the full GitHub Pull Request URL to review").strip()
                if not pr_url or "github.com" not in pr_url or "/pull/" not in pr_url:
                    console.print("[red]❌ Invalid Pull Request URL.[/red]")
                    continue

                result_data = None
                with Status("[cyan]The Inquisitor is reviewing the PR...[/cyan]", spinner="earth"):
                    result_data = api_client.adjudicate_pr(pr_url)

                if result_data and result_data.get("review"):
                    console.print(Panel(Markdown(result_data["review"]), title=f"⚖️ Inquisitor's Review", border_style="blue"))
                    context = {"pr_url": pr_url, "repo_url": pr_url.split("/pull/")[0], **result_data}
                    next_command, context = _present_next_actions(api_client, "review", context)
                elif result_data and result_data.get("error"):
                    console.print(Panel(f"[bold red]Review Failed:[/bold red]\n{result_data['error']}", title="[red]Inquisitor Error[/red]"))
                else: console.print("[red]❌ The Inquisitor did not provide a review.[/red]")
                continue

            elif command.lower() == "harmonize":
                 pr_url = context.get("pr_url")
                 review_text = context.get("review")
                 if not pr_url or not review_text:
                     console.print("[red]Harmonize command requires context from a review. Please run 'review' first.[/red]")
                     continue

                 with Status("[cyan]The Harmonizer is composing a fix...[/cyan]"):
                     fix_data = api_client.harmonize_fix(pr_url, review_text)

                 if fix_data and "modified_files" in fix_data:
                     console.rule("[bold]📝 Review Harmonizer's Proposed Changes[/bold]")
                     dummy_session = AnalysisSession(pr_url.split('/pull/')[0])
                     for filename, new_code in fix_data["modified_files"].items():
                         original_code = fix_data.get("original_contents", {}).get(filename, "")
                         if original_code != new_code:
                             dummy_session._display_diff(original_code, new_code, filename)
                     console.print(Panel("✅ [bold green]Harmonizer's patch generated.[/bold green]", border_style="green"))
                 else:
                     console.print(Panel(f"[red]Harmonizer failed to generate a fix: {fix_data.get('error', 'Unknown error')}[/red]", border_style="red"))

                 context = {}
                 continue

            elif command.lower() in ("dashboard", "d"):
                if not cli_state.get("model"):
                    console.print("[bold red]Please select a model first using the 'config' command.[/bold red]")
                    continue

                repo_id = context.get("repo_id")
                if not repo_id:
                    analyzed_repos = find_analyzed_repos()
                    if not analyzed_repos:
                        console.print("[yellow]No analyzed repositories found. Use 'analyze' to ingest a repo first.[/yellow]")
                        continue
                    console.print(Panel("Select a repository to view its health dashboard.", title="[cyan]🔭 The Sentinel[/cyan]", border_style="cyan"))
                    table = Table(show_header=False, box=None, padding=(0, 2))
                    for i, repo in enumerate(analyzed_repos, 1): table.add_row(f"([bold cyan]{i}[/bold cyan])", repo['display_name'])
                    console.print(table)
                    try:
                        choice = Prompt.ask("Enter choice", choices=[str(i) for i in range(1, len(analyzed_repos) + 1)], show_choices=False, default='1')
                        repo_id = analyzed_repos[int(choice) - 1]['repo_id']
                    except (ValueError, IndexError, KeyboardInterrupt): continue

                with Status("[cyan]The Sentinel is gathering intelligence...[/cyan]"):
                    response = api_client.get_sentinel_briefing(repo_id)

                if response and response.get("briefing"):
                    console.print(Panel(Markdown(response["briefing"]), title=f"[cyan]🔭 Sentinel Health Briefing for {repo_id}[/cyan]", border_style="cyan"))
                    context = {"repo_id": repo_id, "repo_url": f"https://github.com/{repo_id.replace('_', '/')}", **response}
                    next_command, context = _present_next_actions(api_client, "dashboard", context)
                elif response and response.get("error"):
                    console.print(Panel(response['error'], title="[red]Sentinel Error[/red]"))
                else: console.print("[red]❌ The Sentinel did not provide a briefing.[/red]")
                continue

            elif command.lower() in ("analyze", "a"):
                if not cli_state.get("model"):
                    console.print("[bold red]Please select a model first using the 'config' command.[/bold red]")
                    continue

                repo_url_to_analyze = context.get("repo_url")
                if not repo_url_to_analyze:
                    analyzed_repos = find_analyzed_repos()
                    if analyzed_repos:
                        console.print(Panel("Select a repository to analyze or ingest a new one.", title="[cyan]Select Repository for Analysis[/cyan]", border_style="cyan"))
                        table = Table(show_header=False, box=None, padding=(0, 2))
                        for i, repo in enumerate(analyzed_repos, 1): table.add_row(f"([bold cyan]{i}[/bold cyan])", repo['display_name'])
                        table.add_row("([bold yellow]N[/bold yellow])", "Analyze a new repository")
                        console.print(table)
                        choices = [str(i) for i in range(1, len(analyzed_repos) + 1)] + ['n', 'N']
                        try:
                            choice = Prompt.ask("Enter choice", choices=choices, show_choices=False, default='1').lower()
                            if choice == 'n': repo_url_to_analyze = Prompt.ask("Enter GitHub repository URL").strip()
                            else: repo_url_to_analyze = analyzed_repos[int(choice) - 1]['url']
                        except(ValueError, IndexError, KeyboardInterrupt): continue
                    else:
                        repo_url_to_analyze = Prompt.ask("Enter GitHub repository URL").strip()

                if not repo_url_to_analyze: continue

                try:
                    session = AnalysisSession(repo_url_to_analyze)
                    if session.start():
                        session.loop()
                except ValueError as e: console.print(f"[red]{e}[/red]")
                continue

            else:
                console.print("[red]❌ Unknown command. Type 'help' for options.[/red]")

        except KeyboardInterrupt:
            console.print("\n[dim]💤 Interrupted. Type 'exit' to quit.[/dim]")
            next_command = None
            continue
        except EOFError:
            break

if __name__ == "__main__":
    app()
