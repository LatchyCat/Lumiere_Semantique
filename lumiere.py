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
import os
import asyncio
import threading
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.markdown import Markdown
from rich.text import Text
from rich.status import Status
from rich.live import Live
from rich.align import Align
from rich.columns import Columns
from rich.layout import Layout
from rich.syntax import Syntax
from rich.rule import Rule
from rich.spinner import Spinner
from rich.box import ROUNDED, DOUBLE, SIMPLE
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter, FuzzyCompleter, NestedCompleter
from prompt_toolkit.shortcuts import confirm
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from pathlib import Path
import json
import time
from datetime import datetime
import textwrap
from rich.tree import Tree
from fuzzywuzzy import fuzz, process
try:
    import questionary
except ImportError:
    questionary = None

# --- Global Objects & Configuration ---
console = Console()
history_path = Path.home() / ".lumiere" / "history.txt"
config_path = Path.home() / ".lumiere" / "config.json"
history_path.parent.mkdir(parents=True, exist_ok=True)

# --- Centralized API URL ---
API_BASE_URL = "http://127.0.0.1:8002/api/v1"

# Create command completers for better UX
main_commands = ['analyze', 'a', 'ask', 'oracle', 'dance', 'da', 'summon', 'su', 'review', 'dashboard', 'd', 'profile', 'p', 'bom', 'b', 'onboard', 'o', 'repo-mgmt', 'rm', 'quartermaster', 'qm', 'loremaster', 'lm', 'librarian', 'lib', 'config', 'c', 'help', 'h', 'exit', 'x', 'quit', 'list-repos', 'lr']
analysis_commands = ['list', 'l', 'fix', 'f', 'briefing', 'b', 'rca', 'r', 'details', 'd', 'graph', 'g', 'mage', 'm', 'help', 'h', 'back', 'exit', 'quit']
oracle_commands = ['help', 'h', 'back', 'exit', 'quit']
bom_commands = ['overview', 'o', 'dependencies', 'deps', 'services', 's', 'security', 'sec', 'compare', 'c', 'regenerate', 'r', 'help', 'h', 'back', 'exit', 'quit']
onboard_commands = ['scout', 's', 'expert', 'e', 'guide', 'g', 'help', 'h', 'back', 'exit', 'quit']
repo_commands = ['list', 'l', 'status', 's', 'delete', 'd', 'help', 'h', 'back', 'exit', 'quit']

# Basic completers (enhanced completers will be created later)
main_completer = FuzzyCompleter(WordCompleter(main_commands, ignore_case=True))
analysis_completer = FuzzyCompleter(WordCompleter(analysis_commands, ignore_case=True))
oracle_completer = FuzzyCompleter(WordCompleter(oracle_commands, ignore_case=True))
bom_completer = FuzzyCompleter(WordCompleter(bom_commands, ignore_case=True))
onboard_completer = FuzzyCompleter(WordCompleter(onboard_commands, ignore_case=True))
repo_completer = FuzzyCompleter(WordCompleter(repo_commands, ignore_case=True))

# --- Global CLI State ---
cli_state = {
    "model": None,  # Will be populated from config
    "available_models": [],
    "last_repo_url": None,
    "debug_mode": False,
    "theme": "modern",
    "animations_enabled": True,
    "notifications": [],
    "active_operations": [],
    "command_history": [],
}

# --- Modern CLI Enhancements ---

class ThemeManager:
    """Manages different CLI themes for a modern experience."""
    
    THEMES = {
        "modern": {
            "primary": "#00d4aa",
            "secondary": "#ff6b6b", 
            "accent": "#ffd93d",
            "text": "#ffffff",
            "dim": "#888888",
            "success": "#00ff88",
            "warning": "#ffb347",
            "error": "#ff4757",
            "background": "#1a1a1a"
        },
        "cyberpunk": {
            "primary": "#00ffff",
            "secondary": "#ff00ff",
            "accent": "#ffff00", 
            "text": "#ffffff",
            "dim": "#666666",
            "success": "#39ff14",
            "warning": "#ff6600",
            "error": "#ff0040",
            "background": "#0d0d0d"
        },
        "minimal": {
            "primary": "#6366f1",
            "secondary": "#8b5cf6",
            "accent": "#06b6d4",
            "text": "#f8fafc",
            "dim": "#64748b", 
            "success": "#10b981",
            "warning": "#f59e0b",
            "error": "#ef4444",
            "background": "#0f172a"
        }
    }
    
    @classmethod
    def get_style(cls, theme_name: str = "modern") -> Style:
        """Get prompt_toolkit style for the given theme."""
        theme = cls.THEMES.get(theme_name, cls.THEMES["modern"])
        return Style.from_dict({
            'lumiere': f'bold {theme["primary"]}',
            'provider': theme["secondary"],
            'separator': theme["accent"],
            'command': theme["text"],
            'hint': theme["dim"],
            'success': theme["success"],
            'warning': theme["warning"],
            'error': theme["error"]
        })

class AnimationManager:
    """Handles smooth animations and transitions."""
    
    @staticmethod
    def typing_effect(text: str, delay: float = 0.03):
        """Create a typing effect for text output."""
        if not cli_state.get("animations_enabled", True):
            console.print(text)
            return
            
        for char in text:
            console.print(char, end="")
            time.sleep(delay)
        console.print()
    
    @staticmethod 
    def fade_in_panel(content: str, title: str = "", delay: float = 0.1):
        """Fade in a panel with animation."""
        if not cli_state.get("animations_enabled", True):
            console.print(Panel(content, title=title, box=ROUNDED))
            return
            
        lines = content.split('\n')
        with Live(Panel("", title=title, box=ROUNDED), console=console, refresh_per_second=10) as live:
            displayed_lines = []
            for line in lines:
                displayed_lines.append(line)
                live.update(Panel('\n'.join(displayed_lines), title=title, box=ROUNDED))
                time.sleep(delay)

class FuzzyCommandCompleter:
    """Advanced fuzzy matching completer for commands."""
    
    def __init__(self, commands: List[str]):
        self.commands = commands
        self.command_descriptions = {
            'analyze': 'Deep analysis of repository structure and patterns',
            'ask': 'Interactive Q&A about your codebase',
            'oracle': 'AI-powered code insights and recommendations', 
            'dance': 'Dynamic code transformations and refactoring',
            'summon': 'Generate new code components and features',
            'dashboard': 'Overview of repository metrics and health',
            'profile': 'Performance analysis and optimization suggestions',
            'bom': 'Bill of Materials - dependency analysis',
            'onboard': 'Interactive onboarding for new developers',
            'repo-mgmt': 'Repository management and operations',
            'quartermaster': 'Resource and dependency management',
            'loremaster': 'Knowledge base and documentation tools',
            'librarian': 'Code organization and categorization',
            'config': 'Configuration and settings management'
        }
    
    def get_matches(self, text: str, limit: int = 5) -> List[Tuple[str, str]]:
        """Get fuzzy matches with descriptions."""
        if not text:
            return [(cmd, self.command_descriptions.get(cmd, "")) for cmd in self.commands[:limit]]
        
        matches = process.extract(text, self.commands, limit=limit, scorer=fuzz.partial_ratio)
        return [(match[0], self.command_descriptions.get(match[0], "")) for match in matches if match[1] > 30]

class ProgressBarManager:
    """Enhanced progress bars with status updates."""
    
    @staticmethod
    def create_modern_progress() -> Progress:
        """Create a modern-looking progress bar."""
        return Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, style="cyan", complete_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False
        )
    
    @staticmethod
    def show_operation_status(operation: str, steps: List[str]):
        """Show a multi-step operation with progress."""
        with ProgressBarManager.create_modern_progress() as progress:
            task = progress.add_task(operation, total=len(steps))
            for i, step in enumerate(steps):
                progress.update(task, description=f"[cyan]{step}[/cyan]", completed=i)
                time.sleep(0.5)  # Simulate work
            progress.update(task, description=f"[green]✓ {operation} Complete[/green]", completed=len(steps))

class NotificationManager:
    """Real-time notification system."""
    
    @staticmethod
    def add_notification(message: str, type: str = "info", duration: int = 5):
        """Add a notification to the queue."""
        notification = {
            "message": message,
            "type": type,
            "timestamp": datetime.now(),
            "duration": duration
        }
        cli_state["notifications"].append(notification)
        NotificationManager.show_notification(notification)
    
    @staticmethod
    def show_notification(notification: Dict):
        """Display a notification with styling."""
        type_styles = {
            "info": "blue",
            "success": "green", 
            "warning": "yellow",
            "error": "red"
        }
        style = type_styles.get(notification["type"], "blue")
        icon = {"info": "ℹ", "success": "✓", "warning": "⚠", "error": "✗"}[notification["type"]]
        
        console.print(f"[{style}]{icon} {notification['message']}[/{style}]")

class KeyboardShortcuts:
    """Keyboard shortcuts and hotkeys."""
    
    @staticmethod
    def create_key_bindings() -> KeyBindings:
        """Create custom key bindings for enhanced UX."""
        kb = KeyBindings()
        
        @kb.add('c-d')  # Ctrl+D for dashboard
        def _(event):
            event.app.exit(result='dashboard')
        
        @kb.add('c-h')  # Ctrl+H for help
        def _(event):
            event.app.exit(result='help')
        
        @kb.add('c-r')  # Ctrl+R for repo management
        def _(event):
            event.app.exit(result='repo-mgmt')
        
        @kb.add('c-a')  # Ctrl+A for analyze
        def _(event):
            event.app.exit(result='analyze')
        
        return kb

class CommandPreview:
    """Live command preview and validation."""
    
    @staticmethod
    def preview_command(command: str) -> str:
        """Generate a preview of what the command will do."""
        previews = {
            'analyze': "🔍 Start deep analysis of repository structure",
            'ask': "💭 Enter interactive Q&A mode", 
            'oracle': "🔮 Get AI-powered insights",
            'dance': "💫 Begin code transformation wizard",
            'summon': "⚡ Open code generation interface",
            'dashboard': "📊 Display repository dashboard",
            'profile': "⚡ Show performance analysis",
            'bom': "📋 Generate Bill of Materials",
            'onboard': "🎓 Start onboarding assistant",
            'repo-mgmt': "🗂️ Open repository management",
            'config': "⚙️ Configure settings"
        }
        
        base_cmd = command.split()[0] if command else ""
        return previews.get(base_cmd, f"Execute: {command}" if command else "")

class ModernDashboard:
    """Enhanced dashboard with real-time metrics."""
    
    @staticmethod
    def create_layout() -> Layout:
        """Create a modern dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        return layout
    
    @staticmethod
    def show_dashboard():
        """Display the modern dashboard."""
        layout = ModernDashboard.create_layout()
        
        # Header
        layout["header"].update(Panel(
            Align.center("[bold cyan]Lumière Sémantique[/bold cyan] [dim]- Modern Code Intelligence Platform[/dim]"),
            style="cyan"
        ))
        
        # Left panel - Quick actions
        quick_actions = Table(title="Quick Actions", box=SIMPLE)
        quick_actions.add_column("Shortcut", style="cyan")
        quick_actions.add_column("Action", style="white")
        quick_actions.add_row("Ctrl+A", "Analyze Repository")
        quick_actions.add_row("Ctrl+D", "Dashboard")
        quick_actions.add_row("Ctrl+H", "Help")
        quick_actions.add_row("Ctrl+R", "Repo Management")
        
        layout["left"].update(Panel(quick_actions, title="🚀 Quick Access"))
        
        # Right panel - Recent activity
        activity = Table(title="Recent Activity", box=SIMPLE)
        activity.add_column("Time", style="dim")
        activity.add_column("Activity", style="white")
        
        for cmd in cli_state.get("command_history", [])[-5:]:
            activity.add_row(cmd.get("time", ""), cmd.get("command", ""))
        
        layout["right"].update(Panel(activity, title="📈 Activity"))
        
        # Footer
        layout["footer"].update(Panel(
            Align.center("[dim]Use arrow keys to navigate • Press Enter to select • Type to search[/dim]"),
            style="dim"
        ))
        
        console.print(layout)

class SyntaxHighlighter:
    """Syntax highlighting for commands and code."""
    
    @staticmethod
    def highlight_command(command: str) -> str:
        """Add syntax highlighting to commands."""
        keywords = ['analyze', 'ask', 'oracle', 'dance', 'summon', 'dashboard', 'profile', 'bom', 'onboard', 'repo-mgmt']
        flags = ['-v', '--verbose', '-h', '--help', '-f', '--force']
        
        highlighted = command
        for keyword in keywords:
            highlighted = highlighted.replace(keyword, f"[bold cyan]{keyword}[/bold cyan]")
        for flag in flags:
            highlighted = highlighted.replace(flag, f"[yellow]{flag}[/yellow]")
        
        return highlighted
    
    @staticmethod
    def highlight_code(code: str, language: str = "python") -> Syntax:
        """Create syntax highlighted code display."""
        return Syntax(code, language, theme="monokai", line_numbers=True, background_color="default")

class EnhancedHelp:
    """Contextual help system with search."""
    
    HELP_CONTENT = {
        'analyze': {
            'description': 'Deep analysis of repository structure and patterns',
            'usage': 'analyze [options] [repository]',
            'examples': [
                'analyze https://github.com/user/repo',
                'analyze --verbose ./local-repo',
                'analyze -f --force-refresh'
            ],
            'flags': {
                '-v, --verbose': 'Enable verbose output',
                '-f, --force': 'Force re-analysis of existing repo',
                '--depth <n>': 'Set analysis depth (1-5)'
            }
        },
        'dashboard': {
            'description': 'Interactive dashboard with repository metrics',
            'usage': 'dashboard [options]',
            'examples': [
                'dashboard',
                'dashboard --theme cyberpunk',
                'dashboard --refresh-rate 5'
            ],
            'flags': {
                '--theme <name>': 'Set dashboard theme (modern, cyberpunk, minimal)',
                '--refresh-rate <n>': 'Auto-refresh interval in seconds'
            }
        }
    }
    
    @staticmethod
    def show_contextual_help(command: str = None):
        """Show contextual help based on current context."""
        if command and command in EnhancedHelp.HELP_CONTENT:
            help_data = EnhancedHelp.HELP_CONTENT[command]
            
            panel_content = f"""[bold]{help_data['description']}[/bold]

[cyan]Usage:[/cyan]
  {help_data['usage']}

[cyan]Examples:[/cyan]"""
            
            for example in help_data['examples']:
                panel_content += f"\n  {SyntaxHighlighter.highlight_command(example)}"
            
            if help_data.get('flags'):
                panel_content += "\n\n[cyan]Flags:[/cyan]"
                for flag, desc in help_data['flags'].items():
                    panel_content += f"\n  [yellow]{flag}[/yellow] - {desc}"
            
            AnimationManager.fade_in_panel(panel_content, f"Help: {command}", 0.05)
        else:
            EnhancedHelp.show_general_help()
    
    @staticmethod
    def show_general_help():
        """Show general help with all commands."""
        help_table = Table(title="Lumière Sémantique Commands", box=ROUNDED)
        help_table.add_column("Command", style="cyan", width=15)
        help_table.add_column("Description", style="white", width=40)
        help_table.add_column("Shortcut", style="yellow", width=10)
        
        commands_info = [
            ("analyze", "Deep repository analysis", "a"),
            ("ask", "Interactive Q&A about code", ""),
            ("oracle", "AI-powered insights", ""),
            ("dance", "Code transformation", "da"),
            ("summon", "Generate new code", "su"),
            ("dashboard", "Metrics overview", "d"),
            ("profile", "Performance analysis", "p"),
            ("bom", "Bill of Materials", "b"),
            ("onboard", "Developer onboarding", "o"),
            ("repo-mgmt", "Repository management", "rm"),
        ]
        
        for cmd, desc, shortcut in commands_info:
            help_table.add_row(cmd, desc, shortcut)
        
        console.print("\n")
        console.print(help_table)
        console.print("[dim]💡 Use [cyan]help <command>[/cyan] for detailed information[/dim]")
        console.print("[dim]🎯 Keyboard shortcuts: Ctrl+A (analyze), Ctrl+D (dashboard), Ctrl+H (help), Ctrl+R (repo)[/dim]\n")

class CommandHistory:
    """Enhanced command history with search."""
    
    @staticmethod
    def add_to_history(command: str):
        """Add command to history with timestamp."""
        entry = {
            "command": command,
            "time": datetime.now().strftime("%H:%M:%S"),
            "timestamp": datetime.now()
        }
        cli_state["command_history"].append(entry)
        
        # Keep only last 100 commands
        if len(cli_state["command_history"]) > 100:
            cli_state["command_history"] = cli_state["command_history"][-100:]
    
    @staticmethod
    def search_history(query: str) -> List[Dict]:
        """Search command history."""
        if not query:
            return cli_state["command_history"][-10:]
        
        results = []
        for entry in cli_state["command_history"]:
            if query.lower() in entry["command"].lower():
                results.append(entry)
        
        return results[-10:]  # Return last 10 matches
    
    @staticmethod
    def show_history(query: str = ""):
        """Display command history with optional search."""
        entries = CommandHistory.search_history(query)
        
        if not entries:
            console.print("[dim]No commands in history[/dim]")
            return
        
        history_table = Table(title=f"Command History{' - Search: ' + query if query else ''}", box=SIMPLE)
        history_table.add_column("Time", style="dim", width=10)
        history_table.add_column("Command", style="white")
        
        for entry in entries:
            history_table.add_row(entry["time"], SyntaxHighlighter.highlight_command(entry["command"]))
        
        console.print(history_table)

# Initialize enhanced prompt session after all classes are defined
def initialize_prompt_session():
    """Initialize the enhanced prompt session with modern features."""
    global prompt_session, prompt_style
    
    # Get theme style
    prompt_style = ThemeManager.get_style(cli_state.get("theme", "modern"))
    
    # Enhanced prompt session with modern features
    prompt_session = PromptSession(
        history=FileHistory(str(history_path)),
        completer=main_completer,
        style=prompt_style,
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=KeyboardShortcuts.create_key_bindings(),
        mouse_support=True,
        complete_style='multi-column'
    )

# Initialize the prompt session
initialize_prompt_session()

# --- NEW UTILITY FUNCTIONS for managing analyzed repos ---

def check_if_repo_is_analyzed(repo_id: str) -> bool:
    """Checks repository analysis status via API instead of filesystem."""
    api_client = LumiereAPIClient()
    try:
        response = api_client.get_repository_status(repo_id)
        return response and response.get("status") == "complete"
    except Exception:
        # Fallback to filesystem check if API fails
        cloned_repos_dir = Path("backend/cloned_repositories")
        repo_dir = cloned_repos_dir / repo_id
        if not repo_dir.is_dir():
            return False

        cortex_file = repo_dir / f"{repo_id}_cortex.json"
        faiss_file = repo_dir / f"{repo_id}_faiss.index"
        map_file = repo_dir / f"{repo_id}_id_map.json"

        return all([cortex_file.exists(), faiss_file.exists(), map_file.exists()])

def find_analyzed_repos() -> List[Dict[str, str]]:
    """Fetches analyzed repositories from the backend API instead of scanning filesystem."""
    api_client = LumiereAPIClient()
    try:
        response = api_client.list_repositories()
        return response if response else []
    except Exception:
        # Fallback to filesystem scan if API fails
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
    def get_graph(self, repo_id: str): return self._request("GET", f"graph/{repo_id}/")
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
    def get_sentinel_briefing(self, repo_id: str): return self._request("GET", f"sentinel/briefing/{repo_id}/")
    # --- NEW: Method for the Mission Controller ---
    def get_next_actions(self, last_action: str, result_data: dict): return self._request("POST", "suggest-actions/", json={"last_action": last_action, "result_data": result_data})
    
    # --- Repository Management Methods ---
    def list_repositories(self): return self._request("GET", "repositories/")
    def get_repository_detail(self, repo_id: str): return self._request("GET", f"repositories/{repo_id}/")
    def delete_repository(self, repo_id: str): return self._request("DELETE", f"repositories/{repo_id}/")
    def get_repository_status(self, repo_id: str): return self._request("GET", f"repositories/{repo_id}/status/")
    
    # --- BOM (Bill of Materials) Methods ---
    def get_bom_data(self, repo_id: str, format_type: str = "json"): return self._request("GET", f"bom/?repo_id={repo_id}&format={format_type}")
    def get_bom_dependencies(self, repo_id: str, **filters): 
        params = "&".join([f"{k}={v}" for k, v in filters.items() if v is not None])
        url = f"bom/dependencies/?repo_id={repo_id}"
        if params: url += "&" + params
        return self._request("GET", url)
    def get_bom_services(self, repo_id: str, service_type: str = None): 
        url = f"bom/services/?repo_id={repo_id}"
        if service_type: url += f"&service_type={service_type}"
        return self._request("GET", url)
    def get_bom_security(self, repo_id: str, severity: str = None): 
        url = f"bom/security/?repo_id={repo_id}"
        if severity: url += f"&severity={severity}"
        return self._request("GET", url)
    def regenerate_bom(self, repo_id: str, force: bool = False): return self._request("POST", "bom/regenerate/", json={"repo_id": repo_id, "force": force})
    def compare_bom(self, repo_id_1: str, repo_id_2: str, comparison_type: str = "dependencies"): return self._request("POST", "bom/compare/", json={"repo_id_1": repo_id_1, "repo_id_2": repo_id_2, "comparison_type": comparison_type})
    
    # --- Metrics History Method ---
    def get_sentinel_metrics_history(self, repo_id: str): return self._request("GET", f"sentinel/metrics/{repo_id}/")


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


# --- NEW: BOM Session Manager ---
class BOMSession:
    def __init__(self, repo_id: str, repo_url: str):
        self.repo_id = repo_id
        self.repo_url = repo_url
        self.api = LumiereAPIClient()
        console.print(Panel(
            f"[bold blue]📦 Bill of Materials Analysis[/bold blue]\n"
            f"[yellow]Repository:[/yellow] {self.repo_id}\n"
            f"[yellow]URL:[/yellow] {self.repo_url}\n\n"
            "[dim]Analyze dependencies, services, security, and more. Type 'help' for commands.[/dim]",
            border_style="blue"
        ))

    def loop(self):
        """Main interactive loop for BOM analysis."""
        global prompt_session
        prompt_session = PromptSession(
            history=FileHistory(str(history_path)),
            completer=bom_completer,
            style=prompt_style
        )

        while True:
            try:
                bom_prompt_text = [
                    ('class:lumiere', 'Lumière'),
                    ('class:provider', f' (BOM/{self.repo_id})'),
                    ('class:separator', ' > '),
                ]

                command = prompt_session.prompt(bom_prompt_text).strip()

                if not command:
                    continue
                if command.lower() in ("q", "quit", "exit", "back"):
                    break
                if command.lower() in ("h", "help"):
                    self.display_help()
                    continue

                self.handle_bom_command(command)

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' or 'back' to return to the main menu.[/yellow]")
                continue
            except EOFError:
                break

        console.print("[blue]📦 BOM session ended.[/blue]")
        # Restore main completer
        prompt_session = PromptSession(
            history=FileHistory(str(history_path)),
            completer=main_completer,
            style=prompt_style
        )

    def display_help(self):
        """Display BOM help commands."""
        help_table = Table(title="[bold blue]📦 BOM Analysis Commands[/bold blue]", border_style="blue")
        help_table.add_column("Command", style="bold cyan")
        help_table.add_column("Description", style="white")
        
        help_table.add_row("overview / o", "Complete BOM overview")
        help_table.add_row("dependencies / deps", "Analyze dependencies")
        help_table.add_row("services / s", "View services and infrastructure")
        help_table.add_row("security / sec", "Security analysis")
        help_table.add_row("compare / c", "Compare with another repository")
        help_table.add_row("regenerate / r", "Regenerate BOM data")
        help_table.add_row("help / h", "Show this help menu")
        help_table.add_row("back / exit / quit", "Return to main menu")
        
        console.print(help_table)

    def handle_bom_command(self, command: str):
        """Handle BOM commands."""
        cmd = command.lower().strip()
        
        if cmd in ("overview", "o"):
            self.show_overview()
        elif cmd in ("dependencies", "deps"):
            self.show_dependencies()
        elif cmd in ("services", "s"):
            self.show_services()
        elif cmd in ("security", "sec"):
            self.show_security()
        elif cmd in ("compare", "c"):
            self.show_compare()
        elif cmd in ("regenerate", "r"):
            self.regenerate_bom()
        else:
            console.print("[red]❌ Unknown command. Type 'help' for available commands.[/red]")

    def show_overview(self):
        """Show complete BOM overview."""
        with Status("[cyan]📦 Retrieving BOM overview...[/cyan]"):
            bom_data = self.api.get_bom_data(self.repo_id, "summary")
        
        if not bom_data:
            console.print("[red]❌ Could not retrieve BOM data. Repository may not be analyzed.[/red]")
            return
        
        summary = bom_data.get('summary', {})
        
        overview_content = f"""[bold]📊 Repository Summary[/bold]
• Primary Language: {summary.get('primary_language', 'Unknown')}
• Total Dependencies: {summary.get('total_dependencies', 0)}
• Total Services: {summary.get('total_services', 0)}
• Build Tools: {summary.get('total_build_tools', 0)}
• Languages Detected: {summary.get('languages_detected', 0)}
• Ecosystems: {', '.join(summary.get('ecosystems', []))}"""
        
        console.print(Panel(overview_content, title="[bold blue]📦 BOM Overview[/bold blue]", border_style="blue"))

    def show_dependencies(self):
        """Show dependency analysis with filtering options."""
        try:
            ecosystem = Prompt.ask("Filter by ecosystem (python/javascript/docker or press Enter for all)", default="")
            dependency_type = Prompt.ask("Filter by type (application/development/testing or press Enter for all)", default="")
            
            filters = {}
            if ecosystem: filters['ecosystem'] = ecosystem
            if dependency_type: filters['dependency_type'] = dependency_type
            
            with Status("[cyan]📦 Analyzing dependencies...[/cyan]"):
                deps_data = self.api.get_bom_dependencies(self.repo_id, **filters)
            
            if not deps_data:
                console.print("[red]❌ Could not retrieve dependency data.[/red]")
                return
            
            dependencies = deps_data.get('dependencies', [])
            stats = deps_data.get('statistics', {})
            
            if not dependencies:
                console.print("[yellow]📭 No dependencies found with the specified filters.[/yellow]")
                return
            
            table = Table(title=f"[bold cyan]📦 Dependencies ({len(dependencies)} found)[/bold cyan]", border_style="cyan")
            table.add_column("Name", style="white")
            table.add_column("Version", style="green")
            table.add_column("Type", style="blue")
            table.add_column("Ecosystem", style="yellow")
            
            for dep in dependencies[:20]:  # Show first 20
                table.add_row(
                    dep.get('name', 'Unknown'),
                    dep.get('version', 'Unknown'),
                    dep.get('category', 'Unknown'),
                    dep.get('ecosystem', 'Unknown')
                )
            
            if len(dependencies) > 20:
                table.add_row("...", f"({len(dependencies) - 20} more)", "...", "...")
            
            console.print(table)
            
            if stats:
                stats_content = f"📊 **Statistics:**\n"
                for ecosystem, count in stats.get('ecosystems', {}).items():
                    stats_content += f"• {ecosystem.capitalize()}: {count} dependencies\n"
                console.print(Panel(stats_content, title="[cyan]📊 Dependency Statistics[/cyan]", border_style="cyan"))
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Dependency analysis cancelled.[/yellow]")

    def show_services(self):
        """Show services and infrastructure analysis."""
        with Status("[cyan]🏗️ Analyzing services and infrastructure...[/cyan]"):
            services_data = self.api.get_bom_services(self.repo_id)
        
        if not services_data:
            console.print("[red]❌ Could not retrieve services data.[/red]")
            return
        
        services = services_data.get('services', [])
        infrastructure = services_data.get('infrastructure_analysis', {})
        
        if not services:
            console.print("[yellow]🏗️ No services detected in this repository.[/yellow]")
        else:
            table = Table(title="[bold green]🏗️ Detected Services[/bold green]", border_style="green")
            table.add_column("Service", style="white")
            table.add_column("Version", style="green")
            table.add_column("Type", style="blue")
            table.add_column("Source", style="yellow")
            
            for service in services:
                table.add_row(
                    service.get('name', 'Unknown'),
                    service.get('version', 'Unknown'),
                    service.get('service_type', 'Unknown'),
                    service.get('source', 'Unknown')
                )
            
            console.print(table)
        
        if infrastructure:
            infra_content = "🏗️ **Infrastructure Analysis:**\n"
            if infrastructure.get('containerized'):
                infra_content += "• ✅ Containerized deployment detected\n"
            
            for infra_type, items in infrastructure.items():
                if isinstance(items, list) and items:
                    infra_content += f"• {infra_type.replace('_', ' ').title()}: {len(items)} found\n"
            
            console.print(Panel(infra_content, title="[green]🏗️ Infrastructure[/green]", border_style="green"))

    def show_security(self):
        """Show security analysis."""
        with Status("[cyan]🔒 Performing security analysis...[/cyan]"):
            security_data = self.api.get_bom_security(self.repo_id)
        
        if not security_data:
            console.print("[red]❌ Could not retrieve security data.[/red]")
            return
        
        summary = security_data.get('summary', {})
        recommendations = security_data.get('security_recommendations', [])
        compliance = security_data.get('compliance_status', {})
        
        security_content = f"""🔒 **Security Summary:**
• Total Dependencies Scanned: {summary.get('total_dependencies', 0)}
• High Risk Dependencies: {summary.get('high_risk_dependencies', 0)}
• Last Scan: {summary.get('last_scan', 'Never')}

🛡️ **Compliance Status:**
• Overall Compliant: {'✅ Yes' if compliance.get('compliant') else '❌ No'}"""
        
        if compliance.get('checks'):
            for check, status in compliance.get('checks', {}).items():
                emoji = "✅" if status == "pass" else "❌"
                security_content += f"\n• {check.replace('_', ' ').title()}: {emoji} {status}"
        
        console.print(Panel(security_content, title="[bold red]🔒 Security Analysis[/bold red]", border_style="red"))
        
        if recommendations:
            rec_content = "\n".join([f"• {rec}" for rec in recommendations])
            console.print(Panel(rec_content, title="[yellow]💡 Security Recommendations[/yellow]", border_style="yellow"))

    def show_compare(self):
        """Compare BOM with another repository."""
        try:
            analyzed_repos = find_analyzed_repos()
            other_repos = [repo for repo in analyzed_repos if repo['repo_id'] != self.repo_id]
            
            if not other_repos:
                console.print("[yellow]No other analyzed repositories found for comparison.[/yellow]")
                return
            
            console.print("Select a repository to compare with:")
            table = Table(show_header=False, box=None, padding=(0, 2))
            for i, repo in enumerate(other_repos, 1):
                table.add_row(f"([bold cyan]{i}[/bold cyan])", repo['display_name'])
            console.print(table)
            
            choice = Prompt.ask("Enter choice", choices=[str(i) for i in range(1, len(other_repos) + 1)], show_choices=False, default='1')
            other_repo_id = other_repos[int(choice) - 1]['repo_id']
            
            comparison_type = Prompt.ask("Comparison type", choices=["dependencies", "services", "languages", "comprehensive"], default="dependencies")
            
            with Status(f"[cyan]📊 Comparing {self.repo_id} with {other_repo_id}...[/cyan]"):
                comparison_data = self.api.compare_bom(self.repo_id, other_repo_id, comparison_type)
            
            if not comparison_data:
                console.print("[red]❌ Could not perform comparison.[/red]")
                return
            
            comparison = comparison_data.get('comparison', {})
            
            comp_content = f"""📊 **BOM Comparison Results:**
**Repositories:** {self.repo_id} vs {other_repo_id}
**Comparison Type:** {comparison_type.title()}

🔍 **Common Items:** {len(comparison.get('common', []))}
🔹 **Unique to {self.repo_id}:** {len(comparison.get('unique_to_repo_1', []))}
🔸 **Unique to {other_repo_id}:** {len(comparison.get('unique_to_repo_2', []))}"""
            
            console.print(Panel(comp_content, title="[bold magenta]📊 BOM Comparison[/bold magenta]", border_style="magenta"))
            
        except (KeyboardInterrupt, ValueError, IndexError):
            console.print("\n[yellow]Comparison cancelled.[/yellow]")

    def regenerate_bom(self):
        """Regenerate BOM data."""
        try:
            force = Confirm.ask("Force regeneration even if BOM is recent?", default=False)
            
            with Status("[cyan]🔄 Regenerating BOM data...[/cyan]"):
                result = self.api.regenerate_bom(self.repo_id, force)
            
            if result and result.get('regenerated'):
                console.print("[green]✅ BOM data regenerated successfully.[/green]")
            elif result and result.get('message'):
                console.print(Panel(result['message'], title="[yellow]BOM Regeneration[/yellow]", border_style="yellow"))
            else:
                console.print("[red]❌ Failed to regenerate BOM data.[/red]")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]BOM regeneration cancelled.[/yellow]")


# --- NEW: Repository Management Session ---
class RepositoryManagementSession:
    def __init__(self):
        self.api = LumiereAPIClient()
        console.print(Panel(
            f"[bold green]🗃️ Repository Management[/bold green]\n\n"
            "[dim]Manage analyzed repositories. Type 'help' for commands.[/dim]",
            border_style="green"
        ))

    def loop(self):
        """Main interactive loop for repository management."""
        global prompt_session
        prompt_session = PromptSession(
            history=FileHistory(str(history_path)),
            completer=repo_completer,
            style=prompt_style
        )

        while True:
            try:
                repo_prompt_text = [
                    ('class:lumiere', 'Lumière'),
                    ('class:provider', ' (Repo-Mgmt)'),
                    ('class:separator', ' > '),
                ]

                command = prompt_session.prompt(repo_prompt_text).strip()

                if not command:
                    continue
                if command.lower() in ("q", "quit", "exit", "back"):
                    break
                if command.lower() in ("h", "help"):
                    self.display_help()
                    continue

                self.handle_repo_command(command)

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' or 'back' to return to the main menu.[/yellow]")
                continue
            except EOFError:
                break

        console.print("[green]🗃️ Repository management session ended.[/green]")
        # Restore main completer
        prompt_session = PromptSession(
            history=FileHistory(str(history_path)),
            completer=main_completer,
            style=prompt_style
        )

    def display_help(self):
        """Display repository management help commands."""
        help_table = Table(title="[bold green]🗃️ Repository Management Commands[/bold green]", border_style="green")
        help_table.add_column("Command", style="bold cyan")
        help_table.add_column("Description", style="white")
        
        help_table.add_row("list / l", "List all analyzed repositories")
        help_table.add_row("status / s", "Check repository analysis status")
        help_table.add_row("delete / d", "Delete repository data")
        help_table.add_row("help / h", "Show this help menu")
        help_table.add_row("back / exit / quit", "Return to main menu")
        
        console.print(help_table)

    def handle_repo_command(self, command: str):
        """Handle repository management commands."""
        cmd = command.lower().strip()
        
        if cmd in ("list", "l"):
            self.list_repositories()
        elif cmd in ("status", "s"):
            self.check_status()
        elif cmd in ("delete", "d"):
            self.delete_repository()
        else:
            console.print("[red]❌ Unknown command. Type 'help' for available commands.[/red]")

    def list_repositories(self):
        """List all analyzed repositories."""
        with Status("[cyan]🗃️ Fetching repository list...[/cyan]"):
            repos = self.api.list_repositories()
        
        if not repos:
            console.print("[yellow]📭 No analyzed repositories found.[/yellow]")
            return
        
        table = Table(title=f"[bold green]🗃️ Analyzed Repositories ({len(repos)} found)[/bold green]", border_style="green")
        table.add_column("Repository", style="white")
        table.add_column("Display Name", style="cyan")
        table.add_column("URL", style="blue")
        
        for repo in repos:
            table.add_row(
                repo.get('repo_id', 'Unknown'),
                repo.get('display_name', 'Unknown'),
                repo.get('url', 'Unknown')
            )
        
        console.print(table)

    def check_status(self):
        """Check repository analysis status."""
        try:
            repo_id = Prompt.ask("Enter repository ID to check")
            
            with Status(f"[cyan]🔍 Checking status of {repo_id}...[/cyan]"):
                status_data = self.api.get_repository_status(repo_id)
                detail_data = self.api.get_repository_detail(repo_id)
            
            if not status_data:
                console.print(f"[red]❌ Repository '{repo_id}' not found.[/red]")
                return
            
            status = status_data.get('status', 'unknown')
            status_emoji = "✅" if status == "complete" else "❌"
            
            status_content = f"""🔍 **Repository Status**
• Repository ID: {repo_id}
• Analysis Status: {status_emoji} {status.title()}"""
            
            if detail_data:
                metadata = detail_data.get('metadata', {})
                status_content += f"""
• Files Analyzed: {metadata.get('total_files', 'Unknown')}
• Analysis Date: {metadata.get('analysis_date', 'Unknown')}
• Primary Language: {metadata.get('primary_language', 'Unknown')}"""
            
            console.print(Panel(status_content, title=f"[bold cyan]🔍 Status: {repo_id}[/bold cyan]", border_style="cyan"))
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Status check cancelled.[/yellow]")

    def delete_repository(self):
        """Delete repository data."""
        try:
            analyzed_repos = find_analyzed_repos()
            if not analyzed_repos:
                console.print("[yellow]📭 No repositories available to delete.[/yellow]")
                return
            
            console.print("Select a repository to delete:")
            table = Table(show_header=False, box=None, padding=(0, 2))
            for i, repo in enumerate(analyzed_repos, 1):
                table.add_row(f"([bold red]{i}[/bold red])", repo['display_name'])
            console.print(table)
            
            choice = Prompt.ask("Enter choice", choices=[str(i) for i in range(1, len(analyzed_repos) + 1)], show_choices=False, default='1')
            selected_repo = analyzed_repos[int(choice) - 1]
            
            if not Confirm.ask(f"[bold red]Are you sure you want to delete '{selected_repo['display_name']}'?[/bold red]", default=False):
                console.print("[yellow]Deletion cancelled.[/yellow]")
                return
            
            with Status(f"[red]🗑️ Deleting {selected_repo['repo_id']}...[/red]"):
                result = self.api.delete_repository(selected_repo['repo_id'])
            
            if result is not None:  # 204 No Content returns None but is success
                console.print(f"[green]✅ Repository '{selected_repo['display_name']}' deleted successfully.[/green]")
            else:
                console.print(f"[red]❌ Failed to delete repository.[/red]")
                
        except (KeyboardInterrupt, ValueError, IndexError):
            console.print("\n[yellow]Repository deletion cancelled.[/yellow]")


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

# --- Onboarding Session Manager ---
class OnboardingSession:
    def __init__(self, repo_id: str, repo_url: str):
        self.repo_id = repo_id
        self.repo_url = repo_url
        self.api = LumiereAPIClient()
        console.print(Panel(
            f"[bold green]🎓 Onboarding Concierge[/bold green]\n"
            f"[yellow]Repository:[/yellow] {self.repo_id}\n"
            f"[yellow]URL:[/yellow] {self.repo_url}\n\n"
            "[dim]Helping new developers get started. Type 'help' for commands.[/dim]",
            border_style="green"
        ))

    def loop(self):
        """Main interactive loop for onboarding assistance."""
        global prompt_session
        prompt_session = PromptSession(
            history=FileHistory(str(history_path)),
            completer=onboard_completer,
            style=prompt_style
        )

        while True:
            try:
                onboard_prompt_text = [
                    ('class:lumiere', 'Lumière'),
                    ('class:provider', f' (Onboard/{self.repo_id})'),
                    ('class:separator', ' > '),
                ]

                command = prompt_session.prompt(onboard_prompt_text).strip()

                if not command:
                    continue
                if command.lower() in ("q", "quit", "exit", "back"):
                    break
                if command.lower() in ("h", "help"):
                    display_interactive_help('onboard')
                    continue

                self.handle_onboard_command(command)

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' or 'back' to return to the main menu.[/yellow]")
                continue
            except EOFError:
                break

        console.print("[green]🎓 Onboarding session ended.[/green]")
        # Restore main completer
        prompt_session = PromptSession(
            history=FileHistory(str(history_path)),
            completer=main_completer,
            style=prompt_style
        )

    def handle_onboard_command(self, command: str):
        """Handle onboarding-specific commands."""
        parts = command.split()
        if not parts:
            return

        cmd = parts[0].lower()

        if cmd in ("scout", "s"):
            self.scout_good_first_issues()
        elif cmd in ("expert", "e"):
            if len(parts) < 2:
                console.print("[red]Usage: expert <file_path>[/red]")
                return
            file_path = " ".join(parts[1:])
            self.find_expert_for_file(file_path)
        elif cmd in ("guide", "g"):
            if len(parts) < 2:
                console.print("[red]Usage: guide <issue_number>[/red]")
                return
            try:
                issue_number = int(parts[1])
                self.generate_onboarding_guide(issue_number)
            except ValueError:
                console.print("[red]Issue number must be a valid integer.[/red]")
        else:
            console.print(f"[red]Unknown command: {cmd}. Type 'help' for available commands.[/red]")

    def scout_good_first_issues(self):
        """Find and display issues suitable for newcomers."""
        console.print("\n[bold cyan]🔍 Scouting for good first issues...[/bold cyan]")
        
        with Status("[bold green]Analyzing issues with onboarding suitability scoring...", console=console):
            try:
                # Call the strategist to get prioritized issues with onboarding scores
                response = self.api._request("POST", "strategist/prioritize/", {
                    "repo_url": self.repo_url
                })
                
                if not response or 'prioritized_issues' not in response:
                    console.print("[red]❌ Failed to fetch issues for analysis.[/red]")
                    return
                
                issues = response['prioritized_issues']
                # Filter for issues with high onboarding suitability
                suitable_issues = [
                    issue for issue in issues 
                    if issue.get('onboarding_suitability_score', 0) > 70
                ]
                
                if not suitable_issues:
                    console.print("[yellow]No issues found with high onboarding suitability scores.[/yellow]")
                    return
                
                # Display results in a nice table
                table = Table(title="🌟 Good First Issues", border_style="green")
                table.add_column("Issue #", style="bold cyan", width=8)
                table.add_column("Title", style="white", width=40)
                table.add_column("Onboarding Score", style="bold green", width=15)
                table.add_column("Blast Radius", style="yellow", width=12)
                
                for issue in suitable_issues[:10]:  # Show top 10
                    table.add_row(
                        f"#{issue['number']}",
                        issue['title'][:35] + "..." if len(issue['title']) > 35 else issue['title'],
                        f"{issue.get('onboarding_suitability_score', 0):.1f}/100",
                        str(issue.get('blast_radius', 'N/A'))
                    )
                
                console.print(table)
                
                if len(suitable_issues) > 10:
                    console.print(f"[dim]... and {len(suitable_issues) - 10} more suitable issues[/dim]")
                
                console.print(f"\n[green]✅ Found {len(suitable_issues)} issues suitable for newcomers![/green]")
                
            except Exception as e:
                console.print(f"[red]❌ Error during issue analysis: {str(e)}[/red]")

    def find_expert_for_file(self, file_path: str):
        """Find experts for a specific file."""
        console.print(f"\n[bold cyan]👨‍💻 Finding experts for: {file_path}[/bold cyan]")
        
        with Status("[bold green]Analyzing file expertise...", console=console):
            try:
                response = self.api._request("POST", "expertise/find/", {
                    "repo_id": self.repo_id,
                    "file_path": file_path,
                    "type": "file"
                })
                
                if not response or 'experts' not in response:
                    console.print("[red]❌ Failed to find experts for this file.[/red]")
                    return
                
                experts = response['experts']
                
                if not experts:
                    console.print(f"[yellow]No experts found for {file_path}.[/yellow]")
                    return
                
                # Display experts in a table
                table = Table(title=f"🎯 Experts for {file_path}", border_style="blue")
                table.add_column("Rank", style="bold cyan", width=6)
                table.add_column("Expert", style="white", width=25)
                table.add_column("Email", style="yellow", width=30)
                table.add_column("Score", style="bold green", width=8)
                table.add_column("Lines", style="magenta", width=8)
                
                for i, expert in enumerate(experts[:5], 1):  # Show top 5
                    table.add_row(
                        str(i),
                        expert['author'],
                        expert['email'],
                        f"{expert['score']:.1f}",
                        str(expert['details']['blame_lines'])
                    )
                
                console.print(table)
                console.print(f"\n[green]✅ Found {len(experts)} experts for this file![/green]")
                console.print("[dim]💡 Tip: Reach out to the top expert for guidance on this file.[/dim]")
                
            except Exception as e:
                console.print(f"[red]❌ Error finding experts: {str(e)}[/red]")

    def generate_onboarding_guide(self, issue_number: int):
        """Generate a personalized onboarding guide for an issue."""
        console.print(f"\n[bold cyan]📚 Generating onboarding guide for issue #{issue_number}[/bold cyan]")
        
        with Status("[bold green]Creating personalized learning path...", console=console):
            try:
                response = self.api._request("POST", "onboarding/guide/", {
                    "repo_id": self.repo_id,
                    "issue_number": issue_number
                })
                
                if not response or not response.get('onboarding_guide'):
                    console.print("[red]❌ Failed to generate onboarding guide.[/red]")
                    if response and response.get('error'):
                        console.print(f"[red]Error: {response['error']}[/red]")
                    return
                
                # Display the guide
                guide_content = response['onboarding_guide']
                
                # Show summary first
                console.print(Panel(
                    f"[bold green]📖 Onboarding Guide Generated![/bold green]\n"
                    f"[yellow]Issue:[/yellow] #{issue_number} - {response.get('issue_title', 'Unknown')}\n"
                    f"[yellow]Learning Steps:[/yellow] {response.get('learning_path_steps', 0)}\n"
                    f"[yellow]Core Files:[/yellow] {len(response.get('locus_files', []))}\n",
                    border_style="green"
                ))
                
                # Display the full guide using Rich Markdown
                console.print(Markdown(guide_content))
                
                console.print(f"\n[green]✅ Your personalized onboarding guide is ready![/green]")
                console.print("[dim]💡 Follow the steps above to understand this issue. Good luck! 🍀[/dim]")
                
            except Exception as e:
                console.print(f"[red]❌ Error generating guide: {str(e)}[/red]")


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

        if command in ('m', 'mage'):
            self.execute_action(command, {}) # Pass empty dict, issue not needed
            console.print("\n[dim]💡 Type [bold]list[/bold] to see issues, or [bold]help[/bold] for commands.[/dim]")
            return

        if command in ('f', 'fix') and self.last_rca_report:
             issue = next((iss for iss in self.issues if iss.get('number') == self.last_rca_issue_num), None)
             if issue:
                 self.execute_action(command, issue)
                 console.print("\n[dim]💡 Type [bold]list[/bold] to see issues, or [bold]help[/bold] for commands.[/dim]")
                 return

        if command not in ('f', 'fix', 'b', 'briefing', 'r', 'rca', 'd', 'details', 'm', 'mage'):
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

        if command in ('m', 'mage'):
            self.handle_mage_session()
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

    def handle_mage_session(self):
        """Handle The Mage interactive session for code transformation."""
        # Import mage service
        sys.path.append('backend')
        from backend.lumiere_core.services import mage_service
        
        console.print(Panel(
            f"[bold magenta]🔮 The Mage's Sanctum[/bold magenta]\n"
            f"[yellow]Repository:[/yellow] {self.repo_id}\n"
            f"[yellow]URL:[/yellow] {self.repo_url}\n\n"
            "[dim]Master of code transmutation. Cast spells to transform your code.[/dim]",
            border_style="magenta"
        ))
        
        # Show available spells
        spells = mage_service.list_available_spells()
        console.print("\n[bold magenta]📜 Available Spells:[/bold magenta]")
        
        spells_table = Table(show_header=True, header_style="bold cyan", border_style="magenta")
        spells_table.add_column("Spell", style="cyan")
        spells_table.add_column("Description", style="white")
        
        for spell_name, description in spells.items():
            spells_table.add_row(spell_name, description)
        
        console.print(spells_table)
        
        # Interactive spell casting loop
        mage_completer = WordCompleter(list(spells.keys()) + ['help', 'h', 'list', 'l', 'back', 'exit', 'quit'], ignore_case=True)
        mage_session = PromptSession(
            history=FileHistory(str(history_path)),
            completer=mage_completer,
            style=prompt_style
        )
        
        console.print("\n[dim]💡 Usage: cast <spell_name> <file_path> <target_identifier> [options][/dim]")
        console.print("[dim]Example: cast translate_contract src/models.py UserProfile --target=typescript[/dim]")
        
        while True:
            try:
                mage_prompt_text = [
                    ('class:lumiere', 'Lumière'),
                    ('class:provider', f' (Mage/{self.repo_id})'),
                    ('class:separator', ' > '),
                ]
                
                command = mage_session.prompt(mage_prompt_text).strip()
                
                if not command:
                    continue
                if command.lower() in ("q", "quit", "exit", "back"):
                    break
                if command.lower() in ("h", "help"):
                    console.print(spells_table)
                    continue
                if command.lower() in ("l", "list"):
                    console.print(spells_table)
                    continue
                
                # Parse cast command
                if command.startswith("cast "):
                    self._handle_cast_command(command[5:].strip())
                else:
                    console.print("[yellow]💡 Start your command with 'cast' followed by spell name, file path, and target.[/yellow]")
                    console.print("[dim]Type 'help' to see available spells, or 'back' to return.[/dim]")
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'back' or 'exit' to return to analysis session.[/yellow]")
                continue
            except EOFError:
                break
        
        console.print("[magenta]🔮 The Mage session ended.[/magenta]")
    
    def _handle_cast_command(self, command_args: str):
        """Handle casting a specific spell."""
        try:
            # Import mage service
            from backend.lumiere_core.services import mage_service
            
            # Parse arguments
            parts = shlex.split(command_args)
            if len(parts) < 3:
                console.print("[red]❌ Usage: cast <spell_name> <file_path> <target_identifier> [--option=value][/red]")
                return
            
            spell_name = parts[0]
            file_path = parts[1]
            target_identifier = parts[2]
            
            # Parse additional options
            kwargs = {}
            for part in parts[3:]:
                if part.startswith("--"):
                    if "=" in part:
                        key, value = part[2:].split("=", 1)
                        kwargs[key] = value
                    else:
                        kwargs[part[2:]] = True
            
            console.print(f"\n[magenta]🔮 Casting {spell_name} on {target_identifier} in {file_path}...[/magenta]")
            
            # Cast the spell
            with Status("[magenta]✨ The Mage is weaving magic...[/magenta]"):
                result = mage_service.cast_transformation_spell(
                    self.repo_id, spell_name, file_path, target_identifier, **kwargs
                )
            
            if "error" in result:
                console.print(f"[red]❌ Spell failed: {result['error']}[/red]")
                return
            
            # Display the transformation
            console.print(Panel(
                f"[bold green]✨ Spell Cast Successfully![/bold green]\n"
                f"[yellow]Spell:[/yellow] {result['spell_name']}\n"
                f"[yellow]Target:[/yellow] {result['target_identifier']}\n"
                f"[yellow]File:[/yellow] {result['file_path']}\n"
                f"[yellow]Type:[/yellow] {result.get('transformation_type', 'unknown')}\n"
                f"[yellow]Description:[/yellow] {result.get('description', 'No description')}",
                title="[green]🔮 Magic Complete[/green]",
                border_style="green"
            ))
            
            # Show before and after
            console.print("\n[bold]📜 Original Code:[/bold]")
            console.print(Panel(result['original_code'], border_style="red", title="[red]Before[/red]"))
            
            console.print("\n[bold]✨ Transformed Code:[/bold]")
            console.print(Panel(result['transformed_code'], border_style="green", title="[green]After[/green]"))
            
            # Ask if user wants to apply the transformation
            try:
                if Confirm.ask("\n[bold]Apply this transformation to the file?[/bold]", default=False):
                    with Status("[magenta]📝 Applying transformation...[/magenta]"):
                        apply_result = mage_service.apply_code_transformation(self.repo_id, result)
                    
                    if apply_result.get("success"):
                        console.print(Panel(
                            f"[bold green]✅ Transformation Applied![/bold green]\n"
                            f"{apply_result.get('message', 'Code has been updated successfully.')}",
                            border_style="green"
                        ))
                    else:
                        console.print(f"[red]❌ Failed to apply transformation: {apply_result.get('error', 'Unknown error')}[/red]")
                else:
                    console.print("[yellow]Transformation not applied. The spell result is shown above for your reference.[/yellow]")
            except KeyboardInterrupt:
                console.print("\n[yellow]Transformation not applied.[/yellow]")
                
        except Exception as e:
            console.print(f"[red]❌ Error casting spell: {str(e)}[/red]")


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
        help_table.add_row("dance / da", "🕺 The Masked Dancer - visualize execution flow")
        help_table.add_row("summon / su", "🧙 The Summoner - generate code patterns")
        help_table.add_row("review", "Perform an AI-powered review of a Pull Request")
        help_table.add_row("dashboard / d", "View the project health dashboard")
        help_table.add_row("profile / p", "Get GitHub user profile analysis")
        help_table.add_row("bom / b", "Bill of Materials analysis for a repository")
        help_table.add_row("onboard / o", "Onboarding Concierge - help new developers")
        help_table.add_row("repo-mgmt / rm", "Repository management (list, delete, status)")
        help_table.add_row("quartermaster / qm", "⚖️ Supply-chain management & vulnerability scanning")
        help_table.add_row("loremaster / lm", "📚 API documentation & client code generation")
        help_table.add_row("librarian / lib", "📚 Local directory archives & knowledge management")

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
        help_table.add_row("mage / m", "🔮 The Mage - intelligent code transformation")
        help_table.add_row("help / h", "Show this help menu")
        help_table.add_row("back / exit / quit", "Return to main menu")
    elif context == 'onboard':
        help_table.add_row("scout / s", "Find 'good first issues' with onboarding scores")
        help_table.add_row("expert <file_path>", "Find experts for a specific file")
        help_table.add_row("guide <issue_number>", "Generate personalized onboarding guide")
        help_table.add_row("help / h", "Show this help menu")
        help_table.add_row("back / exit / quit", "Return to main menu")

    console.print(help_table)

# --- Main Entry Point ---
app = typer.Typer()

@app.command()
def run():
    """Launch Lumière interactive shell with modern features."""
    # Display startup animation
    AnimationManager.typing_effect("🚀 Initializing Lumière Sémantique...", 0.02)
    
    api_client = LumiereAPIClient()

    # Enhanced health check with progress
    with ProgressBarManager.create_modern_progress() as progress:
        task = progress.add_task("[cyan]Connecting to backend...", total=3)
        progress.update(task, completed=1)
        
        health_status = api_client._request("GET", "health/")
        progress.update(task, completed=2)
        
        if health_status is None:
            progress.update(task, description="[red]✗ Connection failed", completed=3)
            console.print("[bold red]Lumière CLI cannot start without a backend connection.[/bold red]")
            sys.exit(1)
        
        progress.update(task, description="[green]✓ Backend connected", completed=3)

    load_config()
    
    # Modern welcome screen with theme
    theme = cli_state.get("theme", "modern")
    welcome_content = f"""[bold]✨ Welcome to Lumière Sémantique ✨[/bold]

[cyan]🎯 Modern Code Intelligence Platform[/cyan]
[dim]The Conversational Mission Controller is active[/dim]

[green]Backend:[/green] Online at {API_BASE_URL}
[green]Theme:[/green] {theme.title()}
[green]Features:[/green] Fuzzy search, Smart completion, Live preview

[yellow]💡 Quick Start:[/yellow]
• Type [cyan]dashboard[/cyan] or press [yellow]Ctrl+D[/yellow] for overview
• Use [cyan]analyze <repo>[/cyan] for deep analysis  
• Press [yellow]Ctrl+H[/yellow] for help anytime"""

    AnimationManager.fade_in_panel(welcome_content, "🌟 Lumière Sémantique", 0.08)
    
    # Show enhanced help
    EnhancedHelp.show_general_help()

    next_command = None
    context = {}

    while True:
        try:
            if next_command is None:
                prompt_text = get_prompt_text()
                
                # Show command preview as user types
                def get_preview(text):
                    if text.strip():
                        preview = CommandPreview.preview_command(text.strip())
                        return f"[dim]{preview}[/dim]"
                    return ""
                
                # Enhanced prompt with live preview
                try:
                    command = prompt_session.prompt(prompt_text).strip()
                except KeyboardInterrupt:
                    result = prompt_session.app.result
                    if result == 'dashboard':
                        command = 'dashboard'
                    elif result == 'help':
                        command = 'help'
                    elif result == 'analyze':
                        command = 'analyze'
                    elif result == 'repo-mgmt':
                        command = 'repo-mgmt'
                    else:
                        continue
            else:
                command = next_command
                console.print(f"\n[dim]Executing suggested action: [bold]{SyntaxHighlighter.highlight_command(command)}[/bold]...[/dim]")

            next_command = None

            if not command:
                continue
            
            # Add to command history
            CommandHistory.add_to_history(command)

            if command.lower() in ("exit", "quit", "x"):
                NotificationManager.add_notification("Thanks for using Lumière! 👋", "info")
                AnimationManager.typing_effect("Goodbye! 👋", 0.05)
                break

            elif command.lower() == "back":
                console.print()
                continue

            elif command.lower() in ("help", "h"):
                # Enhanced help with contextual information
                parts = command.split()
                if len(parts) > 1:
                    EnhancedHelp.show_contextual_help(parts[1])
                else:
                    EnhancedHelp.show_general_help()
                continue
            
            elif command.lower() in ("history", "hist"):
                # New command history feature
                parts = command.split()
                search_query = " ".join(parts[1:]) if len(parts) > 1 else ""
                CommandHistory.show_history(search_query)
                continue
            
            elif command.lower() in ("theme"):
                # New theme switching feature
                parts = command.split()
                if len(parts) > 1:
                    new_theme = parts[1].lower()
                    if new_theme in ThemeManager.THEMES:
                        cli_state["theme"] = new_theme
                        # Update theme dynamically
                        new_style = ThemeManager.get_style(new_theme)
                        prompt_session.style = new_style
                        NotificationManager.add_notification(f"Theme changed to {new_theme}", "success")
                    else:
                        available_themes = ", ".join(ThemeManager.THEMES.keys())
                        console.print(f"[red]Unknown theme. Available: {available_themes}[/red]")
                else:
                    console.print(f"[yellow]Current theme: {cli_state.get('theme', 'modern')}[/yellow]")
                    console.print(f"[dim]Available themes: {', '.join(ThemeManager.THEMES.keys())}[/dim]")
                continue
            
            elif command.lower() in ("dashboard", "d"):
                # Enhanced dashboard
                ModernDashboard.show_dashboard()
                continue
            
            elif command.lower() in ("animate"):
                # Toggle animations
                cli_state["animations_enabled"] = not cli_state.get("animations_enabled", True)
                status = "enabled" if cli_state["animations_enabled"] else "disabled"
                NotificationManager.add_notification(f"Animations {status}", "info")
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

            elif command.lower() in ("dance", "da"):
                if not cli_state.get("model"):
                    console.print("[bold red]Please select a model first using the 'config' command.[/bold red]")
                    continue
                
                # Import dance service
                sys.path.append('backend')
                from backend.lumiere_core.services import dance_service
                
                # Get repository to analyze
                repo_url = context.get("repo_url")
                if not repo_url:
                    analyzed_repos = find_analyzed_repos()
                    if not analyzed_repos:
                        console.print("[yellow]No analyzed repositories found. Use 'analyze' to ingest a repo first.[/yellow]")
                        continue
                    console.print(Panel("Select a repository to dance with.", title="[magenta]💃 The Masked Dancer[/magenta]", border_style="magenta"))
                    table = Table(show_header=False, box=None, padding=(0, 2))
                    for i, repo in enumerate(analyzed_repos, 1): 
                        table.add_row(f"([bold cyan]{i}[/bold cyan])", repo['display_name'])
                    console.print(table)
                    try:
                        choice = Prompt.ask("Enter choice", choices=[str(i) for i in range(1, len(analyzed_repos) + 1)], show_choices=False, default='1')
                        repo_url = analyzed_repos[int(choice) - 1]['url']
                    except (ValueError, IndexError, KeyboardInterrupt): 
                        continue
                
                repo_id = repo_url.replace("https://github.com/", "").replace("/", "_")
                
                # Get user query for what to trace
                console.print("\n[bold magenta]💃 The Masked Dancer[/bold magenta] reveals the secret choreography of your application.")
                console.print("[dim]Describe what execution flow you'd like to trace (e.g., 'user login API call', 'data processing pipeline').[/dim]")
                
                try:
                    query = Prompt.ask("\n[bold]What dance would you like to see?[/bold]").strip()
                    if not query:
                        console.print("[yellow]No query provided. Dance cancelled.[/yellow]")
                        continue
                except KeyboardInterrupt:
                    continue
                
                # Ask for output format
                try:
                    format_choice = Prompt.ask(
                        "\n[bold]Choose visualization format[/bold]",
                        choices=["cli", "svg"],
                        default="cli"
                    )
                except KeyboardInterrupt:
                    continue
                
                with Status("[magenta]🔍 The Oracle is finding the best starting point...[/magenta]"):
                    entry_point_result = dance_service.find_entry_point(repo_id, query)
                
                if "error" in entry_point_result:
                    console.print(f"[red]❌ Could not find starting point: {entry_point_result['error']}[/red]")
                    continue
                
                suggested_node = entry_point_result.get("suggested_node")
                confidence = entry_point_result.get("confidence", 0)
                
                console.print(Panel(
                    f"[bold green]🎯 Starting Point Found[/bold green]\n"
                    f"[yellow]Function/Method:[/yellow] {suggested_node}\n"
                    f"[yellow]File:[/yellow] {entry_point_result.get('file_path', 'Unknown')}\n"
                    f"[yellow]Confidence:[/yellow] {confidence:.2f}\n\n"
                    f"[dim]Context: {entry_point_result.get('context', '')[:100]}...[/dim]",
                    border_style="green"
                ))
                
                try:
                    if not Confirm.ask(f"[bold]Begin dance with '{suggested_node}'?[/bold]", default=True):
                        # Show alternatives
                        alternatives = entry_point_result.get("alternatives", [])
                        if alternatives:
                            console.print("\n[bold]Alternative starting points:[/bold]")
                            for i, alt in enumerate(alternatives[:5], 1):
                                console.print(f"  {i}. {alt['node_id']} (confidence: {alt['confidence']:.2f})")
                            
                            try:
                                alt_choice = Prompt.ask("Choose alternative (1-5) or press Enter to cancel", default="")
                                if alt_choice and alt_choice.isdigit() and 1 <= int(alt_choice) <= len(alternatives):
                                    suggested_node = alternatives[int(alt_choice) - 1]['node_id']
                                else:
                                    console.print("[yellow]Dance cancelled.[/yellow]")
                                    continue
                            except KeyboardInterrupt:
                                continue
                        else:
                            console.print("[yellow]Dance cancelled.[/yellow]")
                            continue
                except KeyboardInterrupt:
                    continue
                
                # Perform the dance!
                with Status("[magenta]💃 The Masked Dancer is tracing the execution flow...[/magenta]"):
                    if format_choice == "svg":
                        output_file = f"dance_of_{suggested_node.replace('.', '_')}.svg"
                        dance_result = dance_service.visualize_dance(repo_id, suggested_node, "svg", output_file)
                        console.print(Panel(
                            f"[bold green]✨ Dance visualization complete![/bold green]\n"
                            f"[yellow]SVG file saved:[/yellow] {output_file}\n\n"
                            "[dim]Open the SVG file in a browser to see the animated execution flow![/dim]",
                            title="[green]🎭 Dance Complete[/green]",
                            border_style="green"
                        ))
                    else:
                        dance_result = dance_service.visualize_dance(repo_id, suggested_node, "cli")
                        console.print("\n" + dance_result)
                
                context = {"repo_url": repo_url, "repo_id": repo_id}
                continue

            elif command.lower() in ("summon", "su"):
                if not cli_state.get("model"):
                    console.print("[bold red]Please select a model first using the 'config' command.[/bold red]")
                    continue
                
                # Import summoner service
                sys.path.append('backend')
                from backend.lumiere_core.services import summoner_service
                
                # Get repository to analyze
                repo_url = context.get("repo_url")
                if not repo_url:
                    analyzed_repos = find_analyzed_repos()
                    if not analyzed_repos:
                        console.print("[yellow]No analyzed repositories found. Use 'analyze' to ingest a repo first.[/yellow]")
                        continue
                    console.print(Panel("Select a repository to summon patterns for.", title="[cyan]🧙 The Summoner[/cyan]", border_style="cyan"))
                    table = Table(show_header=False, box=None, padding=(0, 2))
                    for i, repo in enumerate(analyzed_repos, 1): 
                        table.add_row(f"([bold cyan]{i}[/bold cyan])", repo['display_name'])
                    console.print(table)
                    try:
                        choice = Prompt.ask("Enter choice", choices=[str(i) for i in range(1, len(analyzed_repos) + 1)], show_choices=False, default='1')
                        repo_url = analyzed_repos[int(choice) - 1]['url']
                    except (ValueError, IndexError, KeyboardInterrupt): 
                        continue
                
                repo_id = repo_url.replace("https://github.com/", "").replace("/", "_")
                
                console.print("\n[bold cyan]🧙 The Summoner[/bold cyan] materializes code patterns from the architectural DNA.")
                
                # Show available recipes
                recipes = summoner_service.list_summoning_recipes()
                console.print("\n[bold cyan]📜 Available Summoning Recipes:[/bold cyan]")
                
                recipes_table = Table(show_header=True, header_style="bold cyan", border_style="cyan")
                recipes_table.add_column("Recipe", style="cyan")
                recipes_table.add_column("Description", style="white")
                
                for recipe_name, recipe_info in recipes.items():
                    recipes_table.add_row(recipe_name, recipe_info["description"])
                
                console.print(recipes_table)
                
                # Get recipe choice
                try:
                    recipe_choice = Prompt.ask(
                        "\n[bold]Choose a recipe to summon[/bold]",
                        choices=list(recipes.keys()),
                        show_choices=False
                    )
                except KeyboardInterrupt:
                    continue
                
                recipe_info = recipes[recipe_choice]
                
                # Get parameters for the recipe
                console.print(f"\n[bold]Recipe: {recipe_info['name']}[/bold]")
                console.print(f"[dim]{recipe_info['description']}[/dim]")
                
                parameters = {}
                for param in recipe_info.get("parameters", []):
                    try:
                        if param == "path":
                            value = Prompt.ask(f"Enter API path (e.g., /users)", default="/items")
                        elif param == "methods":
                            value = Prompt.ask(f"Enter HTTP methods (comma-separated)", default="get,post")
                        elif param == "model_name":
                            value = Prompt.ask(f"Enter model/entity name", default="Item")
                        elif param == "component_name":
                            value = Prompt.ask(f"Enter component name", default="MyComponent")
                        else:
                            value = Prompt.ask(f"Enter {param}")
                        parameters[param] = value
                    except KeyboardInterrupt:
                        console.print("\n[yellow]Summoning cancelled.[/yellow]")
                        break
                else:
                    # All parameters collected, perform summoning
                    with Status("[cyan]🔍 Resonating with the architecture...[/cyan]"):
                        summoning_result = summoner_service.summon_code_pattern(repo_id, recipe_choice, **parameters)
                    
                    if "error" in summoning_result:
                        console.print(f"[red]❌ Summoning failed: {summoning_result['error']}[/red]")
                        continue
                    
                    # Display the summoning plan
                    console.print(Panel(
                        f"[bold green]🧙 Summoning Plan Ready[/bold green]\n"
                        f"[yellow]Recipe:[/yellow] {summoning_result['recipe_name']}\n"
                        f"[yellow]Description:[/yellow] {summoning_result['recipe_description']}\n\n"
                        f"[bold]📜 The Blueprint:[/bold]",
                        border_style="green"
                    ))
                    
                    # Show surgical plan
                    surgical_plan = summoning_result.get("surgical_plan", {})
                    operations = surgical_plan.get("operations", [])
                    
                    plan_table = Table(show_header=True, header_style="bold yellow", border_style="yellow")
                    plan_table.add_column("Operation", style="cyan")
                    plan_table.add_column("File", style="white")
                    plan_table.add_column("Description", style="dim white")
                    
                    for op in operations:
                        op_type = op["type"]
                        if op_type == "CREATE_FILE":
                            op_display = "CREATE"
                        elif op_type == "MODIFY_FILE":
                            op_display = "MODIFY"
                        else:
                            op_display = "INSERT"
                        
                        plan_table.add_row(
                            op_display,
                            op["file_path"],
                            op.get("description", "")
                        )
                    
                    console.print(plan_table)
                    
                    console.print(f"\n[bold]Summary:[/bold] {surgical_plan.get('summary', 'Unknown')}")
                    console.print(f"[yellow]Files to create:[/yellow] {surgical_plan.get('files_created', 0)}")
                    console.print(f"[yellow]Files to modify:[/yellow] {surgical_plan.get('files_modified', 0)}")
                    
                    # Ask for confirmation
                    try:
                        if Confirm.ask("\n[bold]Proceed with the summoning?[/bold]", default=False):
                            with Status("[cyan]🧙 The Summoner is weaving the ritual...[/cyan]"):
                                execution_result = summoner_service.execute_summoning_ritual(repo_id, summoning_result)
                            
                            if execution_result.get("success"):
                                console.print(Panel(
                                    f"[bold green]✨ Summoning Complete![/bold green]\n"
                                    f"[yellow]Operations completed:[/yellow] {execution_result.get('operations_completed', 0)}/{execution_result.get('total_operations', 0)}\n"
                                    f"[yellow]Files created:[/yellow] {len(execution_result.get('created_files', []))}\n"
                                    f"[yellow]Files modified:[/yellow] {len(execution_result.get('modified_files', []))}\n\n"
                                    f"[bold]Created files:[/bold]\n" + "\n".join(f"  • {f}" for f in execution_result.get('created_files', [])) + "\n\n" +
                                    f"[bold]Modified files:[/bold]\n" + "\n".join(f"  • {f}" for f in execution_result.get('modified_files', [])),
                                    title="[green]🧙 Summoning Successful[/green]",
                                    border_style="green"
                                ))
                            else:
                                console.print(Panel(
                                    f"[bold red]❌ Summoning Failed[/bold red]\n"
                                    f"Operations completed: {execution_result.get('operations_completed', 0)}/{execution_result.get('total_operations', 0)}\n"
                                    f"Some operations may have succeeded. Check the file system for partial results.",
                                    border_style="red"
                                ))
                        else:
                            console.print("[yellow]Summoning cancelled. The plan remains available for future use.[/yellow]")
                    except KeyboardInterrupt:
                        console.print("\n[yellow]Summoning cancelled.[/yellow]")
                
                context = {"repo_url": repo_url, "repo_id": repo_id}
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
                    
                    # Offer to show historical metrics
                    try:
                        if Confirm.ask("[dim]Would you like to view historical metrics data?[/dim]", default=False):
                            with Status("[cyan]📊 Retrieving metrics history...[/cyan]"):
                                metrics_history = api_client.get_sentinel_metrics_history(repo_id)
                            
                            if metrics_history and len(metrics_history) > 1:
                                # Show trends over time
                                latest = metrics_history[-1]
                                oldest = metrics_history[0] 
                                
                                trends_content = f"""📊 **Metrics History Analysis**
**Data Points:** {len(metrics_history)} snapshots
**Time Span:** {oldest.get('timestamp', 'Unknown')} → {latest.get('timestamp', 'Now')}

**Key Trends:**"""
                                
                                # Calculate trends for numeric fields
                                for key in latest.keys():
                                    if isinstance(latest.get(key), (int, float)) and key in oldest:
                                        old_val = oldest[key]
                                        new_val = latest[key] 
                                        if old_val != 0:
                                            change = ((new_val - old_val) / abs(old_val)) * 100
                                            trend_emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                                            trends_content += f"\n• {key.replace('_', ' ').title()}: {old_val} → {new_val} {trend_emoji} ({change:+.1f}%)"
                                
                                console.print(Panel(trends_content, title="[bold yellow]📊 Historical Metrics[/bold yellow]", border_style="yellow"))
                            elif metrics_history:
                                console.print("[yellow]📊 Only one metrics snapshot available. Historical analysis requires multiple data points.[/yellow]")
                            else:
                                console.print("[red]❌ Could not retrieve metrics history.[/red]")
                    except KeyboardInterrupt:
                        pass
                    
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

            elif command.lower() in ("bom", "b"):
                if not cli_state.get("model"):
                    console.print("[bold red]Please select a model first using the 'config' command.[/bold red]")
                    continue

                analyzed_repos = find_analyzed_repos()
                if not analyzed_repos:
                    console.print("[yellow]No analyzed repositories found. Use 'analyze' to ingest a repo first.[/yellow]")
                    continue
                
                console.print(Panel("Select a repository for BOM analysis.", title="[blue]📦 Bill of Materials Analysis[/blue]", border_style="blue"))
                table = Table(show_header=False, box=None, padding=(0, 2))
                for i, repo in enumerate(analyzed_repos, 1): 
                    table.add_row(f"([bold cyan]{i}[/bold cyan])", repo['display_name'])
                console.print(table)
                
                try:
                    choice = Prompt.ask("Enter choice", choices=[str(i) for i in range(1, len(analyzed_repos) + 1)], show_choices=False, default='1')
                    selected_repo = analyzed_repos[int(choice) - 1]
                    
                    bom_session = BOMSession(selected_repo['repo_id'], selected_repo['url'])
                    bom_session.loop()
                    context = {}
                except (ValueError, IndexError, KeyboardInterrupt): 
                    continue

            elif command.lower() in ("onboard", "o"):
                if not cli_state.get("model"):
                    console.print("[bold red]Please select a model first using the 'config' command.[/bold red]")
                    continue

                analyzed_repos = find_analyzed_repos()
                if not analyzed_repos:
                    console.print("[yellow]No analyzed repositories found. Use 'analyze' to ingest a repo first.[/yellow]")
                    continue
                
                console.print(Panel("Select a repository for onboarding assistance.", title="[green]🎓 Onboarding Concierge[/green]", border_style="green"))
                table = Table(show_header=False, box=None, padding=(0, 2))
                for i, repo in enumerate(analyzed_repos, 1): 
                    table.add_row(f"([bold cyan]{i}[/bold cyan])", repo['display_name'])
                console.print(table)
                
                try:
                    choice = Prompt.ask("Enter choice", choices=[str(i) for i in range(1, len(analyzed_repos) + 1)], show_choices=False, default='1')
                    selected_repo = analyzed_repos[int(choice) - 1]
                    
                    onboard_session = OnboardingSession(selected_repo['repo_id'], selected_repo['url'])
                    onboard_session.loop()
                    context = {}
                except (ValueError, IndexError, KeyboardInterrupt): 
                    continue

            elif command.lower() in ("librarian", "lib"):
                if not cli_state.get("model"):
                    console.print("[bold red]Please select a model first using the 'config' command.[/bold red]")
                    continue

                # Show Librarian menu
                console.print("\n[bold purple]📚 The Librarian's Archives[/bold purple] - Universal Knowledge Management")
                console.print("[dim]Managing both remote repositories and local directories as knowledge archives.[/dim]\n")
                
                try:
                    # Get archives list
                    with Status("[purple]Loading archive collection...[/purple]"):
                        archives_response = api_client._request("GET", "librarian/archives/")
                    
                    if archives_response and "archives" in archives_response:
                        archives = archives_response["archives"]
                        
                        # Display archives
                        if archives:
                            archives_table = Table(title="📚 Knowledge Archives", show_header=True, header_style="bold purple")
                            archives_table.add_column("Archive ID", style="cyan")
                            archives_table.add_column("Type", style="yellow")
                            archives_table.add_column("Source", style="white")
                            archives_table.add_column("Files", style="dim white")
                            
                            for archive in archives[:10]:  # Show top 10
                                archives_table.add_row(
                                    archive.get("archive_id", "unknown"),
                                    archive.get("source_type", "unknown"),
                                    archive.get("source_path", "unknown")[:40] + "...",
                                    str(archive.get("file_count", 0))
                                )
                            console.print(archives_table)
                        else:
                            console.print("[yellow]📚 No archives found. Use 'ingest' to add local directories.[/yellow]")
                        
                        # Show available actions
                        actions_panel = Panel(
                            "[bold]Available Actions:[/bold]\n"
                            "[yellow]ingest[/yellow] - Ingest a local directory as an archive\n"
                            "[yellow]list[/yellow] - List all available archives\n"
                            "[yellow]details[/yellow] - View archive details\n"
                            "[yellow]ask[/yellow] - Ask questions about archives\n"
                            "[dim]Type action name or 'back' to exit[/dim]",
                            title="📚 Librarian Commands",
                            border_style="purple"
                        )
                        console.print(actions_panel)
                        
                        # Action selection loop
                        while True:
                            try:
                                action = Prompt.ask("\n[bold purple]Librarian Action[/bold purple]").strip().lower()
                                
                                if action in ("back", "exit", "quit"):
                                    break
                                elif action == "ingest":
                                    directory_path = Prompt.ask("Local directory path to ingest")
                                    if not directory_path.strip():
                                        console.print("[yellow]No directory path provided.[/yellow]")
                                        continue
                                    
                                    # Expand user path
                                    directory_path = os.path.expanduser(directory_path.strip())
                                    
                                    if not os.path.exists(directory_path):
                                        console.print(f"[red]Directory does not exist: {directory_path}[/red]")
                                        continue
                                    
                                    with Status(f"[purple]Ingesting directory: {directory_path}...[/purple]"):
                                        ingest_response = api_client._request("POST", "librarian/ingest/", 
                                                                            json={"directory_path": directory_path})
                                    
                                    if ingest_response and "archive_id" in ingest_response:
                                        result = ingest_response
                                        ingest_panel = Panel(
                                            f"[bold]Directory Successfully Archived[/bold]\n"
                                            f"[yellow]Archive ID:[/yellow] {result.get('archive_id', 'Unknown')}\n"
                                            f"[yellow]Source Path:[/yellow] {result.get('source_path', 'Unknown')}\n"
                                            f"[yellow]Files Processed:[/yellow] {result.get('files_processed', 0)}\n"
                                            f"[yellow]Processing Time:[/yellow] {result.get('processing_time_seconds', 0):.2f}s\n"
                                            f"[yellow]Status:[/yellow] ✅ {result.get('status', 'Unknown')}",
                                            title="📚 Archive Created",
                                            border_style="green"
                                        )
                                        console.print(ingest_panel)
                                    else:
                                        error_msg = ingest_response.get("error", "Unknown error") if ingest_response else "No response"
                                        console.print(f"[red]❌ Ingestion failed: {error_msg}[/red]")
                                
                                elif action == "list":
                                    with Status("[purple]Refreshing archive list...[/purple]"):
                                        archives_response = api_client._request("GET", "librarian/archives/")
                                    if archives_response and "archives" in archives_response:
                                        archives = archives_response["archives"]
                                        console.print(f"[green]✅ Found {len(archives)} archives.[/green]")
                                        
                                elif action == "details":
                                    if not archives:
                                        console.print("[yellow]No archives available.[/yellow]")
                                        continue
                                    
                                    # Show archive choices
                                    console.print("\n[bold]Select an archive for details:[/bold]")
                                    for i, archive in enumerate(archives[:10], 1):
                                        console.print(f"[cyan]{i}[/cyan]. {archive.get('archive_id', 'unknown')}")
                                    
                                    try:
                                        choice = int(Prompt.ask("Archive number")) - 1
                                        if 0 <= choice < len(archives):
                                            archive_id = archives[choice]["archive_id"]
                                            with Status(f"[purple]Loading archive details...[/purple]"):
                                                detail_response = api_client._request("GET", f"librarian/archives/{archive_id}/")
                                            
                                            if detail_response and "archive" in detail_response:
                                                archive_info = detail_response["archive"]
                                                metadata = archive_info.get("archive_metadata", {})
                                                detail_panel = Panel(
                                                    f"[bold]Archive Details[/bold]\n"
                                                    f"[yellow]Archive ID:[/yellow] {archive_info.get('archive_id', 'Unknown')}\n"
                                                    f"[yellow]Source Type:[/yellow] {metadata.get('source_type', 'Unknown')}\n"
                                                    f"[yellow]Source Path:[/yellow] {metadata.get('source_path', 'Unknown')}\n"
                                                    f"[yellow]Ingested At:[/yellow] {metadata.get('ingested_at', 'Unknown')}\n"
                                                    f"[yellow]Total Files:[/yellow] {archive_info.get('total_files', 0)}\n"
                                                    f"[yellow]Directory Size:[/yellow] {metadata.get('directory_stats', {}).get('directory_size_mb', 0)} MB",
                                                    title=f"📁 Archive: {archive_id}",
                                                    border_style="purple"
                                                )
                                                console.print(detail_panel)
                                        else:
                                            console.print("[red]Invalid choice.[/red]")
                                    except (ValueError, KeyboardInterrupt):
                                        continue
                                
                                elif action == "ask":
                                    if not archives:
                                        console.print("[yellow]No archives available for questioning.[/yellow]")
                                        continue
                                    
                                    # Show archive choices
                                    console.print("\n[bold]Select an archive to query:[/bold]")
                                    for i, archive in enumerate(archives[:10], 1):
                                        console.print(f"[cyan]{i}[/cyan]. {archive.get('archive_id', 'unknown')}")
                                    
                                    try:
                                        choice = int(Prompt.ask("Archive number")) - 1
                                        if 0 <= choice < len(archives):
                                            archive_id = archives[choice]["archive_id"]
                                            question = Prompt.ask(f"\n[bold]Ask a question about archive '{archive_id}'[/bold]")
                                            
                                            if question.strip():
                                                with Status("[purple]The Librarian is consulting the archives...[/purple]"):
                                                    ask_response = api_client._request("POST", "librarian/ask/", 
                                                                                     json={"archive_id": archive_id, "question": question})
                                                
                                                if ask_response and "answer" in ask_response:
                                                    answer = ask_response["answer"]
                                                    sources = ask_response.get("sources", [])
                                                    
                                                    answer_panel = Panel(
                                                        f"[bold]Answer from Archive '{archive_id}'[/bold]\n\n"
                                                        f"{answer}\n\n"
                                                        f"[dim]Sources: {', '.join(sources[:3])}[/dim]" if sources else "",
                                                        title="🔮 Archive Oracle Response",
                                                        border_style="purple"
                                                    )
                                                    console.print(answer_panel)
                                                else:
                                                    error_msg = ask_response.get("error", "Unknown error") if ask_response else "No response"
                                                    console.print(f"[red]❌ Query failed: {error_msg}[/red]")
                                        else:
                                            console.print("[red]Invalid choice.[/red]")
                                    except (ValueError, KeyboardInterrupt):
                                        continue
                                else:
                                    console.print("[yellow]Unknown action. Available: ingest, list, details, ask[/yellow]")
                            except KeyboardInterrupt:
                                break
                    else:
                        console.print("[red]❌ Failed to load archives[/red]")
                        
                except Exception as e:
                    console.print(f"[red]❌ Librarian error: {e}[/red]")
                
                context = {}
                continue

            elif command.lower() in ("loremaster", "lm"):
                if not cli_state.get("model"):
                    console.print("[bold red]Please select a model first using the 'config' command.[/bold red]")
                    continue

                # Get repository to analyze
                repo_url = context.get("repo_url")
                if not repo_url:
                    analyzed_repos = find_analyzed_repos()
                    if not analyzed_repos:
                        console.print("[yellow]No analyzed repositories found. Use 'analyze' to ingest a repo first.[/yellow]")
                        continue
                    console.print(Panel("Select a repository for API documentation generation.", title="[blue]📚 The Loremaster[/blue]", border_style="blue"))
                    table = Table(show_header=False, box=None, padding=(0, 2))
                    for i, repo in enumerate(analyzed_repos, 1): 
                        table.add_row(f"([bold cyan]{i}[/bold cyan])", repo['display_name'])
                    console.print(table)
                    try:
                        choice = Prompt.ask("Enter choice", choices=[str(i) for i in range(1, len(analyzed_repos) + 1)], show_choices=False, default='1')
                        repo_url = analyzed_repos[int(choice) - 1]['url']
                    except (ValueError, IndexError, KeyboardInterrupt): 
                        continue

                repo_id = repo_url.replace("https://github.com/", "").replace("/", "_")
                
                # Show Loremaster menu
                console.print("\n[bold blue]📚 The Loremaster's Codex[/bold blue] - Interactive API Documentation")
                console.print("[dim]Generating living documentation and client code from your API architecture.[/dim]\n")
                
                try:
                    # Get API inventory
                    with Status("[blue]Cataloging API endpoints...[/blue]"):
                        inventory_response = api_client._request("GET", f"loremaster/{repo_id}/inventory/")
                    
                    if inventory_response and "endpoints" in inventory_response:
                        endpoints = inventory_response["endpoints"]
                        
                        # Display API inventory
                        if endpoints:
                            api_table = Table(title="🔗 API Endpoint Inventory", show_header=True, header_style="bold blue")
                            api_table.add_column("Method", style="cyan")
                            api_table.add_column("Path", style="white")
                            api_table.add_column("Function", style="yellow")
                            api_table.add_column("File", style="dim white")
                            
                            for endpoint in endpoints[:15]:  # Show top 15
                                api_table.add_row(
                                    endpoint.get("method", "GET"),
                                    endpoint.get("path", "/unknown"),
                                    endpoint.get("function_name", "unknown"),
                                    endpoint.get("file_path", "unknown")[:30] + "..."
                                )
                            console.print(api_table)
                        else:
                            console.print("[yellow]⚠️ No API endpoints detected in this repository[/yellow]")
                        
                        # Show available actions
                        actions_panel = Panel(
                            "[bold]Available Actions:[/bold]\n"
                            "[yellow]spec[/yellow] - Generate OpenAPI 3.0 specification\n"
                            "[yellow]docs[/yellow] - Generate interactive documentation page\n"
                            "[yellow]client[/yellow] - Generate client code snippet\n"
                            "[yellow]inventory[/yellow] - Refresh endpoint inventory\n"
                            "[dim]Type action name or 'back' to exit[/dim]",
                            title="📚 Loremaster Commands",
                            border_style="blue"
                        )
                        console.print(actions_panel)
                        
                        # Action selection loop
                        while True:
                            try:
                                action = Prompt.ask("\n[bold blue]Loremaster Action[/bold blue]").strip().lower()
                                
                                if action in ("back", "exit", "quit"):
                                    break
                                elif action == "spec":
                                    with Status("[blue]Generating OpenAPI specification...[/blue]"):
                                        spec_response = api_client._request("GET", f"loremaster/{repo_id}/openapi-spec/")
                                    if spec_response and "openapi_spec" in spec_response:
                                        spec = spec_response["openapi_spec"]
                                        spec_panel = Panel(
                                            f"[bold]OpenAPI 3.0 Specification Generated[/bold]\n"
                                            f"[yellow]Title:[/yellow] {spec.get('info', {}).get('title', 'API')}\n"
                                            f"[yellow]Version:[/yellow] {spec.get('info', {}).get('version', '1.0.0')}\n"
                                            f"[yellow]Endpoints:[/yellow] {len(spec.get('paths', {}))}\n"
                                            f"[yellow]Components:[/yellow] {len(spec.get('components', {}).get('schemas', {}))}\n\n"
                                            f"[dim]Full specification available via API endpoint[/dim]",
                                            title="📋 OpenAPI Spec",
                                            border_style="blue"
                                        )
                                        console.print(spec_panel)
                                elif action == "docs":
                                    with Status("[blue]Generating interactive documentation...[/blue]"):
                                        docs_response = api_client._request("GET", f"loremaster/{repo_id}/documentation/")
                                    if docs_response and "documentation_url" in docs_response:
                                        docs_url = docs_response["documentation_url"]
                                        docs_panel = Panel(
                                            f"[bold]Interactive Documentation Generated[/bold]\n"
                                            f"[yellow]Documentation URL:[/yellow] {docs_url}\n"
                                            f"[yellow]Features:[/yellow] Swagger UI, Try-it-out functionality\n"
                                            f"[yellow]Status:[/yellow] ✅ Ready to use\n\n"
                                            f"[dim]Open the URL in your browser to explore the interactive docs[/dim]",
                                            title="🌐 Interactive Docs",
                                            border_style="blue"
                                        )
                                        console.print(docs_panel)
                                elif action == "client":
                                    # Get client preferences
                                    language = Prompt.ask("Client language", choices=["python", "javascript", "curl"], default="python")
                                    framework = ""
                                    if language == "python":
                                        framework = Prompt.ask("Python framework", choices=["requests", "httpx", "aiohttp"], default="requests")
                                    elif language == "javascript":
                                        framework = Prompt.ask("JavaScript framework", choices=["fetch", "axios", "node-fetch"], default="fetch")
                                    
                                    with Status(f"[blue]Generating {language} client code...[/blue]"):
                                        client_response = api_client._request("POST", f"loremaster/{repo_id}/client-snippet/", 
                                                                            json={"language": language, "framework": framework})
                                    if client_response and "client_code" in client_response:
                                        client_code = client_response["client_code"]
                                        console.print(Panel(
                                            f"[bold]{language.title()} Client Code Snippet[/bold]\n\n"
                                            f"```{language}\n{client_code}\n```",
                                            title=f"🔧 {language.title()} Client",
                                            border_style="blue"
                                        ))
                                elif action == "inventory":
                                    with Status("[blue]Refreshing API inventory...[/blue]"):
                                        inventory_response = api_client._request("GET", f"loremaster/{repo_id}/inventory/")
                                    if inventory_response and "endpoints" in inventory_response:
                                        endpoints = inventory_response["endpoints"]
                                        console.print(f"[green]✅ Inventory refreshed. Found {len(endpoints)} endpoints.[/green]")
                                else:
                                    console.print("[yellow]Unknown action. Available: spec, docs, client, inventory[/yellow]")
                            except KeyboardInterrupt:
                                break
                    else:
                        console.print("[red]❌ Failed to load API inventory[/red]")
                        
                except Exception as e:
                    console.print(f"[red]❌ Loremaster error: {e}[/red]")
                
                context = {}
                continue

            elif command.lower() in ("quartermaster", "qm"):
                if not cli_state.get("model"):
                    console.print("[bold red]Please select a model first using the 'config' command.[/bold red]")
                    continue

                # Get repository to analyze
                repo_url = context.get("repo_url")
                if not repo_url:
                    analyzed_repos = find_analyzed_repos()
                    if not analyzed_repos:
                        console.print("[yellow]No analyzed repositories found. Use 'analyze' to ingest a repo first.[/yellow]")
                        continue
                    console.print(Panel("Select a repository for supply-chain management.", title="[green]⚖️ The Quartermaster[/green]", border_style="green"))
                    table = Table(show_header=False, box=None, padding=(0, 2))
                    for i, repo in enumerate(analyzed_repos, 1): 
                        table.add_row(f"([bold cyan]{i}[/bold cyan])", repo['display_name'])
                    console.print(table)
                    try:
                        choice = Prompt.ask("Enter choice", choices=[str(i) for i in range(1, len(analyzed_repos) + 1)], show_choices=False, default='1')
                        repo_url = analyzed_repos[int(choice) - 1]['url']
                    except (ValueError, IndexError, KeyboardInterrupt): 
                        continue

                repo_id = repo_url.replace("https://github.com/", "").replace("/", "_")
                
                # Show Quartermaster dashboard
                console.print("\n[bold green]⚖️ The Quartermaster's Inventory[/bold green] - Supply-Chain Management Dashboard")
                console.print("[dim]Managing dependencies, vulnerabilities, and compliance for your digital supply chain.[/dim]\n")
                
                try:
                    # Get dashboard data
                    with Status("[green]Loading supply-chain health dashboard...[/green]"):
                        response = api_client._request("GET", f"quartermaster/{repo_id}/dashboard/")
                    
                    if response and "error" not in response:
                        # Display dashboard
                        dashboard = response.get("dashboard", {})
                        
                        # Health overview
                        health_table = Table(title="🎯 Supply-Chain Health Overview", show_header=True, header_style="bold green")
                        health_table.add_column("Metric", style="cyan")
                        health_table.add_column("Status", style="white")
                        health_table.add_column("Count", style="yellow")
                        
                        health_table.add_row("🔍 Dependencies Scanned", "✅ Complete", str(dashboard.get("total_dependencies", 0)))
                        health_table.add_row("🚨 Critical Vulnerabilities", "⚠️ Found" if dashboard.get("critical_vulnerabilities", 0) > 0 else "✅ Clean", str(dashboard.get("critical_vulnerabilities", 0)))
                        health_table.add_row("📄 License Violations", "⚠️ Found" if dashboard.get("license_violations", 0) > 0 else "✅ Compliant", str(dashboard.get("license_violations", 0)))
                        health_table.add_row("📊 Overall Health Score", "✅ Good" if dashboard.get("health_score", 0) >= 80 else "⚠️ Needs Attention", f"{dashboard.get('health_score', 0)}/100")
                        
                        console.print(health_table)
                        
                        # Show available actions
                        actions_panel = Panel(
                            "[bold]Available Actions:[/bold]\n"
                            "[yellow]vuln[/yellow] - Check vulnerabilities\n"
                            "[yellow]license[/yellow] - Check license compliance\n"
                            "[yellow]upgrade[/yellow] - Simulate dependency upgrades\n"
                            "[yellow]risk[/yellow] - Generate risk assessment report\n"
                            "[dim]Type action name or 'back' to exit[/dim]",
                            title="⚖️ Quartermaster Commands",
                            border_style="green"
                        )
                        console.print(actions_panel)
                        
                        # Action selection loop
                        while True:
                            try:
                                action = Prompt.ask("\n[bold green]Quartermaster Action[/bold green]").strip().lower()
                                
                                if action in ("back", "exit", "quit"):
                                    break
                                elif action == "vuln":
                                    with Status("[green]Scanning for vulnerabilities...[/green]"):
                                        vuln_response = api_client._request("GET", f"quartermaster/{repo_id}/vulnerabilities/")
                                    if vuln_response and "vulnerabilities" in vuln_response:
                                        vulnerabilities = vuln_response["vulnerabilities"]
                                        if vulnerabilities:
                                            vuln_table = Table(title="🚨 Vulnerability Report", show_header=True, header_style="bold red")
                                            vuln_table.add_column("Severity", style="red")
                                            vuln_table.add_column("Package", style="cyan")
                                            vuln_table.add_column("CVE ID", style="yellow")
                                            vuln_table.add_column("Summary", style="white")
                                            
                                            for vuln in vulnerabilities[:10]:  # Show top 10
                                                vuln_table.add_row(
                                                    vuln.get("severity", "Unknown"),
                                                    vuln.get("package_name", "Unknown"),
                                                    vuln.get("cve_id", "N/A"),
                                                    vuln.get("summary", "No summary available")[:50] + "..."
                                                )
                                            console.print(vuln_table)
                                        else:
                                            console.print("[green]✅ No vulnerabilities found![/green]")
                                elif action == "license":
                                    with Status("[green]Checking license compliance...[/green]"):
                                        license_response = api_client._request("POST", f"quartermaster/{repo_id}/check-license-compliance/", 
                                                                             json={"policy": {"allowed_licenses": ["MIT", "Apache-2.0", "BSD-3-Clause"]}})
                                    if license_response and "violations" in license_response:
                                        violations = license_response["violations"]
                                        if violations:
                                            license_table = Table(title="📄 License Compliance Report", show_header=True, header_style="bold yellow")
                                            license_table.add_column("Package", style="cyan")
                                            license_table.add_column("License", style="red")
                                            license_table.add_column("Severity", style="yellow")
                                            license_table.add_column("Reason", style="white")
                                            
                                            for violation in violations[:10]:
                                                license_table.add_row(
                                                    violation.get("package_name", "Unknown"),
                                                    violation.get("license", "Unknown"),
                                                    violation.get("severity", "Unknown"),
                                                    violation.get("reason", "Policy violation")
                                                )
                                            console.print(license_table)
                                        else:
                                            console.print("[green]✅ All licenses are compliant![/green]")
                                elif action == "upgrade":
                                    package_name = Prompt.ask("Package to simulate upgrade for")
                                    target_version = Prompt.ask("Target version (optional)", default="latest")
                                    
                                    with Status("[green]Simulating upgrade...[/green]"):
                                        upgrade_response = api_client._request("POST", f"quartermaster/{repo_id}/simulate-upgrade/", 
                                                                             json={"package_name": package_name, "target_version": target_version})
                                    if upgrade_response and "simulation_result" in upgrade_response:
                                        result = upgrade_response["simulation_result"]
                                        upgrade_panel = Panel(
                                            f"[bold]Upgrade Simulation Results[/bold]\n"
                                            f"[yellow]Package:[/yellow] {result.get('package_name', 'Unknown')}\n"
                                            f"[yellow]Current Version:[/yellow] {result.get('current_version', 'Unknown')}\n"
                                            f"[yellow]Target Version:[/yellow] {result.get('target_version', 'Unknown')}\n"
                                            f"[yellow]Success:[/yellow] {'✅ Yes' if result.get('success') else '❌ No'}\n"
                                            f"[yellow]Impact:[/yellow] {result.get('impact_summary', 'No impact summary available')}",
                                            title="🔄 Upgrade Simulation",
                                            border_style="blue"
                                        )
                                        console.print(upgrade_panel)
                                elif action == "risk":
                                    with Status("[green]Generating risk assessment...[/green]"):
                                        risk_response = api_client._request("GET", f"quartermaster/{repo_id}/risk-report/")
                                    if risk_response and "risk_assessment" in risk_response:
                                        assessment = risk_response["risk_assessment"]
                                        risk_panel = Panel(
                                            f"[bold]Risk Assessment Report[/bold]\n"
                                            f"[yellow]Overall Risk Level:[/yellow] {assessment.get('overall_risk_level', 'Unknown')}\n"
                                            f"[yellow]Supply Chain Score:[/yellow] {assessment.get('supply_chain_score', 0)}/100\n"
                                            f"[yellow]Key Recommendations:[/yellow]\n{chr(10).join(f'• {rec}' for rec in assessment.get('recommendations', []))}",
                                            title="📊 Risk Assessment",
                                            border_style="red"
                                        )
                                        console.print(risk_panel)
                                else:
                                    console.print("[yellow]Unknown action. Available: vuln, license, upgrade, risk[/yellow]")
                            except KeyboardInterrupt:
                                break
                    else:
                        console.print("[red]❌ Failed to load Quartermaster dashboard[/red]")
                        
                except Exception as e:
                    console.print(f"[red]❌ Quartermaster error: {e}[/red]")
                
                context = {}
                continue

            elif command.lower() in ("repo-mgmt", "rm"):
                if not cli_state.get("model"):
                    console.print("[bold red]Please select a model first using the 'config' command.[/bold red]")
                    continue

                repo_mgmt_session = RepositoryManagementSession()
                repo_mgmt_session.loop()
                context = {}
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
