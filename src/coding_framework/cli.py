"""
Command Line Interface for the VERL + LangGraph Multi-Agent Coding Framework.

Provides a user-friendly CLI for interacting with the multi-agent coding system,
supporting both inference and training modes.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import click
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from coding_framework.orchestration import CodingSupervisor
from coding_framework.utils import load_config, setup_logging

console = Console()
logger = structlog.get_logger()


@click.group()
@click.option("--config", default="config/default.yaml", help="Configuration file path")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx: click.Context, config: str, log_level: str, verbose: bool) -> None:
    """
    VERL + LangGraph Multi-Agent Coding Framework CLI.

    A sophisticated multi-agent system for code generation, review, and execution.
    """
    ctx.ensure_object(dict)

    # Setup logging
    setup_logging(level=log_level, verbose=verbose)

    # Store configuration in context
    ctx.obj["config_path"] = config
    ctx.obj["verbose"] = verbose

    # Display welcome message
    if verbose:
        console.print(
            Panel.fit(
                "[bold blue]VERL + LangGraph Multi-Agent Coding Framework[/bold blue]\n"
                "[dim]Intelligent code generation through collaborative AI agents[/dim]",
                border_style="blue",
            )
        )


@main.command()
@click.argument("problem", required=True)
@click.option("--language", "-l", default="python", help="Target programming language")
@click.option("--style", "-s", default="clean", help="Code style preference")
@click.option("--include-tests", is_flag=True, help="Include unit tests")
@click.option("--focus-areas", help="Review focus areas (comma-separated)")
@click.option("--output", "-o", help="Output file path")
@click.option("--timeout", default=300, help="Operation timeout in seconds")
@click.pass_context
def solve(
    ctx: click.Context,
    problem: str,
    language: str,
    style: str,
    include_tests: bool,
    focus_areas: Optional[str],
    output: Optional[str],
    timeout: int,
) -> None:
    """
    Solve a coding problem using the multi-agent system.

    PROBLEM: Description of the coding problem to solve
    """
    asyncio.run(
        _solve_problem(
            ctx.obj,
            problem,
            language,
            style,
            include_tests,
            focus_areas,
            output,
            timeout,
        )
    )


async def _solve_problem(
    ctx_obj: dict[str, Any],
    problem: str,
    language: str,
    style: str,
    include_tests: bool,
    focus_areas: Optional[str],
    output: Optional[str],
    timeout: int,
) -> None:
    """Execute the problem-solving workflow."""
    try:
        # Load configuration
        config_path = ctx_obj["config_path"]
        verbose = ctx_obj["verbose"]

        if not Path(config_path).exists():
            console.print(f"[red]Configuration file not found: {config_path}[/red]")
            console.print("Please create a configuration file or use --config to specify one.")
            sys.exit(1)

        config = load_config(config_path)

        # Initialize supervisor
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Initializing agents...", total=None)

            supervisor = CodingSupervisor(config)
            await supervisor.initialize()

            progress.update(task, description="Agents initialized ✓")

        # Prepare problem context
        context = {
            "language": language,
            "style": style,
            "include_tests": include_tests,
            "focus_areas": focus_areas.split(",") if focus_areas else None,
        }

        console.print(f"\n[bold]Problem:[/bold] {problem}")
        console.print(f"[bold]Language:[/bold] {language}")
        console.print(f"[bold]Style:[/bold] {style}")

        if include_tests:
            console.print("[bold]Tests:[/bold] Included")

        if focus_areas:
            console.print(f"[bold]Focus Areas:[/bold] {focus_areas}")

        # Execute problem solving
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Solving problem...", total=None)

            result = await asyncio.wait_for(
                supervisor.solve_problem(problem, context),
                timeout=timeout,
            )

            progress.update(task, description="Problem solved ✓", completed=1, total=1)

        # Display results
        _display_solution_results(result, verbose)

        # Save output if requested
        if output:
            _save_output(result, output)
            console.print(f"\n[green]Results saved to: {output}[/green]")

    except asyncio.TimeoutError:
        console.print(f"[red]Operation timed out after {timeout} seconds[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@main.command()
@click.option("--algorithm", default="ppo", help="RL algorithm to use")
@click.option("--episodes", default=100, help="Number of training episodes")
@click.option("--data-path", help="Training data path")
@click.option("--checkpoint-dir", help="Checkpoint directory")
@click.option("--wandb-project", help="Weights & Biases project name")
@click.pass_context
def train(
    ctx: click.Context,
    algorithm: str,
    episodes: int,
    data_path: Optional[str],
    checkpoint_dir: Optional[str],
    wandb_project: Optional[str],
) -> None:
    """
    Train the agents using VERL reinforcement learning.
    """
    asyncio.run(
        _train_agents(
            ctx.obj,
            algorithm,
            episodes,
            data_path,
            checkpoint_dir,
            wandb_project,
        )
    )


async def _train_agents(
    ctx_obj: dict[str, Any],
    algorithm: str,
    episodes: int,
    data_path: Optional[str],
    checkpoint_dir: Optional[str],
    wandb_project: Optional[str],
) -> None:
    """Execute the training workflow."""
    try:
        config_path = ctx_obj["config_path"]
        verbose = ctx_obj["verbose"]

        config = load_config(config_path)

        # Override training parameters
        if data_path:
            config.training.data_path = data_path
        if checkpoint_dir:
            config.training.checkpoint_dir = checkpoint_dir
        if wandb_project:
            config.training.wandb_project = wandb_project

        console.print(f"[bold]Starting VERL training with {algorithm.upper()}[/bold]")
        console.print(f"Episodes: {episodes}")
        console.print(f"Data path: {config.training.data_path}")

        # Initialize supervisor
        supervisor = CodingSupervisor(config)

        # Start training
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Training agents...", total=None)

            training_results = await supervisor.train_agents(
                algorithm=algorithm,
                episodes=episodes,
            )

            progress.update(task, description="Training completed ✓", completed=1, total=1)

        # Display training results
        _display_training_results(training_results, verbose)

    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@main.command()
@click.option("--agent", help="Specific agent to check (generator, reviewer, executor)")
@click.pass_context
def health(ctx: click.Context, agent: Optional[str]) -> None:
    """
    Check health status of agents and system components.
    """
    asyncio.run(_health_check(ctx.obj, agent))


async def _health_check(ctx_obj: dict[str, Any], agent: Optional[str]) -> None:
    """Perform system health check."""
    try:
        config_path = ctx_obj["config_path"]
        config = load_config(config_path)

        supervisor = CodingSupervisor(config)
        await supervisor.initialize()

        console.print("[bold]System Health Check[/bold]\n")

        if agent:
            # Check specific agent
            health_status = await supervisor.check_agent_health(agent)
            _display_health_status({agent: health_status})
        else:
            # Check all agents
            health_status = await supervisor.health_check()
            _display_health_status(health_status)

    except Exception as e:
        console.print(f"[red]Health check failed: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--focus", help="Review focus areas")
@click.option("--severity", default="all", help="Issue severity filter")
@click.pass_context
def review(
    ctx: click.Context,
    file_path: str,
    focus: Optional[str],
    severity: str,
) -> None:
    """
    Review code from a file using the reviewer agent.

    FILE_PATH: Path to the code file to review
    """
    asyncio.run(_review_code_file(ctx.obj, file_path, focus, severity))


async def _review_code_file(
    ctx_obj: dict[str, Any],
    file_path: str,
    focus: Optional[str],
    severity: str,
) -> None:
    """Review code from a file."""
    try:
        # Read code file
        with open(file_path, encoding="utf-8") as f:
            code = f.read()

        config_path = ctx_obj["config_path"]
        config = load_config(config_path)

        supervisor = CodingSupervisor(config)
        await supervisor.initialize()

        # Prepare context
        context = {
            "file_path": file_path,
            "focus_areas": focus.split(",") if focus else None,
        }

        console.print(f"[bold]Reviewing file:[/bold] {file_path}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing code...", total=None)

            result = await supervisor.review_code(code, context, severity_filter=severity)

            progress.update(task, description="Review completed ✓", completed=1, total=1)

        _display_review_results(result)

    except Exception as e:
        console.print(f"[red]Code review failed: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("code_file", type=click.Path(exists=True))
@click.option("--language", "-l", help="Programming language (auto-detected if not specified)")
@click.option("--timeout", default=30, help="Execution timeout")
@click.option("--tests", is_flag=True, help="Run tests if present in code")
@click.pass_context
def execute(
    ctx: click.Context,
    code_file: str,
    language: Optional[str],
    timeout: int,
    tests: bool,
) -> None:
    """
    Execute code from a file using the executor agent.

    CODE_FILE: Path to the code file to execute
    """
    asyncio.run(_execute_code_file(ctx.obj, code_file, language, timeout, tests))


async def _execute_code_file(
    ctx_obj: dict[str, Any],
    code_file: str,
    language: Optional[str],
    timeout: int,
    tests: bool,
) -> None:
    """Execute code from a file."""
    try:
        # Read code file
        with open(code_file, encoding="utf-8") as f:
            code = f.read()

        # Auto-detect language if not specified
        if not language:
            language = _detect_language_from_extension(Path(code_file).suffix)

        config_path = ctx_obj["config_path"]
        config = load_config(config_path)

        supervisor = CodingSupervisor(config)
        await supervisor.initialize()

        context = {
            "language": language,
            "file_path": code_file,
            "include_tests": tests,
        }

        console.print(f"[bold]Executing file:[/bold] {code_file}")
        console.print(f"[bold]Language:[/bold] {language}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Executing code...", total=None)

            result = await asyncio.wait_for(
                supervisor.execute_code(code, context),
                timeout=timeout,
            )

            progress.update(task, description="Execution completed ✓", completed=1, total=1)

        _display_execution_results(result)

    except asyncio.TimeoutError:
        console.print(f"[red]Execution timed out after {timeout} seconds[/red]")
    except Exception as e:
        console.print(f"[red]Code execution failed: {e}[/red]")
        sys.exit(1)


def _display_solution_results(result: dict[str, Any], verbose: bool) -> None:
    """Display problem solution results."""
    console.print("\n" + "=" * 60)
    console.print("[bold green]Solution Results[/bold green]")
    console.print("=" * 60)

    # Display generated code
    if "code" in result:
        syntax = Syntax(result["code"], result.get("language", "python"), theme="github-dark")
        console.print("\n[bold]Generated Code:[/bold]")
        console.print(syntax)

    # Display review feedback
    if "review" in result and result["review"]:
        console.print("\n[bold]Code Review:[/bold]")
        console.print(result["review"])

    # Display execution results
    if "execution" in result and result["execution"]:
        console.print("\n[bold]Execution Results:[/bold]")
        console.print(result["execution"])

    # Display metrics
    if verbose and "metrics" in result:
        _display_metrics_table(result["metrics"])


def _display_training_results(results: dict[str, Any], verbose: bool) -> None:
    """Display training results."""
    console.print("\n[bold green]Training Results[/bold green]")

    table = Table(title="Training Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in results.get("metrics", {}).items():
        table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)


def _display_health_status(health_status: dict[str, Any]) -> None:
    """Display system health status."""
    table = Table(title="Agent Health Status")
    table.add_column("Agent", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Response Time", style="yellow")
    table.add_column("Details")

    for agent_name, status in health_status.items():
        status_color = "green" if status.get("status") == "healthy" else "red"
        status_text = f"[{status_color}]{status.get('status', 'unknown').upper()}[/{status_color}]"

        response_time = f"{status.get('response_time', 0):.3f}s"
        details = status.get("error", "OK")

        table.add_row(agent_name, status_text, response_time, details)

    console.print(table)


def _display_review_results(result: dict[str, Any]) -> None:
    """Display code review results."""
    console.print("\n[bold blue]Code Review Results[/bold blue]")

    if "content" in result:
        console.print(result["content"])

    # Display metrics if available
    if "metadata" in result:
        metadata = result["metadata"]
        console.print(f"\n[dim]Issues found: {metadata.get('issues_found', 0)}[/dim]")
        console.print(f"[dim]Overall score: {metadata.get('overall_score', 0)}/100[/dim]")


def _display_execution_results(result: dict[str, Any]) -> None:
    """Display code execution results."""
    console.print("\n[bold cyan]Execution Results[/bold cyan]")

    if "content" in result:
        console.print(result["content"])

    # Display execution metrics
    if "metadata" in result:
        metadata = result["metadata"]
        success_icon = "✅" if metadata.get("success") else "❌"
        console.print(f"\n[dim]{success_icon} Success: {metadata.get('success')}[/dim]")
        console.print(f"[dim]⏱️ Execution time: {metadata.get('execution_time', 0):.3f}s[/dim]")


def _display_metrics_table(metrics: dict[str, Any]) -> None:
    """Display metrics in a formatted table."""
    table = Table(title="Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in metrics.items():
        formatted_key = key.replace("_", " ").title()
        formatted_value = f"{value:.3f}" if isinstance(value, float) else str(value)
        table.add_row(formatted_key, formatted_value)

    console.print("\n")
    console.print(table)


def _save_output(result: dict[str, Any], output_path: str) -> None:
    """Save results to output file."""
    output_data = {
        "timestamp": result.get("timestamp"),
        "problem": result.get("problem"),
        "solution": result.get("code"),
        "language": result.get("language"),
        "review": result.get("review"),
        "execution": result.get("execution"),
        "metrics": result.get("metrics"),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def _detect_language_from_extension(extension: str) -> str:
    """Detect programming language from file extension."""
    extensions = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".go": "go",
        ".rs": "rust",
    }

    return extensions.get(extension.lower(), "python")


if __name__ == "__main__":
    main()
