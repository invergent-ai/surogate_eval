# surogate_eval/results.py
"""Utilities for viewing and analyzing evaluation results."""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from rich.console import Console
from rich.table import Table
from rich import box

from surogate_eval.utils.logger import get_logger

logger = get_logger()
console = Console()


# ------------------------------------------------------------------------------
# Helpers (schema-agnostic)
# ------------------------------------------------------------------------------

def _get_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    return result.get("summary", {}) or {}


def _get_targets(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    # New schema
    if "targets" in result and isinstance(result["targets"], list):
        return result["targets"]

    # Legacy schema fallback
    if "results" in result and isinstance(result["results"], list):
        return result["results"]

    return []


def _get_evaluations(target: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "evaluations" in target and isinstance(target["evaluations"], list):
        return target["evaluations"]

    # Legacy fallback
    if "metrics_summary" in target:
        return [target]

    return []


# ------------------------------------------------------------------------------
# Listing results
# ------------------------------------------------------------------------------

def list_results(results_dir: str = "eval_results") -> List[Path]:
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.warning(f"Results directory not found: {results_dir}")
        return []

    return sorted(results_path.glob("eval_*.json"), reverse=True)


def display_results_list(results: List[Path], results_dir: str):
    if not results:
        console.print("[yellow]No evaluation results found[/yellow]")
        return

    console.print(f"\n[bold cyan]Available Evaluation Results[/bold cyan] ({results_dir})\n")

    table = Table(box=box.ROUNDED)
    table.add_column("#", style="dim", width=4)
    table.add_column("Filename", style="cyan")
    table.add_column("Date", style="green")
    table.add_column("Targets", justify="right")
    table.add_column("Evaluations", justify="right")
    table.add_column("Test Cases", justify="right")

    for i, result_file in enumerate(results, 1):
        try:
            with open(result_file, "r") as f:
                data = json.load(f)

            summary = _get_summary(data)

            table.add_row(
                str(i),
                result_file.name,
                data.get("timestamp", "N/A")[:19],
                str(summary.get("total_targets", "N/A")),
                str(summary.get("total_evaluations", "N/A")),
                str(summary.get("total_test_cases", "N/A")),
            )
        except Exception:
            table.add_row(str(i), result_file.name, "Error", "-", "-", "-")

    console.print(table)
    console.print("\n[dim]Use 'surogate-eval eval --view <filename>' to view details[/dim]")


# ------------------------------------------------------------------------------
# Loading & viewing results
# ------------------------------------------------------------------------------

def load_result(filepath: str) -> Optional[Dict[str, Any]]:
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load result: {e}")
        return None


def display_results(filepath: str):
    result = load_result(filepath)
    if not result:
        return

    console.print("\n[bold cyan]📊 Evaluation Report[/bold cyan]")
    console.print(f"[dim]File: {filepath}[/dim]\n")

    project = result.get("project", {})
    summary = _get_summary(result)

    console.print(
        f"[bold]Project:[/bold] {project.get('name', 'N/A')} "
        f"(v{project.get('version', 'N/A')})"
    )
    console.print(f"[bold]Timestamp:[/bold] {result.get('timestamp', 'N/A')}")
    console.print(f"[bold]Total Test Cases:[/bold] {summary.get('total_test_cases', 0)}\n")

    for target in _get_targets(result):
        console.rule(
            f"[bold green]🎯 Target: {target.get('name', 'Unknown')}[/bold green] "
            f"[dim]({target.get('model', 'N/A')})[/dim]"
        )

        for eval_run in _get_evaluations(target):
            console.print(f"\n[bold]Evaluation:[/bold] {eval_run.get('name', 'N/A')}")
            console.print(
                f"[dim]Dataset: {eval_run.get('dataset', 'N/A')} "
                f"({eval_run.get('dataset_type', 'N/A')})[/dim]"
            )

            metrics = eval_run.get("metrics_summary", {})
            if not metrics:
                console.print("[yellow]No metrics available[/yellow]")
                continue

            table = Table(box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Avg Score", justify="right")
            table.add_column("Success Rate", justify="right")
            table.add_column("Status", justify="center")

            for metric_name, metric_data in metrics.items():
                if "error" in metric_data:
                    table.add_row(metric_name, "N/A", "N/A", "[red]❌ Failed[/red]")
                    continue

                avg = metric_data.get("avg_score", 0.0)
                rate = metric_data.get("success_rate", 0.0)

                status = (
                    "[green]✅ Excellent[/green]" if rate >= 0.8 else
                    "[yellow]⚠️ Good[/yellow]" if rate >= 0.6 else
                    "[red]❌ Needs Work[/red]"
                )

                table.add_row(
                    metric_name,
                    f"{avg:.3f}",
                    f"{rate * 100:.1f}%",
                    status,
                )

            console.print(table)

    console.print()


# ------------------------------------------------------------------------------
# Comparing results
# ------------------------------------------------------------------------------

def compare_results(filepath1: str, filepath2: str):
    r1 = load_result(filepath1)
    r2 = load_result(filepath2)

    if not r1 or not r2:
        logger.error("Failed to load one or both results")
        return

    t1 = _get_targets(r1)
    t2 = _get_targets(r2)

    if not t1 or not t2:
        logger.error("No comparable targets found")
        return

    e1 = _get_evaluations(t1[0])
    e2 = _get_evaluations(t2[0])

    if not e1 or not e2:
        logger.error("No comparable evaluations found")
        return

    m1 = e1[0].get("metrics_summary", {})
    m2 = e2[0].get("metrics_summary", {})

    console.print("\n[bold cyan]📊 Comparison Report[/bold cyan]\n")
    console.print(f"[dim]File 1: {Path(filepath1).name}[/dim]")
    console.print(f"[dim]File 2: {Path(filepath2).name}[/dim]\n")

    table = Table(box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Result 1", justify="right")
    table.add_column("Result 2", justify="right")
    table.add_column("Δ", justify="right")

    for metric in m1.keys() & m2.keys():
        s1 = m1[metric].get("avg_score", 0.0)
        s2 = m2[metric].get("avg_score", 0.0)
        delta = s2 - s1

        arrow = "↑" if delta > 0.01 else "↓" if delta < -0.01 else "→"
        color = "green" if delta > 0.01 else "red" if delta < -0.01 else "white"

        table.add_row(
            metric,
            f"{s1:.3f}",
            f"{s2:.3f}",
            f"[{color}]{arrow} {delta:+.3f}[/{color}]",
        )

    console.print(table)
    console.print()
