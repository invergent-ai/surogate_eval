"""Report generation utilities."""

from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape

from surogate_eval.utils.logger import get_logger

logger = get_logger()

TEMPLATES_DIR = Path(__file__).parent / "templates"


class ReportGenerator:
    """Generates evaluation reports from results."""

    def __init__(self):
        self.env = Environment(
            loader=FileSystemLoader(TEMPLATES_DIR),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._register_filters()

    def _register_filters(self):
        """Register custom Jinja2 filters."""

        def get(d: dict, key: str, default=0):
            """Safe dict get for templates."""
            if isinstance(d, dict):
                return d.get(key, default)
            return default

        def format_percent(value, decimals=1):
            """Format value as percentage."""
            try:
                return f"{float(value) * 100:.{decimals}f}%"
            except (ValueError, TypeError):
                return "N/A"

        def format_score(value, decimals=3):
            """Format score value."""
            try:
                return f"{float(value):.{decimals}f}"
            except (ValueError, TypeError):
                return "N/A"

        self.env.filters["get"] = get
        self.env.filters["format_percent"] = format_percent
        self.env.filters["format_score"] = format_score

    def _get_template_context(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Build template context from results."""
        return {
            "project": results.get("project", {}),
            "timestamp": results.get("timestamp"),
            "summary": results.get("summary", {}),
            "targets": results.get("targets", []),
        }

    def generate_markdown(self, results: Dict[str, Any]) -> str:
        """
        Generate markdown report from results.

        Args:
            results: Consolidated evaluation results dict

        Returns:
            Markdown string
        """
        template = self.env.get_template("report.md.j2")
        return template.render(**self._get_template_context(results))

    def save_markdown(self, results: Dict[str, Any], output_path: Path) -> Path:
        """
        Generate and save markdown report.

        Args:
            results: Consolidated evaluation results dict
            output_path: Path to save the report

        Returns:
            Path to saved report
        """
        content = self.generate_markdown(results)
        output_path.write_text(content)
        logger.success(f"Markdown report saved to: {output_path}")
        return output_path

    def generate_html(self, results: Dict[str, Any]) -> str:
        """
        Generate HTML report from results.

        Args:
            results: Consolidated evaluation results dict

        Returns:
            HTML string
        """
        template = self.env.get_template("report.html.j2")
        return template.render(**self._get_template_context(results))

    def save_pdf(self, results: Dict[str, Any], output_path: Path) -> Path:
        """
        Generate and save PDF report.

        Args:
            results: Consolidated evaluation results dict
            output_path: Path to save the report

        Returns:
            Path to saved report
        """
        try:
            from weasyprint import HTML
        except ImportError:
            logger.error("weasyprint not installed. Install with: pip install weasyprint")
            raise ImportError("weasyprint is required for PDF generation")

        html_content = self.generate_html(results)
        HTML(string=html_content, base_url=str(TEMPLATES_DIR)).write_pdf(output_path)
        logger.success(f"PDF report saved to: {output_path}")
        return output_path