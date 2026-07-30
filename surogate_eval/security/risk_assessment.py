# surogate/eval/security/risk_assessment.py
"""Risk assessment for red-teaming results."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json

from .base import SeverityLevel
from ..utils.logger import get_logger

logger = get_logger()


@dataclass
class VulnerabilityResult:
    """Result for a single vulnerability type."""
    vulnerability_name: str  # ADD THIS - "PII Leakage", "Bias", etc.
    vulnerability_type: str  # "api_and_database_access", "religion", etc.
    total_attacks: int
    successful_attacks: int
    failed_attacks: int
    success_rate: float
    severity: SeverityLevel
    attack_breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass
class RiskAssessment:
    """Risk assessment from red-teaming."""
    target_name: str
    vulnerabilities: List[VulnerabilityResult]
    overview: Optional[str] = None
    test_cases: Optional[List[Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)

    def result_counts(self) -> Tuple[int, int]:
        """Countable units for the run outcome, as ``(scored, errored)``.

        One unit is one attack put to the target. An attack the judge
        scored is a measurement whether or not the target resisted it -
        "the model was vulnerable" is a result, not an error. An attack
        that came back without a score was never judged, so it counts as
        errored.
        """
        if self.detailed_results:
            errored = sum(1 for case in self.detailed_results if case.get('score') is None)
            return len(self.detailed_results) - errored, errored

        # No per-attack detail to count: fall back to the per-vulnerability
        # totals so a scan is still positive evidence that it measured.
        return sum(v.total_attacks for v in self.vulnerabilities), 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        scored_n, errored_n = self.result_counts()
        return {
            'target_name': self.target_name,
            'timestamp': self.timestamp,
            'scored_n': scored_n,
            'errored_n': errored_n,
            'vulnerabilities': [
                {
                    'vulnerability_name': v.vulnerability_name,
                    'vulnerability_type': v.vulnerability_type,
                    'total_attacks': v.total_attacks,
                    'successful_attacks': v.successful_attacks,
                    'failed_attacks': v.failed_attacks,
                    'success_rate': v.success_rate,
                    'severity': v.severity.value,
                    'attack_breakdown': v.attack_breakdown
                }
                for v in self.vulnerabilities
            ],
            'overview': self.overview,
            'detailed_results': self.detailed_results,
        }

    def save(self, path: str):
        """Save risk assessment to file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Risk assessment saved to: {output_path}")

    def get_critical_vulnerabilities(self) -> List[VulnerabilityResult]:
        """Get critical severity vulnerabilities."""
        return [
            v for v in self.vulnerabilities
            if v.severity == SeverityLevel.CRITICAL
        ]

    def get_high_risk_vulnerabilities(self) -> List[VulnerabilityResult]:
        """Get high and critical severity vulnerabilities."""
        return [
            v for v in self.vulnerabilities
            if v.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
        ]