# surogate/eval/security/vulnerabilities.py - COMPLETE FIXED VERSION
"""Vulnerability scanner wrapper with comprehensive vulnerability definitions."""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from .base import VulnerabilityType, SeverityLevel
from ..utils.logger import get_logger

logger = get_logger()


@dataclass
class VulnerabilityDefinition:
    """Definition of a vulnerability type."""
    name: str
    description: str
    subtypes: List[str]
    default_severity: SeverityLevel
    category: str
    compliance_frameworks: List[str]  # e.g., ["OWASP", "NIST", "GDPR"]


class VulnerabilityScanner:
    """Comprehensive vulnerability scanner with extensive vulnerability definitions."""

    # Comprehensive vulnerability subtypes mapping - ALIGNED WITH DEEPTEAM
    VULNERABILITY_SUBTYPES = {
        # =========================================================================
        # DATA PRIVACY
        # =========================================================================
        VulnerabilityType.PII_LEAKAGE: [
            "api_and_database_access", "direct_disclosure",
            "session_leak", "social_manipulation"
        ],

        VulnerabilityType.PROMPT_LEAKAGE: [
            "secrets_and_credentials", "instructions",
            "guard_exposure", "permissions_and_roles"
        ],

        # =========================================================================
        # RESPONSIBLE AI
        # =========================================================================
        VulnerabilityType.BIAS: [
            "religion", "politics", "gender", "race"
        ],

        VulnerabilityType.TOXICITY: [
            "profanity", "insults", "threats", "mockery"
        ],

        VulnerabilityType.FAIRNESS: [
            "equality_consistency", "procedural_opportunity", "temporal_outcome"
        ],

        VulnerabilityType.ETHICS: [
            "moral_integrity", "responsible_transparency", "harm_prevention"
        ],

        # =========================================================================
        # SAFETY
        # =========================================================================
        VulnerabilityType.ILLEGAL_ACTIVITY: [
            "weapons", "illegal_drugs", "violent_crimes",
            "non_violent_crimes", "sex_crimes", "cybercrime",
            "child_exploitation"
        ],

        VulnerabilityType.GRAPHIC_CONTENT: [
            "sexual_content", "graphic_content", "pornographic_content"
        ],

        VulnerabilityType.PERSONAL_SAFETY: [
            "bullying", "self_harm", "unsafe_practices",
            "dangerous_challenges", "stalking"
        ],

        VulnerabilityType.CHILD_PROTECTION: [
            "age_verification", "data_privacy", "exposure_interaction"
        ],

        # =========================================================================
        # BUSINESS
        # =========================================================================
        VulnerabilityType.MISINFORMATION: [
            "factual_errors", "unsupported_claims", "expertize_misrepresentation"
        ],

        VulnerabilityType.INTELLECTUAL_PROPERTY: [
            "imitation", "copyright_violations",
            "trademark_infringement", "patent_disclosure"
        ],

        VulnerabilityType.COMPETITION: [
            "competitor_mention", "market_manipulation",
            "discreditation", "confidential_strategies"
        ],

        # =========================================================================
        # SECURITY
        # =========================================================================
        VulnerabilityType.BFLA: [
            "privilege_escalation", "function_bypass", "authorization_bypass"
        ],

        VulnerabilityType.BOLA: [
            "object_access_bypass", "cross_customer_access",
            "unauthorized_object_manipulation"
        ],

        VulnerabilityType.RBAC: [
            "role_bypass", "privilege_escalation", "unauthorized_role_assumption"
        ],

        VulnerabilityType.DEBUG_ACCESS: [
            "debug_mode_bypass", "development_endpoint_access",
            "administrative_interface_exposure"
        ],

        VulnerabilityType.SHELL_INJECTION: [
            "command_injection", "system_command_execution", "shell_escape_sequences"
        ],

        VulnerabilityType.SQL_INJECTION: [
            "blind_sql_injection", "union_based_injection", "error_based_injection"
        ],

        VulnerabilityType.SSRF: [
            "internal_service_access", "cloud_metadata_access", "port_scanning"
        ],

        # =========================================================================
        # AGENTIC
        # =========================================================================
        VulnerabilityType.GOAL_THEFT: [
            "escalating_probing", "cooperative_dialogue", "social_engineering"
        ],

        VulnerabilityType.RECURSIVE_HIJACKING: [
            "self_modifying_goals", "recursive_objective_chaining",
            "goal_propagation_attacks"
        ],

        VulnerabilityType.ROBUSTNESS: [
            "input_overreliance", "hijacking"
        ],

        VulnerabilityType.EXCESSIVE_AGENCY: [
            "functionality", "permissions", "autonomy"
        ],
    }

    # Vulnerability definitions with metadata
    VULNERABILITY_DEFINITIONS = {
        VulnerabilityType.PII_LEAKAGE: VulnerabilityDefinition(
            name="PII Leakage",
            description="Exposure of Personally Identifiable Information",
            subtypes=VULNERABILITY_SUBTYPES[VulnerabilityType.PII_LEAKAGE],
            default_severity=SeverityLevel.CRITICAL,
            category="Data Privacy",
            compliance_frameworks=["GDPR", "CCPA", "HIPAA", "PCI-DSS"]
        ),
        VulnerabilityType.BIAS: VulnerabilityDefinition(
            name="Bias",
            description="Unfair treatment based on protected characteristics",
            subtypes=VULNERABILITY_SUBTYPES[VulnerabilityType.BIAS],
            default_severity=SeverityLevel.HIGH,
            category="Fairness",
            compliance_frameworks=["EU AI Act", "EEOC", "Fair Housing Act"]
        ),
        VulnerabilityType.TOXICITY: VulnerabilityDefinition(
            name="Toxicity",
            description="Harmful, abusive, or offensive content",
            subtypes=VULNERABILITY_SUBTYPES[VulnerabilityType.TOXICITY],
            default_severity=SeverityLevel.HIGH,
            category="Content Safety",
            compliance_frameworks=["Trust & Safety", "Community Guidelines"]
        ),
        VulnerabilityType.ILLEGAL_ACTIVITY: VulnerabilityDefinition(
            name="Illegal Activity",
            description="Content promoting illegal activities",
            subtypes=VULNERABILITY_SUBTYPES[VulnerabilityType.ILLEGAL_ACTIVITY],
            default_severity=SeverityLevel.CRITICAL,
            category="Content Safety",
            compliance_frameworks=["Legal Compliance", "Law Enforcement"]
        ),
        VulnerabilityType.SQL_INJECTION: VulnerabilityDefinition(
            name="SQL Injection",
            description="Database injection attacks",
            subtypes=VULNERABILITY_SUBTYPES[VulnerabilityType.SQL_INJECTION],
            default_severity=SeverityLevel.CRITICAL,
            category="Security",
            compliance_frameworks=["OWASP Top 10", "PCI-DSS"]
        ),
        VulnerabilityType.CHILD_PROTECTION: VulnerabilityDefinition(
            name="Child Protection",
            description="Content harmful to minors",
            subtypes=VULNERABILITY_SUBTYPES[VulnerabilityType.CHILD_PROTECTION],
            default_severity=SeverityLevel.CRITICAL,
            category="Content Safety",
            compliance_frameworks=["COPPA", "CSAM Laws"]
        ),
        VulnerabilityType.INTELLECTUAL_PROPERTY: VulnerabilityDefinition(
            name="Intellectual Property",
            description="IP violations and disclosures",
            subtypes=VULNERABILITY_SUBTYPES[VulnerabilityType.INTELLECTUAL_PROPERTY],
            default_severity=SeverityLevel.HIGH,
            category="Business",
            compliance_frameworks=["Copyright Law", "Patent Law"]
        ),
    }

    @staticmethod
    def get_available_vulnerabilities() -> List[str]:
        """Get list of all available vulnerability types."""
        return [v.value for v in VulnerabilityType]

    @staticmethod
    def get_vulnerability_categories() -> Dict[str, List[str]]:
        """Get vulnerabilities grouped by category."""
        categories = {}
        for vuln_type in VulnerabilityType:
            if vuln_type in VulnerabilityScanner.VULNERABILITY_DEFINITIONS:
                definition = VulnerabilityScanner.VULNERABILITY_DEFINITIONS[vuln_type]
                category = definition.category
                if category not in categories:
                    categories[category] = []
                categories[category].append(vuln_type.value)
        return categories

    @staticmethod
    def get_subtypes(vulnerability: str, custom_subtypes: Optional[List[str]] = None) -> List[str]:
        """
        Get subtypes for a vulnerability.

        If custom_subtypes provided, use those. Otherwise use predefined subtypes.
        Validates custom subtypes against known subtypes and warns if unknown.

        Args:
            vulnerability: Vulnerability type name
            custom_subtypes: Optional custom subtypes from user config

        Returns:
            List of subtype names
        """
        try:
            vuln_enum = VulnerabilityType(vulnerability.lower())
        except ValueError:
            logger.warning(f"Unknown vulnerability type: {vulnerability}")
            return custom_subtypes or []

        predefined_subtypes = VulnerabilityScanner.VULNERABILITY_SUBTYPES.get(vuln_enum, [])

        # If custom subtypes provided, use them
        if custom_subtypes:
            # Validate custom subtypes
            unknown_subtypes = [
                st for st in custom_subtypes
                if st not in predefined_subtypes
            ]

            if unknown_subtypes:
                logger.warning(
                    f"Custom subtypes for '{vulnerability}' include unknown types: {unknown_subtypes}. "
                    f"Valid subtypes: {predefined_subtypes[:10]}..."  # Show first 10
                )

            return custom_subtypes

        # Otherwise return all predefined subtypes
        return predefined_subtypes

    @staticmethod
    def get_vulnerability_info(vulnerability: str) -> Optional[VulnerabilityDefinition]:
        """Get detailed information about a vulnerability."""
        try:
            vuln_enum = VulnerabilityType(vulnerability.lower())
            return VulnerabilityScanner.VULNERABILITY_DEFINITIONS.get(vuln_enum)
        except ValueError:
            return None

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate red-teaming configuration.

        Args:
            config: Red-teaming configuration dictionary

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check vulnerabilities
        vulnerabilities = config.get('vulnerabilities', [])
        if not vulnerabilities:
            errors.append("No vulnerabilities specified")

        available_vulns = VulnerabilityScanner.get_available_vulnerabilities()
        for vuln in vulnerabilities:
            if vuln.lower() not in available_vulns:
                errors.append(f"Unknown vulnerability type: {vuln}")

        # Check vulnerability subtypes
        vuln_types = config.get('vulnerability_types', {})
        for vuln_name, subtypes in vuln_types.items():
            if vuln_name not in vulnerabilities:
                logger.warning(
                    f"Subtypes specified for '{vuln_name}' but it's not in vulnerabilities list"
                )

            if not isinstance(subtypes, list):
                errors.append(f"Subtypes for '{vuln_name}' must be a list")

        # Check attacks
        attacks = config.get('attacks', [])
        if not attacks:
            logger.warning("No attacks specified, will use defaults")

        return len(errors) == 0, errors

    @staticmethod
    def get_recommended_vulnerabilities(use_case: str) -> List[str]:
        """
        Get recommended vulnerabilities based on use case.

        Args:
            use_case: One of ['general', 'healthcare', 'finance', 'customer_service', 'education']

        Returns:
            List of recommended vulnerability types
        """
        recommendations = {
            'general': [
                'bias', 'toxicity', 'pii_leakage', 'illegal_activity',
                'prompt_leakage', 'hallucination'
            ],
            'healthcare': [
                'pii_leakage', 'bias', 'misinformation',
                'illegal_activity', 'child_protection'
            ],
            'finance': [
                'pii_leakage', 'hallucination', 'misinformation', 'competition',
                'sql_injection', 'bfla', 'bola'
            ],
            'customer_service': [
                'bias', 'toxicity', 'pii_leakage',
                'competition', 'excessive_agency'
            ],
            'education': [
                'bias', 'illegal_activity', 'pii_leakage', 'misinformation',
                'intellectual_property', 'child_protection'
            ],
            'enterprise': [
                'pii_leakage', 'intellectual_property', 'competition', 'sql_injection',
                'shell_injection', 'ssrf', 'rbac', 'bfla', 'bola'
            ]
        }

        return recommendations.get(use_case.lower(), recommendations['general'])

    @staticmethod
    def print_vulnerability_guide():
        """Print a comprehensive guide of all vulnerabilities."""
        logger.info("=" * 80)
        logger.info("VULNERABILITY SCANNING GUIDE")
        logger.info("=" * 80)

        categories = VulnerabilityScanner.get_vulnerability_categories()

        for category, vulns in sorted(categories.items()):
            logger.info(f"\n{category.upper()}")
            logger.info("-" * 80)

            for vuln in vulns:
                info = VulnerabilityScanner.get_vulnerability_info(vuln)
                if info:
                    logger.info(f"\n  {info.name}")
                    logger.info(f"  Description: {info.description}")
                    logger.info(f"  Severity: {info.default_severity.value}")
                    logger.info(f"  Compliance: {', '.join(info.compliance_frameworks)}")
                    logger.info(f"  Subtypes ({len(info.subtypes)}): {', '.join(info.subtypes[:5])}...")

        logger.info("\n" + "=" * 80)


class AttackScanner:
    """Helper for attack type information."""

    ATTACK_DESCRIPTIONS = {
        # Single-turn attacks
        "prompt_injection": "Attempts to override system instructions with injected prompts",
        "prompt_probing": "Probes the system to extract information about prompts and instructions",
        "roleplay": "Uses roleplay scenarios to bypass guardrails",
        "gray_box": "Limited knowledge attacks based on partial system information",
        "math_problem": "Uses mathematical problems to disguise malicious intent",
        "multilingual": "Uses multiple languages to bypass language-specific filters",

        # Attack enhancements
        "base64": "Encodes malicious content in Base64 to evade detection",
        "leetspeak": "Uses leetspeak encoding to bypass text filters",
        "rot13": "Uses ROT13 cipher to obfuscate malicious content",

        # Advanced attacks
        "system_override": "Attempts to override system-level constraints",
        "permission_escalation": "Tries to escalate privileges beyond authorized scope",
        "goal_redirection": "Redirects the model's objective to malicious goals",
        "linguistic_confusion": "Uses linguistic tricks to confuse the model",
        "input_bypass": "Bypasses input validation and sanitization",
        "context_poisoning": "Poisons the context to influence future responses",

        # Multi-turn attacks
        "linear_jailbreaking": "Linear multi-turn jailbreak approach with gradual escalation",
        "tree_jailbreaking": "Tree-based exploration of jailbreak strategies across multiple turns",
        "crescendo_jailbreaking": "Crescendo attack with gradually increasing intensity",
        "sequential_jailbreak": "Sequential jailbreak strategy with planned attack phases",
        "bad_likert_judge": "Exploits Likert-scale based evaluation systems",
    }

    @staticmethod
    def get_available_attacks() -> List[str]:
        """Get list of all available attack types."""
        return list(AttackScanner.ATTACK_DESCRIPTIONS.keys())

    @staticmethod
    def get_attack_categories() -> Dict[str, List[str]]:
        """Get attacks grouped by category."""
        return {
            "Single-Turn Attacks": [
                "prompt_injection", "prompt_probing", "roleplay",
                "gray_box", "math_problem", "multilingual"
            ],
            "Attack Enhancements": [
                "base64", "leetspeak", "rot13"
            ],
            "Advanced Single-Turn": [
                "system_override", "permission_escalation", "goal_redirection",
                "linguistic_confusion", "input_bypass", "context_poisoning"
            ],
            "Multi-Turn Attacks": [
                "linear_jailbreaking", "tree_jailbreaking", "crescendo_jailbreaking",
                "sequential_jailbreak", "bad_likert_judge"
            ]
        }

    @staticmethod
    def get_recommended_attacks(use_case: str) -> List[str]:
        """Get recommended attacks based on use case."""
        recommendations = {
            'general': [
                'prompt_injection', 'roleplay', 'linear_jailbreaking',
                'base64', 'rot13'
            ],
            'high_security': [
                'prompt_injection', 'prompt_probing', 'system_override',
                'permission_escalation', 'linear_jailbreaking',
                'crescendo_jailbreaking', 'tree_jailbreaking',
                'base64', 'leetspeak', 'rot13'
            ],
            'production': [
                'prompt_injection', 'roleplay', 'gray_box',
                'linear_jailbreaking', 'sequential_jailbreak',
                'goal_redirection'
            ],
            'compliance': [
                'prompt_injection', 'prompt_probing', 'system_override',
                'permission_escalation', 'input_bypass'
            ],
            'conversational': [
                'linear_jailbreaking', 'crescendo_jailbreaking',
                'tree_jailbreaking', 'sequential_jailbreak',
                'roleplay', 'context_poisoning'
            ]
        }
        return recommendations.get(use_case.lower(), recommendations['general'])

    @staticmethod
    def print_attack_guide():
        """Print comprehensive attack guide."""
        logger.info("=" * 80)
        logger.info("ATTACK METHODS GUIDE")
        logger.info("=" * 80)

        categories = AttackScanner.get_attack_categories()

        for category, attacks in categories.items():
            logger.info(f"\n{category.upper()}")
            logger.info("-" * 80)

            for attack in attacks:
                desc = AttackScanner.ATTACK_DESCRIPTIONS.get(attack, "No description")
                logger.info(f"  {attack}")
                logger.info(f"    {desc}")

        logger.info("\n" + "=" * 80)