"""Input safety checks and runtime compliance helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import yaml


class InputSafetyError(ValueError):
    """Raised when a request is blocked by input safety policy."""


@dataclass(frozen=True)
class InputSafetyDecision:
    """Result of prompt injection inspection."""

    allowed: bool
    normalized_text: str
    risk_score: int
    flags: List[str]


@dataclass(frozen=True)
class RuntimeComplianceDecision:
    """Result of runtime compliance enforcement for one response."""

    enforced: bool
    recorded: bool
    high_risk_topics: List[str]
    oversight_file: Optional[str]


class PromptInjectionGuard:
    """Conservative detection of direct instruction override attempts."""

    _SEVERE_PATTERNS = (
        (
            re.compile(
                r"(?is)\b(ignore|disregard|override)\b.{0,120}\b(previous|prior|above)\b.{0,80}\b"
                r"(instruction|prompt|system|developer)\b"
            ),
            "instruction_override_attempt",
        ),
        (
            re.compile(
                r"(?is)\b(reveal|show|print|dump|leak|expose)\b.{0,120}\b"
                r"(system prompt|developer prompt|hidden prompt|api key|secret|token|password)\b"
            ),
            "secret_exfiltration_attempt",
        ),
        (
            re.compile(
                r"(?is)\b(bypass|disable|evade)\b.{0,100}\b(guardrail|safety|policy|filter)\b"
            ),
            "safety_bypass_attempt",
        ),
    )

    _WARNING_PATTERNS = (
        (
            re.compile(r"(?is)\b(jailbreak|prompt injection)\b"),
            "injection_context_detected",
        ),
        (
            re.compile(r"(?is)\b(act as|pretend to be)\b.{0,80}\b(system|developer|admin|root)\b"),
            "role_escalation_language",
        ),
    )

    _SECRET_OUTPUT_PATTERNS = (
        re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"ghp_[A-Za-z0-9]{30,}"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
    )

    def inspect_query(self, query: str, *, strict: bool = True) -> InputSafetyDecision:
        normalized = (query or "").strip()
        risk_score = 0
        flags: List[str] = []

        for pattern, flag in self._SEVERE_PATTERNS:
            if pattern.search(normalized):
                flags.append(flag)
                risk_score += 3

        for pattern, flag in self._WARNING_PATTERNS:
            if pattern.search(normalized):
                flags.append(flag)
                risk_score += 1

        allowed = risk_score < (3 if strict else 6)
        return InputSafetyDecision(
            allowed=allowed,
            normalized_text=normalized,
            risk_score=risk_score,
            flags=sorted(set(flags)),
        )

    def enforce_query(self, query: str, *, strict: bool = True) -> InputSafetyDecision:
        decision = self.inspect_query(query, strict=strict)
        if not decision.allowed:
            labels = ", ".join(decision.flags) if decision.flags else "policy_violation"
            raise InputSafetyError(f"Request blocked by input safety policy ({labels})")
        return decision

    def sanitize_generated_text(self, text: str) -> str:
        sanitized = text or ""
        for pattern in self._SECRET_OUTPUT_PATTERNS:
            sanitized = pattern.sub("[redacted]", sanitized)
        return sanitized


class RuntimeComplianceGate:
    """Runtime enforcement hooks for high-risk response paths."""

    _HIGH_RISK_TOPIC_KEYWORDS = {
        "medical": ("diagnosis", "treatment", "medical", "clinical", "patient", "drug dosage"),
        "legal": ("legal advice", "lawsuit", "criminal", "court strategy", "legal opinion"),
        "employment": ("hiring", "candidate ranking", "firing decision", "employment eligibility"),
        "credit": ("loan approval", "credit score", "creditworthiness", "mortgage eligibility"),
        "biometric": ("biometric identification", "facial recognition", "voiceprint", "fingerprint"),
    }

    def __init__(self, root_path: Optional[Path] = None) -> None:
        self.root = root_path or Path(__file__).resolve().parents[2]
        self.profile_path = self.root / "compliance" / "european_union_artificial_intelligence_act_profile.yaml"

    def enforce_before_response(self, *, query: str, answer: str) -> RuntimeComplianceDecision:
        profile = self._load_profile()
        category = str(
            (profile.get("risk_classification") or {}).get("category", "")
        ).strip().lower()
        if category != "high_risk":
            return RuntimeComplianceDecision(
                enforced=False,
                recorded=False,
                high_risk_topics=[],
                oversight_file=None,
            )

        combined_text = f"{query}\n{answer}".lower()
        topics = self._detect_high_risk_topics(combined_text)
        if not topics:
            return RuntimeComplianceDecision(
                enforced=False,
                recorded=False,
                high_risk_topics=[],
                oversight_file=None,
            )

        controls = profile.get("controls") or {}
        raw_path = str(controls.get("human_oversight_file", "")).strip()
        if not raw_path:
            raise RuntimeError("High-risk runtime compliance requires human_oversight_file path")

        oversight_path = self._resolve_profile_path(raw_path)
        oversight_path.parent.mkdir(parents=True, exist_ok=True)

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "topics": topics,
            "query_hash": sha256(query.encode("utf-8")).hexdigest(),
            "answer_hash": sha256(answer.encode("utf-8")).hexdigest(),
            "event": "runtime_human_oversight_required",
        }
        line = "\n" + json.dumps(event, ensure_ascii=True)
        with oversight_path.open("a", encoding="utf-8") as handle:
            handle.write(line)

        return RuntimeComplianceDecision(
            enforced=True,
            recorded=True,
            high_risk_topics=topics,
            oversight_file=str(oversight_path),
        )

    def _load_profile(self) -> Dict[str, Any]:
        if not self.profile_path.exists():
            return {}
        try:
            return yaml.safe_load(self.profile_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}

    def _resolve_profile_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate
        return (self.root / candidate).resolve()

    def _detect_high_risk_topics(self, text: str) -> List[str]:
        matched: List[str] = []
        for topic, keywords in self._HIGH_RISK_TOPIC_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                matched.append(topic)
        return matched
