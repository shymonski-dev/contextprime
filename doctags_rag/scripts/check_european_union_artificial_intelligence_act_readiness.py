#!/usr/bin/env python3
"""
Readiness checks for Regulation (EU) 2024/1689 controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List

import yaml


@dataclass
class CheckResult:
    status: str
    check: str
    detail: str


class ReadinessChecker:
    def __init__(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.report_path = self.root / "reports" / "european_union_artificial_intelligence_act_readiness.json"
        self.results: List[CheckResult] = []
        self.profile: Dict[str, Any] = {}

    def run(self) -> int:
        self._check_profile()
        self._check_literacy_register()
        self._check_incident_process()
        self._check_transparency_notice()
        self._check_environment_security()
        self._check_runtime_defaults()
        self._check_high_risk_documents()
        self._write_report()
        return 1 if any(item.status == "fail" for item in self.results) else 0

    def _record(self, status: str, check: str, detail: str) -> None:
        self.results.append(CheckResult(status=status, check=check, detail=detail))

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate
        return (self.root / candidate).resolve()

    def _check_profile(self) -> None:
        profile_path = self.root / "compliance" / "european_union_artificial_intelligence_act_profile.yaml"
        if not profile_path.exists():
            self._record("fail", "profile_file", f"Missing profile: {profile_path}")
            return

        try:
            self.profile = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            self._record("fail", "profile_parse", f"Profile parse failed: {exc}")
            return

        for section in ("regulation", "system", "risk_classification", "controls"):
            if section not in self.profile:
                self._record("fail", "profile_structure", f"Missing profile section: {section}")
                return

        category = str(self.profile.get("risk_classification", {}).get("category", "")).strip().lower()
        allowed_categories = {"minimal_risk", "limited_risk", "high_risk", "prohibited"}
        if category not in allowed_categories:
            self._record("fail", "risk_category", f"Invalid risk category: {category}")
        elif category == "prohibited":
            self._record("fail", "risk_category", "Declared category is prohibited and cannot be deployed")
        else:
            self._record("pass", "risk_category", f"Risk category is declared as: {category}")

        last_review_raw = str(
            self.profile.get("risk_classification", {}).get("last_review_date", "")
        ).strip()
        try:
            last_review = datetime.strptime(last_review_raw, "%Y-%m-%d").date()
            days_old = (date.today() - last_review).days
            if days_old > 365:
                self._record(
                    "fail",
                    "risk_review_freshness",
                    f"Risk review is stale ({days_old} days old)",
                )
            elif days_old > 90:
                self._record(
                    "warn",
                    "risk_review_freshness",
                    f"Risk review should be refreshed soon ({days_old} days old)",
                )
            else:
                self._record(
                    "pass",
                    "risk_review_freshness",
                    f"Risk review date is recent ({days_old} days old)",
                )
        except ValueError:
            self._record("fail", "risk_review_freshness", "Invalid or missing last_review_date")

    def _check_literacy_register(self) -> None:
        controls = self.profile.get("controls", {})
        register_raw = str(controls.get("literacy_register_file", "")).strip()
        if not register_raw:
            self._record("fail", "literacy_register", "literacy_register_file is not configured")
            return

        register_path = self._resolve_path(register_raw)
        if not register_path.exists():
            self._record("fail", "literacy_register", f"Missing literacy register: {register_path}")
            return

        with register_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = [row for row in reader if any(cell.strip() for cell in row)]
        if len(rows) < 2:
            self._record("fail", "literacy_register", "Literacy register has no training entries")
            return

        self._record("pass", "literacy_register", f"Literacy register entries: {len(rows) - 1}")

    def _check_incident_process(self) -> None:
        controls = self.profile.get("controls", {})
        raw_path = str(controls.get("incident_process_file", "")).strip()
        if not raw_path:
            self._record("fail", "incident_process", "incident_process_file is not configured")
            return

        doc_path = self._resolve_path(raw_path)
        if not doc_path.exists():
            self._record("fail", "incident_process", f"Missing incident process file: {doc_path}")
            return

        content = doc_path.read_text(encoding="utf-8").lower()
        required_terms = ("incident", "serious event")
        missing = [term for term in required_terms if term not in content]
        if missing:
            self._record(
                "fail",
                "incident_process",
                f"Incident process is missing required terms: {', '.join(missing)}",
            )
            return

        self._record("pass", "incident_process", "Incident and serious event process is documented")

    def _check_transparency_notice(self) -> None:
        controls = self.profile.get("controls", {})
        raw_path = str(controls.get("transparency_notice_file", "")).strip()
        if not raw_path:
            self._record("fail", "transparency_notice", "transparency_notice_file is not configured")
            return

        ui_path = self._resolve_path(raw_path)
        if not ui_path.exists():
            self._record("fail", "transparency_notice", f"Missing user interface file: {ui_path}")
            return

        content = ui_path.read_text(encoding="utf-8")
        expected_text = "You are interacting with an artificial intelligence system."
        if expected_text not in content:
            self._record(
                "fail",
                "transparency_notice",
                "Required transparency notice text is missing from user interface",
            )
            return

        self._record("pass", "transparency_notice", "Transparency notice text is present")

    def _read_env_file(self) -> Dict[str, str]:
        env_path = self.root / ".env"
        values: Dict[str, str] = {}
        if not env_path.exists():
            self._record("fail", "environment_file", f"Missing environment file: {env_path}")
            return values

        pattern = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*)\s*$")
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = pattern.match(line)
            if not match:
                continue
            key = match.group(1).strip()
            value = match.group(2).strip().strip("\"").strip("'")
            values[key] = value
        self._record("pass", "environment_file", f"Loaded environment values from {env_path}")
        return values

    def _is_truthy(self, value: str) -> bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _check_environment_security(self) -> None:
        env_values = self._read_env_file()
        if not env_values:
            return

        require_token = env_values.get("SECURITY__REQUIRE_ACCESS_TOKEN", "")
        if not self._is_truthy(require_token):
            self._record(
                "fail",
                "require_access_token",
                "SECURITY__REQUIRE_ACCESS_TOKEN must be true for production",
            )
        else:
            self._record("pass", "require_access_token", "Production access control is enabled")

        auth_mode = env_values.get("SECURITY__AUTH_MODE", "").strip().lower()
        if auth_mode != "jwt":
            self._record(
                "fail",
                "auth_mode",
                "SECURITY__AUTH_MODE must be jwt for signed identity and permission control",
            )
        else:
            self._record("pass", "auth_mode", "Signed token mode is enabled")

        jwt_secret = env_values.get("SECURITY__JWT_SECRET", "")
        if len(jwt_secret) < 32:
            self._record(
                "fail",
                "jwt_secret",
                "SECURITY__JWT_SECRET must be set and at least 32 characters",
            )
        else:
            self._record("pass", "jwt_secret", "Signed token secret length is acceptable")

    def _check_runtime_defaults(self) -> None:
        config_path = self.root / "config" / "config.yaml"
        if not config_path.exists():
            self._record("fail", "config_file", f"Missing config file: {config_path}")
            return

        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            self._record("fail", "config_parse", f"Config parse failed: {exc}")
            return

        readiness_enabled = bool(
            (config.get("startup_readiness") or {}).get("enabled", False)
        )
        if not readiness_enabled:
            self._record(
                "fail",
                "startup_readiness",
                "startup_readiness.enabled must be true",
            )
        else:
            self._record("pass", "startup_readiness", "Dependency readiness checks are enabled")

        rate_limit = int((config.get("api") or {}).get("rate_limit", 0))
        if rate_limit <= 0:
            self._record("fail", "rate_limit", "api.rate_limit must be greater than zero")
        else:
            self._record("pass", "rate_limit", f"api.rate_limit is set to {rate_limit}")

    def _check_high_risk_documents(self) -> None:
        category = str(self.profile.get("risk_classification", {}).get("category", "")).strip().lower()
        controls = self.profile.get("controls", {})

        high_risk_files = [
            "technical_documentation_file",
            "risk_management_file",
            "data_governance_file",
            "record_keeping_file",
            "human_oversight_file",
            "accuracy_robustness_security_file",
            "quality_management_file",
            "post_market_monitoring_file",
            "deployer_fundamental_rights_assessment_file",
        ]

        if category != "high_risk":
            self._record(
                "warn",
                "high_risk_documents",
                "System is not declared as high risk; high risk document completion is not enforced",
            )
            return

        missing: List[str] = []
        for key in high_risk_files:
            raw_path = str(controls.get(key, "")).strip()
            if not raw_path:
                missing.append(f"{key}=<missing>")
                continue
            path = self._resolve_path(raw_path)
            if not path.exists() or path.stat().st_size < 80:
                missing.append(str(path))

        if missing:
            self._record(
                "fail",
                "high_risk_documents",
                "Missing high risk documentation files: " + ", ".join(missing),
            )
            return

        self._record("pass", "high_risk_documents", "High risk documentation files are present")

    def _write_report(self) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "results": [
                {"status": item.status, "check": item.check, "detail": item.detail}
                for item in self.results
            ],
            "summary": {
                "pass": sum(1 for item in self.results if item.status == "pass"),
                "warn": sum(1 for item in self.results if item.status == "warn"),
                "fail": sum(1 for item in self.results if item.status == "fail"),
            },
        }
        self.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        for item in self.results:
            print(f"[{item.status.upper()}] {item.check}: {item.detail}")
        print(f"\nReport written to {self.report_path}")


def main() -> int:
    checker = ReadinessChecker()
    return checker.run()


if __name__ == "__main__":
    sys.exit(main())
