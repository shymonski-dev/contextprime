#!/usr/bin/env python3
"""
Single-command real-world end-to-end flow for the source document.

Workflow:
1. Upload the supplied document through the API document route.
2. Execute a curated query set with community-aware retrieval.
3. Submit automated feedback events for each query.
4. Build a benchmark dataset from the query set.
5. Run retrieval policy benchmark and publish trend summary.
6. Run feedback learning cycle for context selector updates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, List, Any
from base64 import urlsafe_b64encode


@dataclass
class QuerySpec:
    query: str
    expected_terms: List[str]
    answer_terms: List[str]


QUERY_SPECS: List[QuerySpec] = [
    QuerySpec(
        query="What are the main topics covered in this document?",
        expected_terms=["document", "section", "topic", "overview", "summary"],
        answer_terms=["main", "topics", "overview"],
    ),
    QuerySpec(
        query="Which procedures or step-by-step instructions are described?",
        expected_terms=["procedure", "steps", "instructions", "process", "method"],
        answer_terms=["procedure", "steps", "instructions"],
    ),
    QuerySpec(
        query="What warnings, cautions, or safety notes are included?",
        expected_terms=["warning", "caution", "safety", "risk", "notice"],
        answer_terms=["warning", "caution", "safety"],
    ),
    QuerySpec(
        query="What tools, materials, or parts are required?",
        expected_terms=["tools", "materials", "parts", "required", "equipment"],
        answer_terms=["tools", "materials", "parts"],
    ),
    QuerySpec(
        query="Are there maintenance intervals or schedules mentioned?",
        expected_terms=["maintenance", "interval", "schedule", "service", "period"],
        answer_terms=["maintenance", "interval", "schedule"],
    ),
    QuerySpec(
        query="What troubleshooting or diagnostic guidance is present?",
        expected_terms=["troubleshooting", "diagnostic", "fault", "issue", "check"],
        answer_terms=["troubleshooting", "diagnostic", "guidance"],
    ),
]


DEFAULT_PDF_PATH = os.getenv("REALWORLD_SOURCE_PDF", "").strip()


def _run_command(cmd: List[str], cwd: Path) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def _run_json_command(cmd: List[str], cwd: Path) -> Dict[str, Any]:
    output = _run_command(cmd, cwd=cwd)
    try:
        return dict(json.loads(output))
    except json.JSONDecodeError as err:
        raise RuntimeError(f"Expected JSON output, got: {output[:800]}") from err


def _base64url_json(value: Dict[str, Any]) -> str:
    encoded = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return urlsafe_b64encode(encoded).rstrip(b"=").decode("ascii")


def _build_signed_token(secret: str, subject: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": subject, "roles": ["admin"], "scopes": ["api:read", "api:write"]}
    parts = [_base64url_json(header), _base64url_json(payload)]
    signing_input = ".".join(parts).encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    parts.append(urlsafe_b64encode(signature).rstrip(b"=").decode("ascii"))
    return ".".join(parts)


def _resolve_auth_token(explicit_token: str, subject: str) -> str:
    token = explicit_token.strip()
    if token:
        return token

    jwt_secret = (
        os.getenv("SECURITY__JWT_SECRET")
        or os.getenv("SECURITY_JWT_SECRET")
        or ""
    ).strip()
    if jwt_secret:
        return _build_signed_token(secret=jwt_secret, subject=subject)

    access_token = (
        os.getenv("SECURITY__ACCESS_TOKEN")
        or os.getenv("SECURITY_ACCESS_TOKEN")
        or ""
    ).strip()
    return access_token


def _curl_with_auth(base_cmd: List[str], auth_token: str) -> List[str]:
    cmd = list(base_cmd)
    if auth_token:
        cmd.extend(["-H", f"Authorization: Bearer {auth_token}"])
    return cmd


def _build_settings_payload() -> str:
    payload = {
        "enable_ocr": True,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "chunking_method": "structure",
        "extract_entities": True,
        "build_raptor": False,
    }
    return json.dumps(payload, separators=(",", ":"))


def _build_search_payload(query: str) -> Dict[str, Any]:
    return {
        "query": query,
        "top_k": 8,
        "strategy": "hybrid",
        "vector_weight": 0.7,
        "graph_weight": 0.3,
        "use_reranking": True,
        "use_query_expansion": True,
        "graph_policy": "community",
        "include_graph_context": True,
    }


def _run_search_request(
    *,
    api_base_url: str,
    payload: Dict[str, Any],
    auth_token: str,
    cwd: Path,
) -> Dict[str, Any]:
    return _run_json_command(
        _curl_with_auth([
            "curl",
            "-sS",
            "-X",
            "POST",
            f"{api_base_url}/api/search/hybrid",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(payload),
        ], auth_token),
        cwd=cwd,
    )


def _build_fallback_queries(spec: QuerySpec) -> List[str]:
    answer_phrase = " ".join(spec.answer_terms[:2]).strip()
    expected_phrase = " ".join(spec.expected_terms[:3]).strip()
    compact_query = re.sub(r"\s+", " ", spec.query).strip()

    candidates = [
        compact_query,
        f"{answer_phrase} procedure details".strip(),
        f"{expected_phrase} guidance".strip(),
        "document overview and key points",
    ]

    unique: List[str] = []
    seen = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(normalized)
    return unique


def _term_coverage(text: str, terms: List[str]) -> float:
    if not terms:
        return 0.0
    haystack = text.lower()
    hits = 0
    for term in terms:
        token = term.strip().lower()
        if token and token in haystack:
            hits += 1
    return hits / float(len(terms))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-world end-to-end flow for a source document")
    parser.add_argument(
        "--pdf-path",
        default=DEFAULT_PDF_PATH,
        help="Absolute path to the source PDF document",
    )
    parser.add_argument(
        "--api-base-url",
        default="http://localhost:8000",
        help="API base URL",
    )
    parser.add_argument(
        "--output-root",
        default="reports",
        help="Root directory for run artifacts",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip retrieval policy benchmark and trend publishing",
    )
    parser.add_argument(
        "--skip-feedback-learning",
        action="store_true",
        help="Skip context selector feedback learning cycle",
    )
    parser.add_argument(
        "--auth-token",
        default="",
        help=(
            "Bearer token for protected routes. If omitted, the script derives one "
            "from SECURITY__JWT_SECRET or SECURITY__ACCESS_TOKEN."
        ),
    )
    parser.add_argument(
        "--auth-subject",
        default="realworld-runner",
        help="Subject claim used when generating a signed token from SECURITY__JWT_SECRET.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    pdf_path_input = str(args.pdf_path or "").strip()
    if not pdf_path_input:
        raise ValueError(
            "Provide --pdf-path or set REALWORLD_SOURCE_PDF with an absolute PDF path."
        )

    pdf_path = Path(pdf_path_input).expanduser()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (project_root / args.output_root / f"realworld_e2e_{timestamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    auth_token = _resolve_auth_token(explicit_token=args.auth_token, subject=args.auth_subject)

    print(f"[1/8] Checking API health at {args.api_base_url} ...")
    health = _run_json_command(
        ["curl", "-sS", f"{args.api_base_url}/api/health"],
        cwd=project_root,
    )
    _write_json(output_dir / "health.json", health)
    print(f"      API status: {health.get('status')}")

    print(f"[2/8] Uploading source PDF: {pdf_path}")
    upload_response = _run_json_command(
        _curl_with_auth([
            "curl",
            "-sS",
            "-X",
            "POST",
            f"{args.api_base_url}/api/documents",
            "-F",
            f"file=@{pdf_path}",
            "-F",
            f"settings={_build_settings_payload()}",
        ], auth_token),
        cwd=project_root,
    )
    _write_json(output_dir / "upload_response.json", upload_response)
    doc_info = dict(upload_response.get("document") or {})
    doc_id = str(doc_info.get("id", ""))
    if not doc_id:
        raise RuntimeError(f"Upload did not return document id: {upload_response}")
    print(f"      Uploaded document id: {doc_id}")

    print("[3/8] Running retrieval query set and submitting feedback...")
    benchmark_rows: List[Dict[str, Any]] = []
    query_run_rows: List[Dict[str, Any]] = []
    feedback_rows: List[Dict[str, Any]] = []

    for index, spec in enumerate(QUERY_SPECS, start=1):
        payload = _build_search_payload(spec.query)
        search_response = _run_search_request(
            api_base_url=args.api_base_url,
            payload=payload,
            auth_token=auth_token,
            cwd=project_root,
        )
        primary_search_response = dict(search_response)
        fallback_query_used = ""
        fallback_attempt = 0

        results = list(search_response.get("results") or [])
        if not results:
            for attempt, fallback_query in enumerate(_build_fallback_queries(spec), start=1):
                fallback_payload = _build_search_payload(fallback_query)
                # Recovery queries are deterministic and use global graph scan.
                fallback_payload["use_query_expansion"] = False
                fallback_payload["graph_policy"] = "global"
                fallback_response = _run_search_request(
                    api_base_url=args.api_base_url,
                    payload=fallback_payload,
                    auth_token=auth_token,
                    cwd=project_root,
                )
                fallback_results = list(fallback_response.get("results") or [])
                if fallback_results:
                    search_response = fallback_response
                    results = fallback_results
                    fallback_query_used = fallback_query
                    fallback_attempt = attempt
                    break

        if fallback_query_used:
            _write_json(output_dir / f"search_{index:02d}_primary.json", primary_search_response)

        metadata = dict(search_response.get("metadata") or {})
        if fallback_query_used:
            metadata["fallback_query_used"] = fallback_query_used
            metadata["fallback_attempt"] = fallback_attempt
            search_response["metadata"] = metadata

        _write_json(output_dir / f"search_{index:02d}.json", search_response)

        query_id = str(metadata.get("query_id", "")).strip()
        result_ids = [str(item.get("id", "")).strip() for item in results if str(item.get("id", "")).strip()]
        combined_text = " ".join(str(item.get("content", "")) for item in results)
        expected_coverage = _term_coverage(combined_text, spec.expected_terms)
        answer_coverage = _term_coverage(combined_text, spec.answer_terms)
        helpful = bool(results) and answer_coverage >= 0.34

        selected_ids = result_ids[:2]
        result_labels: List[Dict[str, Any]] = []
        if selected_ids:
            result_labels.append({"result_id": selected_ids[0], "label": 1, "note": "top evidence"})
        if len(result_ids) >= 3:
            result_labels.append({"result_id": result_ids[-1], "label": 0, "note": "weak evidence"})

        feedback_payload = {
            "query_id": query_id,
            "helpful": helpful,
            "selected_result_ids": selected_ids,
            "result_labels": result_labels,
            "comment": "Automated end-to-end feedback capture run",
            "metadata": {
                "run_tag": "realworld_e2e",
                "expected_term_coverage": expected_coverage,
                "answer_term_coverage": answer_coverage,
                "query_index": index,
            },
        }

        feedback_response: Dict[str, Any]
        if query_id:
            feedback_response = _run_json_command(
                _curl_with_auth([
                    "curl",
                    "-sS",
                    "-X",
                    "POST",
                    f"{args.api_base_url}/api/feedback/retrieval",
                    "-H",
                    "Content-Type: application/json",
                    "-d",
                    json.dumps(feedback_payload),
                ], auth_token),
                cwd=project_root,
            )
            _write_json(output_dir / f"feedback_{index:02d}.json", feedback_response)
        else:
            feedback_response = {"success": False, "reason": "missing_query_id"}

        query_run_rows.append(
            {
                "query_index": index,
                "query": spec.query,
                "query_id": query_id,
                "result_count": len(results),
                "selected_result_ids": selected_ids,
                "expected_term_coverage": expected_coverage,
                "answer_term_coverage": answer_coverage,
                "helpful": helpful,
                "fallback_query_used": fallback_query_used or None,
                "fallback_attempt": fallback_attempt if fallback_attempt else None,
            }
        )
        feedback_rows.append(
            {
                "query_index": index,
                "query_id": query_id,
                "feedback_response": feedback_response,
            }
        )
        benchmark_rows.append(
            {
                "query": spec.query,
                "expected_ids": selected_ids,
                "expected_terms": spec.expected_terms,
                "answer_terms": spec.answer_terms,
            }
        )

        print(
            f"      Query {index}: results={len(results)}, "
            f"expected_coverage={expected_coverage:.2f}, answer_coverage={answer_coverage:.2f}, "
            f"fallback={'yes' if fallback_query_used else 'no'}"
        )

    _write_json(output_dir / "query_run_summary.json", {"queries": query_run_rows})
    _write_json(output_dir / "feedback_run_summary.json", {"feedback": feedback_rows})

    benchmark_dataset_path = output_dir / "realworld_benchmark_dataset.jsonl"
    _write_jsonl(benchmark_dataset_path, benchmark_rows)
    print(f"[4/8] Benchmark dataset created: {benchmark_dataset_path}")

    if not args.skip_benchmark:
        print("[5/8] Running retrieval policy benchmark and publishing trend history...")
        benchmark_output_path = output_dir / "retrieval_policy_benchmark.json"
        benchmark_cmd = [
            sys.executable,
            str(project_root / "scripts" / "benchmark_retrieval_policies.py"),
            "--dataset",
            str(benchmark_dataset_path),
            "--output",
            str(benchmark_output_path),
            "--publish-trends",
            "--trend-history",
            str(project_root / "reports" / "retrieval_policy_trend_history.jsonl"),
            "--trend-markdown",
            str(project_root / "reports" / "retrieval_policy_trends.md"),
        ]
        _run_command(benchmark_cmd, cwd=project_root)
        print(f"      Benchmark report: {benchmark_output_path}")
    else:
        print("[5/8] Skipped benchmark and trend publishing by request.")

    if not args.skip_feedback_learning:
        print("[6/8] Running feedback learning cycle for context selector update...")
        learning_cmd = [
            sys.executable,
            str(project_root / "scripts" / "run_feedback_learning_cycle.py"),
            "--query-events",
            str(project_root / "data" / "feedback" / "retrieval_query_events.jsonl"),
            "--feedback-events",
            str(project_root / "data" / "feedback" / "retrieval_feedback_events.jsonl"),
            "--dataset-out",
            str(project_root / "data" / "feedback" / "context_selector_feedback_dataset.jsonl"),
            "--model-path",
            str(project_root / "models" / "context_selector.json"),
            "--min-examples",
            "5",
            "--holdout-ratio",
            "0.2",
        ]
        _run_command(learning_cmd, cwd=project_root)
    else:
        print("[6/8] Skipped feedback learning cycle by request.")

    run_summary = {
        "run_timestamp": timestamp,
        "source_pdf": str(pdf_path),
        "api_base_url": args.api_base_url,
        "document_id": doc_id,
        "query_count": len(QUERY_SPECS),
        "artifacts_dir": str(output_dir),
        "benchmark_dataset": str(benchmark_dataset_path),
        "feedback_logs_dir": str(project_root / "data" / "feedback"),
        "trend_markdown": str(project_root / "reports" / "retrieval_policy_trends.md"),
    }
    _write_json(output_dir / "run_summary.json", run_summary)

    print("[7/8] Writing final run summary artifact...")
    print("[8/8] Completed.")
    print(f"      Artifacts directory: {output_dir}")
    print(f"      Run summary: {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
