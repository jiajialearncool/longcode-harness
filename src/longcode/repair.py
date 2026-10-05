from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import Subtask
from .verifiers import VerificationOutcome


def build_repair_packet(
    subtask: Subtask,
    outcome: VerificationOutcome,
    *,
    changed_paths: list[str],
) -> dict[str, Any]:
    """Turn verifier evidence into a bounded, executable repair contract."""

    failures = [
        item
        for item in outcome.results
        if item.required and item.verdict != "pass"
    ]
    payload = {
        "criterion_id": subtask.criterion_id,
        "criterion": subtask.criterion,
        "attempt": subtask.attempt,
        "failed_verifiers": [
            {
                "id": item.verifier_id,
                "kind": item.kind,
                "verdict": item.verdict,
                "fault_code": item.fault_code,
                "summary": item.summary,
                "evidence": list(item.evidence),
                "details": dict(item.details),
            }
            for item in failures
        ],
        "changed_paths": list(changed_paths),
        "preserve_constraints": list(subtask.constraints),
        "regression_checks": list(subtask.checks),
        "repair_instructions": (
            "Continue from the preserved candidate. Reproduce the failed behavioral obligation, "
            "make the smallest corrective change, run the targeted probe-equivalent checks, then "
            "run every regression check. Preserve behavior already demonstrated as passing."
        ),
    }
    fingerprint_source = json.dumps(
        payload["failed_verifiers"], ensure_ascii=False, sort_keys=True
    )
    payload["failure_fingerprint"] = hashlib.sha256(
        fingerprint_source.encode("utf-8")
    ).hexdigest()[:20]
    return payload
