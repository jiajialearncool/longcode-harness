from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable


@dataclass(frozen=True)
class EvaluationRecord:
    case_id: str
    arm: str
    suite: str
    success: bool
    false_completed: bool = False
    recoverable_fault: bool = False
    recovered: bool | None = None
    committed_progress_lost: bool | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    wall_seconds: float = 0.0
    hidden_flow_pass: bool | None = None
    human_acceptance: bool | None = None
    base_case_id: str | None = None
    repetition: int = 1
    run_exit_code: int | None = None
    timed_out: bool = False
    snapshot_id: str | None = None
    token_metrics_available: bool = True
    budget_enforced: bool = True
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    model_calls: int = 0
    budget_exhausted: bool = False
    budget_breach: bool = False
    observed_models: list[str] | None = None
    system_fingerprints: list[str] | None = None
    provider_route_requests: int = 0
    condition_id: str | None = None
    hidden_checks_isolated: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationRecord":
        return cls(**value)


@dataclass(frozen=True)
class ArmMetrics:
    arm: str
    cases: int
    independent_base_cases: int
    success_rate: float
    success_rate_95pct_ci: tuple[float, float]
    false_completion_rate: float
    false_completion_rate_95pct_ci: tuple[float, float]
    recovery_rate: float | None
    recovery_measurement_coverage: float | None
    progress_loss_rate: float | None
    progress_loss_measurement_coverage: float
    hidden_flow_pass_rate: float | None
    human_acceptance_rate: float | None
    average_total_tokens: float | None
    average_cached_input_tokens: float | None
    average_reasoning_tokens: float | None
    average_model_calls: float | None
    token_metrics_coverage: float
    budget_enforcement_rate: float
    budget_breach_rate: float
    average_wall_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_records(path: Path | str) -> list[EvaluationRecord]:
    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        values = json.loads(text)
    else:
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    records = [EvaluationRecord.from_dict(item) for item in values]
    _validate_pairs(records)
    return records


def summarize(records: list[EvaluationRecord]) -> dict[str, ArmMetrics]:
    grouped: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        grouped.setdefault(record.arm, []).append(record)
    return {arm: _metrics(arm, values) for arm, values in sorted(grouped.items())}


def comparison_report(
    records: list[EvaluationRecord],
    *,
    candidate: str = "longcode",
    baseline: str = "lh",
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260817,
    minimum_independent_cases: int = 30,
) -> dict[str, Any]:
    _validate_pairs(records)
    metrics = summarize(records)
    candidate_metrics = metrics.get(candidate)
    baseline_metrics = metrics.get(baseline)
    deltas: dict[str, float | None] | None = None
    claims: dict[str, bool | None] = {
        "paired_baseline_present": baseline_metrics is not None,
        "candidate_present": candidate_metrics is not None,
        "zero_false_completion": None,
        "recoverable_fault_recovery_at_least_90pct": None,
        "zero_committed_progress_loss": None,
        "recovery_measurement_complete": None,
        "progress_loss_measurement_complete": None,
        "success_not_materially_below_lh": None,
        "success_higher_or_cost_20pct_lower": None,
        "matched_budget_enforced": bool(records)
        and all(item.budget_enforced and not item.budget_breach for item in records),
    }
    if candidate_metrics:
        claims["zero_false_completion"] = candidate_metrics.false_completion_rate == 0
        claims["recoverable_fault_recovery_at_least_90pct"] = (
            None
            if candidate_metrics.recovery_rate is None
            else candidate_metrics.recovery_rate >= 0.9
        )
        claims["zero_committed_progress_loss"] = (
            None
            if candidate_metrics.progress_loss_rate is None
            else candidate_metrics.progress_loss_rate == 0
        )
        claims["recovery_measurement_complete"] = (
            None
            if candidate_metrics.recovery_measurement_coverage is None
            else candidate_metrics.recovery_measurement_coverage == 1
        )
        claims["progress_loss_measurement_complete"] = (
            candidate_metrics.progress_loss_measurement_coverage == 1
        )

    paired = _paired_records(records, candidate, baseline)
    paired_statistics = _paired_statistics(
        paired,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    if candidate_metrics and baseline_metrics:
        deltas = {
            "success_rate": candidate_metrics.success_rate - baseline_metrics.success_rate,
            "false_completion_rate": (
                candidate_metrics.false_completion_rate
                - baseline_metrics.false_completion_rate
            ),
            "average_total_tokens": (
                candidate_metrics.average_total_tokens
                - baseline_metrics.average_total_tokens
                if candidate_metrics.average_total_tokens is not None
                and baseline_metrics.average_total_tokens is not None
                else None
            ),
            "average_wall_seconds": (
                candidate_metrics.average_wall_seconds
                - baseline_metrics.average_wall_seconds
            ),
        }
        claims["success_not_materially_below_lh"] = deltas["success_rate"] >= -0.02
        cost_ratio = (
            candidate_metrics.average_total_tokens
            / baseline_metrics.average_total_tokens
            if baseline_metrics.average_total_tokens is not None
            and candidate_metrics.average_total_tokens is not None
            and baseline_metrics.average_total_tokens > 0
            else None
        )
        claims["success_higher_or_cost_20pct_lower"] = (
            deltas["success_rate"] > 0
            or (
                candidate_metrics.success_rate >= baseline_metrics.success_rate - 0.02
                and cost_ratio is not None
                and cost_ratio <= 0.8
            )
        )

    independent_cases = len(
        {
            record.base_case_id or record.case_id
            for record in records
            if record.arm in {candidate, baseline}
        }
    )
    matched_snapshots = _matched_nonempty_value(records, "snapshot_id")
    matched_conditions = _matched_nonempty_value(records, "condition_id")
    token_metrics_complete = bool(records) and all(
        item.token_metrics_available for item in records
    )
    model_observation = _model_observation(records)
    success_ci = paired_statistics["success_rate_delta_95pct_ci"]
    token_ratio_ci = paired_statistics["total_token_ratio_95pct_ci"]
    superiority = success_ci is not None and success_ci[0] > 0
    noninferiority = success_ci is not None and success_ci[0] >= -0.02
    token_efficient = token_ratio_ci is not None and token_ratio_ci[1] <= 0.8
    efficacy = superiority or (noninferiority and token_efficient)
    formal_checks = {
        "minimum_independent_cases": independent_cases >= minimum_independent_cases,
        "all_cases_paired": bool(paired)
        and len(paired)
        == len([item for item in records if item.arm == candidate])
        == len([item for item in records if item.arm == baseline]),
        "matched_snapshots": matched_snapshots,
        "matched_conditions": matched_conditions,
        "hidden_checks_isolated": bool(records)
        and all(
            item.hidden_checks_isolated
            for item in records
            if item.arm in {candidate, baseline}
        ),
        "matched_budget_enforced": bool(claims["matched_budget_enforced"]),
        "token_metrics_complete": token_metrics_complete,
        "observed_model_matched": model_observation["matched"],
        "system_fingerprint_matched": model_observation["fingerprint_matched"],
        "zero_false_completion": bool(claims["zero_false_completion"]),
        "progress_loss_measurement_complete": bool(
            candidate_metrics
            and candidate_metrics.progress_loss_measurement_coverage == 1
        ),
        "zero_committed_progress_loss": claims["zero_committed_progress_loss"] is True,
        "fault_recovery_measured_and_at_least_90pct": bool(
            candidate_metrics
            and (
                candidate_metrics.recovery_measurement_coverage is None
                or (
                    candidate_metrics.recovery_measurement_coverage == 1
                    and candidate_metrics.recovery_rate is not None
                    and candidate_metrics.recovery_rate >= 0.9
                )
            )
        ),
        "efficacy_superior_or_noninferior_and_20pct_cheaper": efficacy,
    }
    formal_supported = all(formal_checks.values())
    return {
        "metrics": {arm: item.to_dict() for arm, item in metrics.items()},
        "candidate": candidate,
        "baseline": baseline,
        "deltas": deltas,
        "paired_statistics": paired_statistics,
        "claims": claims,
        "all_testable_claims_pass": all(
            value for value in claims.values() if value is not None
        ),
        "formal_claim": {
            "supported": formal_supported,
            "checks": formal_checks,
            "independent_base_cases": independent_cases,
            "minimum_independent_cases": minimum_independent_cases,
            "superiority_95pct": superiority,
            "noninferiority_margin": -0.02,
            "noninferior_95pct": noninferiority,
            "token_ratio_at_most_0_8_95pct": token_efficient,
            "model_observation": model_observation,
            "failed_checks": [key for key, passed in formal_checks.items() if not passed],
        },
        "note": (
            "The legacy gates summarize recorded runs. Only formal_claim.supported combines "
            "paired uncertainty, sample size, snapshots, hidden-check isolation, conditions, "
            "provider usage, and budget enforcement."
        ),
    }


def aa_validation_report(
    records: list[EvaluationRecord],
    *,
    token_tolerance: int = 0,
) -> dict[str, Any]:
    _validate_pairs(records)
    grouped: dict[str, dict[str, EvaluationRecord]] = {}
    for record in records:
        grouped.setdefault(record.case_id, {})[record.arm] = record
    required = {"direct", "lh", "longcode"}
    complete = bool(grouped) and all(set(values) == required for values in grouped.values())
    outcome_mismatches: list[str] = []
    token_mismatches: list[str] = []
    call_mismatches: list[str] = []
    for case_id, values in sorted(grouped.items()):
        if set(values) != required:
            continue
        rows = [values[name] for name in sorted(required)]
        if len({item.success for item in rows}) != 1:
            outcome_mismatches.append(case_id)
        totals = [item.input_tokens + item.output_tokens for item in rows]
        if max(totals) - min(totals) > token_tolerance:
            token_mismatches.append(case_id)
        if len({item.model_calls for item in rows}) != 1:
            call_mismatches.append(case_id)
    checks = {
        "complete_three_arm_pairs": complete,
        "identical_outcomes": not outcome_mismatches,
        "identical_token_accounting": not token_mismatches,
        "identical_model_calls": not call_mismatches,
        "matched_snapshots": _matched_nonempty_value(records, "snapshot_id"),
        "matched_conditions": _matched_nonempty_value(records, "condition_id"),
        "matched_budget_enforced": bool(records)
        and all(item.budget_enforced and not item.budget_breach for item in records),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "cases": len(grouped),
        "outcome_mismatches": outcome_mismatches,
        "token_mismatches": token_mismatches,
        "call_mismatches": call_mismatches,
        "token_tolerance": token_tolerance,
    }


def _metrics(arm: str, records: list[EvaluationRecord]) -> ArmMetrics:
    recoverable = [item for item in records if item.recoverable_fault]
    measured_recoveries = [item for item in recoverable if item.recovered is not None]
    measured_progress_loss = [
        item for item in records if item.committed_progress_lost is not None
    ]
    flows = [item.hidden_flow_pass for item in records if item.hidden_flow_pass is not None]
    acceptance = [
        item.human_acceptance for item in records if item.human_acceptance is not None
    ]
    token_records = [item for item in records if item.token_metrics_available]
    success_count = sum(item.success for item in records)
    false_count = sum(item.false_completed for item in records)
    return ArmMetrics(
        arm=arm,
        cases=len(records),
        independent_base_cases=len(
            {item.base_case_id or item.case_id for item in records}
        ),
        success_rate=_rate([item.success for item in records]),
        success_rate_95pct_ci=_wilson_interval(success_count, len(records)),
        false_completion_rate=_rate([item.false_completed for item in records]),
        false_completion_rate_95pct_ci=_wilson_interval(false_count, len(records)),
        recovery_rate=(
            _rate([bool(item.recovered) for item in measured_recoveries])
            if measured_recoveries
            else None
        ),
        recovery_measurement_coverage=(
            len(measured_recoveries) / len(recoverable) if recoverable else None
        ),
        progress_loss_rate=(
            _rate([bool(item.committed_progress_lost) for item in measured_progress_loss])
            if measured_progress_loss
            else None
        ),
        progress_loss_measurement_coverage=(
            len(measured_progress_loss) / len(records) if records else 0.0
        ),
        hidden_flow_pass_rate=_rate(flows) if flows else None,
        human_acceptance_rate=_rate(acceptance) if acceptance else None,
        average_total_tokens=(
            mean(item.input_tokens + item.output_tokens for item in token_records)
            if token_records
            else None
        ),
        average_cached_input_tokens=(
            mean(item.cached_input_tokens for item in token_records)
            if token_records
            else None
        ),
        average_reasoning_tokens=(
            mean(item.reasoning_tokens for item in token_records)
            if token_records
            else None
        ),
        average_model_calls=(
            mean(item.model_calls for item in token_records) if token_records else None
        ),
        token_metrics_coverage=(len(token_records) / len(records) if records else 0.0),
        budget_enforcement_rate=_rate([item.budget_enforced for item in records]),
        budget_breach_rate=_rate([item.budget_breach for item in records]),
        average_wall_seconds=(mean(item.wall_seconds for item in records) if records else 0.0),
    )


def _paired_records(
    records: list[EvaluationRecord], candidate: str, baseline: str
) -> list[tuple[EvaluationRecord, EvaluationRecord]]:
    grouped: dict[str, dict[str, EvaluationRecord]] = {}
    for record in records:
        if record.arm in {candidate, baseline}:
            grouped.setdefault(record.case_id, {})[record.arm] = record
    return [
        (values[candidate], values[baseline])
        for _, values in sorted(grouped.items())
        if candidate in values and baseline in values
    ]


def _paired_statistics(
    pairs: list[tuple[EvaluationRecord, EvaluationRecord]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not pairs:
        return {
            "pairs": 0,
            "candidate_only_success": 0,
            "baseline_only_success": 0,
            "mcnemar_exact_two_sided_p": None,
            "success_rate_delta": None,
            "success_rate_delta_95pct_ci": None,
            "total_token_ratio": None,
            "total_token_ratio_95pct_ci": None,
            "bootstrap_samples": bootstrap_samples,
        }
    candidate_only = sum(candidate.success and not baseline.success for candidate, baseline in pairs)
    baseline_only = sum(baseline.success and not candidate.success for candidate, baseline in pairs)
    delta = mean(float(candidate.success) - float(baseline.success) for candidate, baseline in pairs)
    token_pairs = [
        pair
        for pair in pairs
        if pair[0].token_metrics_available
        and pair[1].token_metrics_available
        and pair[1].input_tokens + pair[1].output_tokens > 0
    ]
    token_ratio = _aggregate_token_ratio(token_pairs)
    clusters: dict[str, list[tuple[EvaluationRecord, EvaluationRecord]]] = {}
    for pair in pairs:
        cluster = pair[0].base_case_id or pair[0].case_id
        clusters.setdefault(cluster, []).append(pair)
    success_ci = _cluster_bootstrap_interval(
        list(clusters.values()),
        lambda values: mean(
            float(candidate.success) - float(baseline.success)
            for candidate, baseline in values
        ),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    token_clusters: dict[str, list[tuple[EvaluationRecord, EvaluationRecord]]] = {}
    for pair in token_pairs:
        cluster = pair[0].base_case_id or pair[0].case_id
        token_clusters.setdefault(cluster, []).append(pair)
    token_ci = _cluster_bootstrap_interval(
        list(token_clusters.values()),
        _aggregate_token_ratio,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )
    return {
        "pairs": len(pairs),
        "independent_base_cases": len(clusters),
        "candidate_only_success": candidate_only,
        "baseline_only_success": baseline_only,
        "mcnemar_exact_two_sided_p": _mcnemar_exact_p(candidate_only, baseline_only),
        "success_rate_delta": delta,
        "success_rate_delta_95pct_ci": success_ci,
        "token_pairs": len(token_pairs),
        "total_token_ratio": token_ratio,
        "total_token_ratio_95pct_ci": token_ci,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }


def _aggregate_token_ratio(
    pairs: list[tuple[EvaluationRecord, EvaluationRecord]],
) -> float:
    baseline = sum(item.input_tokens + item.output_tokens for _, item in pairs)
    candidate = sum(item.input_tokens + item.output_tokens for item, _ in pairs)
    return candidate / baseline if baseline else math.inf


def _cluster_bootstrap_interval(
    clusters: list[list[tuple[EvaluationRecord, EvaluationRecord]]],
    statistic: Callable[[list[tuple[EvaluationRecord, EvaluationRecord]]], float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float] | None:
    if not clusters:
        return None
    if samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        selected: list[tuple[EvaluationRecord, EvaluationRecord]] = []
        for _ in range(len(clusters)):
            selected.extend(clusters[rng.randrange(len(clusters))])
        value = statistic(selected)
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None
    values.sort()
    return (_percentile(values, 0.025), _percentile(values, 0.975))


def _mcnemar_exact_p(candidate_only: int, baseline_only: int) -> float:
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    smaller = min(candidate_only, baseline_only)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, center - half), min(1.0, center + half))


def _percentile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    position = fraction * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _matched_nonempty_value(records: list[EvaluationRecord], field: str) -> bool:
    grouped: dict[str, list[Any]] = {}
    for record in records:
        grouped.setdefault(record.case_id, []).append(getattr(record, field))
    return bool(grouped) and all(
        len(values) == 3 and all(value not in (None, "") for value in values)
        and len(set(values)) == 1
        for values in grouped.values()
    )


def _model_observation(records: list[EvaluationRecord]) -> dict[str, Any]:
    observed: set[str] = set()
    fingerprints: set[str] = set()
    coverage = 0
    for record in records:
        models = record.observed_models or []
        if models:
            coverage += 1
            observed.update(models)
        fingerprints.update(record.system_fingerprints or [])
    return {
        "matched": bool(records) and coverage == len(records) and len(observed) == 1,
        "coverage": coverage / len(records) if records else 0.0,
        "models": sorted(observed),
        "system_fingerprints": sorted(fingerprints),
        "fingerprint_matched": len(fingerprints) <= 1,
    }


def _rate(values: list[bool]) -> float:
    return sum(bool(value) for value in values) / len(values) if values else 0.0


def _validate_pairs(records: list[EvaluationRecord]) -> None:
    seen: set[tuple[str, str]] = set()
    grouped: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        key = (record.case_id, record.arm)
        if key in seen:
            raise ValueError(
                f"duplicate evaluation record: case={record.case_id} arm={record.arm}"
            )
        seen.add(key)
        if record.arm not in {"direct", "lh", "longcode"}:
            raise ValueError(f"unknown evaluation arm: {record.arm}")
        grouped.setdefault(record.case_id, []).append(record)
    for case_id, values in grouped.items():
        snapshots = {item.snapshot_id for item in values if item.snapshot_id}
        if len(snapshots) > 1:
            raise ValueError(f"snapshot mismatch inside paired case: {case_id}")
        conditions = {item.condition_id for item in values if item.condition_id}
        if len(conditions) > 1:
            raise ValueError(f"condition mismatch inside paired case: {case_id}")
