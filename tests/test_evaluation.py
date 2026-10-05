from __future__ import annotations

import unittest

from longcode.evaluation import (
    EvaluationRecord,
    aa_validation_report,
    comparison_report,
)


class EvaluationTests(unittest.TestCase):
    def test_report_compares_matched_lh_and_longcode_runs(self):
        records = [
            EvaluationRecord("A", "lh", "coding", True, input_tokens=100, output_tokens=100),
            EvaluationRecord("B", "lh", "coding", False, false_completed=True, input_tokens=100, output_tokens=100),
            EvaluationRecord("A", "longcode", "coding", True, input_tokens=80, output_tokens=70),
            EvaluationRecord(
                "B",
                "longcode",
                "coding",
                True,
                recoverable_fault=True,
                recovered=True,
                input_tokens=80,
                output_tokens=70,
            ),
        ]
        report = comparison_report(records)
        self.assertEqual(report["deltas"]["success_rate"], 0.5)
        self.assertTrue(report["claims"]["zero_false_completion"])
        self.assertTrue(report["claims"]["recoverable_fault_recovery_at_least_90pct"])
        self.assertTrue(report["claims"]["success_higher_or_cost_20pct_lower"])

    def test_missing_budget_enforcement_and_tokens_cannot_look_like_zero_cost(self):
        records = [
            EvaluationRecord(
                "A",
                "lh",
                "coding",
                True,
                token_metrics_available=False,
                budget_enforced=False,
            ),
            EvaluationRecord(
                "A",
                "longcode",
                "coding",
                True,
                token_metrics_available=False,
                budget_enforced=False,
            ),
        ]

        report = comparison_report(records)

        self.assertIsNone(report["metrics"]["lh"]["average_total_tokens"])
        self.assertEqual(report["metrics"]["lh"]["token_metrics_coverage"], 0)
        self.assertFalse(report["claims"]["matched_budget_enforced"])
        self.assertFalse(report["all_testable_claims_pass"])

    def test_formal_claim_requires_and_uses_paired_confidence_intervals(self):
        records = []
        for index in range(30):
            case_id = f"case-{index:02d}#r1"
            common = {
                "case_id": case_id,
                "base_case_id": f"case-{index:02d}",
                "repetition": 1,
                "suite": "coding",
                "snapshot_id": f"snapshot-{index:02d}",
                "condition_id": "condition-same",
                "budget_enforced": True,
                "token_metrics_available": True,
                "observed_models": ["same-model-snapshot"],
                "system_fingerprints": ["same-fingerprint"],
                "committed_progress_lost": False,
                "hidden_checks_isolated": True,
            }
            records.extend(
                [
                    EvaluationRecord(
                        arm="direct", success=False, input_tokens=80, output_tokens=20,
                        model_calls=2, **common
                    ),
                    EvaluationRecord(
                        arm="lh", success=False, input_tokens=80, output_tokens=20,
                        model_calls=2, **common
                    ),
                    EvaluationRecord(
                        arm="longcode", success=True, input_tokens=55, output_tokens=15,
                        model_calls=2, **common
                    ),
                ]
            )

        report = comparison_report(records, bootstrap_samples=1000)

        self.assertGreater(
            report["paired_statistics"]["success_rate_delta_95pct_ci"][0], 0
        )
        self.assertTrue(report["formal_claim"]["superiority_95pct"])
        self.assertTrue(report["formal_claim"]["supported"])

        unisolated = [
            EvaluationRecord(**{**item.__dict__, "hidden_checks_isolated": False})
            for item in records
        ]
        unsafe_report = comparison_report(unisolated, bootstrap_samples=1000)
        self.assertFalse(unsafe_report["formal_claim"]["supported"])
        self.assertIn(
            "hidden_checks_isolated", unsafe_report["formal_claim"]["failed_checks"]
        )

    def test_aa_report_rejects_no_difference_only_when_accounting_differs(self):
        records = []
        for arm in ("direct", "lh", "longcode"):
            records.append(
                EvaluationRecord(
                    "A#r1",
                    arm,
                    "aa",
                    True,
                    base_case_id="A",
                    snapshot_id="same-snapshot",
                    condition_id="same-condition",
                    input_tokens=40,
                    output_tokens=10,
                    model_calls=1,
                    token_metrics_available=True,
                    budget_enforced=True,
                )
            )

        self.assertTrue(aa_validation_report(records)["passed"])
        altered = [
            item
            if item.arm != "longcode"
            else EvaluationRecord(**{**item.__dict__, "output_tokens": 11})
            for item in records
        ]
        report = aa_validation_report(altered)
        self.assertFalse(report["passed"])
        self.assertEqual(report["token_mismatches"], ["A#r1"])


if __name__ == "__main__":
    unittest.main()
