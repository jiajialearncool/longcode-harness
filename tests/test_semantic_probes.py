from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from longcode.models import Subtask, TaskContract, TaskState
from longcode.semantic_probes import SemanticContractVerifier
from longcode.verifiers import VerificationContext


IDENTITY_CLI = """
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
source = Path(args.input)
target = Path(args.output)
data = json.loads(source.read_text())
rendered = json.dumps(data, sort_keys=True, indent=2) + "\\n"
if args.dry_run:
    raise SystemExit(0)
if target.exists() and target.read_text() != rendered:
    print("conflict: incompatible output", file=sys.stderr)
    raise SystemExit(3)
target.write_text(rendered)
"""


MIGRATING_CLI = IDENTITY_CLI.replace(
    "data = json.loads(source.read_text())",
    "data = json.loads(source.read_text())\ndata['version'] = int(data.get('version', 0)) + 1",
)


def migration_context(root: Path) -> VerificationContext:
    contract = TaskContract.create(
        "Implement a JSON migration CLI with --input and --output",
        [
            "Support --dry-run without mutation; rerunning the migration must be idempotent"
        ],
        checks=["python3 -m py_compile automation.py"],
    )
    state = TaskState.create(contract)
    subtask = Subtask(
        id="ST-0001",
        round=1,
        criterion_id="AC-001",
        criterion=contract.acceptance_criteria[0].description,
        objective=contract.objective,
        constraints=[],
        allowed_paths=["**"],
        forbidden_paths=[".git/**"],
        checks=contract.checks,
        attempt=1,
        max_attempts=2,
    )
    return VerificationContext(
        workspace=root,
        contract=contract,
        state=state,
        subtask=subtask,
        changed_paths=["automation.py"],
        executor_report={},
    )


class SemanticContractVerifierTests(unittest.TestCase):
    def test_json_migration_identity_is_rejected_with_actionable_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.json").write_text(
                '{"version": 1, "unknown": {"keep": true}}\n', encoding="utf-8"
            )
            (root / "automation.py").write_text(
                textwrap.dedent(IDENTITY_CLI), encoding="utf-8"
            )

            result = SemanticContractVerifier().verify(migration_context(root))

            self.assertEqual(result.verdict, "fail")
            self.assertEqual(result.fault_code, "semantic_behavior_failed")
            failed = [
                item
                for item in result.details["obligations"]
                if item["status"] == "fail"
            ]
            self.assertEqual(
                [item["id"] for item in failed],
                ["migration_transforms_semantics"],
            )
            self.assertIn("semantically identical", failed[0]["observed"])

    def test_json_migration_transform_passes_probe_and_regression_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.json").write_text(
                '{"version": 1, "unknown": {"keep": true}}\n', encoding="utf-8"
            )
            (root / "automation.py").write_text(
                textwrap.dedent(MIGRATING_CLI), encoding="utf-8"
            )

            result = SemanticContractVerifier().verify(migration_context(root))

            self.assertEqual(result.verdict, "pass")
            self.assertTrue(
                all(
                    item["status"] == "pass"
                    for item in result.details["obligations"]
                )
            )


if __name__ == "__main__":
    unittest.main()
