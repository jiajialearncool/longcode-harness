from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from longcode.transactions import TransactionManager, UnsafeCandidatePathError


class TransactionTests(unittest.TestCase):
    def test_repair_candidate_uses_failed_seed_but_keeps_formal_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "value.txt").write_text("original\n", encoding="utf-8")
            manager = TransactionManager(root / "runtime")
            failed = manager.begin(workspace, round_number=1)
            (failed.candidate_workspace / "value.txt").write_text(
                "partial\n", encoding="utf-8"
            )

            repair = manager.begin_repair(
                workspace,
                seed_workspace=failed.candidate_workspace,
                round_number=2,
            )

            self.assertEqual(
                (repair.candidate_workspace / "value.txt").read_text(),
                "partial\n",
            )
            self.assertEqual((workspace / "value.txt").read_text(), "original\n")
            self.assertEqual(repair.diff(), ["value.txt"])
    def test_candidate_changes_do_not_touch_workspace_until_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "a.txt").write_text("original")
            transaction = TransactionManager(root / "runtime").begin(workspace, round_number=1)
            (transaction.candidate_workspace / "a.txt").write_text("candidate")
            (transaction.candidate_workspace / "b.txt").write_text("new")
            self.assertEqual((workspace / "a.txt").read_text(), "original")
            self.assertFalse((workspace / "b.txt").exists())
            paths = transaction.diff()
            self.assertEqual(paths, ["a.txt", "b.txt"])
            result = transaction.promote(paths)
            self.assertTrue(result.passed)
            self.assertEqual((workspace / "a.txt").read_text(), "candidate")
            self.assertEqual((workspace / "b.txt").read_text(), "new")

    def test_rollback_discards_candidate_without_workspace_pollution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            transaction = TransactionManager(root / "runtime").begin(workspace, round_number=1)
            (transaction.candidate_workspace / "bad.txt").write_text("bad")
            transaction.rollback()
            self.assertFalse((workspace / "bad.txt").exists())
            self.assertFalse(transaction.transaction_root.exists())

    def test_concurrent_source_change_is_not_overwritten_by_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            target = workspace / "value.txt"
            target.write_text("base\n", encoding="utf-8")
            transaction = TransactionManager(root / "runtime").begin(workspace, round_number=1)
            (transaction.candidate_workspace / "value.txt").write_text("agent\n", encoding="utf-8")
            target.write_text("user-change\n", encoding="utf-8")

            result = transaction.promote(transaction.diff())
            self.assertFalse(result.passed)
            self.assertIn("Source workspace changed", result.summary)
            self.assertEqual(target.read_text(encoding="utf-8"), "user-change\n")

    def test_escaping_candidate_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside.txt"
            outside.write_text("secret\n", encoding="utf-8")
            transaction = TransactionManager(root / "runtime").begin(workspace, round_number=1)
            (transaction.candidate_workspace / "escape.txt").symlink_to(outside)
            with self.assertRaises(UnsafeCandidatePathError):
                transaction.diff()


if __name__ == "__main__":
    unittest.main()
