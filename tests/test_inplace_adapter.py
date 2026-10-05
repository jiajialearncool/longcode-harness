from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from longcode.inplace_adapter import InPlaceWorkspaceAdapter


class InPlaceWorkspaceAdapterTests(unittest.TestCase):
    def test_rollback_restores_changed_deleted_and_added_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            runtime = root / "runtime"
            workspace.mkdir()
            (workspace / "kept.txt").write_text("before", encoding="utf-8")
            (workspace / "deleted.txt").write_text("restore me", encoding="utf-8")
            tx = InPlaceWorkspaceAdapter(runtime).begin(workspace, round_number=1)
            (workspace / "kept.txt").write_text("after", encoding="utf-8")
            (workspace / "deleted.txt").unlink()
            (workspace / "added.txt").write_text("remove me", encoding="utf-8")
            self.assertEqual(
                tx.diff(),
                ["added.txt", "deleted.txt", "kept.txt"],
            )
            tx.rollback()
            self.assertEqual((workspace / "kept.txt").read_text(), "before")
            self.assertEqual((workspace / "deleted.txt").read_text(), "restore me")
            self.assertFalse((workspace / "added.txt").exists())

    def test_promote_keeps_validated_in_place_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "answer.txt").write_text("old", encoding="utf-8")
            tx = InPlaceWorkspaceAdapter(root / "runtime").begin(workspace, round_number=1)
            (workspace / "answer.txt").write_text("new", encoding="utf-8")
            result = tx.promote(tx.diff())
            self.assertTrue(result.passed)
            tx.close()
            self.assertEqual((workspace / "answer.txt").read_text(), "new")


if __name__ == "__main__":
    unittest.main()
