from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from longcode.goals import revise_goal
from longcode.models import TaskContract
from longcode.storage import RuntimeStore


class GoalTests(unittest.TestCase):
    def _store(self, root: Path) -> RuntimeStore:
        workspace = root / "workspace"
        workspace.mkdir()
        contract = TaskContract.create(
            "Original durable goal",
            ["Feature works", "Tests stay green"],
            checks=["true"],
        )
        store = RuntimeStore(root / "runtime")
        store.initialize(contract, workspace)
        return store

    def test_revision_preserves_original_and_appends_history(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(Path(directory))
            original = store.load_original_contract().to_dict()
            revised = revise_goal(
                store,
                objective="Revised durable goal",
                add_constraints=["Do not edit tests"],
                add_acceptance=["No scope drift"],
            )
            self.assertEqual(revised.version, 2)
            self.assertEqual(store.load_original_contract().to_dict(), original)
            self.assertEqual(store.load_contract().objective, "Revised durable goal")
            state = store.load_state()
            self.assertEqual(state.criteria["AC-001"].status, "needs_revalidation")
            self.assertEqual(state.criteria["AC-003"].status, "pending")
            revisions = [
                json.loads(line)
                for line in store.revisions_path.read_text().splitlines()
                if line
            ]
            self.assertEqual([item["type"] for item in revisions], ["goal_created", "goal_revised"])

    def test_adding_criterion_does_not_invalidate_unaffected_verified_work(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(Path(directory))
            state = store.load_state()
            state.criteria["AC-001"].status = "verified"
            store.save_state(state)
            revise_goal(store, add_acceptance=["New criterion"])
            state = store.load_state()
            self.assertEqual(state.criteria["AC-001"].status, "verified")
            self.assertEqual(state.criteria["AC-003"].status, "pending")

    def test_initialize_refuses_nonempty_runtime_to_protect_original_goal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            runtime = root / "runtime"
            runtime.mkdir()
            (runtime / "existing.txt").write_text("do not overwrite")
            store = RuntimeStore(runtime)
            contract = TaskContract.create("Goal", ["Works"], checks=["true"])
            with self.assertRaises(FileExistsError):
                store.initialize(contract, workspace)
            self.assertEqual((runtime / "existing.txt").read_text(), "do not overwrite")


if __name__ == "__main__":
    unittest.main()
