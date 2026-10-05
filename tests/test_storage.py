from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from longcode.models import TaskContract
from longcode.storage import RuntimeStore


class StorageTests(unittest.TestCase):
    def test_event_sequence_is_strictly_monotonic_across_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            store = RuntimeStore(root / "runtime")
            store.initialize(TaskContract.create("Goal", ["Done"], checks=["true"]), workspace)
            first_run = store.begin_run()
            store.append_event("one", {})
            store.end_run(status="paused")
            second_run = store.begin_run()
            store.append_event("two", {})
            store.end_run(status="completed")

            events = store.iter_events()
            sequences = [item["sequence"] for item in events]
            self.assertEqual(sequences, list(range(1, len(events) + 1)))
            self.assertNotEqual(first_run, second_run)
            self.assertEqual(events[-1]["run_id"], second_run)


if __name__ == "__main__":
    unittest.main()
