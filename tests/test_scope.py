from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from longcode.scope import check_scope, changed_paths, path_matches, snapshot_tree


class ScopeTests(unittest.TestCase):
    def test_path_patterns_and_forbidden_precedence(self):
        self.assertTrue(path_matches("src/pkg/a.py", "src/**"))
        self.assertTrue(path_matches("README.md", "**"))
        self.assertFalse(path_matches("tests/a.py", "src/**"))
        result = check_scope(
            ["src/a.py", "tests/test_a.py"],
            ["src/**", "tests/**"],
            ["tests/**"],
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.violations, ["forbidden:tests/test_a.py"])

    def test_snapshot_detects_added_modified_and_deleted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("before")
            (root / "gone.txt").write_text("gone")
            before = snapshot_tree(root)
            (root / "a.txt").write_text("after")
            (root / "b.txt").write_text("new")
            (root / "gone.txt").unlink()
            after = snapshot_tree(root)
            self.assertEqual(changed_paths(before, after), ["a.txt", "b.txt", "gone.txt"])


if __name__ == "__main__":
    unittest.main()
