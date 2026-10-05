from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from longcode.memory import MemoryStore


class MemoryTests(unittest.TestCase):
    def test_untrusted_episode_and_evidence_backed_fact_are_separated(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryStore(Path(directory))
            episode = memory.add_episode(
                "hypothesis", "Maybe the cache is stale", source="executor"
            )
            trusted = memory.promote(
                "fact",
                "Cache invalidation test passes",
                source="ST-0001",
                evidence=["evidence/checks.json"],
            )
            self.assertEqual(episode.trust, "untrusted")
            self.assertEqual(trusted.trust, "trusted")
            self.assertEqual([item.content for item in memory.trusted()], [trusted.content])
            self.assertEqual([item.content for item in memory.episodes()], [episode.content])

    def test_trusted_memory_cannot_be_written_without_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryStore(Path(directory))
            with self.assertRaises(ValueError):
                memory.promote("fact", "Unsupported claim", source="executor", evidence=[])


if __name__ == "__main__":
    unittest.main()
