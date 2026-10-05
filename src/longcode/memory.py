from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import utc_now


@dataclass
class MemoryRecord:
    id: str
    kind: str
    content: str
    trust: str
    source: str
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryRecord":
        return cls(**value)


class MemoryStore:
    """Separate durable trusted facts from disposable episodic observations."""

    def __init__(self, runtime_root: Path | str):
        self.root = Path(runtime_root).expanduser().resolve() / "memory"
        self.trusted_path = self.root / "trusted.json"
        self.episodic_path = self.root / "episodic.jsonl"

    def add_episode(
        self,
        kind: str,
        content: str,
        *,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        record = MemoryRecord(
            id=f"MEM-{uuid4()}",
            kind=kind,
            content=content,
            trust="untrusted",
            source=source,
            metadata=dict(metadata or {}),
        )
        self.root.mkdir(parents=True, exist_ok=True)
        with self.episodic_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def promote(
        self,
        kind: str,
        content: str,
        *,
        source: str,
        evidence: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        if not evidence:
            raise ValueError("trusted memory requires evidence")
        record = MemoryRecord(
            id=f"MEM-{uuid4()}",
            kind=kind,
            content=content,
            trust="trusted",
            source=source,
            evidence=list(evidence),
            metadata=dict(metadata or {}),
        )
        records = self.trusted()
        records.append(record)
        self._atomic_json([item.to_dict() for item in records])
        return record

    def trusted(self) -> list[MemoryRecord]:
        if not self.trusted_path.exists():
            return []
        values = json.loads(self.trusted_path.read_text(encoding="utf-8"))
        return [MemoryRecord.from_dict(item) for item in values]

    def episodes(self, *, tail: int | None = None) -> list[MemoryRecord]:
        if not self.episodic_path.exists():
            return []
        records = [
            MemoryRecord.from_dict(json.loads(line))
            for line in self.episodic_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        return records if tail is None else records[-max(0, tail):]

    def context(self, *, episode_tail: int = 5) -> dict[str, list[dict[str, Any]]]:
        return {
            "trusted": [item.to_dict() for item in self.trusted()],
            "recent_episodes": [item.to_dict() for item in self.episodes(tail=episode_tail)],
        }

    def _atomic_json(self, value: list[dict[str, Any]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.trusted_path.with_name(f".{self.trusted_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.trusted_path)
