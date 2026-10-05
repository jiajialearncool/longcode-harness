from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import TaskContract, TaskState, utc_now


class RuntimeStore:
    """Durable, append-oriented state for one long-running goal."""

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().resolve()
        self.goals_dir = self.root / "goals"
        self.versions_dir = self.goals_dir / "versions"
        self.evidence_dir = self.root / "evidence"
        self.original_goal_path = self.goals_dir / "original.json"
        self.current_goal_path = self.goals_dir / "current.json"
        self.revisions_path = self.goals_dir / "revisions.jsonl"
        self.state_path = self.root / "state.json"
        self.events_path = self.root / "events.jsonl"
        self.metadata_path = self.root / "metadata.json"
        self.sequence_path = self.root / "event-sequence"
        self.checkpoints_dir = self.root / "checkpoints"
        self.faults_path = self.root / "faults.jsonl"
        self.transactions_dir = self.root / "transactions"

    @property
    def exists(self) -> bool:
        return self.current_goal_path.exists() and self.state_path.exists()

    def initialize(self, contract: TaskContract, workspace: Path | str) -> None:
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"runtime directory is not empty: {self.root}")
        workspace_path = Path(workspace).expanduser().resolve()
        if not workspace_path.is_dir():
            raise NotADirectoryError(f"workspace does not exist: {workspace_path}")
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.transactions_dir.mkdir(parents=True, exist_ok=True)
        payload = contract.to_dict()
        self._atomic_json(self.original_goal_path, payload)
        self._atomic_json(self.current_goal_path, payload)
        self._atomic_json(self._version_path(contract.version), payload)
        self._atomic_json(
            self.metadata_path,
            {
                "workspace": str(workspace_path),
                "runtime": str(self.root),
                "created_at": utc_now(),
                "schema_version": 2,
                "active_run_id": None,
            },
        )
        self.save_state(TaskState.create(contract))
        self.append_revision(
            {
                "type": "goal_created",
                "version": contract.version,
                "objective": contract.objective,
            }
        )
        self.append_event(
            "goal_created",
            {
                "goal_id": contract.goal_id,
                "contract_version": contract.version,
                "acceptance_ids": [item.id for item in contract.acceptance_criteria],
            },
        )

    def load_contract(self) -> TaskContract:
        return TaskContract.from_dict(self._read_json(self.current_goal_path))

    def load_original_contract(self) -> TaskContract:
        return TaskContract.from_dict(self._read_json(self.original_goal_path))

    def save_contract(self, contract: TaskContract) -> None:
        payload = contract.to_dict()
        version_path = self._version_path(contract.version)
        if version_path.exists():
            raise FileExistsError(f"goal version already exists and is immutable: {version_path}")
        self._atomic_json(version_path, payload)
        self._atomic_json(self.current_goal_path, payload)

    def load_state(self) -> TaskState:
        state = TaskState.from_dict(self._read_json(self.state_path))
        # v0.1 runtimes have no typed graph. Migration is deterministic and is persisted by the
        # next state transition rather than mutating a runtime during a read.
        from .state_graph import ensure_graph

        ensure_graph(self.load_contract(), state)
        return state

    def save_state(self, state: TaskState) -> None:
        state.updated_at = utc_now()
        self._atomic_json(self.state_path, state.to_dict())

    def metadata(self) -> dict[str, Any]:
        return self._read_json(self.metadata_path)

    def workspace(self) -> Path:
        return Path(self.metadata()["workspace"]).resolve()

    def append_event(self, event_type: str, data: dict[str, Any]) -> None:
        metadata = self.metadata() if self.metadata_path.exists() else {}
        sequence = self._next_sequence()
        self._append_jsonl(
            self.events_path,
            {
                "schema_version": 2,
                "sequence": sequence,
                "event_id": str(uuid4()),
                "run_id": metadata.get("active_run_id"),
                "timestamp": utc_now(),
                "type": event_type,
                "data": data,
            },
        )

    def append_fault(self, fault: dict[str, Any]) -> None:
        self._append_jsonl(self.faults_path, fault)

    def update_transaction(self, candidate_id: str, **updates: Any) -> dict[str, Any]:
        path = self._transaction_path(candidate_id)
        current = self._read_json(path) if path.exists() else {
            "schema_version": 1,
            "candidate_id": candidate_id,
            "created_at": utc_now(),
        }
        next_phase = updates.get("phase")
        if next_phase and next_phase != current.get("phase"):
            history = list(current.get("phase_history", []))
            history.append({"phase": next_phase, "timestamp": utc_now()})
            current["phase_history"] = history
        current.update(updates)
        current["updated_at"] = utc_now()
        self._atomic_json(path, current)
        return current

    def load_transaction(self, candidate_id: str) -> dict[str, Any] | None:
        path = self._transaction_path(candidate_id)
        return self._read_json(path) if path.exists() else None

    def iter_transactions(self) -> list[dict[str, Any]]:
        if not self.transactions_dir.exists():
            return []
        return [
            self._read_json(path)
            for path in sorted(self.transactions_dir.glob("*.json"))
        ]

    def begin_run(self) -> str:
        metadata = self.metadata()
        run_id = f"RUN-{uuid4()}"
        metadata["active_run_id"] = run_id
        metadata["last_run_started_at"] = utc_now()
        self._atomic_json(self.metadata_path, metadata)
        return run_id

    def end_run(self, *, status: str) -> None:
        metadata = self.metadata()
        metadata["last_run_id"] = metadata.get("active_run_id")
        metadata["last_run_status"] = status
        metadata["last_run_ended_at"] = utc_now()
        metadata["active_run_id"] = None
        self._atomic_json(self.metadata_path, metadata)

    def checkpoint_state(self, state: TaskState, *, label: str) -> str:
        safe_label = "".join(character if character.isalnum() or character in "-_" else "-" for character in label)
        path = self.checkpoints_dir / f"{state.round:04d}-{safe_label}-{uuid4().hex[:8]}.json"
        self._atomic_json(path, state.to_dict())
        return str(path.relative_to(self.root))

    def append_revision(self, data: dict[str, Any]) -> None:
        payload = {"timestamp": utc_now(), **data}
        self._append_jsonl(self.revisions_path, payload)

    def write_evidence(self, relative: str, content: str | dict[str, Any] | list[Any]) -> str:
        path = self.evidence_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            self._atomic_text(path, content)
        else:
            self._atomic_text(path, json.dumps(content, ensure_ascii=False, indent=2) + "\n")
        return str(path.relative_to(self.root))

    def iter_events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        return [json.loads(line) for line in self.events_path.read_text().splitlines() if line]

    def iter_faults(self) -> list[dict[str, Any]]:
        if not self.faults_path.exists():
            return []
        return [json.loads(line) for line in self.faults_path.read_text().splitlines() if line]

    def _version_path(self, version: int) -> Path:
        return self.versions_dir / f"v{version:04d}.json"

    def _transaction_path(self, candidate_id: str) -> Path:
        safe = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in candidate_id
        )
        return self.transactions_dir / f"{safe}.json"

    def _next_sequence(self) -> int:
        current = 0
        if self.sequence_path.exists():
            try:
                current = int(self.sequence_path.read_text(encoding="utf-8").strip() or "0")
            except ValueError:
                current = max(
                    (int(item.get("sequence", 0)) for item in self.iter_events()),
                    default=0,
                )
        elif self.events_path.exists():
            current = max(
                (int(item.get("sequence", 0)) for item in self.iter_events()),
                default=0,
            )
        next_value = current + 1
        self._atomic_text(self.sequence_path, f"{next_value}\n")
        return next_value

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"missing runtime file: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        RuntimeStore._atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    @staticmethod
    def _atomic_text(path: Path, value: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
