"""Private durable conversations and append-only traces (not task acceptance state)."""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from uuid import uuid4

from .agent_config import atomic_json
from .models import utc_now


def redact(value):
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if k.lower() in {"api_key", "apikey", "access_token", "refresh_token", "authorization", "password", "key"} else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"\b(?:sk-[A-Za-z0-9_-]{8,}|eyJ[A-Za-z0-9_.-]{30,})", "[REDACTED]", value)
    return value


class SessionStore:
    def __init__(self, home: Path):
        self.home = home.resolve()
        self.root = self.home / "sessions"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.RLock()
        self.sequences = {}

    def folder(self, session_id):
        if not re.fullmatch(r"[a-f0-9]{32}", session_id):
            raise ValueError("无效的会话编号")
        return self.root / session_id

    def create(self, workspace: str, *, mode="chat"):
        root = Path(workspace).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("请选择已存在的项目目录")
        if root == Path.home() or root == Path(root.anchor) or self.home.is_relative_to(root) or root.is_relative_to(self.home):
            raise ValueError("请选择具体项目目录，不能使用主目录、磁盘根目录或 LongCode 内部目录")
        sid = uuid4().hex
        data = {"id": sid, "workspace": str(root), "mode": mode, "title": root.name,
                "status": "idle", "created_at": utc_now(), "updated_at": utc_now(),
                "messages": [], "requests": [], "sequence": 0, "error": None}
        self.save(data)
        return data

    def read(self, session_id):
        with self.lock:
            return json.loads((self.folder(session_id) / "session.json").read_text())

    def save(self, data):
        with self.lock:
            data = {**data, "updated_at": utc_now()}
            atomic_json(self.folder(data["id"]) / "session.json", redact(data))

    def update(self, session_id, **changes):
        with self.lock:
            data = self.read(session_id)
            data.update(changes)
            self.save(data)
            return data

    def list(self):
        result = []
        for path in self.root.glob("*/session.json"):
            try:
                data = self.read(path.parent.name)
                result.append({k: v for k, v in data.items() if k not in {"messages", "requests"}})
            except (ValueError, OSError):
                continue
        return sorted(result, key=lambda x: x["updated_at"], reverse=True)

    def append(self, session_id, kind, data):
        if kind == "conversation_checkpoint":
            self.update(session_id, messages=data["messages"])
            return
        with self.lock:
            state = self.read(session_id)
            # The event file is authoritative if a crash occurs between append and state save.
            if session_id not in self.sequences:
                self.sequences[session_id] = max(state["sequence"], self.last_sequence(session_id))
            sequence = self.sequences[session_id] + 1
            event = {"sequence": sequence, "timestamp": utc_now(), "type": kind, "data": redact(data)}
            path = self.folder(session_id) / "trace.jsonl"
            import os
            if path.exists():
                with path.open("rb") as previous:
                    previous.seek(0, 2)
                    size = previous.tell()
                    if size:
                        previous.seek(-1, 2)
                        needs_newline = previous.read(1) != b"\n"
                    else:
                        needs_newline = False
            else:
                needs_newline = False
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as output:
                if needs_newline:
                    output.write("\n")
                output.write(json.dumps(event, ensure_ascii=False) + "\n")
                output.flush()
                if kind != "text_delta":
                    os.fsync(output.fileno())
            self.sequences[session_id] = sequence
            state["sequence"] = sequence
            self.save(state)

    def last_sequence(self, session_id):
        path = self.folder(session_id) / "trace.jsonl"
        if not path.exists():
            return 0
        # Only inspect the final record, not the entire model trace on every token.
        with path.open("rb") as stream:
            stream.seek(0, 2)
            end = stream.tell()
            size = min(end, 16_000_000)
            stream.seek(end - size)
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                return int(json.loads(line)["sequence"])
            except (ValueError, KeyError):
                continue
        return 0

    def events(self, session_id, *, after=0, limit=500):
        path = self.folder(session_id) / "trace.jsonl"
        if not path.exists():
            return []
        items = []
        with path.open() as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except ValueError:
                    continue  # an interrupted final append is not a trusted event
                if item["sequence"] > after:
                    items.append(item)
                if len(items) >= limit:
                    break
        return items

    def export(self, session_id, format="md"):
        path = self.folder(session_id) / "trace.jsonl"
        lines = path.read_text().splitlines() if path.exists() else []
        valid = []
        for line in lines:
            try:
                valid.append(json.loads(line))
            except ValueError:
                continue
        if format == "jsonl":
            return "\n".join(json.dumps(x, ensure_ascii=False) for x in valid) + "\n"
        state = self.read(session_id)
        blocks = [f"# LongCode 运行记录\n\n项目：{state['workspace']}\n"]
        for event in valid:
            if event["type"] == "text_delta":
                continue
            value = json.dumps(event["data"], ensure_ascii=False, indent=2)
            fence = "`" * max(3, max((len(x) for x in re.findall(r"`+", value)), default=0) + 1)
            blocks.append(f"## {event['sequence']} · {event['type']}\n\n{event['timestamp']}\n\n{fence}json\n{value}\n{fence}\n")
        return "\n".join(blocks)


def reconcile_messages(messages):
    """Missing result = unknown execution outcome, never automatic tool replay."""
    result = []
    pending = {}
    for message in messages:
        if message.get("role") in {"user", "assistant"} and pending:
            result.extend(interrupted_result(call) for call in pending.values())
            pending = {}
        result.append(message)
        if message.get("role") == "assistant":
            pending = {b["id"]: b for b in message.get("content", []) if b.get("type") == "toolCall"}
        elif message.get("role") == "toolResult":
            pending.pop(message["toolCallId"], None)
    result.extend(interrupted_result(call) for call in pending.values())
    return result


def interrupted_result(call):
    return {"role": "toolResult", "toolCallId": call["id"], "toolName": call["name"],
            "content": [{"type": "text", "text": "操作期间中断，结果未知。先检查实际文件和环境，不得直接重放命令。"}],
            "isError": True, "timestamp": 0}
