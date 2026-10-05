"""Cancellable JSON-lines connection to the pinned Pi model/auth worker."""
from __future__ import annotations

import fcntl
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from uuid import uuid4

from .agent_config import agent_home
from .agent_runtime import Cancellation, stop_process


class ModelBridge:
    def __init__(self, home: Path | None = None):
        self.home = home or agent_home()
        self.worker = Path(__file__).parent / "provider" / "worker.mjs"

    def request(self, op: str, *, cancel: Cancellation | None = None, emit=None,
                ask=None, timeout: int = 1800, **payload):
        cancel = cancel or Cancellation()
        node = shutil.which("node")
        if not node:
            raise RuntimeError("自主模式需要 Node.js >=22.19；请先安装 Node.js")
        if not (self.worker.parent / "node_modules" / "@earendil-works" / "pi-ai").is_dir():
            raise RuntimeError("模型组件尚未安装，请运行 longcode setup")
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Covers all workers/processes, not just threads. A cancelled waiter never blocks UI.
        deadline = time.monotonic() + timeout
        with (self.home / "auth.lock").open("a+") as lock:
            while True:
                cancel.check()
                if time.monotonic() > deadline:
                    raise TimeoutError("等待模型连接超时")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    cancel.event.wait(.1)
            try:
                return self._request(node, op, payload, cancel, emit, ask, deadline)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _request(self, node, op, payload, cancel, emit, ask, deadline):
        env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "TMPDIR", "LANG", "SYSTEMROOT"}}
        process = subprocess.Popen([node, str(self.worker)], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, bufsize=1, env=env, start_new_session=True)
        events = queue.Queue(maxsize=512)
        errors = []
        send_lock = threading.Lock()
        def send(value):
            with send_lock:
                if process.poll() is None:
                    process.stdin.write(json.dumps(value, ensure_ascii=False) + "\n")
                    process.stdin.flush()
        def read():
            try:
                for line in process.stdout:
                    if len(line) > 16_000_000:
                        continue
                    while process.poll() is None:
                        try:
                            events.put(line, timeout=.1)
                            break
                        except queue.Full:
                            if cancel.event.is_set():
                                return
                    else:
                        events.put_nowait(line)
            finally:
                try:
                    events.put(None, timeout=.1)
                except queue.Full:
                    pass
        def read_errors():
            for line in process.stderr:
                errors.append(line[-2000:])
                del errors[:-4]
        reader = threading.Thread(target=read, daemon=True)
        err_reader = threading.Thread(target=read_errors, daemon=True)
        reader.start(); err_reader.start()
        try:
            send({"id": uuid4().hex, "op": op, "home": str(self.home), **payload})
            while True:
                cancel.check()
                if time.monotonic() > deadline:
                    raise TimeoutError("模型调用或登录超过时间上限")
                try:
                    line = events.get(timeout=.1)
                except queue.Empty:
                    continue
                if line is None:
                    raise RuntimeError("模型组件意外退出；请运行 longcode setup 检查依赖")
                event = json.loads(line)
                kind = event.pop("type")
                event.pop("id", None)
                if kind == "result":
                    return event["value"]
                if kind == "error":
                    raise RuntimeError(event["message"])
                if kind == "auth_prompt":
                    if not ask:
                        raise RuntimeError("登录需要交互，请从终端或网页的登录入口操作")
                    # Callback races can cancel manual-code prompts while login continues.
                    def answer(item):
                        try:
                            value = ask(item["prompt"], item["prompt_id"])
                            send({"op": "reply", "prompt_id": item["prompt_id"], "value": value})
                        except (RuntimeError, BrokenPipeError, OSError):
                            pass
                    threading.Thread(target=answer, args=(event,), daemon=True).start()
                elif emit:
                    emit(kind, event)
        finally:
            stop_process(process)
            reader.join(timeout=2); err_reader.join(timeout=2)
            for pipe in (process.stdin, process.stdout, process.stderr):
                pipe.close()
