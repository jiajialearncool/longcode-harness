"""Cancellation, bounded subprocess IO and project locks shared by both UIs."""
from __future__ import annotations

import fcntl
import hashlib
import os
import selectors
import signal
import subprocess
import threading
import time
import tempfile
from contextlib import contextmanager
from pathlib import Path


class Cancelled(RuntimeError):
    pass


class Cancellation:
    def __init__(self):
        self.event = threading.Event()

    def cancel(self):
        self.event.set()

    def check(self):
        if self.event.is_set():
            raise Cancelled("任务已停止；已执行的修改不会自动撤销，请检查后再继续")


def stop_process(process: subprocess.Popen):
    # A shell may exit before its children. Clean the whole group even then.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        if process.poll() is None:
            process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            # Some enclosing sandboxes deny signaling an already-empty group.
            if process.poll() is None:
                process.kill()
        process.wait()


def run_process(argv: list[str], *, cwd: Path, cancel: Cancellation, timeout: float,
                env: dict | None = None, stdin: str | None = None, emit=None,
                limit: int = 40000) -> tuple[int, str, str]:
    cancel.check()
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    chunks = {"stdout": "", "stderr": ""}
    emitted = {"stdout": 0, "stderr": 0}
    deadline = time.monotonic() + timeout
    # Write input on a separate thread: a large prompt cannot block cancellation.
    def feed():
        try:
            process.stdin.write(stdin.encode())
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    writer = threading.Thread(target=feed, daemon=True) if stdin is not None else None
    if writer:
        writer.start()
    try:
        with selectors.DefaultSelector() as selector:
            for name in chunks:
                pipe = getattr(process, name)
                selector.register(pipe, selectors.EVENT_READ, name)
            import codecs
            decoders = {name: codecs.getincrementaldecoder("utf-8")("replace") for name in chunks}
            while selector.get_map():
                cancel.check()
                if time.monotonic() >= deadline:
                    raise TimeoutError("操作超过时间上限")
                for key, _ in selector.select(.1):
                    raw = os.read(key.fileobj.fileno(), 8192)
                    if not raw:
                        selector.unregister(key.fileobj)
                        continue
                    value = decoders[key.data].decode(raw)
                    chunks[key.data] = (chunks[key.data] + value)[-limit:]
                    if emit and emitted[key.data] < limit:
                        visible = value[:limit - emitted[key.data]]
                        emitted[key.data] += len(visible)
                        emit("command_output", {"stream": key.data, "text": visible})
                        if emitted[key.data] == limit:
                            emit("command_output", {"stream": key.data, "text": "\n（实时输出达到上限；命令结果保留末尾内容）"})
            return process.wait(timeout=1), chunks["stdout"], chunks["stderr"]
    finally:
        stop_process(process)
        if writer:
            writer.join(timeout=3)
        for pipe in (process.stdin, process.stdout, process.stderr):
            if pipe and not pipe.closed:
                pipe.close()


@contextmanager
def project_lock(home: Path, workspace: Path):
    # Shared even when two clients use different LONGCODE_HOME directories.
    folder = Path(tempfile.gettempdir()) / f"longcode-project-locks-{os.getuid()}"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    if folder.is_symlink() or folder.stat().st_uid != os.getuid() or folder.stat().st_mode & 0o077:
        raise PermissionError("项目锁目录权限不安全")
    key = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()
    with (folder / (key + ".lock")).open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("这个项目已有任务正在修改文件，请等待或停止该任务")
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
