"""Loopback-only authenticated HTTP API; intentionally no remote hosting mode."""
from __future__ import annotations

import argparse
import fcntl
import hmac
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .agent_config import agent_home, atomic_json, load_settings, save_settings
from .agent_service import AgentService


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, home, port=0, *, service=None):
        self.token = secrets.token_urlsafe(32)
        self.app = service or AgentService(home)
        self.home = home
        super().__init__(("127.0.0.1", port), Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "LongCode/2"
    def log_message(self, *_):
        pass  # Request bodies, credentials and URLs never go into access logs.

    def reply(self, status, data, content_type="application/json; charset=utf-8", *, download=None):
        raw = (json.dumps(data, ensure_ascii=False) if content_type.startswith("application/json;") else data).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def allowed(self, *, auth=True):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            self.reply(403, {"error": "拒绝非本机主机名"}); return False
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://" + h for h in hosts}:
            self.reply(403, {"error": "拒绝其他网站发来的请求"}); return False
        if auth and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + self.server.token):
            self.reply(401, {"error": "请通过 longcode web 打开的链接访问，或输入本地访问令牌"}); return False
        return True

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if not self.allowed(auth=path.startswith("/api/")):
            return
        try:
            app = self.server.app
            query = urllib.parse.parse_qs(parsed.query)
            if path == "/api/health":
                self.reply(200, {"status": "ok", "version": "2", "pid": os.getpid()})
            elif path == "/api/settings":
                self.reply(200, load_settings(self.server.home))
            elif path == "/api/auth/status":
                self.reply(200, app.bridge.request("status", provider=query.get("provider", ["openai-codex"])[0], timeout=15))
            elif path.startswith("/api/auth/"):
                aid = path.rsplit("/", 1)[1]
                job = app.auth_jobs[aid]
                self.reply(200, {**job.public(), "events": job.events})
            elif path == "/api/sessions":
                self.reply(200, app.store.list())
            elif path.startswith("/api/sessions/"):
                parts = path.split("/")
                sid = parts[3]
                action = parts[4] if len(parts) > 4 else ""
                if not action:
                    self.reply(200, app.session(sid))
                elif action == "events":
                    self.reply(200, app.store.events(sid, after=max(0, int(query.get("after", [0])[0]))))
                elif action == "export":
                    fmt = query.get("format", ["md"])[0]
                    if fmt not in {"md", "jsonl"}:
                        raise ValueError("导出格式仅支持 md 或 jsonl")
                    self.reply(200, app.store.export(sid, fmt), "text/plain; charset=utf-8", download=f"longcode-{sid}.{fmt}")
                elif action == "task-events":
                    from .storage import RuntimeStore
                    runtime = RuntimeStore(app.store.folder(sid) / "task")
                    self.reply(200, runtime.iter_events()[-300:] if runtime.exists else [])
                else:
                    self.reply(404, {"error": "未知路径"})
            else:
                files = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"), "/style.css": ("style.css", "text/css")}
                if path not in files:
                    self.reply(404, {"error": "未知路径"}); return
                name, mime = files[path]
                self.reply(200, (Path(__file__).parent / "web" / name).read_text(), mime + "; charset=utf-8")
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            self.reply(400, {"error": str(exc)})

    def do_POST(self):
        if not self.allowed():
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2_000_000:
                raise ValueError("请求过大或为空")
            if "application/json" not in self.headers.get("Content-Type", ""):
                raise ValueError("请求需要 JSON 格式")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("请求需要 JSON 对象")
            app = self.server.app
            path = urllib.parse.urlparse(self.path).path
            if path == "/api/settings":
                result = save_settings(data, self.server.home)
            elif path == "/api/sessions":
                result = app.store.create(data["workspace"])
            elif path == "/api/auth/start":
                result = app.auth_start(data["provider"], data.get("method", "browser"))
            elif path in {"/api/auth/key", "/api/auth/logout"}:
                op = path.rsplit("/", 1)[1]
                result = app.bridge.request(op, provider=data["provider"], **({"key": data["key"]} if op == "key" else {}), timeout=30)
            elif path.startswith("/api/auth/"):
                _, _, _, aid, action = path.split("/")
                job = app.auth_jobs[aid]
                if action == "answer":
                    job.answer(data["id"], data["value"])
                elif action == "cancel":
                    job.cancel.cancel()
                else:
                    raise ValueError("未知操作")
                result = {"ok": True}
            elif path.startswith("/api/sessions/"):
                _, _, _, sid, action = path.split("/")
                if action == "message":
                    result = app.message(sid, data["text"], data["request_id"])
                elif action == "prepare":
                    result = app.prepare(sid, data["objective"], data["request_id"])
                elif action == "task":
                    result = app.start_task(sid, data["draft"], data["request_id"])
                elif action == "resume":
                    result = app.start_task(sid, {}, data["request_id"], resume=True)
                elif action == "stop":
                    result = app.stop(sid)
                elif action == "answer":
                    app.jobs[sid].answer(data["id"], data["value"])
                    result = {"ok": True}
                else:
                    raise ValueError("未知操作")
            else:
                self.reply(404, {"error": "未知路径"}); return
            self.reply(200, result)
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
            self.reply(400, {"error": str(exc)})


def api(descriptor, path, data=None):
    request = urllib.request.Request(descriptor["url"] + path,
        data=json.dumps(data, ensure_ascii=False).encode() if data is not None else None,
        headers={"Authorization": "Bearer " + descriptor["token"], "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=35) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(json.loads(exc.read()).get("error", str(exc))) from None


def ensure_service(home=None):
    home = home or agent_home()
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = home / "server.json"
    def running():
        if descriptor.exists():
            try:
                value = json.loads(descriptor.read_text())
                # Do not send local tokens to an edited arbitrary address.
                parsed = urllib.parse.urlparse(value["url"])
                if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
                    return None
                request = urllib.request.Request(value["url"] + "/api/health", headers={"Authorization": "Bearer " + value["token"]})
                with urllib.request.urlopen(request, timeout=.5) as response:
                    if json.loads(response.read())["pid"] == value["pid"]:
                        return value
            except (OSError, ValueError, KeyError):
                pass
        return None
    if value := running():
        return value
    log_fd = os.open(home / "server.log", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(log_fd, "a") as log:
        subprocess.Popen([sys.executable, "-m", "longcode.local_server", "--home", str(home)],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    for _ in range(60):
        if value := running():
            return value
        time.sleep(.1)
    raise RuntimeError(f"本地服务未能启动，请查看 {home / 'server.log'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, default=agent_home())
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    args.home.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.home, 0o700)
    with (args.home / "server.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        server = LocalServer(args.home, args.port)
        atomic_json(args.home / "server.json", {"url": f"http://127.0.0.1:{server.server_port}", "token": server.token, "pid": os.getpid()})
        def shutdown(*_):
            server.app.shutdown(wait=False)
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        try:
            server.serve_forever(poll_interval=.2)
        finally:
            server.app.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
