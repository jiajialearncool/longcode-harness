"""Interactive client of the same local service used by the browser."""
from __future__ import annotations

import getpass
import json
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from uuid import uuid4

from .agent_config import agent_home
from .local_server import api, ensure_service


def add_commands(commands):
    chat = commands.add_parser("chat", help="与 LongCode 编程助手对话")
    chat.add_argument("--workspace", default=".")
    chat.add_argument("--session", help="继续已保存的会话")
    web = commands.add_parser("web", help="打开本机网页界面")
    web.add_argument("--no-open", action="store_true")
    commands.add_parser("setup", help="安装固定版本的自主执行模型组件")
    config = commands.add_parser("config", help="查看或设置日常使用的后端和模型")
    for name in ("backend", "provider", "model", "reasoning", "base-url"):
        config.add_argument("--" + name)
    login = commands.add_parser("login", help="登录模型服务或外部 CLI")
    login.add_argument("--provider", choices=["openai-codex", "openai", "anthropic", "compatible"], default="openai-codex")
    login.add_argument("--backend", choices=["codex", "claude"])
    login.add_argument("--method", choices=["browser", "device_code"], default="browser")
    logout = commands.add_parser("logout", help="移除 LongCode 保存的模型服务凭据")
    logout.add_argument("--provider", required=True, choices=["openai-codex", "openai", "anthropic", "compatible"])
    export = commands.add_parser("export-session", help="导出会话运行记录到标准输出")
    export.add_argument("session")
    export.add_argument("--format", choices=["md", "jsonl"], default="md")


COMMANDS = {"chat", "web", "setup", "config", "login", "logout", "export-session"}


def terminal_answer(question):
    print("\n" + question.get("message", "需要补充信息"))
    details = {k: v for k, v in question.items() if k not in {"id", "message", "kind", "type", "placeholder"}}
    if details:
        print(json.dumps(details, ensure_ascii=False, indent=2))
    if question.get("kind") == "permission":
        return "allow" if input("允许吗？输入 y 批准，其余拒绝：").strip().lower() == "y" else "deny"
    if question.get("type") in {"secret", "manual_code"}:
        return getpass.getpass("请输入（不显示）：")
    return input("> ")


def wait_session(server, sid):
    cursor = 0
    # Begin at the last persisted event, so reconnecting doesn't reprint all history.
    state = api(server, f"/api/sessions/{sid}")
    cursor = max(0, state.get("sequence", 0) - 2)
    while True:
        try:
            for event in api(server, f"/api/sessions/{sid}/events?after={cursor}"):
                cursor = event["sequence"]
                if event["type"] == "text_delta":
                    print(event["data"]["text"], end="", flush=True)
                elif event["type"] in {"tool_started", "check_finished", "task_stopped", "error"}:
                    print("\n" + json.dumps(event["data"], ensure_ascii=False)[:1000])
            state = api(server, f"/api/sessions/{sid}")
            for question in state.get("pending", []):
                value = terminal_answer(question)
                api(server, f"/api/sessions/{sid}/answer", {"id": question["id"], "value": value})
            if state["status"] != "running":
                print()
                if state.get("error"):
                    print(state["error"])
                return state
            time.sleep(.25)
        except KeyboardInterrupt:
            api(server, f"/api/sessions/{sid}/stop", {})
            print("\n正在停止；项目中的已有修改不会自动撤销。")


def dispatch(args):
    if args.command == "setup":
        npm = shutil.which("npm")
        if not npm:
            raise RuntimeError("请先安装 Node.js >=22.19.0（包含 npm）")
        folder = Path(__file__).parent / "provider"
        return subprocess.call([npm, "ci", "--ignore-scripts", "--no-audit", "--no-fund"], cwd=folder)
    if args.command == "login" and args.backend:
        cli = shutil.which(args.backend)
        if not cli:
            raise FileNotFoundError(f"未安装 {args.backend} CLI")
        return subprocess.call([cli, "login"] if args.backend == "codex" else [cli, "auth", "login"])
    server = ensure_service()
    if args.command == "web":
        url = server["url"] + "/#token=" + server["token"]
        print("本地访问链接（含访问令牌，请勿分享）：\n" + url)
        if not args.no_open:
            webbrowser.open(url)
        return 0
    if args.command == "config":
        settings = api(server, "/api/settings")
        changed = False
        for name in ("backend", "provider", "model", "reasoning", "base_url"):
            if getattr(args, name, None) is not None:
                settings[name] = getattr(args, name)
                changed = True
        if changed:
            api(server, "/api/settings", settings)
        print(json.dumps(settings, ensure_ascii=False, indent=2))
        return 0
    if args.command == "logout":
        api(server, "/api/auth/logout", {"provider": args.provider})
        print("已移除 LongCode 保存的凭据，不影响外部 CLI 的登录。")
        return 0
    if args.command == "login":
        if args.provider != "openai-codex":
            key = getpass.getpass("API key（不会显示或写入运行记录）：")
            api(server, "/api/auth/key", {"provider": args.provider, "key": key})
            print("已保存密钥。请设置对应的 provider 和 model。")
            return 0
        aid = api(server, "/api/auth/start", {"provider": args.provider, "method": args.method})["id"]
        seen = 0
        try:
            while True:
                job = api(server, f"/api/auth/{aid}")
                for event in job["events"][seen:]:
                    value = event.get("data", {}).get("event", {})
                    print(json.dumps(value, ensure_ascii=False))
                    if value.get("type") == "auth_url":
                        webbrowser.open(value["url"])
                seen = len(job["events"])
                for question in job["pending"]:
                    # A browser callback can complete without entering the fallback manual code.
                    if question.get("type") == "manual_code":
                        continue
                    api(server, f"/api/auth/{aid}/answer", {"id": question["id"], "value": terminal_answer(question)})
                if job["status"] != "running":
                    print("登录完成" if job["status"] == "completed" else job["error"])
                    return 0 if job["status"] == "completed" else 1
                time.sleep(.5)
        except KeyboardInterrupt:
            api(server, f"/api/auth/{aid}/cancel", {})
            return 130
    if args.command == "export-session":
        from .session_store import SessionStore
        print(SessionStore(agent_home()).export(args.session, args.format), end="")
        return 0
    if args.command == "chat":
        state = api(server, f"/api/sessions/{args.session}") if args.session else api(server, "/api/sessions", {"workspace": str(Path(args.workspace).resolve())})
        sid = state["id"]
        print(f"LongCode · {state['workspace']}\n会话：{sid}\n/task 启动长程任务；/resume 继续长程任务；/quit 退出。Ctrl-C 停止当前工作。")
        while True:
            try:
                text = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                print(); return 0
            if text == "/quit":
                return 0
            if not text:
                continue
            if text == "/resume":
                api(server, f"/api/sessions/{sid}/resume", {"request_id": uuid4().hex})
            elif text.startswith("/task"):
                objective = text[5:].strip() or input("希望完成什么？")
                api(server, f"/api/sessions/{sid}/prepare", {"objective": objective, "request_id": uuid4().hex})
                state = wait_session(server, sid)
                draft = state.get("draft")
                if not draft:
                    continue
                print(json.dumps(draft, ensure_ascii=False, indent=2))
                acceptance = input("验收要求（用分号分隔，回车保留建议）：").strip()
                checks = input("检查命令（用 ||| 分隔多条命令，回车保留建议）：").strip()
                if acceptance:
                    draft["acceptance"] = [v.strip() for v in acceptance.split(";") if v.strip()]
                if checks:
                    draft["checks"] = [v.strip() for v in checks.split("|||") if v.strip()]
                if input("确认按以上要求修改项目并运行这些检查？输入 y：").strip().lower() != "y":
                    continue
                api(server, f"/api/sessions/{sid}/task", {"draft": draft, "request_id": uuid4().hex})
            else:
                api(server, f"/api/sessions/{sid}/message", {"text": text, "request_id": uuid4().hex})
            wait_session(server, sid)
    return 0
