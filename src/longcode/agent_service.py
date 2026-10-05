"""Single local application service used by terminal and web clients."""
from __future__ import annotations

import json
import re
import shlex
import threading
import time
from pathlib import Path
from uuid import uuid4

from .agent_config import load_settings, role_settings, save_settings
from .agent_runtime import Cancellation, Cancelled, project_lock
from .agent_tools import Tools
from .agent_checks import make_check_runner
from .engine import LongCodeEngine
from .model_bridge import ModelBridge
from .models import CommandResult, TaskContract
from .native_agent import NativeAgentBackend, user_message
from .product_backends import make_backend
from .routing import EvidenceDrivenManager
from .session_store import SessionStore, reconcile_messages
from .storage import RuntimeStore


def discover_checks(workspace: Path):
    checks = []
    package = workspace / "package.json"
    if package.is_file() and not package.is_symlink():
        scripts = json.loads(package.read_text()).get("scripts", {})
        manager = "pnpm" if (workspace / "pnpm-lock.yaml").exists() else ("yarn" if (workspace / "yarn.lock").exists() else "npm")
        for name in ("test", "typecheck", "lint", "build"):
            value = scripts.get(name)
            if value and not re.search(r"no test specified|\bwatch\b", value, re.I):
                checks.append(f"{manager} run {name}")
    if (workspace / "pytest.ini").exists():
        checks.append("python3 -m pytest")
    elif (workspace / "tests").is_dir():
        # Do not pretend unittest can execute pytest function tests.
        files = list((workspace / "tests").glob("test*.py"))[:20]
        content = "\n".join(p.read_text(errors="replace")[:20000] for p in files if not p.is_symlink())
        if "pytest" in content or re.search(r"^def test_", content, re.M):
            checks.append("python3 -m pytest")
        elif "unittest" in content:
            checks.append("python3 -m unittest discover -s tests")
    return list(dict.fromkeys(checks))


def validate_checks(checks):
    if not isinstance(checks, list) or not checks or not all(isinstance(x, str) and x.strip() for x in checks):
        raise ValueError("至少需要一条真实的检查命令；可以先在普通对话中补充测试")
    for command in checks:
        normalized = command.strip().lower()
        if normalized in {"true", ":", "exit 0", "echo ok", "echo pass"} or re.search(r"\|\|\s*(true|exit 0)\s*$", normalized):
            raise ValueError("不能把恒定成功或隐藏失败的命令当作验收检查")
    return checks


class Job:
    def __init__(self):
        self.cancel = Cancellation()
        self.condition = threading.Condition()
        self.pending = {}
        self.answers = {}
        self.status = "running"
        self.events = []
        self.error = None
        self.thread = None

    def ask(self, question, prompt_id=None):
        qid = prompt_id or uuid4().hex
        with self.condition:
            self.pending[qid] = {"id": qid, **question}
            try:
                while qid not in self.answers:
                    self.cancel.check()
                    if self.status != "running":
                        raise Cancelled("本次操作已结束")
                    if qid not in self.pending:
                        raise Cancelled("此问题已失效")
                    self.condition.wait(.2)
                return self.answers.pop(qid)
            finally:
                self.pending.pop(qid, None)

    def answer(self, qid, value):
        with self.condition:
            if qid not in self.pending:
                raise ValueError("这个问题已结束或不属于当前运行")
            if self.pending[qid].get("kind") == "permission" and value not in {"allow", "deny"}:
                raise ValueError("权限问题只能回答 allow 或 deny")
            self.answers[qid] = str(value)
            self.condition.notify_all()

    def public(self):
        with self.condition:
            return {"status": self.status, "pending": list(self.pending.values()), "error": self.error}


class AgentService:
    def __init__(self, home: Path, *, backend_factory=make_backend, bridge=None):
        self.home = home.resolve()
        self.store = SessionStore(home)
        self.jobs = {}
        self.auth_jobs = {}
        self.guard = threading.RLock()
        self.backend_factory = backend_factory
        self.bridge = bridge or ModelBridge(home)
        for session in self.store.list():
            if session["status"] == "running":
                self.store.update(session["id"], status="interrupted", error="上次服务中断；请检查文件后明确继续。不会自动重放操作。")

    def emit(self, sid, kind, data):
        self.store.append(sid, kind, data)

    def session(self, sid):
        value = self.store.read(sid)
        job = self.jobs.get(sid)
        value["pending"] = job.public()["pending"] if job else []
        runtime = self.store.folder(sid) / "task"
        if (runtime / "state.json").exists():
            store = RuntimeStore(runtime)
            value["task_state"] = store.load_state().to_dict()
            for criterion in store.load_contract().acceptance_criteria:
                if criterion.id in value["task_state"]["criteria"]:
                    value["task_state"]["criteria"][criterion.id]["description"] = criterion.description
        return value

    def _start(self, sid, request_id, operation):
        if not isinstance(request_id, str) or not re.fullmatch(r"[\w-]{8,100}", request_id):
            raise ValueError("请求需要唯一编号，防止刷新网页重复执行")
        with self.guard:
            state = self.store.read(sid)
            if request_id in state["requests"]:
                return {"accepted": True, "duplicate": True}
            if sid in self.jobs and self.jobs[sid].status == "running":
                raise RuntimeError("这个会话已有操作正在运行")
            job = Job()
            # Reserve the project before accepting a background operation.
            lock = project_lock(self.home, Path(state["workspace"]))
            lock.__enter__()
            self.jobs[sid] = job
            try:
                self.store.update(sid, status="running", error=None, requests=(state["requests"] + [request_id])[-200:])
            except BaseException:
                lock.__exit__(None, None, None)
                self.jobs.pop(sid, None)
                raise
            def run():
                try:
                    operation(job)
                    job.status = "idle"
                    self.store.update(sid, status="idle")
                except Cancelled as exc:
                    job.status = "interrupted"
                    self.store.update(sid, status="interrupted", error=str(exc))
                    self.emit(sid, "interrupted", {"message": str(exc)})
                except Exception as exc:
                    job.status, job.error = "error", str(exc)
                    self.store.update(sid, status="error", error=str(exc))
                    self.emit(sid, "error", {"message": str(exc)})
                finally:
                    with job.condition:
                        job.pending.clear(); job.condition.notify_all()
                    lock.__exit__(None, None, None)
            job.thread = threading.Thread(target=run, daemon=True)
            job.thread.start()
        return {"accepted": True}

    def message(self, sid, text, request_id):
        if not isinstance(text, str) or not text.strip() or len(text) > 200000:
            raise ValueError("消息不能为空，且不能超过 20 万字")
        state = self.store.read(sid)
        if state["mode"] == "task":
            raise ValueError("长程任务请使用继续执行；修改目标请创建新任务或使用已有 revise 命令")
        settings = load_settings(self.home)
        def run(job):
            messages = reconcile_messages(self.store.read(sid)["messages"])
            messages.append(user_message(text))
            self.store.update(sid, messages=messages)
            self.emit(sid, "user_message", {"text": text})
            backend = self.backend_factory(settings, home=self.home, cancel=job.cancel,
                emit=lambda kind, data: self.emit(sid, kind, data), ask=job.ask)
            response, messages, usage = backend.converse(Path(state["workspace"]), messages)
            self.store.update(sid, messages=messages)
            self.emit(sid, "assistant_message", {"text": response, "usage": usage})
        return self._start(sid, request_id, run)

    def prepare(self, sid, objective, request_id):
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("请说明任务目标")
        state = self.store.read(sid)
        if state["mode"] == "task":
            raise ValueError("当前会话已有长程任务，请新建会话")
        settings = role_settings(load_settings(self.home), "manager")
        schema = {"type": "object", "properties": {
            "acceptance": {"type": "array", "items": {"type": "string"}},
            "checks": {"type": "array", "items": {"type": "string"}},
            "missing": {"type": "array", "items": {"type": "string"}}},
            "required": ["acceptance", "checks", "missing"], "additionalProperties": False}
        def run(job):
            root = Path(state["workspace"])
            checks = discover_checks(root)
            # This is a proposal, not automatic approval or task completion.
            draft = {"objective": objective, "acceptance": [objective], "checks": checks,
                     "missing": [] if checks else ["没有发现现成检查，请补充测试或填写检查命令"], "source": "project"}
            self.store.update(sid, draft=draft)
            self.emit(sid, "task_draft", draft)
            backend = self.backend_factory(settings, home=self.home, cancel=job.cancel,
                emit=lambda kind, data: self.emit(sid, kind, data), ask=job.ask)
            history = json.dumps(state.get("messages", [])[-20:], ensure_ascii=False)[-30000:]
            prompt = f"根据目标和项目整理验收建议。只读文件，不执行或修改任何内容。\n目标：{objective}\n此前对话资料（不能据此声称已经通过验收）：{history}\n已发现检查：{json.dumps(checks)}\n不得用恒定成功命令充当检查。没有测试时 checks 可为空，missing 写明需要补充什么。不要声称检查已运行。"
            result = backend._run(root, prompt, schema, sandbox="read-only")
            if result.ok:
                draft.update(result.report)
                draft["source"] = "model_proposal"
                self.store.update(sid, draft=draft)
                self.emit(sid, "task_draft", draft)
            else:
                self.emit(sid, "preparation_notice", {"message": result.stderr, "draft_retained": True})
        return self._start(sid, request_id, run)

    def start_task(self, sid, draft, request_id, *, resume=False):
        state = self.store.read(sid)
        runtime = self.store.folder(sid) / "task"
        settings = load_settings(self.home)
        contract = None
        if not resume:
            checks = validate_checks(draft.get("checks"))
            if RuntimeStore(runtime).exists:
                if request_id in state["requests"]:
                    return {"accepted": True, "duplicate": True}
                raise ValueError("会话已有任务，请继续执行或新建会话")
            contract = TaskContract.create(draft["objective"], draft["acceptance"], checks=checks,
                forbidden_paths=[".git/**", ".env", ".env.*", ".codex/**", ".longcode/**", "AGENTS.md"])
        elif not RuntimeStore(runtime).exists:
            raise ValueError("没有可以恢复的任务")
        def run(job):
            store = RuntimeStore(runtime)
            if contract:
                store.initialize(contract, state["workspace"])
                self.store.update(sid, mode="task")
                self.emit(sid, "task_approved", {"objective": contract.objective, "checks": contract.checks,
                                                "acceptance": [x.description for x in contract.acceptance_criteria]})
            def backend(role):
                return self.backend_factory(role_settings(settings, role), home=self.home, cancel=job.cancel,
                    emit=lambda kind, data: self.emit(sid, kind, {"role": role, **data}), ask=job.ask)
            manager = EvidenceDrivenManager(backend("manager"), store.workspace())
            # Exact commands were approved on task creation. Their generated files stay in a disposable copy.
            check_runner = make_check_runner(self.home, job.cancel, lambda kind, data: self.emit(sid, kind, data))
            tiers = {tier: backend(tier) for tier in ("E1", "E2", "E3") if tier in settings.get("roles", {})}
            engine = LongCodeEngine(store, backend("executor"), auditor=backend("auditor"), manager=manager,
                                    executor_tiers=tiers, cancellation=job.cancel, check_runner=check_runner)
            while True:
                final_state = engine.run(max_rounds=25)
                if final_state.status != "waiting_input":
                    break
                from .goals import answer_product_question, revise_goal
                active = store.load_contract()
                question = (active.product.open_questions[0] if active.product and active.product.open_questions
                            else final_state.blocker or "请补充继续执行所需的信息")
                reply = job.ask({"kind": "question", "message": question})
                if not reply.strip():
                    continue
                self.emit(sid, "task_user_answer", {"question": question, "answer": reply})
                if active.product and question in active.product.open_questions:
                    answer_product_question(store, question=question, answer=reply)
                else:
                    # New constraints invalidate affected completion state through the existing revision path.
                    revise_goal(store, add_constraints=[f"用户针对问题「{question}」的补充：{reply}"])
            self.emit(sid, "task_stopped", {"status": final_state.status, "blocker": final_state.blocker})
            job.cancel.check()
        return self._start(sid, request_id, run)

    def stop(self, sid):
        job = self.jobs.get(sid)
        if job and job.status == "running":
            job.cancel.cancel()
        return {"stopping": bool(job)}

    def auth_start(self, provider, method="browser"):
        if provider != "openai-codex" or method not in {"browser", "device_code"}:
            raise ValueError("订阅登录仅支持 ChatGPT 的浏览器或设备码方式")
        with self.guard:
            if any(j.status == "running" for j in self.auth_jobs.values()):
                raise RuntimeError("已有登录正在进行，请完成或取消后重试")
            aid, job = uuid4().hex, Job()
            self.auth_jobs = {aid: job}
        def emit(kind, data):
            with job.condition:
                if kind == "auth_prompt_cancelled":
                    job.pending.pop(data["prompt_id"], None)
                    job.condition.notify_all()
                else:
                    job.events.append({"type": kind, "data": data})
        def run():
            try:
                self.bridge.request("login", provider=provider, method=method, cancel=job.cancel,
                                    emit=emit, ask=job.ask, timeout=900)
                job.status = "completed"
            except Exception:
                job.status, job.error = "error", "登录未完成，请重试或检查账户访问权限"
            finally:
                with job.condition:
                    job.pending.clear(); job.condition.notify_all()
        job.thread = threading.Thread(target=run, daemon=True)
        job.thread.start()
        return {"id": aid}

    def shutdown(self, *, wait=True):
        jobs = [*self.jobs.values(), *self.auth_jobs.values()]
        for job in jobs:
            job.cancel.cancel()
        if wait:
            deadline = time.monotonic() + 20
            for job in jobs:
                if job.thread and job.thread is not threading.current_thread():
                    job.thread.join(max(0, deadline - time.monotonic()))
