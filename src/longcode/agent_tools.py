"""Concrete tools with host-enforced role and workspace boundaries."""
from __future__ import annotations

import difflib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from .agent_runtime import Cancellation, run_process

PROTECTED = {".git", ".longcode", ".codex", ".ssh", ".aws", ".agents", "node_modules", "__pycache__", ".venv"}


def hidden(path: Path) -> bool:
    return any(p in PROTECTED or p == ".env" or p.startswith(".env.") for p in path.parts)


def schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required, "additionalProperties": False,
    }}


STRING = {"type": "string"}
TOOLS = [
    schema("read_file", "读取项目中的文字文件。大文件用 offset 和 limit 分段读取。", {"path": STRING, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["path"]),
    schema("list_files", "列出项目文件，不读取凭据、依赖库和版本库内部文件。", {"path": STRING}, []),
    schema("search", "按文字搜索项目文件，返回文件名、行号和匹配内容。", {"text": STRING, "path": STRING}, ["text"]),
    schema("write_file", "创建或整体替换项目文件；优先用 edit_file 做局部修改。", {"path": STRING, "content": STRING}, ["path", "content"]),
    schema("edit_file", "把文件内唯一匹配的 old 文本替换为 new；没有唯一匹配则拒绝修改。", {"path": STRING, "old": STRING, "new": STRING}, ["path", "old", "new"]),
    schema("shell", "在项目中运行命令（包括测试）。需用户批准；默认无网络且限制文件读写。", {"command": STRING, "network": {"type": "boolean"}}, ["command"]),
    schema("ask_user", "缺少用户选择或必要资料时提问。", {"question": STRING}, ["question"]),
]


class Tools:
    def __init__(self, workspace: Path, *, role="executor", cancel=None, emit=None,
                 ask=None, home: Path | None = None, settings=None):
        self.root = workspace.resolve()
        self.role = role
        self.cancel = cancel or Cancellation()
        self.emit = emit or (lambda *_: None)
        self.ask = ask
        self.home = home.resolve() if home else None
        self.settings = settings or {}
        self.changed = []
        self.tests = []
        self.mcp_tools = {}
        self.loaded_instructions = set()

    def definitions(self):
        allowed = {"read_file", "list_files", "search", "ask_user"}
        if self.role in {"executor", "chat"}:
            allowed.update({"write_file", "edit_file", "shell"})
        return [t for t in TOOLS if t["name"] in allowed] + list(self.mcp_tools.values())

    def path(self, value: str, *, write=False):
        p = self.root / value
        if p.is_absolute() and not p.is_relative_to(self.root):
            raise PermissionError("文件必须位于当前项目内")
        relative = p.relative_to(self.root)
        if ".." in relative.parts or hidden(relative):
            raise PermissionError("不允许访问凭据、内部记录、依赖目录或项目外文件")
        for parent in [p, *p.parents]:
            if parent == self.root:
                break
            if parent.is_symlink():
                raise PermissionError("工具不跟随符号链接，请使用项目内真实文件")
        if not p.resolve().is_relative_to(self.root):
            raise PermissionError("文件路径超出项目范围")
        if self.home and not self.root.is_relative_to(self.home) and (p.resolve() == self.home or p.resolve().is_relative_to(self.home)):
            raise PermissionError("不能访问 LongCode 的登录凭据和内部记录")
        if write and p.name in {"AGENTS.md", "SKILL.md"}:
            raise PermissionError("运行期间不修改 Agent 指令文件")
        return p

    def _files(self, root):
        count = 0
        for folder, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not hidden(Path(d)) and not (Path(folder) / d).is_symlink())
            for name in sorted(files):
                file = Path(folder) / name
                if hidden(file.relative_to(self.root)) or file.is_symlink():
                    continue
                yield file
                count += 1
                if count >= 3000:
                    return

    def instructions_for(self, file: Path):
        """Nested instructions are returned before a file's contents or edit."""
        found = []
        for folder in reversed([file.parent, *file.parent.parents]):
            if not folder.is_relative_to(self.root):
                continue
            instruction = folder / "AGENTS.md"
            if instruction.is_file() and not instruction.is_symlink() and instruction not in self.loaded_instructions:
                self.loaded_instructions.add(instruction)
                found.append(f"{instruction.relative_to(self.root)}:\n{instruction.read_text()}")
        return "\n".join(found)

    def call(self, name: str, args: dict):
        self.cancel.check()
        from .backends import _matches_schema
        definition = next((t for t in self.definitions() if t["name"] == name), None)
        if not definition or not _matches_schema(args, definition["parameters"]):
            raise PermissionError("当前岗位不允许此工具，或工具参数不符合格式")
        if name in self.mcp_tools:
            from .mcp_client import call_mcp_tool
            return call_mcp_tool(self, name, args)
        if name == "ask_user":
            if not self.ask:
                raise RuntimeError("需要用户补充信息：" + args["question"])
            return self.ask({"kind": "question", "message": args["question"]})
        if name == "shell":
            return self.command(args["command"], network=args.get("network", False))
        target = self.path(args.get("path", "."), write=name in {"write_file", "edit_file"})
        if name == "list_files":
            return "\n".join(str(f.relative_to(self.root)) for f in self._files(target))[:40000]
        if name == "search":
            matches = []
            for file in self._files(target):
                self.cancel.check()
                if file.stat().st_size > 1_000_000:
                    continue
                for i, line in enumerate(file.read_text(errors="replace").splitlines(), 1):
                    if args["text"] in line:
                        matches.append(f"{file.relative_to(self.root)}:{i}: {line[:1000]}")
                        if len(matches) >= 100:
                            return "\n".join(matches) + "\n（只显示前 100 条）"
            return "\n".join(matches) or "没有匹配内容"
        if target.exists() and target.stat().st_size > 2_000_000:
            raise ValueError("文件超过 2 MB，当前文字工具不处理该文件")
        instructions = self.instructions_for(target)
        before = target.read_text() if target.exists() else ""
        if name == "read_file":
            if not target.is_file():
                raise FileNotFoundError(args["path"])
            start = max(0, args.get("offset", 1) - 1)
            end = start + min(500, max(1, args.get("limit", 200)))
            return instructions + "\n" + "\n".join(f"{i + 1}: {line}" for i, line in enumerate(before.splitlines()) if start <= i < end)[:40000]
        # Do not let an edit precede discovery of a nested AGENTS.md.
        if instructions:
            return "尚未修改文件。先阅读以下目录指令，再重新提交修改：\n" + instructions
        if name == "edit_file":
            if not args["old"] or before.count(args["old"]) != 1:
                raise ValueError("old 必须在文件中恰好出现一次，请重新读取文件后修改")
            after = before.replace(args["old"], args["new"], 1)
        else:
            after = args["content"]
        if len(after.encode()) > 2_000_000:
            raise ValueError("单次文件修改不能超过 2 MB")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(after)
        relative = str(target.relative_to(self.root))
        self.changed.append(relative)
        diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=relative, tofile=relative))
        self.emit("file_changed", {"path": relative, "diff": diff[:60000]})
        return f"已修改 {relative}"

    def command(self, command: str, *, network=False, approved=False, timeout=300):
        if self.role not in {"executor", "chat", "verifier"}:
            raise PermissionError("这个岗位不能执行任意命令")
        if not approved:
            if not self.ask or self.ask({"kind": "permission", "message": "允许在项目内运行这条命令吗？", "command": command, "network": network}) != "allow":
                raise PermissionError("用户未批准执行命令")
        if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").exists():
            raise RuntimeError("当前命令隔离仅支持 macOS，未启用不受隔离的降级执行")
        with tempfile.TemporaryDirectory(prefix="longcode-tool-") as temporary:
            temp = Path(temporary).resolve()
            profile = sandbox_profile(self.root, temp, home=self.home, network=network)
            env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(temp),
                   "TMPDIR": str(temp), "LANG": "en_US.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
                   "PYTHONUNBUFFERED": "1", "CI": "1"}
            code, stdout, stderr = run_process(["/usr/bin/sandbox-exec", "-p", profile, "/bin/sh", "-c", command],
                cwd=self.root, cancel=self.cancel, timeout=timeout, env=env, emit=self.emit)
        self.tests.append(command)
        return {"exit_code": code, "stdout": stdout, "stderr": stderr}


def sandbox_profile(root: Path, temporary: Path, *, home=None, network=False):
    quote = lambda p: json.dumps(str(p))
    read_roots = ["/usr", "/bin", "/sbin", "/System", "/Library", "/dev", "/private/etc", "/private/var/db/dyld", "/private/preboot", str(root), str(temporary)]
    profile = '(version 1) (deny default) (allow process*) (allow sysctl-read) (allow mach-lookup) (allow file-map-executable)\n'
    profile += '(allow file-read-metadata)\n'
    profile += '(allow file-read* (literal "/"))\n'
    profile += '(allow file-read* ' + ' '.join(f'(subpath {quote(p)})' for p in read_roots) + ')\n'
    profile += f'(allow file-write* (subpath {quote(root)}) (subpath {quote(temporary)}) (literal "/dev/null"))\n'
    for name in (".git", ".longcode", ".codex", ".agents", "AGENTS.md"):
        profile += f'(deny file-write* (subpath {quote(root / name)}))\n'
    profile += '(deny file-read* file-write* (regex #"(^|/)\\.env($|[./])"))\n'
    if home:
        profile += f'(deny file-read* file-write* (require-all (subpath {quote(home)}) (require-not (subpath {quote(root)}))))\n'
    if network:
        profile += '(allow network-outbound)\n'
    return profile


def load_instructions(workspace: Path, settings: dict, emit) -> str:
    sections = []
    # Project instructions only; do not ingest arbitrary home files.
    path = workspace / "AGENTS.md"
    if path.is_file() and not path.is_symlink():
        sections.append("项目指令：\n" + path.read_text())
        emit("instructions_loaded", {"path": str(path)})
    for value in settings.get("skills", []):
        skill = Path(value).expanduser().resolve()
        if skill.name != "SKILL.md" or not skill.is_file() or skill.stat().st_size > 200000:
            raise ValueError("Skill 必须指向存在且小于 200 KB 的 SKILL.md")
        sections.append(f"用户配置的 Skill（{skill}）：\n{skill.read_text()}")
        emit("skill_loaded", {"path": str(skill)})
    return "\n\n".join(sections)
