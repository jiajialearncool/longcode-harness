"""User-owned settings, separate from benchmark manifests and CLI credentials."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def agent_home() -> Path:
    return Path(os.environ.get("LONGCODE_HOME", "~/.longcode")).expanduser().resolve()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


DEFAULT_SETTINGS = {
    "backend": "native", "provider": "openai-codex", "model": "",
    "reasoning": "", "base_url": "", "context_window": 64000,
    "max_output_tokens": 8192, "max_turns": 40, "timeout": 1800,
    "roles": {}, "skills": [], "mcp": [],
}


def load_settings(home: Path | None = None) -> dict:
    path = (home or agent_home()) / "settings.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    return validate_settings({**DEFAULT_SETTINGS, **data})


def validate_settings(data: dict) -> dict:
    unknown = set(data) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(f"未知设置：{', '.join(sorted(unknown))}；密钥请通过登录入口保存")
    for name in ("backend", "provider", "model", "reasoning", "base_url"):
        if not isinstance(data.get(name, ""), str):
            raise ValueError(f"{name} 必须是文字")
    if data["backend"] not in {"native", "codex", "claude"}:
        raise ValueError("执行后端必须为 native、codex 或 claude")
    if data["provider"] not in {"openai-codex", "openai", "anthropic", "compatible"}:
        raise ValueError("请选择支持的模型服务")
    if data["base_url"]:
        from urllib.parse import urlparse
        url = urlparse(data["base_url"])
        if url.username or url.password or url.query or url.fragment or not url.hostname:
            raise ValueError("服务地址不能包含凭据、查询参数或片段")
        if url.scheme != "https" and not (url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}):
            raise ValueError("非本机服务必须使用 HTTPS")
    for name, lo, hi in (("max_turns", 1, 500), ("timeout", 1, 86400),
                         ("context_window", 4096, 2000000), ("max_output_tokens", 256, 64000)):
        if not isinstance(data.get(name), int) or not lo <= data[name] <= hi:
            raise ValueError(f"{name} 必须在 {lo}～{hi} 之间")
    if data["max_output_tokens"] >= data["context_window"] // 2:
        raise ValueError("输出上限必须小于上下文容量的一半")
    if data["reasoning"] not in {"", "minimal", "low", "medium", "high", "xhigh"}:
        raise ValueError("不支持的推理强度")
    if not isinstance(data["roles"], dict) or set(data["roles"]) - {"executor", "manager", "auditor", "E1", "E2", "E3"}:
        raise ValueError("岗位设置仅支持 executor、manager、auditor 和 E1/E2/E3")
    role_fields = {"backend", "provider", "model", "reasoning", "base_url"}
    for override in data["roles"].values():
        if not isinstance(override, dict) or set(override) - role_fields:
            raise ValueError("岗位只能设置后端、模型服务、模型、推理强度和服务地址")
        validate_settings({**data, **override, "roles": {}})
    if not isinstance(data["skills"], list) or not all(isinstance(x, str) for x in data["skills"]):
        raise ValueError("skills 必须为 SKILL.md 路径列表")
    if not isinstance(data["mcp"], list):
        raise ValueError("mcp 必须为服务配置列表")
    for server in data["mcp"]:
        if not isinstance(server, dict) or set(server) - {"name", "command", "url", "network", "env_names"}:
            raise ValueError("MCP 配置仅支持 name、command、url、network、env_names")
        if bool(server.get("command")) == bool(server.get("url")):
            raise ValueError("MCP 必须选择 command 或 url 中的一种")
        if "command" in server and (not isinstance(server["command"], list) or not all(isinstance(x, str) for x in server["command"])):
            raise ValueError("MCP command 必须是命令及参数列表")
        if not isinstance(server.get("env_names", []), list) or not all(isinstance(x, str) for x in server.get("env_names", [])):
            raise ValueError("MCP env_names 只能列出明确允许传入的环境变量名")
        if not isinstance(server.get("network", False), bool):
            raise ValueError("MCP network 必须是 true 或 false")
    return data


def save_settings(data: dict, home: Path | None = None) -> dict:
    result = validate_settings({**DEFAULT_SETTINGS, **data})
    atomic_json((home or agent_home()) / "settings.json", result)
    return result


def role_settings(settings: dict, role: str) -> dict:
    return {**settings, **settings.get("roles", {}).get(role, {})}
