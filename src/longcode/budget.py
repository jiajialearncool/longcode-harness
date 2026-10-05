from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping


BROKER_BUDGET_KEYS = frozenset(
    {
        "max_model_calls",
        "max_input_tokens",
        "max_output_tokens",
        "max_input_tokens_per_request",
        "max_output_tokens_per_request",
    }
)
RUNNER_BUDGET_KEYS = frozenset({"max_wall_seconds"})


@dataclass(frozen=True)
class BudgetLimits:
    max_model_calls: int
    max_input_tokens: int
    max_output_tokens: int
    max_input_tokens_per_request: int
    max_output_tokens_per_request: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BudgetLimits":
        required = (
            "max_model_calls",
            "max_input_tokens",
            "max_output_tokens",
        )
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(
                "strict broker budget requires: " + ", ".join(sorted(missing))
            )
        parsed = {key: _positive_int(value[key], key) for key in required}
        input_per_request = _positive_int(
            value.get("max_input_tokens_per_request", parsed["max_input_tokens"]),
            "max_input_tokens_per_request",
        )
        output_per_request = _positive_int(
            value.get("max_output_tokens_per_request", parsed["max_output_tokens"]),
            "max_output_tokens_per_request",
        )
        return cls(
            **parsed,
            max_input_tokens_per_request=min(
                input_per_request, parsed["max_input_tokens"]
            ),
            max_output_tokens_per_request=min(
                output_per_request, parsed["max_output_tokens"]
            ),
        )


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    model: str | None = None
    system_fingerprint: str | None = None


@dataclass(frozen=True)
class Reservation:
    id: str
    input_tokens: int
    output_tokens: int


@dataclass
class BudgetState:
    limits: dict[str, int]
    route_requests: int = 0
    control_requests: int = 0
    model_calls: int = 0
    completed_model_calls: int = 0
    failed_model_calls: int = 0
    rejected_requests: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    discarded_input_tokens: int = 0
    discarded_output_tokens: int = 0
    reserved_input_tokens: int = 0
    reserved_output_tokens: int = 0
    missing_usage_responses: int = 0
    budget_breaches: list[str] = field(default_factory=list)
    observed_models: list[str] = field(default_factory=list)
    system_fingerprints: list[str] = field(default_factory=list)
    upstream_errors: list[str] = field(default_factory=list)
    started_at_unix: float = field(default_factory=time.time)
    updated_at_unix: float = field(default_factory=time.time)


class BudgetLedger:
    """Atomic task-level accounting shared by every model role in one arm."""

    def __init__(self, limits: BudgetLimits, path: Path | str):
        self.limits = limits
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._state = BudgetState(limits=asdict(limits))
        self._reservations: dict[str, Reservation] = {}
        self._persist_locked()

    def reserve(self) -> tuple[Reservation | None, str | None]:
        with self._lock:
            state = self._state
            if state.model_calls >= self.limits.max_model_calls:
                return self._reject_locked("max_model_calls exhausted")
            remaining_input = (
                self.limits.max_input_tokens
                - state.input_tokens
                - state.reserved_input_tokens
            )
            remaining_output = (
                self.limits.max_output_tokens
                - state.output_tokens
                - state.reserved_output_tokens
            )
            if remaining_input <= 0:
                return self._reject_locked("max_input_tokens exhausted")
            if remaining_output <= 0:
                return self._reject_locked("max_output_tokens exhausted")
            reservation = Reservation(
                id=uuid.uuid4().hex,
                input_tokens=min(
                    self.limits.max_input_tokens_per_request, remaining_input
                ),
                output_tokens=min(
                    self.limits.max_output_tokens_per_request, remaining_output
                ),
            )
            state.route_requests += 1
            state.model_calls += 1
            state.reserved_input_tokens += reservation.input_tokens
            state.reserved_output_tokens += reservation.output_tokens
            state.updated_at_unix = time.time()
            self._reservations[reservation.id] = reservation
            self._persist_locked()
            return reservation, None

    def complete(
        self,
        reservation: Reservation,
        usage: Usage | None,
    ) -> bool:
        """Complete one request; False means its model response must not reach the agent."""
        with self._lock:
            current = self._reservations.pop(reservation.id, None)
            if current is None:
                raise RuntimeError("unknown or already completed budget reservation")
            state = self._state
            state.reserved_input_tokens -= current.input_tokens
            state.reserved_output_tokens -= current.output_tokens
            state.completed_model_calls += 1
            allow_response = True
            if usage is None:
                state.missing_usage_responses += 1
                state.budget_breaches.append("successful response omitted provider usage")
                allow_response = False
            else:
                state.input_tokens += usage.input_tokens
                state.cached_input_tokens += usage.cached_input_tokens
                state.output_tokens += usage.output_tokens
                state.reasoning_tokens += usage.reasoning_tokens
                if usage.model and usage.model not in state.observed_models:
                    state.observed_models.append(usage.model)
                if (
                    usage.system_fingerprint
                    and usage.system_fingerprint not in state.system_fingerprints
                ):
                    state.system_fingerprints.append(usage.system_fingerprint)
                if usage.input_tokens > current.input_tokens:
                    state.budget_breaches.append(
                        "provider input usage exceeded the reserved per-request budget"
                    )
                    state.discarded_input_tokens += usage.input_tokens
                    state.discarded_output_tokens += usage.output_tokens
                    allow_response = False
                if usage.output_tokens > current.output_tokens:
                    state.budget_breaches.append(
                        "provider output usage exceeded the reserved per-request budget"
                    )
                    state.discarded_input_tokens += usage.input_tokens
                    state.discarded_output_tokens += usage.output_tokens
                    allow_response = False
                if state.input_tokens > self.limits.max_input_tokens:
                    state.budget_breaches.append("max_input_tokens exceeded")
                    allow_response = False
                if state.output_tokens > self.limits.max_output_tokens:
                    state.budget_breaches.append("max_output_tokens exceeded")
                    allow_response = False
            state.updated_at_unix = time.time()
            self._persist_locked()
            return allow_response

    def fail(self, reservation: Reservation, message: str) -> None:
        with self._lock:
            current = self._reservations.pop(reservation.id, None)
            if current is None:
                return
            state = self._state
            state.reserved_input_tokens -= current.input_tokens
            state.reserved_output_tokens -= current.output_tokens
            state.failed_model_calls += 1
            state.upstream_errors.append(message[:500])
            state.updated_at_unix = time.time()
            self._persist_locked()

    def note_control_request(self) -> None:
        with self._lock:
            self._state.control_requests += 1
            self._state.updated_at_unix = time.time()
            self._persist_locked()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            value = asdict(self._state)
            value["token_metrics_available"] = (
                self._state.completed_model_calls > 0
                and self._state.missing_usage_responses == 0
            )
            value["budget_enforced"] = (
                self._state.route_requests > 0
                and not self._state.budget_breaches
                and self._state.missing_usage_responses == 0
                and not self._reservations
            )
            value["budget_exhausted"] = self._state.rejected_requests > 0
            return value

    def _reject_locked(self, reason: str) -> tuple[None, str]:
        self._state.rejected_requests += 1
        self._state.updated_at_unix = time.time()
        self._persist_locked()
        return None, reason

    def _persist_locked(self) -> None:
        payload = asdict(self._state)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


class BudgetBroker:
    """Local Responses-compatible reverse proxy with a strict task budget."""

    def __init__(
        self,
        *,
        limits: BudgetLimits,
        upstream_base_url: str,
        ledger_path: Path | str,
        upstream_api_key_env: str | None = None,
        upstream_timeout_seconds: int = 900,
        environ: Mapping[str, str] | None = None,
    ):
        parsed = urllib.parse.urlparse(upstream_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("provider_proxy.upstream_base_url must be HTTP(S)")
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.upstream_api_key_env = upstream_api_key_env
        self.upstream_timeout_seconds = _positive_int(
            upstream_timeout_seconds, "provider_proxy.upstream_timeout_seconds"
        )
        self.environment = dict(os.environ if environ is None else environ)
        if upstream_api_key_env and not self.environment.get(upstream_api_key_env):
            raise ValueError(
                f"provider proxy API key environment variable is unset: {upstream_api_key_env}"
            )
        self.ledger = BudgetLedger(limits, ledger_path)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("budget broker has not been started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "BudgetBroker":
        if self._server is not None:
            raise RuntimeError("budget broker is already running")
        broker = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802
                broker.ledger.note_control_request()
                broker._forward_control(self)

            def do_POST(self) -> None:  # noqa: N802
                broker._forward_model(self)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="longcode-budget-broker",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def __enter__(self) -> "BudgetBroker":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _forward_model(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length", "0"))
        raw_body = handler.rfile.read(length)
        try:
            request_body = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError:
            _json_error(handler, 400, "invalid_json", "request body must be JSON")
            return
        if not isinstance(request_body, dict):
            _json_error(handler, 400, "invalid_json", "request body must be an object")
            return
        reservation, reason = self.ledger.reserve()
        if reservation is None:
            _json_error(handler, 429, "budget_exhausted", reason or "budget exhausted")
            return
        requested_max = request_body.get("max_output_tokens")
        if not isinstance(requested_max, int) or requested_max > reservation.output_tokens:
            request_body["max_output_tokens"] = reservation.output_tokens
        forwarded_body = json.dumps(request_body, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self._upstream_url(handler.path),
            data=forwarded_body,
            headers=self._forward_headers(handler.headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.upstream_timeout_seconds
            ) as response:
                response_body = response.read()
                status = response.status
                content_type = response.headers.get(
                    "Content-Type", "application/json"
                )
        except urllib.error.HTTPError as error:
            response_body = error.read()
            status = error.code
            content_type = error.headers.get("Content-Type", "application/json")
            self.ledger.fail(reservation, f"upstream HTTP {status}")
            _raw_response(handler, status, content_type, response_body)
            return
        except (OSError, urllib.error.URLError) as error:
            self.ledger.fail(reservation, f"upstream transport error: {error}")
            _json_error(handler, 502, "upstream_error", str(error))
            return
        usage = _extract_usage(response_body, content_type)
        if status < 200 or status >= 300:
            self.ledger.fail(reservation, f"upstream HTTP {status}")
            _raw_response(handler, status, content_type, response_body)
            return
        if not self.ledger.complete(reservation, usage):
            _json_error(
                handler,
                429,
                "budget_breach",
                "provider response exceeded or omitted the reserved token budget",
            )
            return
        _raw_response(handler, status, content_type, response_body)

    def _forward_control(self, handler: BaseHTTPRequestHandler) -> None:
        request = urllib.request.Request(
            self._upstream_url(handler.path),
            headers=self._forward_headers(handler.headers),
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.upstream_timeout_seconds
            ) as response:
                _raw_response(
                    handler,
                    response.status,
                    response.headers.get("Content-Type", "application/json"),
                    response.read(),
                )
        except urllib.error.HTTPError as error:
            _raw_response(
                handler,
                error.code,
                error.headers.get("Content-Type", "application/json"),
                error.read(),
            )
        except (OSError, urllib.error.URLError) as error:
            _json_error(handler, 502, "upstream_error", str(error))

    def _upstream_url(self, request_path: str) -> str:
        parsed = urllib.parse.urlsplit(request_path)
        upstream_path = urllib.parse.urlsplit(self.upstream_base_url).path.rstrip("/")
        path = parsed.path
        if upstream_path.endswith("/v1") and path.startswith("/v1/"):
            base = self.upstream_base_url[: -len(upstream_path)]
            url = base + path
        else:
            url = self.upstream_base_url + (path if path.startswith("/") else "/" + path)
        return url + (("?" + parsed.query) if parsed.query else "")

    def _forward_headers(self, source: Mapping[str, str]) -> dict[str, str]:
        excluded = {"host", "content-length", "accept-encoding", "connection"}
        headers = {
            key: value for key, value in source.items() if key.lower() not in excluded
        }
        headers["Content-Type"] = "application/json"
        if self.upstream_api_key_env:
            headers["Authorization"] = (
                "Bearer " + self.environment[self.upstream_api_key_env]
            )
        return headers


def broker_result(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "input_tokens": int(snapshot.get("input_tokens", 0)),
        "cached_input_tokens": int(snapshot.get("cached_input_tokens", 0)),
        "output_tokens": int(snapshot.get("output_tokens", 0)),
        "reasoning_tokens": int(snapshot.get("reasoning_tokens", 0)),
        "model_calls": int(snapshot.get("model_calls", 0)),
        "token_metrics_available": bool(snapshot.get("token_metrics_available", False)),
        "budget_enforced": bool(snapshot.get("budget_enforced", False)),
        "budget_exhausted": bool(snapshot.get("budget_exhausted", False)),
        "budget_breach": bool(snapshot.get("budget_breaches")),
        "observed_models": list(snapshot.get("observed_models", [])),
        "system_fingerprints": list(snapshot.get("system_fingerprints", [])),
        "provider_route_requests": int(snapshot.get("route_requests", 0)),
        "broker": dict(snapshot),
    }


def _extract_usage(body: bytes, content_type: str) -> Usage | None:
    payloads: list[dict[str, Any]] = []
    text = body.decode("utf-8", errors="replace")
    if "text/event-stream" in content_type:
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                value = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                payloads.append(value)
    else:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(value, dict):
            payloads.append(value)
    for value in reversed(payloads):
        response = value.get("response") if isinstance(value.get("response"), dict) else value
        usage = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(usage, dict):
            continue
        input_details = usage.get("input_tokens_details", {})
        output_details = usage.get("output_tokens_details", {})
        return Usage(
            input_tokens=_nonnegative_int(usage.get("input_tokens", 0)),
            cached_input_tokens=_nonnegative_int(
                input_details.get("cached_tokens", 0)
                if isinstance(input_details, dict)
                else 0
            ),
            output_tokens=_nonnegative_int(usage.get("output_tokens", 0)),
            reasoning_tokens=_nonnegative_int(
                output_details.get("reasoning_tokens", 0)
                if isinstance(output_details, dict)
                else 0
            ),
            model=str(response.get("model")) if response.get("model") else None,
            system_fingerprint=(
                str(response.get("system_fingerprint"))
                if response.get("system_fingerprint")
                else None
            ),
        )
    return None


def _raw_response(
    handler: BaseHTTPRequestHandler,
    status: int,
    content_type: str,
    body: bytes,
) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(body)


def _json_error(
    handler: BaseHTTPRequestHandler,
    status: int,
    code: str,
    message: str,
) -> None:
    body = json.dumps(
        {"error": {"type": code, "code": code, "message": message}},
        separators=(",", ":"),
    ).encode("utf-8")
    _raw_response(handler, status, "application/json", body)


def _positive_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0
