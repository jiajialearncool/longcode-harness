from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from longcode.budget import BudgetBroker, BudgetLimits


class _FakeUpstreamResponse:
    def __init__(self, body: bytes):
        self._body = body
        self.status = 200
        self.headers = {"Content-Type": "application/json"}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeUpstreamResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


class _FakeHandler:
    def __init__(self, body: dict):
        raw = json.dumps(body).encode()
        self.path = "/v1/responses"
        self.headers = {
            "Content-Length": str(len(raw)),
            "Content-Type": "application/json",
            "Authorization": "Bearer local",
        }
        self.rfile = io.BytesIO(raw)
        self.wfile = io.BytesIO()
        self.status: int | None = None
        self.response_headers: dict[str, str] = {}

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, key: str, value: str) -> None:
        self.response_headers[key] = value

    def end_headers(self) -> None:
        return None


def _provider_body(*, input_tokens: int = 40, output_tokens: int = 10) -> bytes:
    return json.dumps(
        {
            "id": "resp_fake",
            "model": "same-model-snapshot",
            "system_fingerprint": "fp_same",
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_details": {"cached_tokens": 5},
                "output_tokens": output_tokens,
                "output_tokens_details": {"reasoning_tokens": 3},
            },
            "output": [],
        }
    ).encode()


class BudgetBrokerTests(unittest.TestCase):
    def test_proxy_records_provider_usage_and_caps_output(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.json"
            limits = BudgetLimits.from_mapping(
                {
                    "max_model_calls": 2,
                    "max_input_tokens": 100,
                    "max_output_tokens": 20,
                    "max_input_tokens_per_request": 60,
                    "max_output_tokens_per_request": 12,
                }
            )
            broker = BudgetBroker(
                limits=limits,
                upstream_base_url="https://api.example.test/v1",
                ledger_path=ledger,
            )
            handler = _FakeHandler({"model": "same-model", "input": "test"})
            with patch(
                "longcode.budget.urllib.request.urlopen",
                return_value=_FakeUpstreamResponse(_provider_body()),
            ) as urlopen:
                broker._forward_model(handler)
            snapshot = broker.ledger.snapshot()

            forwarded = json.loads(urlopen.call_args.args[0].data)
            self.assertEqual(handler.status, 200)
            self.assertEqual(forwarded["max_output_tokens"], 12)
            self.assertEqual(snapshot["input_tokens"], 40)
            self.assertEqual(snapshot["cached_input_tokens"], 5)
            self.assertEqual(snapshot["output_tokens"], 10)
            self.assertEqual(snapshot["reasoning_tokens"], 3)
            self.assertEqual(snapshot["observed_models"], ["same-model-snapshot"])
            self.assertTrue(snapshot["token_metrics_available"])
            self.assertTrue(snapshot["budget_enforced"])
            self.assertTrue(ledger.is_file())

    def test_call_limit_is_rejected_before_reaching_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            limits = BudgetLimits.from_mapping(
                {
                    "max_model_calls": 1,
                    "max_input_tokens": 100,
                    "max_output_tokens": 20,
                }
            )
            broker = BudgetBroker(
                limits=limits,
                upstream_base_url="https://api.example.test/v1",
                ledger_path=Path(directory) / "ledger.json",
            )
            with patch(
                "longcode.budget.urllib.request.urlopen",
                return_value=_FakeUpstreamResponse(_provider_body()),
            ) as urlopen:
                first = _FakeHandler({"model": "same-model", "input": "test"})
                broker._forward_model(first)
                second = _FakeHandler({"model": "same-model", "input": "test"})
                broker._forward_model(second)
            snapshot = broker.ledger.snapshot()

            self.assertEqual(first.status, 200)
            self.assertEqual(second.status, 429)
            self.assertEqual(urlopen.call_count, 1)
            self.assertTrue(snapshot["budget_exhausted"])
            self.assertTrue(snapshot["budget_enforced"])

    def test_response_over_input_reservation_is_withheld_and_marks_breach(self):
        with tempfile.TemporaryDirectory() as directory:
            limits = BudgetLimits.from_mapping(
                {
                    "max_model_calls": 1,
                    "max_input_tokens": 100,
                    "max_output_tokens": 20,
                    "max_input_tokens_per_request": 60,
                }
            )
            broker = BudgetBroker(
                limits=limits,
                upstream_base_url="https://api.example.test/v1",
                ledger_path=Path(directory) / "ledger.json",
            )
            handler = _FakeHandler({"model": "same-model", "input": "test"})
            with patch(
                "longcode.budget.urllib.request.urlopen",
                return_value=_FakeUpstreamResponse(_provider_body(input_tokens=61)),
            ):
                broker._forward_model(handler)
            snapshot = broker.ledger.snapshot()

            self.assertEqual(handler.status, 429)
            self.assertFalse(snapshot["budget_enforced"])
            self.assertTrue(snapshot["budget_breaches"])
            self.assertEqual(snapshot["discarded_input_tokens"], 61)


if __name__ == "__main__":
    unittest.main()
