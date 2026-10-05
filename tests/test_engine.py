from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from longcode.backends import BackendResult
from longcode.engine import LongCodeEngine
from longcode.models import ManagerDecision, Subtask, TaskContract
from longcode.scope import snapshot_tree
from longcode.semantic_probes import SemanticContractVerifier
from longcode.storage import RuntimeStore

from tests.helpers import FakeBackend


class EngineTests(unittest.TestCase):
    def _runtime(
        self,
        root: Path,
        *,
        checks: list[str],
        allowed: list[str] | None = None,
        forbidden: list[str] | None = None,
        attempts: int = 1,
        acceptance: list[str] | None = None,
    ) -> RuntimeStore:
        workspace = root / "workspace"
        workspace.mkdir()
        contract = TaskContract.create(
            "Keep this exact core goal in every round",
            acceptance or ["Implementation is correct"],
            checks=checks,
            allowed_paths=allowed or ["**"],
            forbidden_paths=forbidden or [".git/**"],
            max_attempts=attempts,
        )
        store = RuntimeStore(root / "runtime")
        store.initialize(contract, workspace)
        return store

    def test_executor_claim_cannot_override_failed_check(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._runtime(Path(directory), checks=["exit 7"])
            backend = FakeBackend(claimed_complete=True)
            state = LongCodeEngine(store, backend, auditor=backend).run(max_rounds=1)
            self.assertEqual(state.status, "blocked")
            self.assertEqual(state.criteria["AC-001"].status, "failed")
            self.assertEqual(backend.audits, [])
            event_types = [item["type"] for item in store.iter_events()]
            self.assertIn("criterion_verification_failed", event_types)
            self.assertNotIn("goal_completed", event_types)

    def test_semantic_failure_becomes_incremental_repair_and_reverification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._runtime(
                root,
                checks=["python3 -m py_compile automation.py"],
                attempts=2,
                acceptance=[
                    "JSON migration supports --input and --output, --dry-run without mutation, "
                    "and idempotent reruns"
                ],
            )
            contract = store.load_contract()
            contract.objective = "Implement a JSON migration CLI"
            store._atomic_json(store.current_goal_path, contract.to_dict())
            store._atomic_json(store._version_path(1), contract.to_dict())
            workspace = root / "workspace"
            (workspace / "input.json").write_text(
                '{"version": 1, "unknown": {"keep": true}}\n', encoding="utf-8"
            )
            calls = 0

            def implement_then_repair(candidate, contract, subtask):
                nonlocal calls
                calls += 1
                script = candidate / "automation.py"
                if calls == 1:
                    script.write_text(
                        "import argparse,json,sys\n"
                        "from pathlib import Path\n"
                        "p=argparse.ArgumentParser();p.add_argument('--input',required=True);"
                        "p.add_argument('--output',required=True);p.add_argument('--dry-run',action='store_true')\n"
                        "a=p.parse_args();src=Path(a.input);dst=Path(a.output);d=json.loads(src.read_text());"
                        "render=json.dumps(d,sort_keys=True,indent=2)+'\\n'\n"
                        "if a.dry_run: raise SystemExit(0)\n"
                        "if dst.exists() and dst.read_text()!=render: print('conflict',file=sys.stderr);raise SystemExit(3)\n"
                        "dst.write_text(render)\n",
                        encoding="utf-8",
                    )
                    return
                self.assertTrue(subtask.repair_context)
                self.assertIn("render=json.dumps", script.read_text(encoding="utf-8"))
                script.write_text(
                    script.read_text(encoding="utf-8").replace(
                        "d=json.loads(src.read_text());render=",
                        "d=json.loads(src.read_text());d['version']=2;render=",
                    ),
                    encoding="utf-8",
                )

            backend = FakeBackend(implement_then_repair)
            state = LongCodeEngine(
                store,
                backend,
                auditor=None,
                verifier_plugins=[SemanticContractVerifier()],
            ).run(max_rounds=2)

            self.assertEqual(state.status, "completed")
            self.assertEqual(calls, 2)
            self.assertEqual(state.repair_packets_created, 1)
            self.assertEqual(state.repair_attempts, 1)
            self.assertEqual(state.repairs_succeeded, 1)
            self.assertEqual(
                state.criteria["AC-001"].proof_obligations[
                    "semantic-contract-probe"
                ]["status"],
                "pass",
            )
            event_types = [item["type"] for item in store.iter_events()]
            self.assertIn("repair_packet_created", event_types)
            self.assertIn("repair_attempt_started", event_types)
            self.assertIn("repair_verified", event_types)

    def test_capability_on_unselected_stronger_executor_cannot_authorize_weaker_tier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._runtime(root, checks=["true"], attempts=2)
            contract = store.load_contract()
            contract.acceptance_criteria[0] = replace(
                contract.acceptance_criteria[0], risk_level="low"
            )
            store._atomic_json(store.current_goal_path, contract.to_dict())
            store._atomic_json(store._version_path(1), contract.to_dict())

            class FixedManager:
                def decide(self, contract, state):
                    return ManagerDecision(
                        action="execute",
                        criterion_id="AC-001",
                        subtask_goal="Use GUI",
                        executor_tier="E1",
                        risk_level="low",
                        verification_profile="default",
                        required_capabilities=["cli", "gui"],
                        reason="GUI task",
                    )

            class GuiBackend(FakeBackend):
                capabilities = frozenset({"cli", "filesystem", "gui"})

            economy = FakeBackend()
            expert = GuiBackend()
            state = LongCodeEngine(
                store,
                economy,
                auditor=None,
                manager=FixedManager(),
                executor_tiers={"E1": economy, "E3": expert},
            ).run(max_rounds=1)

            self.assertEqual(state.status, "waiting_input")
            self.assertIn("gui", state.blocker or "")
            self.assertEqual(state.criteria["AC-001"].attempts, 0)
            self.assertEqual(economy.executions, [])
            self.assertEqual(expert.executions, [])

    def test_e0_recipe_executes_without_calling_model_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._runtime(
                root,
                checks=["test -f metadata.json"],
                allowed=["metadata.json"],
                attempts=2,
            )
            contract = store.load_contract()
            contract.deterministic_recipes = {
                "AC-001": ["printf '{\"ready\": true}\\n' > metadata.json"]
            }
            contract.acceptance_criteria[0] = replace(
                contract.acceptance_criteria[0],
                risk_level="low",
                recipe_id="AC-001",
            )
            store._atomic_json(store.current_goal_path, contract.to_dict())
            store._atomic_json(store._version_path(1), contract.to_dict())
            model_backend = FakeBackend()

            state = LongCodeEngine(
                store,
                model_backend,
                auditor=None,
            ).run(max_rounds=1)
            self.assertEqual(state.status, "completed")
            self.assertEqual(state.criteria["AC-001"].tier_history, ["E0"])
            self.assertEqual(model_backend.executions, [])
            self.assertIn('"ready": true', (root / "workspace" / "metadata.json").read_text())

    def test_reported_tool_fault_restarts_fresh_episode_without_model_escalation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._runtime(
                root,
                checks=["test -f tool-output.txt"],
                allowed=["tool-output.txt"],
                attempts=2,
            )

            class ToolBackend(FakeBackend):
                def __init__(self):
                    super().__init__()
                    self.calls = 0

                def execute(self, workspace, contract, subtask):
                    self.calls += 1
                    self.executions.append((contract.objective, subtask.criterion_id))
                    if self.calls == 1:
                        return BackendResult(
                            ok=True,
                            report={
                                "summary": "browser tool unavailable",
                                "claimed_complete": False,
                                "changed_files": [],
                                "tests_run": [],
                                "remaining_risks": ["tool unavailable"],
                                "fault": {
                                    "category": "tool",
                                    "code": "browser_process_crashed",
                                    "summary": "Browser process crashed before action",
                                    "retryable": True,
                                    "scope": "tool_call",
                                    "severity": "warning",
                                },
                            },
                            stdout="",
                            stderr="",
                            return_code=0,
                            duration_seconds=0.01,
                        )
                    (workspace / "tool-output.txt").write_text("done\n", encoding="utf-8")
                    return super().execute(workspace, contract, subtask)

            backend = ToolBackend()
            state = LongCodeEngine(store, backend, auditor=None).run(max_rounds=2)
            self.assertEqual(state.status, "completed")
            self.assertEqual(state.criteria["AC-001"].tier_history, ["E2", "E2"])
            self.assertEqual(store.iter_faults()[0]["category"], "tool")
            decisions = [
                item for item in store.iter_events() if item["type"] == "recovery_decided"
            ]
            self.assertEqual(decisions[0]["data"]["action"], "restart_tool")
            self.assertNotIn(
                "executor_tier_escalated",
                [item["type"] for item in store.iter_events()],
            )

    def test_scope_drift_blocks_completion_even_when_checks_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._runtime(
                Path(directory),
                checks=["true"],
                allowed=["src/**"],
                forbidden=["tests/**"],
            )

            def drift(workspace, contract, subtask):
                (workspace / "tests").mkdir()
                (workspace / "tests" / "cheat.py").write_text("changed")

            backend = FakeBackend(drift)
            state = LongCodeEngine(store, backend, auditor=backend).run(max_rounds=1)
            self.assertEqual(state.status, "blocked")
            self.assertIn("scope violations", state.blocker or "")
            self.assertEqual(backend.audits, [])

    def test_happy_path_requires_criterion_and_final_audits(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._runtime(
                Path(directory),
                checks=["test -f solution.py"],
                allowed=["solution.py"],
            )
            contract = store.load_contract()
            contract.default_verification_profile = "semantic"
            contract.acceptance_criteria[0] = replace(
                contract.acceptance_criteria[0], verification_profile="semantic"
            )
            store._atomic_json(store.current_goal_path, contract.to_dict())
            store._atomic_json(store._version_path(1), contract.to_dict())

            def solve(workspace, contract, subtask):
                (workspace / "solution.py").write_text("ANSWER = 42\n")

            backend = FakeBackend(solve)
            state = LongCodeEngine(store, backend, auditor=backend).run(max_rounds=2)
            self.assertEqual(state.status, "completed")
            self.assertEqual(state.criteria["AC-001"].status, "verified")
            self.assertEqual(len(backend.audits), 1)
            self.assertEqual(backend.audits[-1][1], "AC-001")
            self.assertTrue(state.final_evidence)
            self.assertTrue(all((store.root / path).exists() for path in state.final_evidence))
            event_types = [item["type"] for item in store.iter_events()]
            self.assertIn("deterministic_verification", event_types)
            self.assertIn("audit_completed", event_types)
            self.assertEqual(state.auditor_calls, 1)
            self.assertGreaterEqual(state.control_level_counts["L2"], 1)
            self.assertEqual(event_types[-1], "goal_completed")

    def test_conclusive_deterministic_path_skips_auditor(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._runtime(Path(directory), checks=["true"])
            backend = FakeBackend(claimed_complete=False)
            state = LongCodeEngine(store, backend, auditor=backend).run(max_rounds=1)
            self.assertEqual(state.status, "completed")
            self.assertEqual(backend.audits, [])
            self.assertEqual(state.auditor_calls, 0)
            events = [item["type"] for item in store.iter_events()]
            self.assertIn("audit_skipped", events)
            self.assertNotIn("audit_completed", events)

    def test_fresh_subtasks_always_receive_active_goal_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._runtime(
                Path(directory),
                checks=["true"],
                attempts=2,
                acceptance=["First", "Second"],
            )
            backend = FakeBackend()
            first = LongCodeEngine(store, backend, auditor=None).run(max_rounds=1)
            self.assertEqual(first.status, "paused")
            second = LongCodeEngine(store, backend, auditor=None).run(max_rounds=2)
            self.assertEqual(second.status, "completed")
            self.assertEqual(
                [objective for objective, _ in backend.executions],
                [
                    "Keep this exact core goal in every round",
                    "Keep this exact core goal in every round",
                ],
            )

    def test_auditor_rejection_prevents_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._runtime(Path(directory), checks=["true"])
            contract = store.load_contract()
            contract.default_verification_profile = "semantic"
            contract.acceptance_criteria[0] = replace(
                contract.acceptance_criteria[0], verification_profile="semantic"
            )
            store._atomic_json(store.current_goal_path, contract.to_dict())
            store._atomic_json(store._version_path(1), contract.to_dict())
            backend = FakeBackend(audit_verdict="uncertain")
            state = LongCodeEngine(store, backend, auditor=backend).run(max_rounds=1)
            self.assertEqual(state.status, "blocked")
            self.assertEqual(state.criteria["AC-001"].status, "failed")
            self.assertNotIn("goal_completed", [item["type"] for item in store.iter_events()])

    def test_interrupted_in_progress_state_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._runtime(Path(directory), checks=["true"], attempts=2)
            state = store.load_state()
            state.status = "running"
            state.criteria["AC-001"].status = "in_progress"
            state.criteria["AC-001"].attempts = 1
            store.save_state(state)
            backend = FakeBackend()
            resumed = LongCodeEngine(store, backend, auditor=None).run(max_rounds=1)
            self.assertEqual(resumed.status, "completed")
            self.assertEqual(resumed.criteria["AC-001"].attempts, 2)

    def test_crash_after_promotion_recovers_by_reverification_without_reexecution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._runtime(
                root,
                checks=["test -f solution.py"],
                allowed=["solution.py"],
                attempts=2,
            )
            workspace = root / "workspace"
            (workspace / "solution.py").write_text("RECOVERED = True\n", encoding="utf-8")
            state = store.load_state()
            state.status = "running"
            state.round = 1
            criterion_state = state.criteria["AC-001"]
            criterion_state.status = "in_progress"
            criterion_state.attempts = 1
            criterion_state.current_tier = "E2"
            criterion_state.tier_attempts = {"E2": 1}
            criterion_state.tier_history = ["E2"]
            criterion_state.candidate_id = "CAND-crash-window"
            state.current_subtask_id = "ST-0001"
            state.current_executor_tier = "E2"
            store.save_state(state)
            executor_evidence = store.write_evidence(
                "round-0001/executor.json", {"ok": True}
            )
            verification_evidence = store.write_evidence(
                "round-0001/verification.json", {"verdict": "pass"}
            )
            promotion_evidence = store.write_evidence(
                "round-0001/promotion.json", {"passed": True}
            )
            contract = store.load_contract()
            subtask = Subtask(
                id="ST-0001",
                round=1,
                criterion_id="AC-001",
                criterion=contract.acceptance_criteria[0].description,
                objective=contract.objective,
                constraints=[],
                allowed_paths=["solution.py"],
                forbidden_paths=[".git/**"],
                checks=contract.checks,
                attempt=1,
                max_attempts=2,
                executor_tier="E2",
            )
            store.update_transaction(
                "CAND-crash-window",
                phase="promoted",
                criterion_id="AC-001",
                subtask_id="ST-0001",
                subtask=subtask.to_dict(),
                changed_paths=["solution.py"],
                executor_report={
                    "summary": "implemented",
                    "claimed_complete": True,
                    "changed_files": ["solution.py"],
                    "tests_run": [],
                    "remaining_risks": [],
                },
                evidence=[executor_evidence, verification_evidence, promotion_evidence],
                environment_context={},
            )
            backend = FakeBackend()

            recovered = LongCodeEngine(
                store,
                backend,
                auditor=None,
            ).run(max_rounds=1)

            self.assertEqual(recovered.status, "completed")
            self.assertEqual(backend.executions, [])
            self.assertEqual(
                store.load_transaction("CAND-crash-window")["phase"],
                "state_committed_recovered",
            )
            self.assertIn(
                "promoted_transaction_recovered",
                [item["type"] for item in store.iter_events()],
            )

    def test_local_promotion_started_is_reconciled_from_hashes_without_reexecution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._runtime(
                root,
                checks=["test -f solution.py"],
                allowed=["solution.py"],
                attempts=2,
            )
            workspace = root / "workspace"
            (workspace / "solution.py").write_text("RECOVERED = True\n", encoding="utf-8")
            state = store.load_state()
            state.status = "running"
            state.round = 1
            criterion_state = state.criteria["AC-001"]
            criterion_state.status = "in_progress"
            criterion_state.attempts = 1
            criterion_state.candidate_id = "CAND-started-local"
            state.current_subtask_id = "ST-0001"
            state.current_executor_tier = "E2"
            store.save_state(state)
            contract = store.load_contract()
            subtask = Subtask(
                id="ST-0001",
                round=1,
                criterion_id="AC-001",
                criterion=contract.acceptance_criteria[0].description,
                objective=contract.objective,
                constraints=[],
                allowed_paths=["solution.py"],
                forbidden_paths=[".git/**"],
                checks=contract.checks,
                attempt=1,
                max_attempts=2,
                executor_tier="E2",
            )
            committed_hash = snapshot_tree(workspace)["solution.py"]
            store.update_transaction(
                "CAND-started-local",
                phase="promotion_started",
                criterion_id="AC-001",
                subtask=subtask.to_dict(),
                changed_paths=["solution.py"],
                base_hashes={"solution.py": None},
                expected_hashes={"solution.py": committed_hash},
                external_environment=False,
                executor_report={
                    "summary": "implemented",
                    "claimed_complete": True,
                    "changed_files": ["solution.py"],
                    "tests_run": [],
                    "remaining_risks": [],
                },
                evidence=[],
                environment_context={},
            )
            backend = FakeBackend()

            recovered = LongCodeEngine(store, backend, auditor=None).run(max_rounds=1)

            self.assertEqual(recovered.status, "completed")
            self.assertEqual(backend.executions, [])
            journal = store.load_transaction("CAND-started-local")
            self.assertEqual(journal["phase"], "state_committed_recovered")
            self.assertIn(
                "promotion_reconciled_committed",
                [item["phase"] for item in journal["phase_history"]],
            )

    def test_unknown_external_commit_waits_and_never_replays_executor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._runtime(
                root,
                checks=["test -f action.txt"],
                allowed=["action.txt"],
                attempts=2,
            )
            workspace = root / "workspace"
            (workspace / "action.txt").write_text("possibly committed\n", encoding="utf-8")
            state = store.load_state()
            state.status = "running"
            state.round = 1
            criterion_state = state.criteria["AC-001"]
            criterion_state.status = "in_progress"
            criterion_state.attempts = 1
            criterion_state.candidate_id = "CAND-started-external"
            state.current_subtask_id = "ST-0001"
            state.current_executor_tier = "E2"
            store.save_state(state)
            store.update_transaction(
                "CAND-started-external",
                phase="promotion_started",
                changed_paths=["action.txt"],
                base_hashes={"action.txt": None},
                expected_hashes={
                    "action.txt": snapshot_tree(workspace)["action.txt"]
                },
                external_environment=True,
                environment_context={"session_id": "session-001"},
            )

            class UnknownExternalEnvironment:
                capabilities = frozenset(
                    {"cli", "filesystem", "transaction", "external_environment"}
                )

                def __init__(self):
                    self.reconciliations = 0

                def reconcile(self, journal, workspace):
                    self.reconciliations += 1
                    return {"status": "unknown", "summary": "controller has no receipt"}

                def begin(self, workspace, *, round_number):
                    raise AssertionError("ambiguous action must not start a new candidate")

            environment = UnknownExternalEnvironment()
            backend = FakeBackend()
            first = LongCodeEngine(
                store,
                backend,
                auditor=None,
                transaction_manager=environment,
            ).run(max_rounds=1)
            second = LongCodeEngine(
                store,
                backend,
                auditor=None,
                transaction_manager=environment,
            ).run(max_rounds=1)

            self.assertEqual(first.status, "waiting_input")
            self.assertEqual(second.status, "waiting_input")
            self.assertIn("will not replay", second.blocker or "")
            self.assertEqual(backend.executions, [])
            self.assertEqual(environment.reconciliations, 2)
            self.assertEqual(
                store.load_transaction("CAND-started-external")["phase"],
                "promotion_reconciliation_required",
            )

    def test_repeated_verified_failure_escalates_and_only_promotes_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._runtime(
                root,
                checks=['test "$(cat answer.txt)" = "correct"'],
                allowed=["answer.txt"],
                attempts=2,
            )
            contract = store.load_contract()
            contract.acceptance_criteria[0] = replace(
                contract.acceptance_criteria[0], risk_level="low"
            )
            # This test owns the freshly-created runtime, so replacing v1 is safe and keeps the
            # setup concise. Production goal revisions remain append-only through revise_goal.
            store._atomic_json(store.current_goal_path, contract.to_dict())
            store._atomic_json(store._version_path(1), contract.to_dict())

            attempts_seen: list[tuple[str, str]] = []

            def fail_at_e1(workspace, contract, subtask):
                attempts_seen.append((subtask.executor_tier, workspace.name))
                (workspace / "answer.txt").write_text("wrong\n", encoding="utf-8")
                self.assertFalse((root / "workspace" / "answer.txt").exists())

            def solve_at_e2(workspace, contract, subtask):
                attempts_seen.append((subtask.executor_tier, workspace.name))
                (workspace / "answer.txt").write_text("correct\n", encoding="utf-8")
                self.assertFalse((root / "workspace" / "answer.txt").exists())

            low = FakeBackend(fail_at_e1)
            standard = FakeBackend(solve_at_e2)
            state = LongCodeEngine(
                store,
                standard,
                auditor=None,
                executor_tiers={"E1": low, "E2": standard},
            ).run(max_rounds=3)

            self.assertEqual(state.status, "completed")
            self.assertEqual(state.criteria["AC-001"].tier_history, ["E1", "E1", "E2"])
            self.assertEqual((root / "workspace" / "answer.txt").read_text(), "correct\n")
            self.assertEqual([tier for tier, _ in attempts_seen], ["E1", "E1", "E2"])
            events = store.iter_events()
            escalations = [item for item in events if item["type"] == "executor_tier_escalated"]
            promotions = [item for item in events if item["type"] == "candidate_promotion"]
            self.assertEqual(len(escalations), 1)
            self.assertEqual(len(promotions), 1)
            self.assertTrue(promotions[0]["data"]["passed"])
            graph_kinds = {node.kind for node in state.graph.nodes.values()}
            self.assertTrue(
                {"subtask", "artifact", "fault", "decision", "evidence", "checkpoint"}
                <= graph_kinds
            )
            relations = {edge.relation for edge in state.graph.edges}
            self.assertIn("failed_because", relations)
            self.assertIn("produced_by", relations)
            self.assertIn("verified_by", relations)

    def test_strongest_tier_replan_gets_a_new_bounded_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._runtime(
                root,
                checks=['test "$(cat result.txt)" = "solved"'],
                allowed=["result.txt"],
                attempts=2,
            )
            contract = store.load_contract()
            contract.acceptance_criteria[0] = replace(
                contract.acceptance_criteria[0], risk_level="high"
            )
            store._atomic_json(store.current_goal_path, contract.to_dict())
            store._atomic_json(store._version_path(1), contract.to_dict())
            calls = 0

            def solve_after_replan(workspace, contract, subtask):
                nonlocal calls
                calls += 1
                value = "solved" if calls == 3 else "same-wrong-result"
                (workspace / "result.txt").write_text(value + "\n", encoding="utf-8")

            expert = FakeBackend(solve_after_replan)
            state = LongCodeEngine(
                store,
                expert,
                auditor=None,
                executor_tiers={"E3": expert},
            ).run(max_rounds=3)
            self.assertEqual(state.status, "completed")
            self.assertEqual(state.criteria["AC-001"].replan_count, 1)
            self.assertEqual(state.criteria["AC-001"].tier_history, ["E3", "E3", "E3"])
            self.assertIn(
                "subtask_replan_requested",
                [item["type"] for item in store.iter_events()],
            )
            self.assertGreaterEqual(state.control_level_counts["L3"], 1)


if __name__ == "__main__":
    unittest.main()
