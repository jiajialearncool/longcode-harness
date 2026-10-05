from __future__ import annotations

import traceback
import time
from dataclasses import replace
from pathlib import Path

from .adapters import LocalWorkspaceAdapter, adapter_capabilities
from .backends import AgentBackend, BackendResult
from .agent_runtime import Cancelled
from .memory import MemoryStore
from .models import (
    FaultEvent,
    ManagerDecision,
    StateEdge,
    StateNode,
    Subtask,
    TaskContract,
    TaskState,
    stable_node_id,
    utc_now,
)
from .policy import PolicyEngine
from .repair import build_repair_packet
from .recovery import FaultClassifier, SafetyBelt, register_fault
from .recipes import RecipeBackend
from .routing import (
    ExecutorRegistry,
    HeuristicAdaptiveManager,
    capabilities_for,
    enforce_tier_policy,
    infer_signals,
    next_tier,
)
from .scope import snapshot_tree
from .state_graph import add_edge, add_node, ensure_graph, goal_coverage, sync_criterion_projection
from .storage import RuntimeStore
from .transactions import TransactionManager, UnsafeCandidatePathError, WorkspaceTransaction
from .verifiers import VerificationOutcome, VerifierMesh, VerifierPlugin


ACTIVE_CRITERION_STATUSES = {"pending", "failed", "needs_revalidation", "in_progress"}
CONTINUE_ACTIONS = {
    "retry",
    "retry_backoff",
    "restart_tool",
    "switch_provider",
    "escalate",
    "strengthen_verifier",
    "replan",
}


class LongCodeEngine:
    """Adaptive A-MEA-R engine; the name remains as a v0.1 compatibility entry point."""

    def __init__(
        self,
        store: RuntimeStore,
        executor: AgentBackend,
        *,
        auditor: AgentBackend | None,
        manager=None,
        executor_tiers: dict[str, AgentBackend] | None = None,
        verifier_plugins: list[VerifierPlugin] | None = None,
        transaction_manager: TransactionManager | None = None,
        safety_belt: SafetyBelt | None = None,
        fault_classifier: FaultClassifier | None = None,
        policy_engine: PolicyEngine | None = None,
        memory_store: MemoryStore | None = None,
        available_capabilities: set[str] | None = None,
        sleep_fn=None,
        cancellation=None,
        check_runner=None,
    ):
        self.store = store
        self.executor = executor
        self.auditor = auditor
        self.manager = manager or HeuristicAdaptiveManager()
        configured_tiers = {"E0": RecipeBackend(), **(executor_tiers or {})}
        self.executors = ExecutorRegistry(executor, configured_tiers)
        self.cancellation = cancellation
        self.verifier_mesh = VerifierMesh(auditor=auditor, plugins=verifier_plugins, check_runner=check_runner)
        self.transactions = transaction_manager or LocalWorkspaceAdapter(store.root)
        self.safety_belt = safety_belt or SafetyBelt()
        self.fault_classifier = fault_classifier or FaultClassifier()
        self.policy_engine = policy_engine or PolicyEngine()
        self.memory = memory_store or MemoryStore(store.root)
        self.environment_capabilities = {"cli", "filesystem"}
        self.environment_capabilities.update(adapter_capabilities(self.transactions))
        self.environment_capabilities.update(available_capabilities or set())
        self.verifier_capabilities: set[str] = set()
        for plugin in verifier_plugins or []:
            kind = str(getattr(plugin, "kind", ""))
            if kind:
                self.verifier_capabilities.add(kind)
        self._last_manager_backend_result = None
        self._manager_evidence_sequence = 0
        self.sleep_fn = sleep_fn or time.sleep

    def run(self, *, max_rounds: int = 10) -> TaskState:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        contract = self.store.load_contract()
        state = self.store.load_state()
        workspace = self.store.workspace()
        if state.contract_version != contract.version:
            raise RuntimeError("state and active goal versions do not match")
        ensure_graph(contract, state)
        if not self._recover_interrupted_state(contract, state, workspace):
            return self.store.load_state()
        run_id = self.store.begin_run()
        state.run_id = run_id
        state.status = "running"
        state.blocker = None
        self.store.save_state(state)
        self.store.append_event(
            "run_started",
            {
                "contract_version": contract.version,
                "max_rounds": max_rounds,
                "auditor_enabled": self.auditor is not None,
                "transactional": True,
                "adaptive_routing": True,
            },
        )
        self._enter_control_level(
            state,
            "L0",
            "Execute the selected task through the configured backend",
            force=True,
        )
        self.store.save_state(state)
        try:
            result = self._run_loop(contract, state, workspace, max_rounds=max_rounds)
        except Cancelled as error:
            state = self.store.load_state()
            state.status = "paused"
            state.blocker = str(error)
            self.store.save_state(state)
            self.store.append_event("run_cancelled", {"reason": str(error)})
            result = state
        except OSError as error:
            result = self._pause_after_control_failure(state, error)
        finally:
            latest_status = self.store.load_state().status
            self.store.end_run(status=latest_status)
        return result

    def _run_loop(
        self,
        contract: TaskContract,
        state: TaskState,
        workspace: Path,
        *,
        max_rounds: int,
    ) -> TaskState:
        rounds_used = 0
        while True:
            if self.cancellation:
                self.cancellation.check()
            decision: ManagerDecision = self.manager.decide(contract, state)
            self.store.append_event("manager_decision", decision.to_dict())
            self._record_manager_backend_if_new(state, decision)
            self._remember_episode(
                "manager_decision",
                decision.reason or decision.action,
                source="manager",
                metadata=decision.to_dict(),
            )
            if decision.action == "ask":
                state.status = "waiting_input"
                state.blocker = decision.reason
                self.store.save_state(state)
                self.store.append_event("user_input_required", {"reason": decision.reason})
                return self.store.load_state()
            if decision.action == "blocked":
                self._block(state, decision.reason or "No ready subtask can advance the goal")
                return self.store.load_state()
            if decision.action == "done":
                self._finalize(contract, state, workspace)
                return self.store.load_state()
            if decision.action != "execute" or not decision.criterion_id:
                self._block(state, f"Invalid manager decision: {decision.action}")
                return self.store.load_state()
            if rounds_used >= max_rounds:
                state.status = "paused"
                state.blocker = f"Run reached max_rounds={max_rounds}; resume with another run"
                self.store.save_state(state)
                self.store.append_event("run_paused", {"reason": state.blocker})
                return self.store.load_state()

            criterion = next(
                (item for item in contract.acceptance_criteria if item.id == decision.criterion_id),
                None,
            )
            if criterion is None:
                self._block(state, f"Manager selected unknown criterion: {decision.criterion_id}")
                return self.store.load_state()
            criterion_state = state.criteria[criterion.id]
            inferred = infer_signals(
                criterion.description,
                criterion.risk_level,
                criterion.depends_on,
                criterion.verification_profile,
            )
            proposed_risk = {"low": 1, "medium": 2, "high": 3}.get(
                decision.risk_level,
                2,
            )
            recipe_available = bool(
                criterion.recipe_id
                and contract.deterministic_recipes.get(criterion.recipe_id)
            )
            policy_tier = enforce_tier_policy(
                decision.executor_tier,
                max(inferred.risk, proposed_risk),
                deterministic_recipe=recipe_available,
            )
            if policy_tier != decision.executor_tier:
                self.store.append_event(
                    "manager_tier_adjusted",
                    {
                        "criterion_id": criterion.id,
                        "proposed_tier": decision.executor_tier,
                        "enforced_tier": policy_tier,
                        "reason": "deterministic contract risk floor",
                    },
                )
            requested_tier = criterion_state.next_tier or policy_tier
            tier, backend = self.executors.resolve(requested_tier)
            tier_attempts = criterion_state.tier_attempts.get(tier, 0)
            if tier_attempts >= contract.max_attempts_per_criterion:
                stronger = next_tier(tier, contract.executor_tiers)
                if stronger:
                    tier, backend = self.executors.resolve(stronger)
                    criterion_state.next_tier = tier
                    self.store.append_event(
                        "executor_tier_escalated",
                        {
                            "criterion_id": criterion.id,
                            "from_tier": requested_tier,
                            "to_tier": tier,
                            "reason": "Current tier attempt budget exhausted before execution",
                        },
                    )
                else:
                    self._block(state, f"{criterion.id} exhausted all configured executor tiers")
                    return self.store.load_state()

            rounds_used += 1
            state.round += 1
            criterion_state.attempts += 1
            criterion_state.tier_attempts[tier] = criterion_state.tier_attempts.get(tier, 0) + 1
            criterion_state.status = "in_progress"
            criterion_state.current_tier = tier
            criterion_state.next_tier = None
            criterion_state.tier_history.append(tier)
            state.current_executor_tier = tier
            required_capabilities = sorted(
                set(decision.required_capabilities)
                | set(
                    capabilities_for(
                        criterion.description,
                        criterion.verification_profile,
                    )
                )
            )
            subtask = Subtask(
                id=f"ST-{state.round:04d}",
                round=state.round,
                criterion_id=criterion.id,
                criterion=criterion.description,
                objective=contract.objective,
                constraints=list(contract.constraints),
                allowed_paths=list(contract.allowed_paths),
                forbidden_paths=list(contract.forbidden_paths),
                checks=list(contract.checks),
                attempt=criterion_state.attempts,
                max_attempts=contract.max_attempts_per_criterion,
                node_id=criterion.id,
                executor_tier=tier,
                risk_level=decision.risk_level,
                verification_profile=criterion.verification_profile,
                required_capabilities=required_capabilities,
                manager_reason=decision.reason,
                budget={"command_timeout_seconds": contract.command_timeout_seconds},
                memory_context=self.memory.context(),
                recipe_id=criterion.recipe_id if tier == "E0" else None,
                repair_context=dict(criterion_state.repair_packet or {}),
            )
            state.current_subtask_id = subtask.id
            self._graph_add_subtask(state, subtask)
            sync_criterion_projection(state)
            self.store.save_state(state)
            self.store.append_event("subtask_created", subtask.to_dict())
            self.store.append_event(
                "executor_routed",
                {
                    "subtask_id": subtask.id,
                    "requested_tier": requested_tier,
                    "resolved_tier": tier,
                    "reason": decision.reason,
                },
            )

            policy = self.policy_engine.authorize(
                contract,
                state,
                subtask,
                available_capabilities=(
                    set(self.environment_capabilities)
                    | set(self.verifier_capabilities)
                    | adapter_capabilities(backend)
                ),
            )
            self.store.append_event(
                "policy_decided",
                {
                    "subtask_id": subtask.id,
                    "verdict": policy.verdict,
                    "reason": policy.reason,
                    "missing_capabilities": list(policy.missing_capabilities),
                },
            )
            if not policy.allowed:
                # A missing capability or human approval is a planning stop, not a failed
                # executor attempt. Preserve the round in the trace but refund attempt budgets.
                criterion_state.status = "pending"
                criterion_state.attempts -= 1
                criterion_state.tier_attempts[tier] -= 1
                if criterion_state.tier_attempts[tier] == 0:
                    criterion_state.tier_attempts.pop(tier)
                if criterion_state.tier_history:
                    criterion_state.tier_history.pop()
                if state.graph and subtask.id in state.graph.nodes:
                    state.graph.nodes[subtask.id].status = "blocked"
                    state.graph.nodes[subtask.id].metadata["policy_reason"] = policy.reason
                state.current_subtask_id = None
                state.current_executor_tier = None
                state.status = "waiting_input" if policy.verdict == "ask" else "blocked"
                state.blocker = policy.reason
                sync_criterion_projection(state)
                self.store.save_state(state)
                self.store.append_event(
                    "policy_blocked_execution",
                    {"subtask_id": subtask.id, "verdict": policy.verdict, "reason": policy.reason},
                )
                return self.store.load_state()

            repair_seed = (
                Path(criterion_state.repair_seed_path)
                if criterion_state.repair_seed_path
                else None
            )
            begin_repair = getattr(self.transactions, "begin_repair", None)
            try:
                if (
                    subtask.repair_context
                    and repair_seed is not None
                    and repair_seed.is_dir()
                    and callable(begin_repair)
                ):
                    transaction = begin_repair(
                        workspace,
                        seed_workspace=repair_seed,
                        round_number=state.round,
                    )
                    criterion_state.repair_attempts += 1
                    state.repair_attempts += 1
                    self.store.append_event(
                        "repair_attempt_started",
                        {
                            "subtask_id": subtask.id,
                            "criterion_id": subtask.criterion_id,
                            "failure_fingerprint": subtask.repair_context.get(
                                "failure_fingerprint"
                            ),
                            "seed_workspace": str(repair_seed),
                        },
                    )
                else:
                    transaction = self.transactions.begin(
                        workspace, round_number=state.round
                    )
            except OSError as error:
                fault = self.fault_classifier.environment_failure(
                    code="candidate_prepare_failed", summary=str(error), subtask=subtask
                )
                self._handle_fault(contract, state, subtask, fault, [])
                if state.status in {"blocked", "waiting_input"}:
                    return self.store.load_state()
                continue
            criterion_state.candidate_id = transaction.candidate_id
            self._graph_add_candidate(state, subtask, transaction.candidate_id)
            environment_context = dict(getattr(transaction, "context", {}) or {})
            if environment_context:
                subtask = replace(subtask, environment_context=environment_context)
                if state.graph and subtask.id in state.graph.nodes:
                    state.graph.nodes[subtask.id].metadata["environment"] = _redact_sensitive(
                        environment_context
                    )
                self.store.append_event(
                    "environment_bound",
                    {
                        "subtask_id": subtask.id,
                        "candidate_id": transaction.candidate_id,
                        "context": _redact_sensitive(environment_context),
                    },
                )
            self.store.update_transaction(
                transaction.candidate_id,
                phase="prepared",
                criterion_id=criterion.id,
                subtask_id=subtask.id,
                round=state.round,
                executor_tier=tier,
                subtask=_safe_subtask_dict(subtask),
                environment_context=_redact_sensitive(environment_context),
            )
            self.store.save_state(state)
            self.store.append_event(
                "candidate_prepared",
                {
                    "subtask_id": subtask.id,
                    "candidate_id": transaction.candidate_id,
                    "candidate_workspace": str(transaction.candidate_workspace),
                },
            )

            self._enter_control_level(
                state,
                "L0",
                "Control returned to the native executor",
                subtask_id=subtask.id,
            )
            executor_result = self._execute_backend(backend, transaction, contract, subtask)
            executor_evidence = self._record_backend_result(state.round, "executor", executor_result)
            self.store.update_transaction(
                transaction.candidate_id,
                phase="executed" if executor_result.ok else "executor_failed",
                executor_evidence=executor_evidence,
                executor_report=executor_result.report,
            )
            self._remember_episode(
                "executor_report",
                str(executor_result.report.get("summary", "executor returned no summary")),
                source=f"executor:{tier}",
                metadata={
                    "subtask_id": subtask.id,
                    "ok": executor_result.ok,
                    "evidence": executor_evidence,
                },
            )
            if not executor_result.ok:
                fault = self.fault_classifier.backend_failure(
                    executor_result, role="executor", subtask=subtask
                )
                transaction.rollback()
                self.store.update_transaction(
                    transaction.candidate_id,
                    phase="rolled_back",
                    failure="executor_failed",
                )
                self._handle_fault(contract, state, subtask, fault, [executor_evidence])
                if state.status in {"blocked", "waiting_input"}:
                    return self.store.load_state()
                continue
            if isinstance(executor_result.report.get("fault"), dict):
                fault = self.fault_classifier.reported_failure(
                    executor_result.report,
                    subtask=subtask,
                    evidence=[executor_evidence],
                )
                transaction.rollback()
                self.store.update_transaction(
                    transaction.candidate_id,
                    phase="rolled_back",
                    failure="reported_executor_fault",
                )
                self._handle_fault(
                    contract,
                    state,
                    subtask,
                    fault,
                    [executor_evidence],
                )
                if state.status in {"blocked", "waiting_input"}:
                    return self.store.load_state()
                continue

            try:
                paths = transaction.diff()
            except UnsafeCandidatePathError as error:
                fault = self.fault_classifier.scope_violation(
                    subtask,
                    [f"unsafe-candidate-path:{error}"],
                    [executor_evidence],
                )
                transaction.rollback()
                self.store.update_transaction(
                    transaction.candidate_id,
                    phase="rolled_back",
                    failure="unsafe_candidate_path",
                )
                self._handle_fault(contract, state, subtask, fault, [executor_evidence])
                return self.store.load_state()
            except OSError as error:
                transaction.rollback()
                self.store.update_transaction(
                    transaction.candidate_id,
                    phase="rolled_back",
                    failure="environment_inspect_failed",
                )
                fault = self.fault_classifier.environment_failure(
                    code="environment_inspect_failed",
                    summary=str(error),
                    subtask=subtask,
                )
                self._handle_fault(contract, state, subtask, fault, [executor_evidence])
                if state.status in {"blocked", "waiting_input"}:
                    return self.store.load_state()
                continue
            inspected_context = dict(getattr(transaction, "context", {}) or {})
            if inspected_context:
                subtask = replace(
                    subtask,
                    environment_context=_redact_sensitive(inspected_context),
                )
                self.store.append_event(
                    "environment_inspected",
                    {
                        "subtask_id": subtask.id,
                        "candidate_id": transaction.candidate_id,
                        "context": _redact_sensitive(inspected_context),
                    },
                )
            self._enter_control_level(
                state,
                "L1",
                "Run deterministic scope, command, and environment checks",
                subtask_id=subtask.id,
            )
            outcome = self.verifier_mesh.verify(
                transaction.candidate_workspace,
                contract,
                state,
                subtask=subtask,
                changed_paths=paths,
                executor_report=executor_result.report,
            )
            self._update_proof_obligations(
                state,
                criterion_state,
                outcome,
                round_number=state.round,
            )
            if outcome.auditor_invoked:
                self._enter_control_level(
                    state,
                    "L2",
                    outcome.escalation_reason or "Semantic evidence requires independent review",
                    subtask_id=subtask.id,
                )
            verification_evidence = self._record_verification(
                state, state.round, subtask, outcome, executor_result.report
            )
            self._graph_add_evidence(
                state,
                owner_id=subtask.id,
                evidence_path=verification_evidence,
            )
            evidence = [executor_evidence, verification_evidence]
            if not outcome.passed:
                fault = self._fault_from_outcome(subtask, outcome, evidence)
                repair_packet = build_repair_packet(
                    subtask,
                    outcome,
                    changed_paths=paths,
                )
                criterion_state.repair_packet = repair_packet
                criterion_state.repair_seed_path = None
                state.repair_packets_created += 1
                seed_preserved = bool(
                    outcome.scope.passed
                    and callable(getattr(self.transactions, "begin_repair", None))
                    and transaction.candidate_workspace.resolve() != workspace.resolve()
                )
                if seed_preserved:
                    transaction.close(preserve=True)
                    criterion_state.repair_seed_path = str(
                        transaction.candidate_workspace.resolve()
                    )
                else:
                    transaction.rollback()
                self.store.update_transaction(
                    transaction.candidate_id,
                    phase=(
                        "repair_seed_preserved"
                        if seed_preserved
                        else "rolled_back"
                    ),
                    failure="verification_failed",
                    verification_evidence=verification_evidence,
                    changed_paths=paths,
                    repair_packet=repair_packet,
                    repair_seed_path=criterion_state.repair_seed_path,
                )
                self.store.append_event(
                    "repair_packet_created",
                    {
                        "subtask_id": subtask.id,
                        "criterion_id": subtask.criterion_id,
                        "failure_fingerprint": repair_packet["failure_fingerprint"],
                        "failed_verifiers": [
                            item["id"]
                            for item in repair_packet["failed_verifiers"]
                        ],
                        "seed_preserved": seed_preserved,
                        "evidence": verification_evidence,
                    },
                )
                self._handle_fault(contract, state, subtask, fault, evidence)
                if state.status in {"blocked", "waiting_input"}:
                    return self.store.load_state()
                continue

            self.store.update_transaction(
                transaction.candidate_id,
                phase="verified",
                subtask=_safe_subtask_dict(subtask),
                changed_paths=paths,
                evidence=list(evidence),
                verification_evidence=verification_evidence,
                environment_context=_redact_sensitive(inspected_context),
            )

            formal_snapshot = snapshot_tree(workspace)
            candidate_snapshot = snapshot_tree(transaction.candidate_workspace)
            self.store.update_transaction(
                transaction.candidate_id,
                phase="promotion_started",
                changed_paths=paths,
                base_hashes={
                    path: formal_snapshot.get(path) for path in paths
                },
                expected_hashes={
                    path: candidate_snapshot.get(path) for path in paths
                },
                external_environment=(
                    "external_environment" in self.environment_capabilities
                ),
            )
            if self.cancellation:
                self.cancellation.check()
            promotion = transaction.promote(paths)
            promotion_evidence = self.store.write_evidence(
                f"round-{state.round:04d}/promotion.json",
                {"candidate_id": transaction.candidate_id, **promotion.to_dict()},
            )
            self.store.append_event(
                "candidate_promotion",
                {
                    "subtask_id": subtask.id,
                    "candidate_id": transaction.candidate_id,
                    **promotion.to_dict(),
                    "evidence": promotion_evidence,
                },
            )
            evidence.append(promotion_evidence)
            self.store.update_transaction(
                transaction.candidate_id,
                phase="promoted" if promotion.passed else "promotion_failed",
                promotion=promotion.to_dict(),
                promotion_evidence=promotion_evidence,
                evidence=list(evidence),
            )
            self._graph_add_evidence(
                state,
                owner_id=subtask.id,
                evidence_path=promotion_evidence,
            )
            if not promotion.passed:
                fault = self.fault_classifier.environment_failure(
                    code="promotion_failed", summary=promotion.summary, subtask=subtask
                )
                transaction.close(preserve=True)
                self._handle_fault(contract, state, subtask, fault, evidence)
                if state.status in {"blocked", "waiting_input"}:
                    return self.store.load_state()
                continue

            criterion_state.status = "verified"
            criterion_state.last_error_type = None
            criterion_state.last_error = None
            criterion_state.last_recovery_action = None
            criterion_state.next_tier = None
            criterion_state.verified_at = utc_now()
            criterion_state.evidence = evidence
            if subtask.repair_context:
                criterion_state.repairs_succeeded += 1
                state.repairs_succeeded += 1
                self.store.append_event(
                    "repair_verified",
                    {
                        "subtask_id": subtask.id,
                        "criterion_id": subtask.criterion_id,
                        "failure_fingerprint": subtask.repair_context.get(
                            "failure_fingerprint"
                        ),
                        "verification_evidence": verification_evidence,
                        "regression_checks": list(subtask.checks),
                    },
                )
            criterion_state.repair_packet = None
            criterion_state.repair_seed_path = None
            for item in inspected_context.get("evidence", []):
                if isinstance(item, dict):
                    safe_item = _redact_sensitive(item)
                    if safe_item not in state.environment_evidence:
                        state.environment_evidence.append(safe_item)
            candidate_id = criterion_state.candidate_id
            criterion_state.candidate_id = None
            state.current_subtask_id = None
            state.current_executor_tier = None
            if state.graph:
                subtask_node = state.graph.nodes.get(subtask.id)
                if subtask_node:
                    subtask_node.status = "completed"
                    subtask_node.trust = "trusted"
                artifact_node = state.graph.nodes.get(f"ARTIFACT-{candidate_id}")
                if artifact_node:
                    artifact_node.status = "verified"
                    artifact_node.trust = "trusted"
                    artifact_node.metadata["promotion"] = "promoted"
                for path in evidence:
                    self._graph_add_evidence(state, owner_id=criterion.id, evidence_path=path)
                state.graph.mark_verified(criterion.id, evidence)
                for edge in state.graph.edges:
                    if edge.active and edge.source == criterion.id and edge.relation == "satisfies":
                        state.graph.mark_verified(edge.target, evidence)
            sync_criterion_projection(state)
            checkpoint = self.store.checkpoint_state(state, label=f"verified-{criterion.id}")
            criterion_state.evidence.append(checkpoint)
            self._graph_add_evidence(state, owner_id=criterion.id, evidence_path=checkpoint, kind="checkpoint")
            if state.graph:
                state.graph.mark_verified(criterion.id, criterion_state.evidence)
            sync_criterion_projection(state)
            self._enter_control_level(
                state,
                "L0",
                "Verified checkpoint committed; resume native execution",
                subtask_id=subtask.id,
            )
            self.store.save_state(state)
            self.store.update_transaction(
                transaction.candidate_id,
                phase="state_committed",
                evidence=list(criterion_state.evidence),
                committed_at=utc_now(),
            )
            trusted_memory_id = self._promote_memory(
                "verified_requirement",
                criterion.description,
                source=subtask.id,
                evidence=list(criterion_state.evidence),
                metadata={"criterion_id": criterion.id, "executor_tier": tier},
            )
            self.store.append_event(
                "criterion_verified",
                {
                    "subtask_id": subtask.id,
                    "criterion_id": criterion.id,
                    "executor_tier": tier,
                    "candidate_id": transaction.candidate_id,
                    "evidence": criterion_state.evidence,
                    "trusted_memory_id": trusted_memory_id,
                },
            )
            transaction.close()

    @staticmethod
    def _execute_backend(
        backend,
        transaction: WorkspaceTransaction,
        contract: TaskContract,
        subtask: Subtask,
    ) -> BackendResult:
        try:
            return backend.execute(transaction.candidate_workspace, contract, subtask)
        except Cancelled:
            raise
        except Exception as error:  # Backend calls are a fault boundary.
            return BackendResult(
                ok=False,
                report={},
                stdout="",
                stderr=f"Backend raised {type(error).__name__}: {error}",
                return_code=None,
                duration_seconds=0.0,
            )

    def _fault_from_outcome(
        self,
        subtask: Subtask,
        outcome: VerificationOutcome,
        evidence: list[str],
    ) -> FaultEvent:
        if not outcome.scope.passed:
            return self.fault_classifier.scope_violation(subtask, outcome.scope.violations, evidence)
        failed_checks = [
            _compact_check_failure(item)
            for item in outcome.commands
            if not item.passed
        ]
        if failed_checks:
            return self.fault_classifier.check_failure(subtask, failed_checks, evidence)
        failing = next(
            (item for item in outcome.results if item.required and item.verdict != "pass"),
            None,
        )
        verdict = failing.verdict if failing else outcome.verdict
        summary = failing.summary if failing else "Verification did not pass"
        if failing and failing.verifier_id != "semantic-auditor":
            return self.fault_classifier.verifier_failure(
                subtask,
                code=failing.fault_code or "verification_rejected",
                summary=summary,
                evidence=evidence,
            )
        return self.fault_classifier.audit_failure(subtask, verdict, summary, evidence)

    def _update_proof_obligations(
        self,
        state: TaskState,
        criterion_state,
        outcome: VerificationOutcome,
        *,
        round_number: int,
    ) -> None:
        for result in outcome.results:
            if result.verifier_id == "semantic-contract-probe" and result.details.get(
                "applicable"
            ):
                state.semantic_probe_batches += 1
            if not result.required and result.verdict in {"skipped", "pass"}:
                continue
            criterion_state.proof_obligations[result.verifier_id] = {
                "status": result.verdict,
                "required": result.required,
                "summary": result.summary,
                "evidence": list(result.evidence),
                "details": dict(result.details),
                "round": round_number,
                "verified_at": utc_now(),
            }

    def _handle_fault(
        self,
        contract: TaskContract,
        state: TaskState,
        subtask: Subtask,
        fault: FaultEvent,
        evidence: list[str],
    ) -> None:
        criterion_state = state.criteria[subtask.criterion_id]
        candidate_id = criterion_state.candidate_id
        tier_attempts = criterion_state.tier_attempts.get(subtask.executor_tier, 0)
        decision = self.safety_belt.decide(
            fault,
            criterion_state,
            current_tier=subtask.executor_tier,
            allowed_tiers=contract.executor_tiers,
            attempts_remaining=tier_attempts < contract.max_attempts_per_criterion,
            recovery_budget_remaining=state.recovery_count < contract.max_total_recoveries,
        )
        register_fault(criterion_state, fault, decision)
        criterion_state.status = "failed"
        criterion_state.evidence = list(evidence)
        criterion_state.candidate_id = None
        state.current_subtask_id = None
        state.current_executor_tier = None
        state.recovery_count += 1
        self._graph_add_fault(
            state,
            subtask,
            fault,
            decision,
            candidate_id=candidate_id,
        )
        sync_criterion_projection(state)
        fault_record = {**fault.to_dict(), "recovery": decision.to_dict()}
        self.store.append_fault(fault_record)
        self.store.append_event(
            "criterion_verification_failed",
            {
                "criterion_id": subtask.criterion_id,
                "attempt": criterion_state.attempts,
                "error_type": criterion_state.last_error_type,
                "error": fault.summary,
                "evidence": evidence,
            },
        )
        self.store.append_event("fault_classified", fault_record)
        self._remember_episode(
            "fault",
            fault.summary,
            source=fault.source_role,
            metadata=fault_record,
        )
        if decision.action == "replan":
            criterion_state.replan_count += 1
            criterion_state.tier_attempts[subtask.executor_tier] = 0
            criterion_state.next_tier = subtask.executor_tier
            self._enter_control_level(
                state,
                "L3",
                decision.reason,
                subtask_id=subtask.id,
            )
            self.store.append_event(
                "subtask_replan_requested",
                {
                    "subtask_id": subtask.id,
                    "criterion_id": subtask.criterion_id,
                    "replan_count": criterion_state.replan_count,
                    "reason": decision.reason,
                },
            )
        self.store.append_event(
            "recovery_decided",
            {"fault_id": fault.id, "subtask_id": subtask.id, **decision.to_dict()},
        )
        if decision.action == "escalate":
            self.store.append_event(
                "executor_tier_escalated",
                {
                    "subtask_id": subtask.id,
                    "from_tier": subtask.executor_tier,
                    "to_tier": decision.next_tier,
                    "fault_id": fault.id,
                    "reason": decision.reason,
                },
            )
        if decision.action in CONTINUE_ACTIONS:
            state.status = "running"
            state.blocker = None
            self.store.save_state(state)
            if decision.backoff_seconds > 0:
                delay = min(decision.backoff_seconds, 30)
                self.store.append_event(
                    "recovery_backoff",
                    {"fault_id": fault.id, "seconds": delay},
                )
                self.sleep_fn(delay)
            return
        if decision.action == "ask":
            state.status = "waiting_input"
            state.blocker = decision.reason
            self.store.save_state(state)
            return
        # Rollback is terminal for a policy violation until the user revises scope/permissions.
        self._block(
            state,
            f"{fault.category}:{fault.code}: {fault.summary}; {decision.reason}",
        )

    def _record_verification(
        self,
        state: TaskState,
        round_number: int,
        subtask: Subtask | None,
        outcome: VerificationOutcome,
        executor_report,
    ) -> str:
        state.deterministic_check_batches += 1
        label = "final" if subtask is None else f"round-{round_number:04d}"
        payload = {
            "subtask": _safe_subtask_dict(subtask) if subtask else None,
            **outcome.to_dict(),
            "executor_report": executor_report,
        }
        path = self.store.write_evidence(f"{label}/verification.json", payload)
        self.store.append_event(
            "deterministic_verification",
            {
                "subtask_id": subtask.id if subtask else None,
                "passed": outcome.scope.passed and all(item.passed for item in outcome.commands),
                "scope": outcome.scope.to_dict(),
                "checks": [
                    {
                        "command": item.command,
                        "passed": item.passed,
                        "exit_code": item.exit_code,
                        "timed_out": item.timed_out,
                    }
                    for item in outcome.commands
                ],
                "evidence": path,
            },
        )
        for result in outcome.results:
            self.store.append_event(
                "verifier_completed",
                {
                    "subtask_id": subtask.id if subtask else None,
                    **result.to_dict(),
                    "evidence_path": path,
                },
            )
        if outcome.auditor_backend_result is not None:
            self._record_backend_result(
                round_number if subtask else 0,
                "auditor" if subtask else "auditor-final",
                outcome.auditor_backend_result,
            )
        if outcome.auditor_invoked and outcome.audit is not None:
            state.auditor_calls += 1
            audit_path = self.store.write_evidence(
                f"{label}/audit.json", outcome.audit.to_dict()
            )
            self.store.append_event(
                "audit_completed",
                {
                    "subtask_id": subtask.id if subtask else None,
                    "verdict": outcome.audit.verdict,
                    "summary": outcome.audit.summary,
                    "reason": outcome.escalation_reason,
                    "evidence": [path, audit_path],
                },
            )
        else:
            self.store.append_event(
                "audit_skipped",
                {
                    "subtask_id": subtask.id if subtask else None,
                    "deterministic_verdict": outcome.deterministic_verdict,
                    "reason": (
                        "hard deterministic failure"
                        if outcome.deterministic_verdict == "fail"
                        else "deterministic evidence is conclusive"
                    ),
                    "evidence": path,
                },
            )
        return path

    def _finalize(self, contract: TaskContract, state: TaskState, workspace: Path) -> None:
        covered, missing = goal_coverage(contract, state)
        if not covered:
            self._block(state, "Finalization lacks goal coverage: " + ", ".join(missing))
            return
        self._enter_control_level(
            state,
            "L1",
            "Run final deterministic verification",
        )
        # Criterion-level semantic evidence is already trusted at this point.
        # Finalization is a global deterministic regression pass, not a duplicate
        # semantic audit over an unchanged checkpoint.
        final_contract = replace(contract, default_verification_profile="default")
        outcome = self.verifier_mesh.verify(
            workspace,
            final_contract,
            state,
            subtask=None,
            changed_paths=[],
            executor_report=None,
        )
        if outcome.auditor_invoked:
            self._enter_control_level(
                state,
                "L2",
                outcome.escalation_reason or "Final semantic evidence requires review",
            )
        verification_path = self._record_verification(state, 0, None, outcome, None)
        audit_path = (
            "evidence/final/audit.json" if outcome.auditor_invoked else None
        )
        goal_node = next(
            (node for node in state.graph.nodes.values() if node.kind == "goal"),
            None,
        ) if state.graph else None
        if goal_node:
            self._graph_add_evidence(
                state,
                owner_id=goal_node.id,
                evidence_path=verification_path,
            )
            if audit_path:
                self._graph_add_evidence(
                    state,
                    owner_id=goal_node.id,
                    evidence_path=audit_path,
                )
        if not outcome.passed:
            for criterion in contract.acceptance_criteria:
                criterion_state = state.criteria[criterion.id]
                criterion_state.status = "needs_revalidation"
                criterion_state.verified_at = None
            if state.graph:
                state.graph.invalidate(
                    [item.id for item in contract.acceptance_criteria],
                    reason="Final global verification failed",
                )
            state.status = "paused"
            state.blocker = "Final Verifier Mesh rejected completion"
            sync_criterion_projection(state)
            self.store.save_state(state)
            self.store.append_event(
                "final_verification_failed",
                {
                    "reason": state.blocker,
                    "verdict": outcome.verdict,
                    "evidence": verification_path,
                },
            )
            return
        state.status = "completed"
        state.blocker = None
        state.completed_at = utc_now()
        state.final_evidence = [verification_path]
        if audit_path:
            state.final_evidence.append(audit_path)
        self._enter_control_level(
            state,
            "L0",
            "Final verification passed; completion is trusted",
        )
        if goal_node:
            goal_node.status = "completed"
            goal_node.trust = "trusted"
            goal_node.evidence = list(state.final_evidence)
        checkpoint = self.store.checkpoint_state(state, label="completed")
        state.final_evidence.append(checkpoint)
        if goal_node:
            self._graph_add_evidence(
                state,
                owner_id=goal_node.id,
                evidence_path=checkpoint,
                kind="checkpoint",
            )
            goal_node.evidence = list(state.final_evidence)
        sync_criterion_projection(state)
        self.store.save_state(state)
        self.store.append_event(
            "goal_completed",
            {"contract_version": contract.version, "evidence": state.final_evidence},
        )

    def _recover_interrupted_state(
        self,
        contract: TaskContract,
        state: TaskState,
        workspace: Path,
    ) -> bool:
        interrupted = [
            criterion_id
            for criterion_id, item in state.criteria.items()
            if item.status == "in_progress"
        ]
        if state.status != "running" and not interrupted:
            return True
        recovered_promotions: list[str] = []
        ambiguous_promotions: list[str] = []
        for criterion_id in interrupted:
            item = state.criteria[criterion_id]
            journal = (
                self.store.load_transaction(item.candidate_id)
                if item.candidate_id
                else None
            )
            if journal and journal.get("phase") in {
                "promoted",
                "promotion_reconciled_committed",
                "committed_recovery_verification_failed",
            }:
                recovered = self._recover_promoted_transaction(
                    contract,
                    state,
                    workspace,
                    criterion_id=criterion_id,
                    journal=journal,
                )
                if recovered:
                    recovered_promotions.append(criterion_id)
                    continue
                if journal.get("external_environment"):
                    item.last_error_type = "COMMITTED_RECOVERY_VERIFICATION_FAILED"
                    item.last_error = (
                        "The external action is committed, but recovery verification did not pass"
                    )
                    item.last_recovery_action = "inspect_or_fix_verifier"
                    ambiguous_promotions.append(criterion_id)
                    continue
            if journal and journal.get("phase") in {
                "promotion_started",
                "promotion_reconciliation_required",
            }:
                disposition, summary, reconciled_journal = self._reconcile_started_promotion(
                    journal,
                    workspace,
                )
                if disposition == "committed":
                    recovered = self._recover_promoted_transaction(
                        contract,
                        state,
                        workspace,
                        criterion_id=criterion_id,
                        journal=reconciled_journal,
                    )
                    if recovered:
                        recovered_promotions.append(criterion_id)
                        continue
                    if reconciled_journal.get("external_environment"):
                        item.last_error_type = "COMMITTED_RECOVERY_VERIFICATION_FAILED"
                        item.last_error = (
                            "The external action is committed, but recovery verification did not pass"
                        )
                        item.last_recovery_action = "inspect_or_fix_verifier"
                        ambiguous_promotions.append(criterion_id)
                        continue
                elif disposition == "unknown":
                    item.last_error_type = "CONTROL_PLANE_COMMIT_AMBIGUOUS"
                    item.last_error = summary
                    item.last_recovery_action = "reconcile_or_ask"
                    ambiguous_promotions.append(criterion_id)
                    continue
            item.status = "failed"
            item.last_error_type = "CONTROL_PLANE_INTERRUPTED"
            item.last_error = "Previous run ended while the subtask was in progress"
            item.candidate_id = None
        if ambiguous_promotions:
            state.status = "waiting_input"
            state.blocker = (
                "Commit outcome is ambiguous for "
                + ", ".join(ambiguous_promotions)
                + "; inspect the transaction journal/environment before retrying. "
                "The Harness will not replay the action automatically."
            )
            self.store.save_state(state)
            self.store.append_event(
                "promotion_reconciliation_required",
                {
                    "criteria": ambiguous_promotions,
                    "reason": state.blocker,
                },
            )
            return False
        state.status = "ready"
        state.blocker = None
        state.current_subtask_id = None
        state.current_executor_tier = None
        self.store.save_state(state)
        self.store.append_event(
            "run_recovered",
            {
                "interrupted_criteria": interrupted,
                "recovered_promotions": recovered_promotions,
                "reason": "Recovered in-flight state",
            },
        )
        return True

    def _reconcile_started_promotion(
        self,
        journal: dict,
        workspace: Path,
    ) -> tuple[str, str, dict]:
        candidate_id = str(journal.get("candidate_id", ""))
        changed_paths = list(journal.get("changed_paths", []))
        base_hashes = journal.get("base_hashes")
        expected_hashes = journal.get("expected_hashes")
        if (
            not candidate_id
            or not changed_paths
            or not isinstance(base_hashes, dict)
            or not isinstance(expected_hashes, dict)
            or any(path not in base_hashes or path not in expected_hashes for path in changed_paths)
        ):
            summary = "Promotion journal lacks the hashes required for safe reconciliation"
            updated = (
                self.store.update_transaction(
                    candidate_id,
                    reconciliation={"status": "unknown", "summary": summary},
                )
                if candidate_id
                else dict(journal)
            )
            return "unknown", summary, updated

        current = snapshot_tree(workspace)
        matches_expected = all(
            current.get(path) == expected_hashes.get(path) for path in changed_paths
        )
        matches_base = all(
            current.get(path) == base_hashes.get(path) for path in changed_paths
        )
        external = bool(journal.get("external_environment"))
        reconciliation: dict = {
            "status": "local_only",
            "summary": "Compared formal workspace with durable base/candidate hashes",
        }
        if external:
            reconcile = getattr(self.transactions, "reconcile", None)
            if not callable(reconcile):
                reconciliation = {
                    "status": "unknown",
                    "summary": "External environment adapter has no reconciliation operation",
                }
            else:
                try:
                    response = reconcile(journal, workspace)
                except Exception as error:  # Reconciliation is a fail-closed boundary.
                    response = {
                        "status": "unknown",
                        "summary": f"External reconciliation failed: {type(error).__name__}: {error}",
                    }
                reconciliation = (
                    _redact_sensitive(response)
                    if isinstance(response, dict)
                    else {"status": "unknown", "summary": "Invalid reconciliation response"}
                )

        external_status = str(reconciliation.get("status", "unknown"))
        if (not external or external_status == "committed") and matches_expected:
            updated = self.store.update_transaction(
                candidate_id,
                phase="promotion_reconciled_committed",
                reconciliation=reconciliation,
            )
            return "committed", "Promotion was reconciled as committed", updated
        if (
            (not external or external_status in {"not_committed", "rolled_back"})
            and matches_base
        ):
            updated = self.store.update_transaction(
                candidate_id,
                phase="promotion_reconciled_not_committed",
                reconciliation=reconciliation,
            )
            return "not_committed", "Promotion was reconciled as not committed", updated

        summary = (
            "Promotion state is ambiguous: formal workspace and external controller "
            "do not provide one consistent committed/not-committed result"
        )
        updated = self.store.update_transaction(
            candidate_id,
            phase="promotion_reconciliation_required",
            reconciliation={**reconciliation, "summary": summary},
        )
        return "unknown", summary, updated

    def _recover_promoted_transaction(
        self,
        contract: TaskContract,
        state: TaskState,
        workspace: Path,
        *,
        criterion_id: str,
        journal: dict,
    ) -> bool:
        subtask_payload = journal.get("subtask")
        if not isinstance(subtask_payload, dict):
            return False
        try:
            subtask = Subtask(**subtask_payload)
        except TypeError:
            return False
        self._enter_control_level(
            state,
            "L1",
            "Revalidate a promoted transaction after recovery",
            subtask_id=subtask.id,
        )
        outcome = self.verifier_mesh.verify(
            workspace,
            contract,
            state,
            subtask=subtask,
            changed_paths=list(journal.get("changed_paths", [])),
            executor_report=journal.get("executor_report"),
        )
        if outcome.auditor_invoked:
            self._enter_control_level(
                state,
                "L2",
                outcome.escalation_reason or "Recovered evidence requires semantic review",
                subtask_id=subtask.id,
            )
        recovery_evidence = self._record_verification(
            state,
            state.round,
            subtask,
            outcome,
            journal.get("executor_report"),
        )
        if not outcome.passed:
            self.store.update_transaction(
                journal["candidate_id"],
                phase=(
                    "committed_recovery_verification_failed"
                    if journal.get("external_environment")
                    else "recovery_verification_failed"
                ),
                recovery_evidence=recovery_evidence,
            )
            return False
        criterion_state = state.criteria[criterion_id]
        evidence = list(dict.fromkeys([
            *journal.get("evidence", []),
            recovery_evidence,
        ]))
        criterion_state.status = "verified"
        criterion_state.verified_at = utc_now()
        criterion_state.evidence = evidence
        criterion_state.last_error_type = None
        criterion_state.last_error = None
        criterion_state.last_recovery_action = "recover_promoted"
        criterion_state.candidate_id = None
        if state.graph:
            state.graph.mark_verified(criterion_id, evidence)
            for edge in state.graph.edges:
                if edge.active and edge.source == criterion_id and edge.relation == "satisfies":
                    state.graph.mark_verified(edge.target, evidence)
            for path in evidence:
                self._graph_add_evidence(
                    state,
                    owner_id=criterion_id,
                    evidence_path=path,
                )
        environment_context = journal.get("environment_context", {})
        if isinstance(environment_context, dict):
            for evidence_item in environment_context.get("evidence", []):
                if isinstance(evidence_item, dict) and evidence_item not in state.environment_evidence:
                    state.environment_evidence.append(evidence_item)
        sync_criterion_projection(state)
        self.store.update_transaction(
            journal["candidate_id"],
            phase="state_committed_recovered",
            recovery_evidence=recovery_evidence,
            evidence=evidence,
            committed_at=utc_now(),
        )
        self.store.append_event(
            "promoted_transaction_recovered",
            {
                "candidate_id": journal["candidate_id"],
                "criterion_id": criterion_id,
                "subtask_id": subtask.id,
                "evidence": evidence,
            },
        )
        return True

    def _pause_after_control_failure(self, state: TaskState, error: OSError) -> TaskState:
        state.status = "paused"
        state.blocker = f"Recoverable control-plane error: {error}"
        self.store.save_state(state)
        self.store.append_event(
            "control_plane_failure",
            {
                "error": str(error),
                "traceback": traceback.format_exc(limit=8),
                "recovery": "resume from last durable state",
            },
        )
        return self.store.load_state()

    def _block(self, state: TaskState, reason: str) -> None:
        state.status = "blocked"
        state.blocker = reason
        self.store.save_state(state)
        self.store.append_event("goal_blocked", {"reason": reason})

    def _record_backend_result(
        self, round_number: int, role: str, result: BackendResult
    ) -> str:
        prefix = "final" if round_number == 0 else f"round-{round_number:04d}"
        payload = {
            "ok": result.ok,
            "return_code": result.return_code,
            "duration_seconds": result.duration_seconds,
            "report": result.report,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        path = self.store.write_evidence(f"{prefix}/{role}.json", payload)
        self.store.append_event(
            "backend_completed",
            {
                "role": role,
                "round": round_number,
                "ok": result.ok,
                "return_code": result.return_code,
                "duration_seconds": result.duration_seconds,
                "evidence": path,
            },
        )
        return path

    def _record_manager_backend_if_new(
        self,
        state: TaskState,
        decision: ManagerDecision,
    ) -> None:
        result = getattr(self.manager, "last_backend_result", None)
        if result is None or result is self._last_manager_backend_result:
            return
        self._last_manager_backend_result = result
        self._manager_evidence_sequence += 1
        state.manager_calls += 1
        run_label = state.run_id or "no-run"
        path = self.store.write_evidence(
            f"manager/{run_label}/decision-{self._manager_evidence_sequence:04d}.json",
            {
                "ok": result.ok,
                "return_code": result.return_code,
                "duration_seconds": result.duration_seconds,
                "report": result.report,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "accepted_decision": decision.to_dict(),
                "fallback_reason": getattr(self.manager, "last_fallback_reason", None),
            },
        )
        self.store.append_event(
            "manager_backend_completed",
            {
                "ok": result.ok,
                "fallback_reason": getattr(self.manager, "last_fallback_reason", None),
                "evidence": path,
            },
        )

    def _enter_control_level(
        self,
        state: TaskState,
        level: str,
        reason: str,
        *,
        subtask_id: str | None = None,
        force: bool = False,
    ) -> None:
        if level not in {"L0", "L1", "L2", "L3"}:
            raise ValueError(f"unknown control level: {level}")
        previous = state.control_level
        if previous == level and not force:
            return
        state.control_level = level
        state.control_level_counts[level] = state.control_level_counts.get(level, 0) + 1
        if level in {"L2", "L3"}:
            state.escalation_reasons.append(reason)
        self.store.append_event(
            "control_level_changed",
            {
                "from": previous,
                "to": level,
                "reason": reason,
                "subtask_id": subtask_id,
            },
        )

    @staticmethod
    def _graph_add_subtask(state: TaskState, subtask: Subtask) -> None:
        if not state.graph:
            return
        if subtask.id not in state.graph.nodes:
            add_node(
                state.graph,
                StateNode(
                    id=subtask.id,
                    kind="subtask",
                    title=subtask.criterion,
                    description=subtask.manager_reason,
                    status="in_progress",
                    trust="untrusted",
                    source_version=state.contract_version,
                    priority=2,
                    metadata={
                        "criterion_id": subtask.criterion_id,
                        "executor_tier": subtask.executor_tier,
                        "attempt": subtask.attempt,
                        "verification_profile": subtask.verification_profile,
                    },
                ),
            )
        add_edge(
            state.graph,
            StateEdge(source=subtask.id, relation="derived_from", target=subtask.criterion_id),
        )

    @staticmethod
    def _graph_add_candidate(state: TaskState, subtask: Subtask, candidate_id: str) -> None:
        if not state.graph:
            return
        artifact_id = f"ARTIFACT-{candidate_id}"
        if artifact_id not in state.graph.nodes:
            add_node(
                state.graph,
                StateNode(
                    id=artifact_id,
                    kind="artifact",
                    title=f"Candidate workspace {candidate_id}",
                    status="in_progress",
                    trust="untrusted",
                    source_version=state.contract_version,
                    metadata={"candidate_id": candidate_id},
                ),
            )
        add_edge(
            state.graph,
            StateEdge(source=artifact_id, relation="produced_by", target=subtask.id),
        )

    @staticmethod
    def _graph_add_evidence(
        state: TaskState,
        *,
        owner_id: str,
        evidence_path: str,
        kind: str = "evidence",
    ) -> None:
        if not state.graph or owner_id not in state.graph.nodes:
            return
        evidence_id = stable_node_id(kind.upper(), evidence_path)
        if evidence_id not in state.graph.nodes:
            add_node(
                state.graph,
                StateNode(
                    id=evidence_id,
                    kind=kind,
                    title=evidence_path,
                    description=evidence_path,
                    status="verified",
                    trust="trusted",
                    source_version=state.contract_version,
                    evidence=[evidence_path],
                ),
            )
        add_edge(
            state.graph,
            StateEdge(source=owner_id, relation="verified_by", target=evidence_id),
        )

    def _graph_add_fault(
        self,
        state: TaskState,
        subtask: Subtask,
        fault: FaultEvent,
        decision,
        *,
        candidate_id: str | None,
    ) -> None:
        if not state.graph:
            return
        subtask_node = state.graph.nodes.get(subtask.id)
        if subtask_node:
            subtask_node.status = "failed"
            subtask_node.metadata["fault_id"] = fault.id
        if candidate_id:
            artifact_node = state.graph.nodes.get(f"ARTIFACT-{candidate_id}")
            if artifact_node:
                if fault.code == "promotion_failed":
                    artifact_node.status = "failed"
                    artifact_node.metadata["promotion"] = "conflict_or_io_failure"
                else:
                    artifact_node.status = "cancelled"
                    artifact_node.metadata["rollback"] = True
        if fault.id not in state.graph.nodes:
            add_node(
                state.graph,
                StateNode(
                    id=fault.id,
                    kind="fault",
                    title=f"{fault.category}:{fault.code}",
                    description=fault.summary,
                    status="active",
                    trust="trusted",
                    source_version=state.contract_version,
                    priority=3,
                    evidence=list(fault.evidence),
                    metadata={
                        "fingerprint": fault.fingerprint,
                        "retryable": fault.retryable,
                        "scope": fault.scope,
                        "severity": fault.severity,
                    },
                ),
            )
        if subtask.id in state.graph.nodes:
            add_edge(
                state.graph,
                StateEdge(source=subtask.id, relation="failed_because", target=fault.id),
            )
        for evidence_path in fault.evidence:
            self._graph_add_evidence(
                state,
                owner_id=fault.id,
                evidence_path=evidence_path,
            )
        recovery_id = f"RECOVERY-{fault.id}"
        if recovery_id not in state.graph.nodes:
            add_node(
                state.graph,
                StateNode(
                    id=recovery_id,
                    kind="decision",
                    title=f"Recovery: {decision.action}",
                    description=decision.reason,
                    status="active",
                    trust="trusted",
                    source_version=state.contract_version,
                    priority=3,
                    metadata=decision.to_dict(),
                ),
            )
        add_edge(
            state.graph,
            StateEdge(source=recovery_id, relation="derived_from", target=fault.id),
        )

    def _remember_episode(
        self,
        kind: str,
        content: str,
        *,
        source: str,
        metadata: dict | None = None,
    ) -> None:
        try:
            self.memory.add_episode(
                kind,
                content,
                source=source,
                metadata=metadata,
            )
        except OSError as error:
            # Memory is a derived context service. Its outage must not terminate the worker.
            self.store.append_event(
                "auxiliary_component_failed",
                {"component": "episodic_memory", "error": str(error)},
            )

    def _promote_memory(
        self,
        kind: str,
        content: str,
        *,
        source: str,
        evidence: list[str],
        metadata: dict | None = None,
    ) -> str | None:
        try:
            return self.memory.promote(
                kind,
                content,
                source=source,
                evidence=evidence,
                metadata=metadata,
            ).id
        except OSError as error:
            self.store.append_event(
                "auxiliary_component_failed",
                {"component": "trusted_memory", "error": str(error)},
            )
            return None


SENSITIVE_CONTEXT_TOKENS = ("token", "password", "secret", "api_key", "authorization", "cookie")


def _compact_check_failure(result) -> str:
    detail = (result.stderr or result.stdout or "").strip().replace("\n", " ")
    if len(detail) > 600:
        detail = detail[:597] + "..."
    suffix = f"; output={detail}" if detail else ""
    return (
        f"{result.command}; exit={result.exit_code}; timed_out={result.timed_out}"
        f"{suffix}"
    )


def _redact_sensitive(value):
    if isinstance(value, dict):
        return {
            key: (
                "<redacted>"
                if any(token in str(key).lower() for token in SENSITIVE_CONTEXT_TOKENS)
                else _redact_sensitive(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def _safe_subtask_dict(subtask: Subtask) -> dict:
    payload = subtask.to_dict()
    payload["environment_context"] = _redact_sensitive(payload.get("environment_context", {}))
    return payload
