from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_node_id(prefix: str, content: str) -> str:
    digest = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()[:12].upper()
    return f"{prefix}-{digest}"


@dataclass(frozen=True)
class AcceptanceCriterion:
    id: str
    description: str
    kind: str = "requirement"
    priority: int = 2
    depends_on: list[str] = field(default_factory=list)
    verification_profile: str = "default"
    risk_level: str = "medium"
    source_version: int = 1
    recipe_id: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AcceptanceCriterion":
        return cls(
            id=value["id"],
            description=value["description"],
            kind=value.get("kind", "requirement"),
            priority=int(value.get("priority", 2)),
            depends_on=list(value.get("depends_on", [])),
            verification_profile=value.get("verification_profile", "default"),
            risk_level=value.get("risk_level", "medium"),
            source_version=int(value.get("source_version", 1)),
            recipe_id=value.get("recipe_id"),
        )


@dataclass
class ProductSpec:
    """Product intent carried beside the technical goal contract."""

    problem: str = ""
    target_users: list[str] = field(default_factory=list)
    key_flows: list[str] = field(default_factory=list)
    must_haves: list[str] = field(default_factory=list)
    visual_expectations: list[str] = field(default_factory=list)
    quality_attributes: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    required_human_gates: list[str] = field(default_factory=list)
    product_checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProductSpec":
        return cls(**value)


@dataclass
class TaskContract:
    goal_id: str
    version: int
    created_at: str
    updated_at: str
    objective: str
    acceptance_criteria: list[AcceptanceCriterion]
    constraints: list[str] = field(default_factory=list)
    non_goals: list[str] = field(default_factory=list)
    allowed_paths: list[str] = field(default_factory=lambda: ["**"])
    forbidden_paths: list[str] = field(default_factory=lambda: [".git/**"])
    checks: list[str] = field(default_factory=list)
    max_attempts_per_criterion: int = 2
    command_timeout_seconds: int = 300
    mode: str = "task"
    product: ProductSpec | None = None
    executor_tiers: list[str] = field(default_factory=lambda: ["E0", "E1", "E2", "E3"])
    default_verification_profile: str = "default"
    max_total_recoveries: int = 20
    deterministic_recipes: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        objective: str,
        acceptance: list[str],
        *,
        constraints: list[str] | None = None,
        non_goals: list[str] | None = None,
        allowed_paths: list[str] | None = None,
        forbidden_paths: list[str] | None = None,
        checks: list[str] | None = None,
        max_attempts: int = 2,
        command_timeout: int = 300,
        mode: str = "task",
        product: ProductSpec | None = None,
        max_total_recoveries: int = 20,
        deterministic_recipes: dict[str, list[str]] | None = None,
    ) -> "TaskContract":
        objective = objective.strip()
        if not objective:
            raise ValueError("objective must not be empty")
        cleaned = [item.strip() for item in acceptance if item.strip()]
        if not cleaned:
            raise ValueError("at least one acceptance criterion is required")
        normalized_checks = [item.strip() for item in (checks or []) if item.strip()]
        if not normalized_checks:
            raise ValueError("at least one deterministic verification command is required")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if command_timeout < 1:
            raise ValueError("command_timeout must be at least 1")
        if mode not in {"task", "product", "auto"}:
            raise ValueError("mode must be task, product, or auto")
        if max_total_recoveries < 1:
            raise ValueError("max_total_recoveries must be at least 1")
        now = utc_now()
        criteria = [
            AcceptanceCriterion(id=f"AC-{index:03d}", description=description)
            for index, description in enumerate(cleaned, start=1)
        ]
        return cls(
            goal_id=str(uuid4()),
            version=1,
            created_at=now,
            updated_at=now,
            objective=objective,
            acceptance_criteria=criteria,
            constraints=list(constraints or []),
            non_goals=list(non_goals or []),
            allowed_paths=list(allowed_paths or ["**"]),
            forbidden_paths=list(forbidden_paths or [".git/**"]),
            checks=normalized_checks,
            max_attempts_per_criterion=max_attempts,
            command_timeout_seconds=command_timeout,
            mode=mode,
            product=product,
            max_total_recoveries=max_total_recoveries,
            deterministic_recipes={
                key: list(commands)
                for key, commands in (deterministic_recipes or {}).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskContract":
        data = dict(value)
        data["acceptance_criteria"] = [
            AcceptanceCriterion.from_dict(item) for item in data["acceptance_criteria"]
        ]
        if isinstance(data.get("product"), dict):
            data["product"] = ProductSpec.from_dict(data["product"])
        return cls(**data)


@dataclass
class CriterionState:
    status: str = "pending"
    attempts: int = 0
    evidence: list[str] = field(default_factory=list)
    last_error_type: str | None = None
    last_error: str | None = None
    verified_at: str | None = None
    current_tier: str | None = None
    next_tier: str | None = None
    tier_history: list[str] = field(default_factory=list)
    tier_attempts: dict[str, int] = field(default_factory=dict)
    fault_fingerprints: list[str] = field(default_factory=list)
    last_recovery_action: str | None = None
    candidate_id: str | None = None
    replan_count: int = 0
    manager_replans_handled: int = 0
    proof_obligations: dict[str, dict[str, Any]] = field(default_factory=dict)
    repair_packet: dict[str, Any] | None = None
    repair_seed_path: str | None = None
    repair_attempts: int = 0
    repairs_succeeded: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CriterionState":
        return cls(**value)


@dataclass
class TaskState:
    contract_version: int
    status: str
    round: int
    criteria: dict[str, CriterionState]
    created_at: str
    updated_at: str
    completed_at: str | None = None
    blocker: str | None = None
    final_evidence: list[str] = field(default_factory=list)
    graph: "TaskGraph | None" = None
    current_subtask_id: str | None = None
    current_executor_tier: str | None = None
    recovery_count: int = 0
    run_id: str | None = None
    last_event_sequence: int = 0
    approved_gates: dict[str, dict[str, str]] = field(default_factory=dict)
    environment_evidence: list[dict[str, Any]] = field(default_factory=list)
    control_level: str = "L0"
    control_level_counts: dict[str, int] = field(
        default_factory=lambda: {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
    )
    deterministic_check_batches: int = 0
    auditor_calls: int = 0
    manager_calls: int = 0
    escalation_reasons: list[str] = field(default_factory=list)
    semantic_probe_batches: int = 0
    repair_packets_created: int = 0
    repair_attempts: int = 0
    repairs_succeeded: int = 0

    @classmethod
    def create(cls, contract: TaskContract) -> "TaskState":
        now = utc_now()
        return cls(
            contract_version=contract.version,
            status="ready",
            round=0,
            criteria={criterion.id: CriterionState() for criterion in contract.acceptance_criteria},
            created_at=now,
            updated_at=now,
            graph=TaskGraph.from_contract(contract),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskState":
        data = dict(value)
        data["criteria"] = {
            key: CriterionState.from_dict(item) for key, item in data["criteria"].items()
        }
        if isinstance(data.get("graph"), dict):
            data["graph"] = TaskGraph.from_dict(data["graph"])
        return cls(**data)


@dataclass
class StateNode:
    id: str
    kind: str
    title: str
    description: str = ""
    status: str = "pending"
    trust: str = "untrusted"
    source_version: int = 1
    priority: int = 2
    confidence: float | None = None
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StateNode":
        return cls(**value)


@dataclass(frozen=True)
class StateEdge:
    source: str
    relation: str
    target: str
    active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StateEdge":
        return cls(**value)


@dataclass
class TaskGraph:
    version: int
    nodes: dict[str, StateNode]
    edges: list[StateEdge]
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def from_contract(cls, contract: TaskContract) -> "TaskGraph":
        goal_node_id = f"GOAL-{contract.goal_id}"
        nodes: dict[str, StateNode] = {
            goal_node_id: StateNode(
                id=goal_node_id,
                kind="goal",
                title=contract.objective,
                description=contract.objective,
                status="active",
                trust="trusted",
                source_version=contract.version,
                priority=3,
            )
        }
        edges: list[StateEdge] = []
        for criterion in contract.acceptance_criteria:
            nodes[criterion.id] = StateNode(
                id=criterion.id,
                kind=criterion.kind,
                title=criterion.description,
                description=criterion.description,
                source_version=criterion.source_version,
                priority=criterion.priority,
                metadata={
                    "verification_profile": criterion.verification_profile,
                    "risk_level": criterion.risk_level,
                },
            )
            edges.append(StateEdge(source=criterion.id, relation="derived_from", target=goal_node_id))
            for dependency in criterion.depends_on:
                edges.append(StateEdge(source=criterion.id, relation="depends_on", target=dependency))
            if criterion.risk_level == "high":
                risk_id = stable_node_id("RISK", criterion.id)
                nodes[risk_id] = StateNode(
                    id=risk_id,
                    kind="risk",
                    title=f"High-risk requirement: {criterion.description}",
                    description=criterion.description,
                    status="active",
                    trust="trusted",
                    source_version=criterion.source_version,
                    priority=3,
                )
                edges.append(StateEdge(source=risk_id, relation="derived_from", target=criterion.id))
        for constraint in contract.constraints:
            constraint_id = stable_node_id("CONSTRAINT", constraint)
            nodes[constraint_id] = StateNode(
                id=constraint_id,
                kind="constraint",
                title=constraint,
                description=constraint,
                status="active",
                trust="trusted",
                source_version=contract.version,
                priority=3,
            )
            edges.append(StateEdge(source=constraint_id, relation="derived_from", target=goal_node_id))
        for non_goal in contract.non_goals:
            non_goal_id = stable_node_id("NON-GOAL", non_goal)
            nodes[non_goal_id] = StateNode(
                id=non_goal_id,
                kind="non_goal",
                title=non_goal,
                description=non_goal,
                status="active",
                trust="trusted",
                source_version=contract.version,
                priority=2,
            )
            edges.append(StateEdge(source=non_goal_id, relation="derived_from", target=goal_node_id))
        if contract.product:
            for flow in contract.product.key_flows:
                flow_id = stable_node_id("FLOW", flow)
                nodes[flow_id] = StateNode(
                    id=flow_id,
                    kind="product_flow",
                    title=flow,
                    description=flow,
                    source_version=contract.version,
                    priority=3,
                    metadata={"verification_profile": "product_flow"},
                )
                edges.append(StateEdge(source=flow_id, relation="derived_from", target=goal_node_id))
                matching = next(
                    (
                        criterion
                        for criterion in contract.acceptance_criteria
                        if criterion.verification_profile == "product_flow"
                        and flow.lower() in criterion.description.lower()
                    ),
                    None,
                )
                if matching:
                    edges.append(StateEdge(source=matching.id, relation="satisfies", target=flow_id))
            for question in contract.product.open_questions:
                question_id = stable_node_id("QUESTION", question)
                nodes[question_id] = StateNode(
                    id=question_id,
                    kind="question",
                    title=question,
                    description=question,
                    status="open",
                    source_version=contract.version,
                    priority=3,
                )
                edges.append(StateEdge(source=question_id, relation="derived_from", target=goal_node_id))
            for gate in contract.product.required_human_gates:
                gate_id = stable_node_id("HUMAN-GATE", gate)
                nodes[gate_id] = StateNode(
                    id=gate_id,
                    kind="human_gate",
                    title=gate,
                    description=gate,
                    status="open",
                    source_version=contract.version,
                    priority=3,
                )
                edges.append(StateEdge(source=gate_id, relation="derived_from", target=goal_node_id))
        return cls(version=contract.version, nodes=nodes, edges=edges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "nodes": {key: value.to_dict() for key, value in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskGraph":
        return cls(
            version=int(value["version"]),
            nodes={key: StateNode.from_dict(item) for key, item in value.get("nodes", {}).items()},
            edges=[StateEdge.from_dict(item) for item in value.get("edges", [])],
            updated_at=value.get("updated_at", utc_now()),
        )

    def dependencies_satisfied(self, node_id: str) -> bool:
        dependencies = [
            edge.target
            for edge in self.edges
            if edge.active and edge.source == node_id and edge.relation == "depends_on"
        ]
        return all(
            dependency in self.nodes and self.nodes[dependency].status in {"verified", "completed"}
            for dependency in dependencies
        )

    def mark_verified(self, node_id: str, evidence: list[str]) -> None:
        node = self.nodes[node_id]
        node.status = "verified"
        node.trust = "trusted"
        node.evidence = list(evidence)
        node.updated_at = utc_now()
        self.updated_at = node.updated_at

    def invalidate(self, node_ids: list[str], *, reason: str) -> None:
        for node_id in node_ids:
            node = self.nodes.get(node_id)
            if not node:
                continue
            node.status = "needs_revalidation"
            node.trust = "untrusted"
            node.evidence = []
            node.metadata["invalidation_reason"] = reason
            node.updated_at = utc_now()
        self.updated_at = utc_now()


@dataclass(frozen=True)
class Subtask:
    id: str
    round: int
    criterion_id: str
    criterion: str
    objective: str
    constraints: list[str]
    allowed_paths: list[str]
    forbidden_paths: list[str]
    checks: list[str]
    attempt: int
    max_attempts: int
    node_id: str | None = None
    executor_tier: str = "E2"
    risk_level: str = "medium"
    verification_profile: str = "default"
    required_capabilities: list[str] = field(default_factory=lambda: ["cli"])
    manager_reason: str = ""
    budget: dict[str, int] = field(default_factory=dict)
    memory_context: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    recipe_id: str | None = None
    environment_context: dict[str, Any] = field(default_factory=dict)
    repair_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CommandResult:
    command: str
    passed: bool
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditResult:
    verdict: str
    summary: str
    evidence: list[str]
    risks: list[str]
    fault_code: str | None = None
    confidence: float | None = None
    verified_claims: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManagerDecision:
    action: str
    criterion_id: str | None
    subtask_goal: str
    executor_tier: str
    risk_level: str
    verification_profile: str
    required_capabilities: list[str]
    reason: str
    recipe_id: str | None = None
    complexity: int = 1
    uncertainty: int = 1
    criticality: int = 1
    verification_difficulty: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FaultEvent:
    id: str
    category: str
    code: str
    severity: str
    retryable: bool
    scope: str
    source_role: str
    subtask_id: str | None
    fingerprint: str
    summary: str
    evidence: list[str]
    occurred_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecoveryDecision:
    action: str
    reason: str
    next_tier: str | None = None
    backoff_seconds: int = 0
    recovery_scope: str = "subtask"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerifierResult:
    verifier_id: str
    kind: str
    verdict: str
    required: bool
    summary: str
    evidence: list[str]
    fault_code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
