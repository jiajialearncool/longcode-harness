from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from .models import ManagerDecision, TaskContract, TaskState
from .product import assess_alignment
from .state_graph import goal_coverage, ready_requirement_ids


TIER_ORDER = ("E0", "E1", "E2", "E3")
TIER_RANK = {tier: index for index, tier in enumerate(TIER_ORDER)}


class ExecutorLike(Protocol):
    pass


@dataclass(frozen=True)
class RoutingSignals:
    complexity: int
    uncertainty: int
    risk: int
    criticality: int
    verification_difficulty: int

    @property
    def highest(self) -> int:
        return max(
            self.complexity,
            self.uncertainty,
            self.risk,
            self.criticality,
            self.verification_difficulty,
        )


class HeuristicAdaptiveManager:
    """Dependency-aware fallback manager and deterministic tier policy floor."""

    def decide(self, contract: TaskContract, state: TaskState) -> ManagerDecision:
        alignment = assess_alignment(contract)
        if not alignment.can_execute:
            return ManagerDecision(
                action="ask",
                criterion_id=None,
                subtask_goal="Resolve product questions before execution",
                executor_tier="E0",
                risk_level="medium",
                verification_profile="alignment",
                required_capabilities=[],
                reason="; ".join(alignment.blocking_questions),
                uncertainty=3,
            )

        ready_ids = ready_requirement_ids(contract, state)
        if not ready_ids:
            covered, missing = goal_coverage(contract, state)
            human_gates = [item for item in missing if item.startswith("human-gate:")]
            action = "done" if covered else ("ask" if human_gates else "blocked")
            return ManagerDecision(
                action=action,
                criterion_id=None,
                subtask_goal="",
                executor_tier="E0",
                risk_level="low",
                verification_profile="final",
                required_capabilities=[],
                reason=(
                    "All requirements have evidence"
                    if covered
                    else (
                        "Human acceptance required: "
                        + "; ".join(item.split(":", 1)[1] for item in human_gates)
                        if human_gates
                        else "; ".join(missing)
                    )
                ),
                complexity=0,
                uncertainty=0,
                criticality=0,
                verification_difficulty=0,
            )

        criterion_id = ready_ids[0]
        criterion = next(item for item in contract.acceptance_criteria if item.id == criterion_id)
        signals = infer_signals(criterion.description, criterion.risk_level, criterion.depends_on, criterion.verification_profile)
        recipe_available = bool(
            criterion.recipe_id
            and contract.deterministic_recipes.get(criterion.recipe_id)
        )
        suggested = "E0" if recipe_available and signals.risk <= 1 else _tier_for_signals(signals)
        criterion_state = state.criteria[criterion_id]
        if criterion_state.next_tier:
            suggested = max_tier(suggested, criterion_state.next_tier)
        suggested = enforce_tier_policy(
            suggested,
            signals.risk,
            deterministic_recipe=recipe_available,
        )
        capabilities = capabilities_for(criterion.description, criterion.verification_profile)
        risk_label = ("low", "low", "medium", "high")[signals.risk]
        return ManagerDecision(
            action="execute",
            criterion_id=criterion_id,
            subtask_goal=criterion.description,
            executor_tier=suggested,
            risk_level=risk_label,
            verification_profile=criterion.verification_profile,
            required_capabilities=capabilities,
            reason=(
                f"ready={criterion_id}; complexity={signals.complexity}; "
                f"uncertainty={signals.uncertainty}; risk={signals.risk}; "
                f"verification={signals.verification_difficulty}"
            ),
            recipe_id=criterion.recipe_id if recipe_available else None,
            complexity=signals.complexity,
            uncertainty=signals.uncertainty,
            criticality=signals.criticality,
            verification_difficulty=signals.verification_difficulty,
        )


class AgentAdaptiveManager:
    """Use an Agent manager, with deterministic readiness and risk checks as a safety floor."""

    def __init__(self, backend, workspace: Path, *, fallback: HeuristicAdaptiveManager | None = None):
        self.backend = backend
        self.workspace = workspace
        self.fallback = fallback or HeuristicAdaptiveManager()
        self.last_backend_result = None
        self.last_fallback_reason: str | None = None

    def decide(self, contract: TaskContract, state: TaskState) -> ManagerDecision:
        safe_default = self.fallback.decide(contract, state)
        if safe_default.action == "ask":
            return safe_default
        manage = getattr(self.backend, "manage", None)
        if not callable(manage):
            self.last_fallback_reason = "Backend has no manager capability"
            return safe_default
        result, proposed = manage(self.workspace, contract, state)
        self.last_backend_result = result
        if not result.ok or proposed is None:
            self.last_fallback_reason = "Manager backend returned no valid decision"
            return safe_default

        ready = set(ready_requirement_ids(contract, state))
        covered, _ = goal_coverage(contract, state)
        if proposed.action == "done" and not covered:
            self.last_fallback_reason = "Manager claimed done without goal coverage"
            return safe_default
        if proposed.action == "execute" and proposed.criterion_id not in ready:
            self.last_fallback_reason = "Manager selected a non-ready criterion"
            return safe_default
        dimensions = [
            proposed.complexity,
            proposed.uncertainty,
            proposed.criticality,
            proposed.verification_difficulty,
        ]
        if any(value < 0 or value > 3 for value in dimensions):
            self.last_fallback_reason = "Manager dimension outside 0..3"
            return safe_default
        criterion = next(
            (
                item
                for item in contract.acceptance_criteria
                if item.id == proposed.criterion_id
            ),
            None,
        )
        proposed_risk = {"low": 1, "medium": 2, "high": 3}.get(proposed.risk_level, 2)
        contract_risk = (
            infer_signals(
                criterion.description,
                criterion.risk_level,
                criterion.depends_on,
                criterion.verification_profile,
            ).risk
            if criterion
            else 2
        )
        risk = max(proposed_risk, contract_risk)
        recipe_id = criterion.recipe_id if criterion else None
        recipe_available = bool(
            recipe_id and contract.deterministic_recipes.get(recipe_id)
        )
        tier = enforce_tier_policy(
            proposed.executor_tier,
            risk,
            deterministic_recipe=recipe_available,
        )
        if proposed.criterion_id:
            next_requested = state.criteria[proposed.criterion_id].next_tier
            if next_requested:
                tier = max_tier(tier, next_requested)
        self.last_fallback_reason = None
        required_capabilities = sorted(
            set(proposed.required_capabilities)
            | set(
                capabilities_for(
                    criterion.description,
                    criterion.verification_profile,
                )
                if criterion
                else []
            )
        )
        return replace(
            proposed,
            executor_tier=tier,
            recipe_id=recipe_id if tier == "E0" else None,
            verification_profile=(
                criterion.verification_profile
                if criterion
                else proposed.verification_profile
            ),
            required_capabilities=required_capabilities,
        )


class EvidenceDrivenManager:
    """Call the model Manager only after the safety belt requests a replan.

    Normal routing remains deterministic.  A replan count is consumed once, so
    resuming an already-handled state does not spend another Manager call.
    """

    def __init__(
        self,
        backend,
        workspace: Path,
        *,
        fallback: HeuristicAdaptiveManager | None = None,
    ):
        self.backend = backend
        self.workspace = workspace
        self.fallback = fallback or HeuristicAdaptiveManager()
        self.last_backend_result = None
        self.last_fallback_reason: str | None = None

    def decide(self, contract: TaskContract, state: TaskState) -> ManagerDecision:
        safe_default = self.fallback.decide(contract, state)
        if safe_default.action != "execute" or not safe_default.criterion_id:
            self.last_backend_result = None
            self.last_fallback_reason = None
            return safe_default

        criterion_state = state.criteria[safe_default.criterion_id]
        if criterion_state.replan_count <= criterion_state.manager_replans_handled:
            self.last_backend_result = None
            self.last_fallback_reason = None
            return safe_default

        criterion_state.manager_replans_handled = criterion_state.replan_count
        agent_manager = AgentAdaptiveManager(
            self.backend,
            self.workspace,
            fallback=self.fallback,
        )
        decision = agent_manager.decide(contract, state)
        self.last_backend_result = agent_manager.last_backend_result
        self.last_fallback_reason = agent_manager.last_fallback_reason
        return decision


class ExecutorRegistry:
    """Map abstract tiers to backends without leaking provider names into the engine."""

    def __init__(self, default: ExecutorLike, tiers: dict[str, ExecutorLike] | None = None):
        self.default = default
        self.tiers = dict(tiers or {})

    def resolve(self, tier: str) -> tuple[str, ExecutorLike]:
        if tier in self.tiers:
            return tier, self.tiers[tier]
        requested_rank = TIER_RANK.get(tier, TIER_RANK["E2"])
        # E0 is an explicit deterministic recipe boundary. It must never become the nearest
        # fallback for an open-ended E1/E2/E3 request.
        eligible_tiers = {
            name: backend
            for name, backend in self.tiers.items()
            if tier == "E0" or name != "E0"
        }
        configured = sorted(
            ((abs(TIER_RANK.get(name, 2) - requested_rank), -TIER_RANK.get(name, 2), name, backend)
             for name, backend in eligible_tiers.items()),
            key=lambda item: (item[0], item[1], item[2]),
        )
        if configured:
            _, _, name, backend = configured[0]
            return name, backend
        return tier, self.default


def infer_signals(
    description: str,
    explicit_risk: str,
    dependencies: list[str],
    verification_profile: str,
) -> RoutingSignals:
    text = description.lower()
    words = text.split()
    complexity = 1
    if len(words) > 25 or dependencies:
        complexity = 2
    if len(words) > 60 or len(dependencies) > 2 or any(
        token in text for token in ("architecture", "migration", "distributed", "跨系统", "架构", "迁移")
    ):
        complexity = 3

    uncertainty = 1
    if any(token in text for token in ("investigate", "explore", "unknown", "研究", "探索", "不确定")):
        uncertainty = 2
    if any(token in text for token in ("ambiguous", "root cause unknown", "模糊", "根因未知")):
        uncertainty = 3

    explicit = {"low": 1, "medium": 2, "high": 3}.get(explicit_risk, 2)
    risky_tokens = (
        "delete", "payment", "production", "credential", "security", "deploy",
        "删除", "支付", "生产", "密钥", "安全", "部署",
    )
    risk = max(explicit, 3 if any(token in text for token in risky_tokens) else 1)

    criticality = 2 if dependencies else 1
    if any(token in text for token in ("core", "authentication", "database", "核心", "登录", "数据库")):
        criticality = 3

    verification = 1
    if verification_profile in {"semantic", "product_flow", "visual", "security"}:
        verification = 2
    if verification_profile in {"visual_strict", "security_strict", "irreversible"}:
        verification = 3
    return RoutingSignals(complexity, uncertainty, risk, criticality, verification)


def enforce_tier_policy(
    tier: str,
    risk: int,
    *,
    deterministic_recipe: bool = False,
) -> str:
    minimum = (
        "E3"
        if risk >= 3
        else ("E2" if risk == 2 else ("E0" if deterministic_recipe else "E1"))
    )
    return max_tier(tier, minimum)


def next_tier(current: str, allowed: list[str] | tuple[str, ...] = TIER_ORDER) -> str | None:
    ordered = sorted(set(allowed), key=lambda item: TIER_RANK.get(item, 99))
    current_rank = TIER_RANK.get(current, 2)
    return next((tier for tier in ordered if TIER_RANK.get(tier, 99) > current_rank), None)


def max_tier(first: str, second: str) -> str:
    return first if TIER_RANK.get(first, 2) >= TIER_RANK.get(second, 2) else second


def _tier_for_signals(signals: RoutingSignals) -> str:
    return ("E0", "E1", "E2", "E3")[max(0, min(3, signals.highest))]


def capabilities_for(description: str, verification_profile: str) -> list[str]:
    text = description.lower()
    capabilities = ["cli"]
    environment_profile = verification_profile in {
        "product_flow",
        "visual",
        "visual_strict",
    }
    gui_requested = environment_profile and any(
        token in text
        for token in (
            "desktop app",
            "gui",
            "spreadsheet",
            "桌面应用",
            "操作表格",
            "演示文稿",
        )
    )
    browser_requested = environment_profile or any(
        token in text
        for token in (
            "use the browser",
            "browser automation",
            "operate the browser",
            "使用浏览器",
            "浏览器操作",
        )
    )
    if gui_requested:
        capabilities.append("gui")
    if browser_requested and not gui_requested:
        capabilities.append("browser")
    return capabilities
