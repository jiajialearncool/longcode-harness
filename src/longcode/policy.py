from __future__ import annotations

from dataclasses import dataclass

from .models import Subtask, TaskContract, TaskState


IRREVERSIBLE_TOKENS = (
    "deploy", "production", "payment", "delete database", "rotate credential",
    "部署", "生产", "支付", "删除数据库", "轮换密钥",
)


@dataclass(frozen=True)
class PolicyDecision:
    verdict: str
    reason: str
    missing_capabilities: tuple[str, ...] = ()

    @property
    def allowed(self) -> bool:
        return self.verdict == "allow"


class PolicyEngine:
    """Fail closed when a plan needs unavailable tools or an unapproved irreversible action."""

    def authorize(
        self,
        contract: TaskContract,
        state: TaskState,
        subtask: Subtask,
        *,
        available_capabilities: set[str],
    ) -> PolicyDecision:
        missing = tuple(sorted(set(subtask.required_capabilities) - available_capabilities))
        if missing:
            return PolicyDecision(
                "ask",
                "Required environment capabilities are unavailable: " + ", ".join(missing),
                missing,
            )
        text = f"{subtask.criterion} {subtask.manager_reason}".lower()
        if any(token in text for token in IRREVERSIBLE_TOKENS):
            configured = contract.product.required_human_gates if contract.product else []
            unapproved = [gate for gate in configured if gate not in state.approved_gates]
            if not configured:
                return PolicyDecision(
                    "ask",
                    "Irreversible/high-impact action requires an explicit human gate in the contract",
                )
            if unapproved:
                return PolicyDecision(
                    "ask",
                    "Human approval required before high-impact action: " + "; ".join(unapproved),
                )
        return PolicyDecision("allow", "Capabilities and human-gate policy satisfied")
