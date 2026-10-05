from __future__ import annotations

from dataclasses import dataclass

from .models import AcceptanceCriterion, ProductSpec, TaskContract


@dataclass(frozen=True)
class AlignmentAssessment:
    mode: str
    ambiguity_score: int
    can_execute: bool
    blocking_questions: list[str]
    assumptions: list[str]
    reason: str


def assess_alignment(contract: TaskContract) -> AlignmentAssessment:
    """Fail closed on explicit product questions while keeping task mode lightweight."""
    if contract.mode == "task" and contract.product is None:
        return AlignmentAssessment(
            mode="task",
            ambiguity_score=0,
            can_execute=True,
            blocking_questions=[],
            assumptions=[],
            reason="Explicit task contract",
        )
    product = contract.product or ProductSpec(problem=contract.objective)
    questions = [item.strip() for item in product.open_questions if item.strip()]
    ambiguity = min(
        3,
        int(not product.problem.strip())
        + int(not product.target_users)
        + int(not product.key_flows)
        + int(bool(questions)),
    )
    return AlignmentAssessment(
        mode="product",
        ambiguity_score=ambiguity,
        can_execute=not questions,
        blocking_questions=questions,
        assumptions=list(product.assumptions),
        reason=(
            "Product contract has unresolved high-impact questions"
            if questions
            else "Product contract is executable"
        ),
    )


def build_product_spec(
    *,
    problem: str,
    target_users: list[str] | None = None,
    key_flows: list[str] | None = None,
    must_haves: list[str] | None = None,
    visual_expectations: list[str] | None = None,
    quality_attributes: list[str] | None = None,
    assumptions: list[str] | None = None,
    open_questions: list[str] | None = None,
    required_human_gates: list[str] | None = None,
    product_checks: list[str] | None = None,
) -> ProductSpec:
    cleaned_users = _clean(target_users)
    cleaned_flows = _clean(key_flows)
    cleaned_must_haves = _clean(must_haves)
    questions = _clean(open_questions)
    if not cleaned_users:
        questions.append("谁是首要目标用户？")
    if not cleaned_flows and not cleaned_must_haves:
        questions.append("必须优先打通的关键用户流程是什么？")
    return ProductSpec(
        problem=problem.strip(),
        target_users=cleaned_users,
        key_flows=cleaned_flows,
        must_haves=cleaned_must_haves,
        visual_expectations=_clean(visual_expectations),
        quality_attributes=_clean(quality_attributes),
        assumptions=_clean(assumptions),
        open_questions=list(dict.fromkeys(questions)),
        required_human_gates=_clean(required_human_gates),
        product_checks=_clean(product_checks),
    )


def product_acceptance(product: ProductSpec, *, start_index: int = 1) -> list[AcceptanceCriterion]:
    """Create explicit product-flow claims when a caller did not provide them."""
    claims = [
        (f"关键用户流程可完成：{flow}", "product_flow")
        for flow in product.key_flows
    ]
    claims.extend(
        (f"必须功能已实现：{item}", "product_flow") for item in product.must_haves
    )
    claims.extend(
        (f"视觉与体验符合预期：{item}", "visual")
        for item in product.visual_expectations
    )
    claims.extend(
        (f"质量属性达标：{item}", "semantic")
        for item in product.quality_attributes
    )
    if not claims and product.problem:
        claims.append((f"产品能够解决：{product.problem}", "product_flow"))
    return [
        AcceptanceCriterion(
            id=f"AC-{index:03d}",
            description=description,
            kind="requirement",
            priority=3,
            verification_profile=profile,
            risk_level="medium",
        )
        for index, (description, profile) in enumerate(claims, start=start_index)
    ]


def _clean(values: list[str] | None) -> list[str]:
    return [item.strip() for item in (values or []) if item.strip()]
