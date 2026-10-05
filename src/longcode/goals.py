from __future__ import annotations

from copy import deepcopy

from .models import AcceptanceCriterion, CriterionState, StateEdge, StateNode, TaskContract, utc_now
from .state_graph import add_edge, add_node, ensure_graph, sync_criterion_projection
from .storage import RuntimeStore


def revise_goal(
    store: RuntimeStore,
    *,
    objective: str | None = None,
    add_constraints: list[str] | None = None,
    remove_constraints: list[str] | None = None,
    add_acceptance: list[str] | None = None,
    remove_acceptance_ids: list[str] | None = None,
) -> TaskContract:
    """Create a new goal version; never mutate the original contract or history."""
    contract = store.load_contract()
    before = contract.to_dict()
    state = store.load_state()
    ensure_graph(contract, state)
    add_constraints = [item.strip() for item in (add_constraints or []) if item.strip()]
    remove_constraints = [item.strip() for item in (remove_constraints or []) if item.strip()]
    add_acceptance = [item.strip() for item in (add_acceptance or []) if item.strip()]
    remove_acceptance_ids = [item.strip() for item in (remove_acceptance_ids or []) if item.strip()]

    changes: list[dict[str, object]] = []
    global_change = False

    if objective is not None and objective.strip() != contract.objective:
        if not objective.strip():
            raise ValueError("objective must not be empty")
        changes.append({"field": "objective", "before": contract.objective, "after": objective.strip()})
        contract.objective = objective.strip()
        global_change = True

    for item in add_constraints:
        if item not in contract.constraints:
            contract.constraints.append(item)
            changes.append({"field": "constraints", "operation": "add", "value": item})
            global_change = True
    for item in remove_constraints:
        if item in contract.constraints:
            contract.constraints.remove(item)
            changes.append({"field": "constraints", "operation": "remove", "value": item})
            global_change = True

    removed_ids: set[str] = set()
    if remove_acceptance_ids:
        known = {item.id for item in contract.acceptance_criteria}
        unknown = sorted(set(remove_acceptance_ids) - known)
        if unknown:
            raise ValueError(f"unknown acceptance IDs: {', '.join(unknown)}")
        removed_ids = set(remove_acceptance_ids)
        contract.acceptance_criteria = [
            item for item in contract.acceptance_criteria if item.id not in removed_ids
        ]
        for item in sorted(removed_ids):
            changes.append({"field": "acceptance_criteria", "operation": "remove", "id": item})

    next_number = _next_acceptance_number(contract)
    added_ids: list[str] = []
    for description in add_acceptance:
        criterion_id = f"AC-{next_number:03d}"
        next_number += 1
        contract.acceptance_criteria.append(
            AcceptanceCriterion(
                id=criterion_id,
                description=description,
                source_version=contract.version + 1,
            )
        )
        added_ids.append(criterion_id)
        changes.append(
            {
                "field": "acceptance_criteria",
                "operation": "add",
                "id": criterion_id,
                "description": description,
            }
        )

    if not contract.acceptance_criteria:
        raise ValueError("a goal must retain at least one acceptance criterion")
    if not changes:
        raise ValueError("the requested revision does not change the active goal")

    contract.version += 1
    contract.updated_at = utc_now()
    state.contract_version = contract.version
    state.status = "ready"
    state.completed_at = None
    state.blocker = None
    state.final_evidence = []

    if global_change:
        for criterion_id, criterion_state in state.criteria.items():
            if criterion_state.status != "cancelled" and criterion_id not in removed_ids:
                criterion_state.status = "needs_revalidation"
                criterion_state.attempts = 0
                criterion_state.last_error_type = "GOAL_REVISED"
                criterion_state.last_error = "Goal-level requirements changed"
                criterion_state.verified_at = None
                criterion_state.evidence = []

    for criterion_id in removed_ids:
        criterion_state = state.criteria.setdefault(criterion_id, CriterionState())
        criterion_state.status = "cancelled"
        criterion_state.last_error_type = "GOAL_REVISED"
        criterion_state.last_error = f"Removed by goal revision v{contract.version}"
        criterion_state.verified_at = None

    for criterion_id in added_ids:
        state.criteria[criterion_id] = CriterionState(status="pending")

    ensure_graph(contract, state)
    if state.graph:
        for criterion_id in removed_ids:
            node = state.graph.nodes.get(criterion_id)
            if node:
                node.status = "cancelled"
                node.trust = "untrusted"
                node.evidence = []
        if global_change:
            state.graph.invalidate(
                [item.id for item in contract.acceptance_criteria if item.id not in removed_ids],
                reason=f"Goal-level requirements changed in v{contract.version}",
            )
    sync_criterion_projection(state)

    store.save_contract(contract)
    store.save_state(state)
    store.append_revision(
        {
            "type": "goal_revised",
            "from_version": contract.version - 1,
            "to_version": contract.version,
            "changes": changes,
            "before": before,
            "after": contract.to_dict(),
        }
    )
    store.append_event(
        "goal_revised",
        {
            "from_version": contract.version - 1,
            "to_version": contract.version,
            "changes": deepcopy(changes),
        },
    )
    return contract


def _next_acceptance_number(contract: TaskContract) -> int:
    numbers = []
    for item in contract.acceptance_criteria:
        try:
            numbers.append(int(item.id.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(numbers, default=0) + 1


def answer_product_question(store: RuntimeStore, *, question: str, answer: str) -> TaskContract:
    contract = store.load_contract()
    state = store.load_state()
    if contract.product is None:
        raise ValueError("active goal has no product contract")
    question = question.strip()
    answer = answer.strip()
    if not question or not answer:
        raise ValueError("question and answer must not be empty")
    try:
        index = contract.product.open_questions.index(question)
    except ValueError as error:
        raise ValueError(f"unknown open product question: {question}") from error
    graph = ensure_graph(contract, state)
    question_node = next(
        (node for node in graph.nodes.values() if node.kind == "question" and node.title == question),
        None,
    )
    before = contract.to_dict()
    contract.product.open_questions.pop(index)
    decision_text = f"{question} => {answer}"
    contract.product.assumptions.append(decision_text)
    contract.version += 1
    contract.updated_at = utc_now()
    state.contract_version = contract.version
    state.status = "ready"
    state.blocker = None
    graph = ensure_graph(contract, state)
    if question_node:
        question_node.status = "verified"
        question_node.trust = "trusted"
        question_node.metadata["answer"] = answer
        question_node.updated_at = utc_now()
    decision_id = f"DECISION-v{contract.version:04d}-{index + 1:03d}"
    add_node(
        graph,
        StateNode(
            id=decision_id,
            kind="decision",
            title=decision_text,
            description=decision_text,
            status="verified",
            trust="trusted",
            source_version=contract.version,
            priority=3,
            metadata={"question": question, "answer": answer},
        ),
    )
    if question_node:
        add_edge(graph, StateEdge(source=decision_id, relation="resolves", target=question_node.id))
    graph.version = contract.version
    store.save_contract(contract)
    store.save_state(state)
    change = {
        "field": "product.open_questions",
        "operation": "resolve",
        "question": question,
        "answer": answer,
        "decision_id": decision_id,
    }
    store.append_revision(
        {
            "type": "product_question_answered",
            "from_version": contract.version - 1,
            "to_version": contract.version,
            "changes": [change],
            "before": before,
            "after": contract.to_dict(),
        }
    )
    store.append_event(
        "product_question_answered",
        {
            "from_version": contract.version - 1,
            "to_version": contract.version,
            **change,
        },
    )
    return contract


def approve_product_gate(store: RuntimeStore, *, gate: str, note: str) -> TaskState:
    """Record explicit human acceptance as trusted state without rewriting the goal."""
    contract = store.load_contract()
    state = store.load_state()
    if contract.product is None:
        raise ValueError("active goal has no product contract")
    gate = gate.strip()
    note = note.strip()
    if gate not in contract.product.required_human_gates:
        raise ValueError(f"unknown product gate: {gate}")
    if not note:
        raise ValueError("approval note must not be empty")
    approved_at = utc_now()
    state.approved_gates[gate] = {"note": note, "approved_at": approved_at}
    graph = ensure_graph(contract, state)
    gate_node = next(
        (node for node in graph.nodes.values() if node.kind == "human_gate" and node.title == gate),
        None,
    )
    if gate_node:
        gate_node.status = "verified"
        gate_node.trust = "trusted"
        gate_node.evidence = [f"human-approval:{approved_at}"]
        gate_node.metadata.update({"note": note, "approved_at": approved_at})
        gate_node.updated_at = approved_at
    state.status = "ready" if state.status in {"waiting_input", "paused", "blocked"} else state.status
    state.blocker = None if state.status == "ready" else state.blocker
    store.save_state(state)
    store.append_event(
        "human_gate_approved",
        {"gate": gate, "note": note, "approved_at": approved_at},
    )
    return state
