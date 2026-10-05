from __future__ import annotations

from collections import deque

from .models import StateEdge, StateNode, TaskContract, TaskGraph, TaskState, utc_now


TERMINAL_NODE_STATUSES = {"verified", "completed", "cancelled"}
ACTIVE_NODE_STATUSES = {"pending", "failed", "needs_revalidation", "in_progress"}


def ensure_graph(contract: TaskContract, state: TaskState) -> TaskGraph:
    """Migrate an old runtime and synchronize contract-backed graph nodes."""
    if state.graph is None:
        state.graph = TaskGraph.from_contract(contract)
    graph = state.graph
    graph.version = contract.version
    desired = TaskGraph.from_contract(contract)

    for node_id, desired_node in desired.nodes.items():
        existing = graph.nodes.get(node_id)
        if existing is None:
            graph.nodes[node_id] = desired_node
            continue
        existing.title = desired_node.title
        existing.description = desired_node.description
        existing.source_version = max(existing.source_version, desired_node.source_version)
        existing.priority = desired_node.priority
        existing.metadata.update(desired_node.metadata)

    desired_edges = {(edge.source, edge.relation, edge.target) for edge in desired.edges}
    existing_edges = {(edge.source, edge.relation, edge.target) for edge in graph.edges}
    graph.edges.extend(
        edge for edge in desired.edges if (edge.source, edge.relation, edge.target) not in existing_edges
    )
    active_criterion_ids = {item.id for item in contract.acceptance_criteria}
    desired_product_node_ids = {
        node_id
        for node_id, node in desired.nodes.items()
        if node.kind in {"question", "product_flow", "human_gate"}
    }
    for node in graph.nodes.values():
        if node.kind == "requirement" and node.id not in active_criterion_ids:
            node.status = "cancelled"
        if (
            node.kind in {"question", "product_flow", "human_gate"}
            and node.id not in desired_product_node_ids
            and node.status in {"open", "pending", "failed", "needs_revalidation"}
        ):
            node.status = "cancelled"
            node.metadata["superseded_in_version"] = contract.version
    for edge in graph.edges:
        if (edge.source, edge.relation, edge.target) not in desired_edges and edge.source in active_criterion_ids:
            # Historical edges remain inspectable; a new graph version can supersede them.
            continue
    graph.updated_at = utc_now()
    return graph


def ready_requirement_ids(contract: TaskContract, state: TaskState) -> list[str]:
    graph = ensure_graph(contract, state)
    ready: list[tuple[int, int, str]] = []
    order = {item.id: index for index, item in enumerate(contract.acceptance_criteria)}
    for criterion in contract.acceptance_criteria:
        criterion_state = state.criteria.get(criterion.id)
        node = graph.nodes.get(criterion.id)
        if not criterion_state or not node:
            continue
        if criterion_state.status not in ACTIVE_NODE_STATUSES:
            continue
        if not graph.dependencies_satisfied(criterion.id):
            continue
        ready.append((-node.priority, order[criterion.id], criterion.id))
    return [node_id for _, _, node_id in sorted(ready)]


def add_node(graph: TaskGraph, node: StateNode) -> None:
    if node.id in graph.nodes:
        raise ValueError(f"state graph node already exists: {node.id}")
    graph.nodes[node.id] = node
    graph.updated_at = utc_now()


def add_edge(graph: TaskGraph, edge: StateEdge) -> None:
    if edge.source not in graph.nodes:
        raise ValueError(f"unknown edge source: {edge.source}")
    if edge.target not in graph.nodes:
        raise ValueError(f"unknown edge target: {edge.target}")
    key = (edge.source, edge.relation, edge.target)
    if any((item.source, item.relation, item.target) == key and item.active for item in graph.edges):
        return
    graph.edges.append(edge)
    graph.updated_at = utc_now()


def impacted_nodes(graph: TaskGraph, changed_ids: set[str]) -> set[str]:
    """Return changed nodes and every active dependent/invalidation target."""
    impacted = set(changed_ids)
    queue = deque(changed_ids)
    while queue:
        current = queue.popleft()
        for edge in graph.edges:
            if not edge.active:
                continue
            should_propagate = (
                edge.relation == "depends_on" and edge.target == current
            ) or (
                edge.relation == "invalidates" and edge.source == current
            )
            candidate = edge.source if edge.relation == "depends_on" else edge.target
            if should_propagate and candidate not in impacted:
                impacted.add(candidate)
                queue.append(candidate)
    return impacted


def invalidate_impacted(graph: TaskGraph, changed_ids: set[str], *, reason: str) -> set[str]:
    affected = impacted_nodes(graph, changed_ids)
    graph.invalidate(sorted(affected), reason=reason)
    return affected


def goal_coverage(contract: TaskContract, state: TaskState) -> tuple[bool, list[str]]:
    graph = ensure_graph(contract, state)
    missing: list[str] = []
    for criterion in contract.acceptance_criteria:
        criterion_state = state.criteria.get(criterion.id)
        node = graph.nodes.get(criterion.id)
        if not criterion_state or criterion_state.status != "verified":
            missing.append(f"{criterion.id}:not-verified")
            continue
        evidence = set(criterion_state.evidence)
        if node:
            evidence.update(node.evidence)
        if not evidence:
            missing.append(f"{criterion.id}:missing-evidence")
    if contract.product:
        for node in graph.nodes.values():
            if node.kind == "product_flow" and node.status != "verified":
                missing.append(f"{node.id}:product-flow-not-verified")
        for node in graph.nodes.values():
            if node.kind == "question" and node.status == "open":
                missing.append(f"{node.id}:open-question")
        for gate in contract.product.required_human_gates:
            if gate not in state.approved_gates:
                missing.append(f"human-gate:{gate}")
    return not missing, missing


def sync_criterion_projection(state: TaskState) -> None:
    if not state.graph:
        return
    for criterion_id, criterion_state in state.criteria.items():
        node = state.graph.nodes.get(criterion_id)
        if not node:
            continue
        node.status = criterion_state.status
        node.evidence = list(criterion_state.evidence)
        node.trust = "trusted" if criterion_state.status == "verified" else "untrusted"
        node.updated_at = utc_now()
    state.graph.updated_at = utc_now()
