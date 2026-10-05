from __future__ import annotations

import unittest
from dataclasses import replace

from longcode.models import StateEdge, TaskContract, TaskState
from longcode.state_graph import impacted_nodes, ready_requirement_ids


class StateGraphTests(unittest.TestCase):
    def test_contract_constraints_non_goals_and_risks_are_typed_nodes(self):
        contract = TaskContract.create(
            "Goal",
            ["Protect production data"],
            constraints=["Do not edit migrations"],
            non_goals=["No redesign"],
            checks=["true"],
        )
        contract.acceptance_criteria[0] = replace(
            contract.acceptance_criteria[0], risk_level="high"
        )
        state = TaskState.create(contract)
        kinds = {node.kind for node in state.graph.nodes.values()}
        self.assertTrue({"constraint", "non_goal", "risk"} <= kinds)

    def test_dependency_blocks_downstream_requirement(self):
        contract = TaskContract.create("Goal", ["Foundation", "Feature"], checks=["true"])
        contract.acceptance_criteria[1] = replace(
            contract.acceptance_criteria[1], depends_on=["AC-001"], priority=3
        )
        state = TaskState.create(contract)
        self.assertEqual(ready_requirement_ids(contract, state), ["AC-001"])
        state.criteria["AC-001"].status = "verified"
        state.graph.mark_verified("AC-001", ["evidence"])
        self.assertEqual(ready_requirement_ids(contract, state), ["AC-002"])

    def test_change_impact_propagates_to_dependents(self):
        contract = TaskContract.create("Goal", ["A", "B", "C"], checks=["true"])
        state = TaskState.create(contract)
        state.graph.edges.extend(
            [
                StateEdge(source="AC-002", relation="depends_on", target="AC-001"),
                StateEdge(source="AC-003", relation="depends_on", target="AC-002"),
            ]
        )
        self.assertEqual(impacted_nodes(state.graph, {"AC-001"}), {"AC-001", "AC-002", "AC-003"})


if __name__ == "__main__":
    unittest.main()
