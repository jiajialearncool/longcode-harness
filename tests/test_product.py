from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from longcode.goals import answer_product_question, approve_product_gate
from longcode.models import TaskContract, TaskState
from longcode.product import assess_alignment, build_product_spec, product_acceptance
from longcode.storage import RuntimeStore
from longcode.state_graph import goal_coverage


class ProductTests(unittest.TestCase):
    def test_open_product_question_blocks_execution_until_answered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            product = build_product_spec(
                problem="Help sales manage customers",
                target_users=["sales"],
                key_flows=["create and edit a customer"],
                open_questions=["Should deletion be permanent?"],
            )
            criteria = product_acceptance(product)
            contract = TaskContract.create(
                "Build a CRM",
                [item.description for item in criteria],
                checks=["true"],
                mode="product",
                product=product,
            )
            contract.acceptance_criteria = criteria
            store = RuntimeStore(root / "runtime")
            store.initialize(contract, workspace)
            self.assertFalse(assess_alignment(contract).can_execute)
            revised = answer_product_question(
                store,
                question="Should deletion be permanent?",
                answer="Use reversible archive",
            )
            self.assertTrue(assess_alignment(revised).can_execute)
            self.assertEqual(revised.version, 2)
            self.assertIn("Use reversible archive", revised.product.assumptions[0])
            decisions = [node for node in store.load_state().graph.nodes.values() if node.kind == "decision"]
            self.assertEqual(len(decisions), 1)
            self.assertEqual(decisions[0].trust, "trusted")

    def test_answering_one_question_does_not_rename_another_question_node(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            product = build_product_spec(
                problem="Help sales",
                target_users=["sales"],
                key_flows=["create a customer"],
                open_questions=["Archive or delete?", "Which fields are required?"],
            )
            criteria = product_acceptance(product)
            contract = TaskContract.create(
                "Build CRM",
                [item.description for item in criteria],
                checks=["true"],
                mode="product",
                product=product,
            )
            contract.acceptance_criteria = criteria
            store = RuntimeStore(root / "runtime")
            store.initialize(contract, workspace)
            before = {
                node.title: node.id
                for node in store.load_state().graph.nodes.values()
                if node.kind == "question"
            }
            answer_product_question(store, question="Archive or delete?", answer="Archive")
            after = {
                node.title: node.id
                for node in store.load_state().graph.nodes.values()
                if node.kind == "question"
            }
            self.assertEqual(
                before["Which fields are required?"],
                after["Which fields are required?"],
            )
            resolved = next(
                node
                for node in store.load_state().graph.nodes.values()
                if node.title == "Archive or delete?"
            )
            self.assertEqual(resolved.status, "verified")

    def test_product_flow_appears_in_typed_state_graph(self):
        product = build_product_spec(
            problem="Analyze sales",
            target_users=["manager"],
            key_flows=["upload a spreadsheet and view a chart"],
        )
        criteria = product_acceptance(product)
        contract = TaskContract.create(
            "Sales dashboard",
            [item.description for item in criteria],
            checks=["true"],
            mode="product",
            product=product,
        )
        contract.acceptance_criteria = criteria
        state = TaskState.create(contract)
        flow_nodes = [node for node in state.graph.nodes.values() if node.kind == "product_flow"]
        self.assertEqual(len(flow_nodes), 1)
        self.assertTrue(any(edge.relation == "satisfies" for edge in state.graph.edges))

    def test_product_builder_exposes_missing_high_impact_context_and_visual_claims(self):
        product = build_product_spec(
            problem="Make data legible",
            visual_expectations=["empty state has a clear next action"],
        )
        self.assertIn("谁是首要目标用户？", product.open_questions)
        self.assertIn("必须优先打通的关键用户流程是什么？", product.open_questions)
        criteria = product_acceptance(product)
        visual = next(item for item in criteria if "视觉与体验" in item.description)
        self.assertEqual(visual.verification_profile, "visual")

    def test_human_gate_is_required_for_goal_coverage_and_is_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            product = build_product_spec(
                problem="Ship a dashboard",
                key_flows=["User opens dashboard"],
                required_human_gates=["PM accepts the release candidate"],
            )
            contract = TaskContract.create(
                "Ship dashboard",
                ["Dashboard flow works"],
                checks=["true"],
                mode="product",
                product=product,
            )
            store = RuntimeStore(root / "runtime")
            store.initialize(contract, workspace)
            state = store.load_state()
            state.criteria["AC-001"].status = "verified"
            state.criteria["AC-001"].evidence = ["evidence/flow.json"]
            state.graph.mark_verified("AC-001", ["evidence/flow.json"])
            store.save_state(state)

            covered, missing = goal_coverage(contract, state)
            self.assertFalse(covered)
            self.assertIn("human-gate:PM accepts the release candidate", missing)

            approved = approve_product_gate(
                store,
                gate="PM accepts the release candidate",
                note="Checked staging flow and visual state",
            )
            self.assertIn("PM accepts the release candidate", approved.approved_gates)
            gate_node = next(
                node for node in approved.graph.nodes.values() if node.kind == "human_gate"
            )
            self.assertEqual(gate_node.status, "verified")
            self.assertEqual(gate_node.trust, "trusted")


if __name__ == "__main__":
    unittest.main()
