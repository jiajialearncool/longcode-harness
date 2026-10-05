from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from longcode.product_backends import ProductCliBackend
from longcode.native_agent import user_message
from tests import test_autonomous_agent as fixtures


class ProductBackendTests(unittest.TestCase):
    setUp = fixtures.AgentTests.setUp
    tearDown = fixtures.AgentTests.tearDown

    def test_missing_cli_explains_native_option(self):
        with patch('longcode.product_backends.shutil.which', return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, '自主执行'):
                ProductCliBackend({**self.settings, 'backend': 'claude'})

    def test_claude_readonly_does_not_reuse_benchmark_permissions(self):
        argv_seen = []
        def run(argv, **kwargs):
            argv_seen.extend(argv)
            return 0, json.dumps({'structured_output': {'ok': True}}), ''
        with patch('longcode.product_backends.shutil.which', return_value='/fixture/claude'), patch('longcode.product_backends.run_process', side_effect=run):
            backend = ProductCliBackend({**self.settings, 'backend': 'claude'}, ask=lambda _: 'allow')
            result = backend._run(self.workspace, 'review', {'type':'object', 'properties':{'ok':{'type':'boolean'}}, 'required':['ok']}, sandbox='read-only')
        self.assertTrue(result.ok, result.stderr)
        self.assertIn('Read,Glob,Grep', argv_seen)
        self.assertIn('--no-session-persistence', argv_seen)
        self.assertEqual(argv_seen[argv_seen.index('--permission-mode')+1], 'default')
        self.assertNotIn('--dangerously-skip-permissions', argv_seen)

    def test_codex_chat_uses_sandbox_and_fresh_session(self):
        argv_seen = []
        def run(argv, **kwargs):
            argv_seen.extend(argv)
            Path(argv[argv.index('--output-last-message')+1]).write_text('已检查')
            return 0, '', ''
        with patch('longcode.product_backends.shutil.which', return_value='/fixture/codex'), patch('longcode.product_backends.run_process', side_effect=run):
            backend = ProductCliBackend({**self.settings, 'backend':'codex'}, ask=lambda _: 'allow')
            text, messages, usage = backend.converse(self.workspace, [user_message('检查项目')])
        self.assertEqual(text, '已检查')
        self.assertIn('--ephemeral', argv_seen)
        self.assertEqual(argv_seen[argv_seen.index('--sandbox')+1], 'workspace-write')
        self.assertNotIn('--dangerously-bypass-approvals-and-sandbox', argv_seen)

    def test_cli_declined_never_runs(self):
        with patch('longcode.product_backends.shutil.which', return_value='/fixture/codex'), patch('longcode.product_backends.run_process') as run:
            backend = ProductCliBackend({**self.settings, 'backend':'codex'}, ask=lambda _: 'deny')
            with self.assertRaises(PermissionError):
                backend.converse(self.workspace, [user_message('修改')])
            run.assert_not_called()
