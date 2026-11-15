"""Operational checks for run_once.sh / sync_results.sh (subprocess only).

No network, no installs, no GPU — dry-run/help paths and early gate failures.
"""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ONCE = ROOT / 'scripts' / 'run_once.sh'
SYNC = ROOT / 'scripts' / 'sync_results.sh'

STAGES = [
    'bash scripts/setup_instance.sh',
    '.venv/bin/python scripts/fetch_assets.py --galleries --dinov3-weights --towers so400m256 gopt dinov3',
    '.venv/bin/python scripts/run_experiment.py all --root',
]


def run(script, *args):
    return subprocess.run(['bash', str(script), *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=60)


class RunOnceTests(unittest.TestCase):
    def test_default_is_dry_run_with_three_stages(self):
        for args in [(), ('--dry-run',)]:
            r = run(RUN_ONCE, *args)
            self.assertEqual(r.returncode, 0, r.stderr)
            for stage in STAGES:
                self.assertIn(stage, r.stdout)
            self.assertIn('experiment_config.json', r.stdout)
            self.assertIn('artifacts/experiment-v1', r.stdout)
            self.assertNotIn('Traceback', r.stderr)

    def test_execute_rejected_off_linux_before_pip(self):
        import platform
        if platform.system() == 'Linux':
            self.skipTest('gate is Linux-only; cannot exercise rejection here')
        r = run(RUN_ONCE, '--execute')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('Linux', r.stderr + r.stdout)

    def test_unknown_flag_and_bad_root_rejected(self):
        r = run(RUN_ONCE, '--bogus')
        self.assertNotEqual(r.returncode, 0)
        r = run(RUN_ONCE, '--root', '/nonexistent-path-xyz')
        self.assertNotEqual(r.returncode, 0)


class RunOnceRepairTests(unittest.TestCase):
    """Dry-run branches for the export-repair / completed states."""

    def _mkroot(self, tmp, report=True, examples=False):
        import json
        run = Path(tmp) / 'artifacts' / 'experiment-v1'
        run.mkdir(parents=True)
        if report:
            (run / 'report.json').write_text(json.dumps({'decision': 'x'}))
        if examples:
            (run / 'demo_examples').mkdir()
            (run / 'demo_examples' / 'examples.json').write_text('[]')
        return Path(tmp)

    def test_report_without_examples_prints_export_repair(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = self._mkroot(tmp, report=True, examples=False)
            r = run(RUN_ONCE, '--root', str(root))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('run_experiment.py export', r.stdout)
            self.assertIn('--execute', r.stdout)
            self.assertNotIn('fetch_assets.py --galleries', r.stdout)

    def test_report_and_examples_is_completed_noop(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = self._mkroot(tmp, report=True, examples=True)
            r = run(RUN_ONCE, '--root', str(root))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('already completed', r.stdout)
            self.assertNotIn('run_experiment.py all', r.stdout)


class SyncResultsTests(unittest.TestCase):
    def test_help_exits_zero_without_ssh(self):
        r = run(SYNC, '--help')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('SSH_ALIAS', r.stdout)
        self.assertIn('rsync', r.stdout)

    def test_missing_alias_rejected(self):
        r = run(SYNC)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('SSH_ALIAS', r.stderr)

    def test_option_like_alias_rejected(self):
        r = run(SYNC, '--watch=bogus')
        # --watch=bogus is an unknown option form -> usage error, no ssh
        self.assertNotEqual(r.returncode, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
