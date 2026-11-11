import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    'eval_libero_remote', Path(__file__).resolve().parents[1] / 'deploy/eval_libero_remote.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class ExperimentRecordingTest(unittest.TestCase):
    def test_failed_worker_preserves_log_configuration_and_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'experiment'

            def run(command, **kwargs):
                if command[0] == 'git':
                    return subprocess.CompletedProcess(command, 0, stdout='revision\n')
                self.assertIn('--worker', command)
                kwargs['stdout'].write('intentional worker error\n')
                return subprocess.CompletedProcess(command, 7)

            with patch('sys.argv', ['eval', '--task-id', '0', '--output', str(out)]), \
                    patch.object(evaluation.subprocess, 'run', side_effect=run):
                self.assertEqual(evaluation.main(), 7)
            self.assertEqual(json.loads((out / 'process-result.json').read_text())['exit_code'], 7)
            config = json.loads((out / 'run-config.json').read_text())
            self.assertEqual(config['code_revision'], 'revision')
            self.assertEqual(len(config['script_sha256']), 64)
            self.assertIn('intentional worker error', (out / 'console.log').read_text())

    def test_existing_experiment_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'results.json'
            marker.write_text('original evidence')
            with patch('sys.argv', ['eval', '--task-id', '0', '--output', directory]), \
                    patch.object(evaluation.subprocess, 'run') as run:
                with self.assertRaises(FileExistsError):
                    evaluation.main()
                run.assert_not_called()
            self.assertEqual(marker.read_text(), 'original evidence')
