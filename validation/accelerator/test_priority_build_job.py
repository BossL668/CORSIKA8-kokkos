import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from launch_priority_build_job import invoke_job, load_job


class BuildJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root/'inputs'
        (self.bundle/'tools').mkdir(parents=True)
        self.gate = self.bundle/'tools/wait_priority_build_isolation.py'
        # Deliberately NOT the real gate: inspect argv and return immediately.
        self.gate.write_text('import json,sys\nfrom pathlib import Path\n'
            'print(json.dumps({"fake_gate": True, "memory_argv": sys.argv, '
            '"os_argv": Path("/proc/self/cmdline").read_bytes().decode().split("\\0")}))\n')
        self.gate_sha = hashlib.sha256(self.gate.read_bytes()).hexdigest()
        manifest = dict(files=[dict(path='tools/wait_priority_build_isolation.py', sha256=self.gate_sha)])
        (self.bundle/'MANIFEST.json').write_text(json.dumps(manifest))
        self.fragment = '/foreign/mountain-workflow-that-must-not-be-in-os-argv'
        self.job = dict(schema=1, gate_bundle=str(self.bundle),
            gate_bundle_sha256=hashlib.sha256((self.bundle/'MANIFEST.json').read_bytes()).hexdigest(),
            gate_sha256=self.gate_sha, gate_arguments=['--bundle', str(self.bundle),
                '--wait-workflow-path', self.fragment, '--execute', '--', '/never-started/build'])
        self.path = self.root/'JOB.json'

    def save(self):
        self.path.write_text(json.dumps(self.job))
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def test_verify_only_never_invokes_gate(self):
        expected = self.save()
        with patch('launch_priority_build_job.runpy.run_path') as run, contextlib.redirect_stdout(io.StringIO()):
            receipt = invoke_job(self.path, expected)
        run.assert_not_called()
        self.assertTrue(receipt['integrity_only'])

    def test_rejects_changed_job_manifest_or_gate(self):
        expected = self.save()
        self.path.write_text(self.path.read_text()+' ')
        with self.assertRaises(ValueError): load_job(self.path, expected)
        expected = self.save()
        self.gate.write_text(self.gate.read_text()+'# changed\n')
        with self.assertRaises(ValueError): load_job(self.path, expected)
        self.gate.write_text('not original')
        self.job['gate_sha256'] = hashlib.sha256(self.gate.read_bytes()).hexdigest()
        with self.assertRaises(ValueError): load_job(self.path, self.save())

    def test_rejects_argument_bundle_mismatch_and_bad_command(self):
        self.job['gate_arguments'][1] = '/different/bundle'
        with self.assertRaises(ValueError): load_job(self.path, self.save())
        self.job['gate_arguments'] = ['--bundle', str(self.bundle), '--execute', '--']
        with self.assertRaises(ValueError): load_job(self.path, self.save())

    def test_memory_argv_restored_even_when_gate_raises(self):
        expected = self.save()
        before = sys.argv
        with patch('launch_priority_build_job.runpy.run_path', side_effect=RuntimeError('fake gate')), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError): invoke_job(self.path, expected, True)
        self.assertIs(sys.argv, before)

    def test_real_proc_argv_does_not_match_old_workflow_predicate(self):
        expected = self.save()
        launcher = Path(__file__).with_name('launch_priority_build_job.py')
        command = [sys.executable, str(launcher), '--job', str(self.path),
                   '--job-sha256', expected, '--execute']
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
        row = next(json.loads(line) for line in result.stdout.splitlines()
                   if json.loads(line).get('fake_gate'))
        self.assertIn(self.fragment, row['memory_argv'])
        # Exactly the frozen v4 workflow_pids() cmdline predicate.
        self.assertFalse(any(self.fragment in arg for arg in row['os_argv']))
        self.assertEqual([arg for arg in row['os_argv'] if arg], command)
        self.assertNotIn('--wait-workflow-path', row['os_argv'])


if __name__ == '__main__':
    unittest.main()
