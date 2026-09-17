"""Validation options only: small temporary fixtures; never a shower or service."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from run_priority_endpoints_acceptance import (
    fe_timing_family, override_fe_energy, positive_fe_energy,
    validate_correctness_reuse)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_reference(root):
    bins = root/'binaries'
    bins.mkdir()
    (bins/'c8_air_shower').write_bytes(b'fake captured application')
    (bins/'testKokkosCooperativeBackend').write_bytes(b'fake captured fixture')
    identity = dict(
        binary_sha256=hashlib.sha256((bins/'c8_air_shower').read_bytes()).hexdigest(),
        fixture_sha256=hashlib.sha256((bins/'testKokkosCooperativeBackend').read_bytes()).hexdigest(),
        before_sha256=hashlib.sha256(b'fake before binary').hexdigest(),
        source='/frozen/source', threads=20)
    write_json(root/'PROVENANCE.json', identity)
    write_json(root/'STATUS.json', {'complete': True})
    write_json(root/'CORRECTNESS_GATES.json', {'passed': True})
    for mode in ('proposal', 'cuda', 'openmp'):
        write_json(root/(mode+'-regression.json'), {'pass': True})
        if mode != 'proposal':
            for side in ('before', 'after'):
                (root/(side+'-'+mode+'-trace.csv')).write_text('fixed same decision tape\n')
    for mode in ('cuda-openmp', 'openmp-cuda'):
        for label in ('fixture-'+mode, 'N32-'+mode):
            write_json(root/(label+'-guard/summary.json'), {'pass': True, 'returncode': 0})
        write_json(root/('N32-'+mode+'-CHECK.json'),
            dict(events=32, timing={'shower_'+str(i): {'closed': True} for i in range(32)},
                 accelerators=[{'accelerator': {'backend': mode}} for _ in range(32)]))
    return identity


class EnergyOptionsTest(unittest.TestCase):
    def test_default_does_not_modify_template_or_label(self):
        command = ['c8', '-E', '100000', '--max-weight', '0', '--emthin', '1e-6']
        original = list(command)
        override_fe_energy(command, None)
        self.assertEqual(command, original)
        self.assertEqual(fe_timing_family(), 'Fe100TeV')

    def test_override_changes_energy_only_and_actual_family(self):
        command = ['c8', '-E', '100000', '--max-weight', '0', '--emthin', '1e-6']
        override_fe_energy(command, 10000)
        self.assertEqual(command, ['c8', '-E', '10000.0', '--max-weight', '0', '--emthin', '1e-6'])
        self.assertEqual(fe_timing_family(10000), 'Fe10TeV')
        self.assertEqual(fe_timing_family(1500), 'Fe1.5TeV')
        self.assertEqual(fe_timing_family(10), 'Fe0.01TeV')
        self.assertNotEqual(fe_timing_family(1e-320), 'Fe0TeV')

    def test_invalid_energy_rejected(self):
        for value in ('nan', 'inf', '-inf', '0', '-1', 'not-energy', None):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                positive_fe_energy(value)
        self.assertEqual(positive_fe_energy('1e4'), 10000.)


class CorrectnessReuseTest(unittest.TestCase):
    def test_completed_matching_reference_returns_hashed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = make_reference(root)
            result = validate_correctness_reuse(root, identity)
            self.assertTrue(result['passed'])
            self.assertEqual(result['reference_run'], str(root.resolve()))
            self.assertEqual(len(result['files']), 18)
            for row in result['files']:
                self.assertEqual(row['sha256'], hashlib.sha256(Path(row['path']).read_bytes()).hexdigest())
            self.assertEqual(result['matching_identity'], identity)
            self.assertIn('no reuse of performance', result['scope'])

    def test_incomplete_failed_or_chained_reference_rejected(self):
        for filename, updates in (
                ('STATUS.json', {'complete': False}),
                ('CORRECTNESS_GATES.json', {'passed': False}),
                ('PROVENANCE.json', {'correctness_reused_from': '/another/run'})):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); identity = make_reference(root)
                value = json.loads((root/filename).read_text()); value.update(updates)
                write_json(root/filename, value)
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    validate_correctness_reuse(root, identity)

    def test_every_current_identity_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); identity = make_reference(root)
            for key in identity:
                changed = dict(identity)
                changed[key] = 21 if key == 'threads' else 'different'
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    validate_correctness_reuse(root, changed)

    def test_modified_archived_binary_or_trace_refused(self):
        for filename in ('binaries/c8_air_shower', 'binaries/testKokkosCooperativeBackend',
                         'after-cuda-trace.csv', 'after-openmp-trace.csv'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); identity = make_reference(root)
                (root/filename).write_bytes(b'changed')
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    validate_correctness_reuse(root, identity)

    def test_failed_regression_guard_and_unclosed_n32_refused(self):
        for filename, update in (
                ('proposal-regression.json', {'pass': False}),
                ('cuda-regression.json', {'pass': False}),
                ('openmp-regression.json', {'pass': False}),
                ('fixture-openmp-cuda-guard/summary.json', {'pass': False}),
                ('fixture-cuda-openmp-guard/summary.json', {'returncode': 1}),
                ('N32-openmp-cuda-guard/summary.json', {'pass': False}),
                ('N32-cuda-openmp-CHECK.json', {'events': 31}),
                ('N32-openmp-cuda-CHECK.json', {'timing': {'shower_0': {'closed': False}}})):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); identity = make_reference(root)
                value = json.loads((root/filename).read_text()); value.update(update)
                write_json(root/filename, value)
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    validate_correctness_reuse(root, identity)

    def test_missing_evidence_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); identity = make_reference(root)
            (root/'N32-openmp-cuda-guard/summary.json').unlink()
            with self.assertRaises(FileNotFoundError):
                validate_correctness_reuse(root, identity)


if __name__ == '__main__':
    unittest.main()

