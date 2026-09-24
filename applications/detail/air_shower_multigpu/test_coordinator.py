import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location('coordinator', Path(__file__).with_name('run_multigpu.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class CoordinatorTests(unittest.TestCase):
    def test_frontier_consumption_must_match_partition(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feed = dict(mode='bounded-high-energy-first', roots_consumed=3,
                        batches=2, maximum_imported_batch=2)
            (root/'FRONTIER_FEED.json').write_text(json.dumps(feed))
            m.validate_frontier_feed(root, 3)
            with self.assertRaises(ValueError):
                m.validate_frontier_feed(root, 4)
            feed['maximum_imported_batch'] = 1000000
            (root/'FRONTIER_FEED.json').write_text(json.dumps(feed))
            with self.assertRaises(ValueError):
                m.validate_frontier_feed(root, 3)
    def test_explicit_ninety_percent_budget(self):
        self.assertEqual(m.memory_fraction(.9), .9)
        self.assertEqual(m.memory_fraction(.5), .5)
        for value in (0, -.1, .91, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                m.memory_fraction(value)

    def test_partition_exact_once_including_empty_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'all.txt'
            rows = ['22 1 0 0 0 8 2 0 0 0 1 0 0 -1', '11 2 1 1 9 2 4 3 0 0 1 0 0 -1']
            source.write_text(m.HEADER + '\n' + '\n'.join(rows) + '\n')
            dest = [root / str(i) for i in range(4)]
            result = m.partition(source, dest)
            self.assertEqual(result['counts'], [1, 1, 0, 0])
            self.assertEqual(result['weighted_energy_GeV'], 24)
            self.assertEqual(sorted(x for p in dest for x in p.read_text().splitlines()[1:]), sorted(rows))

    def test_duplicate_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'all'
            source.write_text(m.HEADER + '\n' + '22 1 0 0 0 8 2 0 0 0 1 0 0 -1\n' * 2)
            with self.assertRaises(ValueError):
                m.partition(source, [Path(tmp) / 'part'])

    def make_outputs(self, root, value):
        for folder in ('profile', 'production_profile', 'energyloss', 'CoREAS', 'ZHS'):
            (root / folder).mkdir(parents=True)
            radio = folder in ('CoREAS', 'ZHS')
            file = 'observers.parquet' if radio else ('dEdX.parquet' if folder == 'energyloss' else 'profile.parquet')
            data = {'shower': np.zeros(3, dtype=np.uint32), 'Time' if radio else 'X': [0., 1., 2.],
                    'Ex' if radio else 'total': [value, value, value]}
            pq.write_table(pa.table(data), root / folder / file)
            (root / folder / 'config.yaml').write_text('type: test\n')
        for folder, file in [('particles', 'particles.parquet'), ('interactions', 'interactions.parquet')]:
            (root / folder).mkdir()
            pq.write_table(pa.table({'shower': [0], 'pdg': [11]}), root / folder / file)
            (root / folder / 'config.yaml').write_text('type: test\n')

    def test_linear_sum_not_power(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = [root / str(i) for i in range(3)]
            for p, value in zip(parts, (3., -2., 4.)):
                self.make_outputs(p, value)
            result = m.merge(parts, root / 'merged')
            self.assertEqual(result['particles/particles.parquet']['rows'], 3)
            self.assertEqual(result['interactions/interactions.parquet']['rows'], 1)
            self.assertEqual(pq.read_table(root / 'merged/CoREAS/observers.parquet')['Ex'].to_pylist(), [5.] * 3)

    def test_config_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = [root / 'a', root / 'b']
            for p in parts:
                self.make_outputs(p, 1.)
            (parts[1] / 'CoREAS/config.yaml').write_text('type: DIFFERENT\n')
            with self.assertRaises(ValueError):
                m.merge(parts, root / 'merged')
            self.assertFalse((root / 'COMPLETE.json').exists())

    def test_empty_radio_is_not_accepted_as_radio_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_outputs(root, 1.)
            path = root / 'CoREAS/observers.parquet'
            table = pq.read_table(path).slice(0, 0)
            pq.write_table(table, path)
            with self.assertRaisesRegex(ValueError, 'Empty required output'):
                m.validate_closed_outputs(root)

    def test_nonfinite_data_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_outputs(root / 'a', float('nan'))
            with self.assertRaises(ValueError):
                m.merge([root / 'a'], root / 'merged')

    def test_changed_time_grid_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for p in (root / 'a', root / 'b'):
                self.make_outputs(p, 1.)
            path = root / 'b/CoREAS/observers.parquet'
            table = pq.read_table(path).to_pydict()
            table['Time'] = [1., 2., 3.]
            pq.write_table(pa.table(table), path)
            with self.assertRaises(ValueError):
                m.merge([root / 'a', root / 'b'], root / 'merged')

class LeaseTests(unittest.TestCase):
    def test_resume_does_not_signal_reused_pid_or_prepaused_job(self):
        spec = importlib.util.spec_from_file_location('lease', Path(__file__).with_name('production_lease.py'))
        lease = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lease)
        items = [dict(pid=1, state='T'), dict(pid=2, state='S'), dict(pid=3, state='S')]
        with patch.object(lease, 'matches', side_effect=lambda p: p['pid'] == 3), patch.object(lease.os, 'kill') as kill:
            lease.resume(items)
            kill.assert_called_once_with(3, lease.signal.SIGCONT)

if __name__ == '__main__':
    unittest.main()
