import copy
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import unittest

from report_cpu_primary_wait_contention import (
    MODES, HASH_KEYS, build_report, canonical_bytes, markdown)


class WaitContentionReportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.guard = dict(returncode=0, failure=None, performance_valid=True,
            exclusive_gpu_required=True, gpu_observation_failures=0,
            foreign_gpu_pids=[], gpu_checks=6, peak_tree_rss_bytes=3*2**30,
            **{'pass': True})
        self.rows = []
        for repeat in range(6):
            for order in range(3):
                mode = MODES[(order+repeat) % 3]
                # The deliberately slow measured outlier MUST remain recorded.
                base = (999. if repeat == 0 else [8., 9., 10., 11., 100.][repeat-1])
                cpu_s = base * {'helper-off': 1., 'default-fence': 1.3,
                                'blocking-event': 1.1}[mode]
                cpu = self.endpoint('cpu', cpu_s)
                gpu = None if mode == 'helper-off' else self.endpoint('gpu', 4.)
                if gpu is not None and mode == 'blocking-event':
                    gpu.update(blocking_wait_enabled=True, blocking_wait_calls=24,
                               blocking_wait_host_seconds=3., driver_thread_cpu_seconds=.2,
                               transport_driver_thread_cpu_seconds=.2)
                row = dict(repeat=repeat, order=order, warmup=repeat == 0, mode=mode,
                    cpu=cpu, gpu=gpu, joint_wall_seconds=max(cpu_s, 4.),
                    joint_process_cpu_seconds=130*cpu_s, rss_bytes=2*2**30,
                    available_memory_bytes=8*2**30)
                if gpu is not None:
                    row['transport_host_call_window_overlap_seconds'] = min(cpu_s, 4.)
                self.rows.append(row)

    @staticmethod
    def endpoint(name, transport):
        counts = dict(particles_advanced=0, photon_transport_records=123, lepton_transport_records=456)
        hashes = {key: hashlib.sha256((name+key).encode()).hexdigest() for key in HASH_KEYS}
        hashes['physical_counts'] = hashlib.sha256(canonical_bytes(counts)).hexdigest()
        return dict(hashes=hashes, physical_counts=counts, transport_seconds=transport,
            photon_transport_records=123, lepton_transport_records=456,
            driver_thread_cpu_seconds=1., transport_driver_thread_cpu_seconds=1.,
            output_download_seconds=.1, blocking_wait_host_seconds=0.,
            blocking_wait_enabled=False, blocking_wait_calls=0)

    def result(self):
        value = dict(schema=2, scope='synthetic test only', threads=130,
            complete=True, warmups_per_mode=1, measured_repeats_per_mode=5,
            same_cpu_workload_and_outputs_exact=True,
            same_gpu_workload_and_outputs_exact=True,
            wait_mode_and_lifecycle_tests_passed=True, runs=copy.deepcopy(self.rows),
            medians={})
        medians = {}
        for mode in MODES:
            rows = [r for r in self.rows if r['mode'] == mode and not r['warmup']]
            medians[mode] = statistics.median(r['cpu']['transport_seconds'] for r in rows)
        for mode in MODES:
            value['medians'][mode] = dict(cpu_transport_seconds=medians[mode],
                cpu_time_ratio_to_helper_off=medians[mode]/medians['helper-off'])
            if mode != 'helper-off':
                value['medians'][mode]['gpu_driver_thread_cpu_seconds'] = statistics.median(
                    r['gpu']['driver_thread_cpu_seconds'] for r in self.rows
                    if r['mode'] == mode and not r['warmup'])
        return value

    def save(self, logged=None, results=None, guard=None):
        logged = self.rows if logged is None else logged
        results = [self.result()] if results is None else results
        lines = ['unrelated PROPOSAL log line']
        lines += ['CONTENTION_REPEAT '+json.dumps(r) for r in logged]
        lines += ['CONTENTION_RESULT '+json.dumps(r) for r in results]
        (self.path/'command.log').write_text('\n'.join(lines)+'\n')
        (self.path/'summary.json').write_text(json.dumps(self.guard if guard is None else guard))
        return build_report(self.path)

    def test_complete_all_repeats_retained_medians_and_separate_scope(self):
        report = self.save()
        self.assertTrue(report['checks']['correctness_passed'])
        self.assertTrue(report['checks']['performance_evidence_valid'])
        self.assertFalse(report['checks']['known_interference_present'])
        self.assertNotIn('known_interference', report['sources'])
        self.assertFalse(report['checks']['whole_shower_performance_accepted'])
        self.assertEqual(len(report['repeats']), 18)
        self.assertEqual(len(report['results']), 1)
        self.assertEqual(report['mode_summaries']['helper-off']['cpu_transport_median_seconds'], 10.)
        self.assertAlmostEqual(report['mode_summaries']['blocking-event']['cpu_transport_ratio_to_helper_off'], 1.1)
        self.assertEqual(report['mode_summaries']['blocking-event']['gpu_blocking_wait_calls_median'], 24)
        self.assertEqual(report['mode_summaries']['blocking-event']['gpu_driver_thread_cpu_seconds_median'], .2)
        self.assertEqual(report['mode_summaries']['helper-off']['cpu_photon_transport_records'], 123)
        text = markdown(report, 'a'*64)
        self.assertIn('100.000', text)  # Retain the slow observation, not just median.
        self.assertIn('999.000', text)  # Warmup is retained but excluded from median.
        self.assertIn('不是完整 shower', text)
        self.assertIn('CPU s', text)
        self.assertIn('RESULT.json', text)
        self.assertEqual(report['sources']['command_log']['sha256'], hashlib.sha256(
            (self.path/'command.log').read_bytes()).hexdigest())

    def test_missing_final_result_keeps_partial_rows_and_refuses_timing(self):
        report = self.save(logged=self.rows[:8], results=[])
        self.assertEqual(len(report['repeats']), 8)
        self.assertFalse(report['checks']['result_complete'])
        self.assertFalse(report['checks']['performance_evidence_valid'])
        self.assertIn('不能声称性能通过', markdown(report))

    def test_duplicate_repeat_not_deduplicated_even_with_matching_final_runs(self):
        result = self.result()
        logged = self.rows+[copy.deepcopy(self.rows[-1])]
        result['runs'] = logged
        report = self.save(logged=logged, results=[result])
        self.assertEqual(len(report['repeats']), 19)
        self.assertTrue(report['duplicates'])
        self.assertFalse(report['checks']['repeat_matrix_complete'])
        self.assertFalse(report['checks']['performance_evidence_valid'])

    def test_duplicate_final_result_refused(self):
        result = self.result()
        report = self.save(results=[result, copy.deepcopy(result)])
        self.assertEqual(len(report['results']), 2)
        self.assertFalse(report['checks']['unique_result'])
        self.assertFalse(report['checks']['performance_evidence_valid'])

    def test_invalid_monitor_keeps_success_and_all_timings_but_not_performance(self):
        self.guard.update(performance_valid=False, gpu_observation_failures=1)
        report = self.save()
        self.assertTrue(report['checks']['guard_passed'])
        self.assertTrue(report['checks']['correctness_passed'])
        self.assertFalse(report['checks']['performance_evidence_valid'])
        self.assertEqual(len(report['repeats']), 18)
        self.assertEqual(report['mode_summaries']['helper-off']['cpu_transport_median_seconds'], 10.)
        self.assertIn('不能声称性能通过', markdown(report))

    def test_cpu_gpu_hashes_independently_checked_not_just_trusted_flags(self):
        self.rows[-1]['cpu']['hashes']['input'] = 'c'*64
        next(r for r in reversed(self.rows) if r['gpu'] is not None)['gpu']['hashes']['call_sequence'] = 'd'*64
        report = self.save()
        self.assertFalse(report['checks']['cpu_hashes_exact'])
        self.assertFalse(report['checks']['gpu_hashes_exact'])
        self.assertFalse(report['checks']['performance_evidence_valid'])

    def test_claimed_median_or_physical_count_tampering_is_rejected(self):
        result = self.result()
        result['medians']['blocking-event']['cpu_transport_seconds'] = .001
        report = self.save(results=[result])
        self.assertTrue(any('reported median disagrees' in p for p in report['issues']))
        self.assertFalse(report['checks']['performance_evidence_valid'])
        self.rows[0]['cpu']['physical_counts']['lepton_transport_records'] += 1
        report = self.save()
        self.assertTrue(any('physical_counts hash' in p for p in report['issues']))
        self.assertFalse(report['checks']['correctness_passed'])

    def test_incomplete_flag_and_guard_success_do_not_override_failure(self):
        result = self.result()
        result['complete'] = False
        report = self.save(results=[result])
        self.assertTrue(report['checks']['guard_passed'])
        self.assertFalse(report['checks']['performance_evidence_valid'])
        bad_guard = dict(self.guard, returncode=1)
        report = self.save(guard=bad_guard)
        self.assertTrue(report['checks']['correctness_passed'])
        self.assertFalse(report['checks']['guard_passed'])
        self.assertFalse(report['checks']['performance_evidence_valid'])

    def test_schema_three_single_snapshot_scope_preserved(self):
        result = self.result()
        result['schema'] = 3
        result['snapshot_decode_validation'] = 'same single integer snapshot, not independent download'
        report = self.save(results=[result])
        self.assertTrue(report['checks']['performance_evidence_valid'])
        self.assertEqual(report['probe_settings']['snapshot_decode_validation'],
                         result['snapshot_decode_validation'])
        self.assertIn('不是独立普通下载路径', markdown(report))

    def test_failed_probe_stderr_is_retained_and_actual_step_counts_checked(self):
        self.save(logged=self.rows[:1], results=[], guard=dict(self.guard, returncode=1))
        with (self.path/'command.log').open('a') as stream:
            stream.write('wait mode changed after transport\n')
        report = build_report(self.path)
        self.assertIn('wait mode changed after transport', markdown(report))
        self.assertFalse(report['checks']['performance_evidence_valid'])
        self.rows[0]['cpu']['photon_transport_records'] += 1
        report = self.save()
        self.assertTrue(any('inconsistent actual photon_transport_records' in p for p in report['issues']))

    def test_known_cpu_interference_overrides_timing_but_not_correctness(self):
        manifest = dict(reason='correctness runner CPU-only imports overlapped the final 3 seconds',
                        interval_utc=['2026-09-12T06:06:44Z', '2026-09-12T06:06:47Z'])
        marker = self.path/'KNOWN_INTERFERENCE.json'
        marker.write_text(json.dumps(manifest))
        report = self.save()
        self.assertTrue(report['checks']['correctness_passed'])
        self.assertTrue(report['checks']['guard_passed'])
        self.assertTrue(report['checks']['guard_performance_valid'])
        self.assertTrue(report['checks']['monitor_valid'])
        self.assertFalse(report['checks']['performance_evidence_valid'])
        self.assertEqual(len(report['repeats']), 18)
        self.assertEqual(report['known_interference']['manifest'], manifest)
        self.assertEqual(report['sources']['known_interference']['sha256'],
                         hashlib.sha256(marker.read_bytes()).hexdigest())
        self.assertIn(manifest['reason'], markdown(report))
        self.assertIn('不能声称性能通过', markdown(report))

    def test_malformed_interference_manifest_fails_closed_and_retains_hash(self):
        for raw in ('{bad json', '[]', '{"reason": NaN}'):
            with self.subTest(raw=raw):
                marker = self.path/'KNOWN_INTERFERENCE.json'
                marker.write_text(raw)
                report = self.save()
                self.assertTrue(report['checks']['correctness_passed'])
                self.assertTrue(report['checks']['known_interference_present'])
                self.assertFalse(report['checks']['performance_evidence_valid'])
                self.assertTrue(report['known_interference']['issues'])
                self.assertEqual(report['sources']['known_interference']['sha256'],
                                 hashlib.sha256(marker.read_bytes()).hexdigest())
                json.dumps(report, allow_nan=False)  # Malformed input remains reportable.

    def test_broken_interference_marker_is_not_treated_as_absent(self):
        (self.path/'KNOWN_INTERFERENCE.json').symlink_to(self.path/'missing-marker-target')
        report = self.save()
        self.assertTrue(report['checks']['correctness_passed'])
        self.assertTrue(report['checks']['known_interference_present'])
        self.assertFalse(report['checks']['performance_evidence_valid'])
        self.assertTrue(report['known_interference']['issues'])


if __name__ == '__main__':
    unittest.main()
