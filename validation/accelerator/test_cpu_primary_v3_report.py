import json
from pathlib import Path
import tempfile
import unittest

from report_cpu_primary_v3 import build_report, markdown


class CpuPrimaryReportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.entries = []
        self.save('PROVENANCE.json', dict(threads=130, binary_sha256='fixture'))

    def save(self, path, value):
        target = self.run/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))

    def add(self, mode, seed, seconds, valid=True, complete=True, energy='100000'):
        label = '{}-{}'.format(seed, mode)
        self.entries.append(dict(label=label, mode=mode, seed=seed, family='Fe100TeV',
                                 guard=dict(process_wall_s=-123)))
        self.save(label+'-guard/summary.json', dict(
            command=['/bin/frozen', '-E', energy, '-s', str(seed), '-f', label,
                     '--kokkos-execution', mode, '--kokkos-num-threads', '130'],
            process_wall_s=seconds, performance_valid=valid, returncode=0,
            failure=None, **{'pass': True}))
        samples = [dict(elapsed_s=i, cpu_seconds=2*i, rss_bytes=1024,
                        available_bytes=10*2**30, device_util_percent=25,
                        device_used_mib=512) for i in range(3)]
        (self.run/(label+'-guard/resources.jsonl')).write_text(
            '\n'.join(json.dumps(s) for s in samples)+'\n')
        stats = dict(gpu_particles=1000, profile=dict(steps=1000),
            resident_photon_wavefronts=10, resident_lepton_wavefronts=90,
            photon_backend_wall_time_ms=100, lepton_backend_wall_time_ms=100,
            accelerator=dict(backend=mode), deferred_cpu_fallbacks_queued=5,
            deferred_cpu_fallbacks_flushed=5, deferred_cpu_fallback_flushes=1,
            maximum_deferred_cpu_fallback_batch=5, hybrid_timing_ms=dict(scalar_stepper=30))
        if mode == 'openmp-cuda':
            stats['accelerator']['cooperative'] = dict(scheduling_policy='cpu-primary-v3-protected-front',
                openmp_wall_ms=2000, cuda_driver_wall_ms=1000, cuda_result_service_delay_ms=100,
                priority_endpoints={e: {k: dict(transport_records=n, resident_wavefronts=10)
                    for k, n in [('photon', photons), ('lepton', leptons)]}
                    for e, photons, leptons in [('openmp', 100, 650), ('cuda', 50, 200)]})
        self.save(label+'/gpu_em/summary.yaml', dict(shower_0=dict(complete=complete, statistics=stats)))
        self.save(label+'/simulation_timing/summary.yaml', dict(shower_0=dict(closed=complete, wall_time_ms=seconds*1000-20)))

    def report(self, complete=False):
        self.save('STATUS.json', dict(complete=complete, phase='testing', records=self.entries))
        return build_report(self.run)

    def test_actual_guard_and_endpoint_call_denominators(self):
        self.add('openmp', 1, 10)
        self.add('openmp-cuda', 1, 12)
        report = self.report()
        pure, dual = report['records']
        self.assertEqual(pure['process_s'], 10)  # Not stale embedded STATUS guard.
        self.assertEqual(pure['endpoints']['openmp']['steps_per_backend_call_s'], 5000)
        self.assertEqual(dual['endpoints']['openmp']['steps_per_backend_call_s'], 375)
        self.assertEqual(dual['endpoints']['cuda']['steps_per_backend_call_s'], 250)
        self.assertEqual(dual['cpu_step_share'], .75)
        self.assertEqual(dual['fallback']['queued'], 5)
        self.assertEqual(dual['resources']['mean_cpu_core_equivalents'], 2)
        self.assertAlmostEqual(report['pairs'][0]['process_ratio_openmp_over_dual'], 10/12)

    def test_slow_monitor_invalid_pair_is_kept(self):
        for mode, seed, seconds, valid in [('openmp', 1, 10, True),
                ('openmp-cuda', 1, 20, True), ('openmp', 2, 20, True),
                ('openmp-cuda', 2, 200, False)]:
            self.add(mode, seed, seconds, valid)
        report = self.report(True)
        self.assertEqual(len(report['records']), 4)
        self.assertEqual(report['paired_groups'][0]['n'], 2)
        self.assertEqual(report['paired_groups'][0]['monitor_valid'], 1)
        self.assertAlmostEqual(report['paired_groups'][0]['ratio_of_paired_median_process_times'], 15/110)
        self.assertFalse(report['performance_gate_certified'])
        self.assertIn('200.000', markdown(report))

    def test_partial_missing_output_and_parameter_mismatch_not_paired(self):
        self.add('openmp', 1, 10)
        self.add('openmp-cuda', 1, 12, energy='1000')
        self.add('openmp-cuda', 2, 30)
        (self.run/'2-openmp-cuda/gpu_em/summary.yaml').unlink()
        report = self.report()
        self.assertEqual(len(report['records']), 3)
        self.assertFalse(report['state_complete'])
        self.assertFalse(report['pairs'][0]['physics_flags_match'])
        self.assertEqual(report['paired_groups'][0]['n'], 0)
        self.assertTrue(report['records'][2]['issues'])
        self.assertFalse(report['records'][2]['completed'])

    def test_duplicate_seed_mode_refused_and_empty_partial_supported(self):
        self.assertEqual(self.report()['records'], [])
        self.add('openmp', 1, 10)
        self.entries.append(dict(self.entries[0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.report()

    def test_v4_helper_counters_read_without_inferring_performance_pass(self):
        self.add('openmp-cuda', 1, 12)
        path = '1-openmp-cuda/gpu_em/summary.yaml'
        summary = json.loads((self.run/path).read_text())
        summary['shower_0']['statistics']['accelerator']['cooperative'].update(
            scheduling_policy='cpu-primary-v4-resident-helper',
            cuda_completion_packets=3, subshower_cuda_autonomous_continuations=4,
            maximum_cuda_packet_calls=4, cuda_completion_peak_bytes=2**20,
            cuda_continuation_stop_order=['completed', 'handoff'],
            cuda_continuation_stops=[1, 2])
        self.save(path, summary)
        report = self.report()
        helper = report['records'][0]['helper']
        self.assertEqual(helper['packets'], 3)
        self.assertEqual(helper['continuations'], 4)
        self.assertEqual(helper['retained_peak_bytes'], 2**20)
        text = markdown(report)
        self.assertIn('cpu-primary-v4-resident-helper', text)
        self.assertIn('completed:1, handoff:2', text)
        self.assertFalse(report['performance_gate_certified'])


    def test_v6_coalescing_report_does_not_claim_saved_work(self):
        self.add('openmp-cuda', 1, 12)
        path = '1-openmp-cuda/gpu_em/summary.yaml'
        summary = json.loads((self.run/path).read_text())
        summary['shower_0']['statistics']['accelerator']['cooperative'].update(
            scheduling_policy='cpu-primary-v6-species-coalescing',
            host_species_coalesces=7, subshower_openmp_epochs=42)
        self.save(path, summary)
        report = self.report()
        self.assertEqual(report['records'][0]['host_species_coalesces'], 7)
        self.assertIn('| 1 | 7 | 42 |', markdown(report))
        self.assertFalse(report['performance_gate_certified'])


if __name__ == '__main__':
    unittest.main()
