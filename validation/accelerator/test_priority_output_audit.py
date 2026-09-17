import copy
from pathlib import Path
import tempfile
import unittest
import yaml

from audit_priority_completed_events import check_termination, physics_identity


class OutputAuditTest(unittest.TestCase):
    def setUp(self):
        self.stats=dict(cross_species=dict(final_pending_photons=0,final_pending_leptons=0),
            deferred_cpu_fallbacks_queued=3,deferred_cpu_fallbacks_flushed=3,
            proposal_native=dict(inverse_failures=0),queue_overflows=0,
            profile=dict(fixed_point_overflows=0),radio=dict(fixed_point_overflows=0),
            energy_ledger=dict(complete_coverage=False,accepted=False,relative_closure_error=.2))

    def test_incomplete_ledger_is_not_erased(self):
        self.assertEqual(check_termination(self.stats),self.stats['energy_ledger'])
        self.assertFalse(check_termination(self.stats)['accepted'])

    def test_leftover_particles_rejected(self):
        for pid in ('photons','leptons'):
            s=copy.deepcopy(self.stats);s['cross_species']['final_pending_'+pid]=1
            with self.assertRaises(AssertionError):check_termination(s)

    def test_missing_fallback_commit_rejected(self):
        self.stats['deferred_cpu_fallbacks_flushed']=2
        with self.assertRaises(AssertionError):check_termination(self.stats)

    def test_device_errors_rejected(self):
        for key,field in (('profile','fixed_point_overflows'),('radio','fixed_point_overflows'),
                          ('proposal_native','inverse_failures')):
            s=copy.deepcopy(self.stats);s[key][field]=1
            with self.assertRaises(AssertionError):check_termination(s)


class PhysicsIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='priority-identity-test-')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        files={'gpu_em/config.yaml':dict(environment=dict(antenna_file='/irrelevant/path',field=[1,2,3]),
                    magnetic_rigidity_GeV_per_T_m=.29979251468343893,scalar_transport_constants_version=1),
               'primary/summary.yaml':dict(shower_0=dict(pdg=1000260560,total_energy=100000)),
               'CoREAS/config.yaml':dict(observers=dict(location=[1,2,3],duration=400)),
               'ZHS/config.yaml':dict(observers=dict(location=[1,2,3],duration=400))}
        for name,value in files.items():
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(yaml.safe_dump(value))

    def test_execution_and_seed_do_not_change_conditions(self):
        a=['bin','-E','100000','-s','1','--kokkos-execution','cuda','--kokkos-num-threads','1','-f','a']
        b=['other','-E','100000','-s','2','--kokkos-execution','openmp-cuda','--kokkos-num-threads','130','-f','b']
        self.assertEqual(physics_identity(self.root,a),physics_identity(self.root,b))

    def test_physical_cli_or_actual_radio_geometry_change_is_detected(self):
        cmd=['bin','-E','100000','--emthin','1e-6']
        first=physics_identity(self.root,cmd)['sha256']
        self.assertNotEqual(first,physics_identity(self.root,['bin','-E','100000','--emthin','1e-4'])['sha256'])
        path=self.root/'CoREAS/config.yaml'
        value=yaml.safe_load(path.read_text());value['observers']['location'][0]+=1
        path.write_text(yaml.safe_dump(value))
        self.assertNotEqual(first,physics_identity(self.root,cmd)['sha256'])


if __name__=='__main__':
    unittest.main()
