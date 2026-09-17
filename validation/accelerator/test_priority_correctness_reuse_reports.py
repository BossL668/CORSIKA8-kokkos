"""Small in-memory/disk report fixtures only; no simulation, GPU or service."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml
import check_priority_lifecycle as lifecycle
import summarize_priority_endpoints as summarize
from run_priority_endpoints_acceptance import validate_correctness_reuse
from test_priority_energy_correctness_reuse import make_reference, write_json


def fixture(folder):
    reference=folder/'reference'; reference.mkdir()
    current=folder/'current'; current.mkdir()
    identity=make_reference(reference)
    for mode in ('cuda-openmp','openmp-cuda'):
        rows={}
        for i in range(32):
            rows['shower_'+str(i)]=dict(complete=True,statistics=dict(
                accelerator=dict(cooperative=dict(subshower_cuda_submissions=0,
                    subshower_cuda_commits=0,openmp_workspace_bytes=456)),
                backend_lifecycle=dict(reused=i>0,shower_ordinal=i+1),
                queue_overflows=0,profile=dict(fixed_point_overflows=0),
                radio=dict(fixed_point_overflows=0),workspace_bytes=123,
                proposal_native=dict(inverse_failures=0,table_sha256='table',aux_sha256='aux')))
        path=reference/('N32-'+mode)/'gpu_em/summary.yaml'
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(rows))
        write_json(reference/('N32-'+mode+'-guard/summary.json'),
            dict(pass_=True, **{'pass':True},returncode=0,
                 minimum_available_bytes=5*2**30,performance_valid=False))
    identity.update(correctness_reused_from=str(reference.resolve()),affinity=[0,1],
                    modes=['openmp','openmp-cuda'])
    write_json(current/'PROVENANCE.json',identity)
    write_json(current/'CORRECTNESS_REUSE.json',validate_correctness_reuse(reference,identity))
    write_json(current/'STATUS.json',dict(complete=True,phase='completed',records=[]))
    return current,reference,identity


class CorrectnessReuseReportsTests(unittest.TestCase):
    def test_legacy_source_does_not_require_new_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            self.assertEqual(lifecycle.correctness_source(path),(path,None))

    def test_n32_audit_reads_reference_but_writes_current(self):
        with tempfile.TemporaryDirectory() as folder:
            current,reference,_=fixture(Path(folder))
            with patch('sys.argv',['checker',str(current)]),contextlib.redirect_stdout(io.StringIO()):
                lifecycle.main()
            audit=json.loads((current/'N32_LIFECYCLE_AUDIT.json').read_text())
            self.assertFalse((reference/'N32_LIFECYCLE_AUDIT.json').exists())
            self.assertFalse((current/'N32-openmp-cuda').exists())
            for row in audit.values():
                self.assertTrue(row['pass_'])
                self.assertTrue(row['correctness_reused'])
                self.assertEqual(row['source_run'],str(reference))
                self.assertEqual(len(row['events']),32)
                self.assertFalse(row['guard']['performance_valid'])

    def test_report_points_to_reference_not_missing_current_regression(self):
        with tempfile.TemporaryDirectory() as folder:
            current,reference,_=fixture(Path(folder)); out=current/'report'
            with patch('sys.argv',['summarize',str(current),'--output',str(out)]),contextlib.redirect_stdout(io.StringIO()):
                summarize.main()
            text=(out/'REPORT_CN.md').read_text()
            self.assertIn(str(reference),text)
            self.assertNotIn('记录另见本运行目录',text)
            result=json.loads((out/'REPORT.json').read_text())
            self.assertEqual(result['correctness_reference_run'],str(reference))
            self.assertTrue(result['correctness_reuse_revalidated'])

    def test_saved_receipt_detects_even_still_valid_reference_change(self):
        with tempfile.TemporaryDirectory() as folder:
            current,reference,_=fixture(Path(folder))
            path=reference/'STATUS.json'; path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'receipt no longer matches'):
                lifecycle.correctness_source(current)

    def test_identity_change_or_missing_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            current,_,identity=fixture(Path(folder))
            changed=dict(identity,threads=21)
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                lifecycle.correctness_source(current,changed)
            (current/'CORRECTNESS_REUSE.json').unlink()
            with self.assertRaisesRegex(ValueError,'without its receipt'):
                lifecycle.correctness_source(current,identity)

    def test_legacy_report_retains_original_reference_text(self):
        with tempfile.TemporaryDirectory() as folder:
            current,_,identity=fixture(Path(folder))
            (current/'CORRECTNESS_REUSE.json').unlink()
            identity.pop('correctness_reused_from');write_json(current/'PROVENANCE.json',identity)
            out=current/'report'
            with patch('sys.argv',['summarize',str(current),'--output',str(out)]),contextlib.redirect_stdout(io.StringIO()):
                summarize.main()
            self.assertIn('记录另见本运行目录',(out/'REPORT_CN.md').read_text())
            self.assertNotIn('correctness_reference_run',json.loads((out/'REPORT.json').read_text()))


if __name__=='__main__':unittest.main()
