"""The memory regression must not accidentally waive physical differences."""
import importlib.util
from pathlib import Path


source = Path(__file__).resolve().parents[2] / 'validation/accelerator/compare_event_memory_metadata.py'
spec = importlib.util.spec_from_file_location('memory_metadata', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_accepts_measured_duration_not_physical_time():
    assert module.allowed(('statistics', 'kernel_time_ms'), 1., 2.)
    assert module.allowed(('statistics', 'hybrid_timing_ms', 'total_run'), 1., 2.)
    assert not module.allowed(('statistics', 'radio', 'weighted_time_residual_s'), 1., 2.)
    assert not module.allowed(('time',), 1., 2.)


def test_counts_versions_hashes_and_schema_are_strict():
    for key in ('gpu_particles', 'table_sha256', 'physics_alignment_revision',
                'proposal_version', 'rng_domain_version', '<keys/order>'):
        assert not module.allowed(('statistics', key), 1, 2)
    assert list(module.walk({'a': 1, 'b': 2}, {'b': 2, 'a': 1}))
    assert list(module.walk({'x': [1]}, {'x': [1, 2]}))
    assert not list(module.walk({'x': float('nan')}, {'x': float('nan')}))
