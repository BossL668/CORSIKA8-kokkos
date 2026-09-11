import csv
import json
from pathlib import Path
import pytest
from analyze_step_replay import audit, endpoint_errors


def fixture(tmp_path, rows, counts):
    (tmp_path/'shower').mkdir();(tmp_path/'guard').mkdir()
    (tmp_path/'shower/terrain_run.yaml').write_text('complete: true\n')
    (tmp_path/'guard/resources.json').write_text(json.dumps(dict(
        returncode=0,stop_reason=None,samples=[dict(rss_kib=1)])))
    (tmp_path/'steps.counts.json').write_text(json.dumps(dict(
        compared_step_pairs=2,fields=100,bit_differences=len(rows),strict_failures=counts)))
    with (tmp_path/'steps.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['mode','stage','field','host','device','ulp','within_gate'])
        w.writeheader();w.writerows(rows)
    return tmp_path


def sample(field='direction',host='1',device='1.0000000000000002',ulp='1',gate='1'):
    return dict(mode='aligned-stages',stage='scattering',field=field,
                host=host,device=device,ulp=ulp,within_gate=gate)


def test_exact_empty_difference_csv_is_valid(tmp_path):
    r=audit(fixture(tmp_path,[],0));assert r['counts']['bit_differences']==0


def test_ulp_is_not_the_relative_tolerance(tmp_path):
    row=sample(host='1e-20',device='2e-20',ulp='4503599627370496')
    g=audit(fixture(tmp_path,[row],0))['groups']['aligned-stages/scattering']
    assert g['above_gate']==0 and g['field_metrics']['direction']['max_relative']==.5


def test_rng_discrete_and_strict_flags_are_not_hidden(tmp_path):
    rows=[sample(field='first_uniform',device='1.1',gate='0'),
          sample(field='status',host='0',device='1',ulp='integer',gate='0')]
    g=audit(fixture(tmp_path,rows,2))['groups']['aligned-stages/scattering']
    assert g['integer_differences']==1 and g['other_rng_field_differences']==1 and g['above_gate']==2


def test_truncated_csv_rejected(tmp_path):
    folder=fixture(tmp_path,[sample()],0)
    (folder/'steps.csv').write_text('mode,stage,field,host,device,ulp,within_gate\n')
    with pytest.raises(ValueError,match='mismatch'):audit(folder)


def test_nan_rejected(tmp_path):
    with pytest.raises(ValueError,match='nonfinite'):
        audit(fixture(tmp_path,[sample(host='nan',gate='0')],1))
