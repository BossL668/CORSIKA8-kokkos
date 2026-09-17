#!/usr/bin/env python3
"""Reuse the recorded genealogy; declare energy-scaled pilot selection cuts."""
import importlib.util
import json
from pathlib import Path
import socket

def classify(folder,summary):
    assert socket.gethostname()=='psrpku2025'
    path=Path(__file__).with_name('legacy_classifier.py')
    spec=importlib.util.spec_from_file_location('legacy_classifier',path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    result=old.classify(folder,summary)
    threshold=.01*float(summary['energy_GeV'])
    for pair in result['pairs']:
        tests={k:v for k,v in pair['checks'].items() if k not in [
            'first_visible_energy_at_least_1PeV','second_visible_energy_at_least_1PeV','separation_at_least_100m']}
        pair['linked_double_vertex']=all(tests.values())
        pair['previous_high_energy_selection_passed']=pair['qualified']
        tests.update(first_visible_energy_at_least_1percent=pair['first_visible_energy_GeV']>=threshold,
            second_visible_energy_at_least_1percent=pair['second_visible_energy_GeV']>=threshold,
            separation_at_least_10m=pair['separation_m']>=10.)
        pair['checks']=tests;pair['qualified']=all(tests.values())
    result.update(topology_passed=any(p['linked_double_vertex'] for p in result['pairs']),
        qualified_doublebang=any(p['qualified'] for p in result['pairs']),
        visible_energy_threshold_GeV=threshold,separation_threshold_m=10.,
        definition='Natural rock CC -> actually transported tau -> nonmuonic shower-producing decay; direct daughters witnessed. Report the linked topology separately. Pilot selection requires >=1% of primary energy in each visible daughter set and >=10 m between vertices; not a detector resolution condition.',
        target_scope='First accepted 1 PeV illustrative event; retain every completed trial. Higher energies are not yet scheduled.',
        radio_double_pulse_confirmed=False,unbiased_event_rate_sample=False)
    (folder/'doublebang_classification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
