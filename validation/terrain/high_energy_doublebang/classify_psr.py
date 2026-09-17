#!/usr/bin/env python3
"""Qualify actual CC -> transported tau -> shower-producing decay genealogy."""
import json
import math
import socket


def shower_child(daughter):
    pid = abs(int(daughter['pdg']))
    return pid in [11, 22] or pid >= 100


def classify(folder, summary):
    assert socket.gethostname() == 'psrpku2025'
    tracks = json.loads((folder / 'tau_tracks.json').read_text())
    witnessed = json.loads((folder / 'vertex_daughter_tracks.json').read_text())
    graph = {}
    for track in tracks:
        graph.setdefault(int(track['parent_history_id']), set()).add(int(track['history_id']))
    pairs = []
    for vertex in summary['neutrino'].get('interactions', []):
        if vertex['current'] != 'CC' or abs(vertex['projectile_pdg']) != 16:
            continue
        first_visible = [d for d in vertex['daughters'] if shower_child(d)]
        tau_ids = {int(d['history_id']) for d in vertex['daughters'] if abs(d['pdg']) == 15}
        initial_tau_ids = set(tau_ids)
        pending = list(tau_ids)
        while pending:
            for child in graph.get(pending.pop(), ()):
                if child not in tau_ids:
                    tau_ids.add(child)
                    pending.append(child)
        initial = [witnessed[str(i)]['first'] for i in initial_tau_ids if str(i) in witnessed]
        for decay in summary['tau'].get('decays', []):
            if int(decay['history_id']) not in tau_ids:
                continue
            second_visible = [d for d in decay['daughters'] if shower_child(d)]
            pdgs = [abs(d['pdg']) for d in decay['daughters']]
            mode = 'muonic' if 13 in pdgs else 'electronic' if 11 in pdgs else 'hadronic'
            separation = math.dist(vertex['position_enu_m'], decay['position_enu_m'])
            first_energy = sum(d['energy_GeV'] for d in first_visible)
            second_energy = sum(d['energy_GeV'] for d in second_visible)
            first_seen = [d['history_id'] for d in first_visible if str(d['history_id']) in witnessed]
            second_seen = [d['history_id'] for d in second_visible if str(d['history_id']) in witnessed]
            tests = dict(natural_CC=not summary['neutrino']['forced_vertex_is_conditional'],
                natural_decay=not summary['tau']['force_decay_called'],
                CC_tau_born_in_rock=bool(initial) and all(t['medium'] == 'rock' for t in initial),
                genealogy_connected=True, nonmuonic_second_shower=mode in ['electronic', 'hadronic'],
                first_visible_energy_at_least_1PeV=first_energy >= 1e6,
                second_visible_energy_at_least_1PeV=second_energy >= 1e6,
                separation_at_least_100m=separation >= 100.,
                actual_first_daughters_transported=bool(first_seen),
                actual_second_daughters_transported=bool(second_seen),
                decay_with_20us_remaining=decay['time_ns'] <= summary['transport_window_ns'] - 20000.,
                completed_full_shower=summary['complete'], no_track_truncation=not summary['diagnostics']['csv_truncated'])
            pairs.append(dict(qualified=all(tests.values()), checks=tests, mode=mode, separation_m=separation,
                first_visible_energy_GeV=first_energy, second_visible_energy_GeV=second_energy,
                CC_history_id=vertex['history_id'], CC_tau_history_ids=sorted(initial_tau_ids),
                tau_decay_history_id=decay['history_id'], tau_lineage_history_count=len(tau_ids),
                CC_position_m=vertex['position_enu_m'], decay_position_m=decay['position_enu_m'],
                CC_time_s=vertex['time_s'], decay_time_s=decay['time_ns'] * 1e-9,
                decay_medium=decay['medium'], first_transported_daughters=first_seen,
                second_transported_daughters=second_seen))
    command = json.loads((folder / 'command.json').read_text())
    seed = int(command[command.index('--seed') + 1])
    topology_passed = any(p['qualified'] for p in pairs)
    new_seed = seed not in [158, 946, 3605]
    result = dict(qualified_doublebang=topology_passed and new_seed,
        topology_passed=topology_passed, eligible_for_new_seed_target=new_seed, seed=seed, pairs=pairs,
        neutrino_interactions=len(summary['neutrino'].get('interactions', [])),
        tau_decays=len(summary['tau'].get('decays', [])),
        definition='Natural in-rock nu_tau CC connected by actual tau ancestry to a nonmuonic decay; both vertices inject >=1 PeV shower-producing daughters that are actually transported, separation >=100 m and >=20 us remaining. These are declared illustrative selection cuts, not a detector double-pulse criterion.',
        target_scope='Require three qualifying NEW seeds per energy; original seeds 158/946/3605 are retained as reference events and never fill the new-seed quota.',
        radio_double_pulse_confirmed=False, unbiased_event_rate_sample=False)
    (folder / 'doublebang_classification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
