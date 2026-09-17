#!/usr/bin/env python3
"""Validate the isolated first-vertex scanner and screen the frozen NE-SW seeds.

All numerical work runs on PSR. A selected CC vertex is a candidate, not a
verified double bang. The production executable and physics are not modified.
"""
import csv
import datetime
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import yaml

BASE = Path('/data/yhlu/CorsikaData/corsika_validation_results')
ROOT = BASE/'beta5_doublebang_100PeV_ne_sw_130cores_20260916'
WORK = ROOT/'first_vertex'
REPORT = ROOT/'report/first_vertex'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def main():
    assert socket.gethostname() == 'psrpku2025'
    os.sched_setaffinity(0, {3, 511})
    REPORT.mkdir(exist_ok=True)
    regression = yaml.safe_load((WORK/'regression_output.yaml').read_text())
    checks = []
    for case, event in zip(regression['input']['cases'], regression['events']):
        expected = yaml.safe_load(Path(case['reference']).read_text())['neutrino']['interactions'][0]
        actual = event['neutrino']['interactions'][0]
        checks.append(dict(seed=case['seed'], exactly_equal=actual == expected,
                           different_fields=[k for k in expected if actual.get(k) != expected[k]],
                           elapsed_s=event['elapsed_s'], reference=case['reference']))
    assert len(checks) == len(regression['input']['cases'])
    save(REPORT/'regression.json', dict(passed=all(x['exactly_equal'] for x in checks), checks=checks))
    assert all(x['exactly_equal'] for x in checks), checks

    manifest = json.loads((ROOT/'campaign.json').read_text())
    cases = []
    for seed in manifest['seeds']:
        cmd = next(c['command'] for c in manifest['cases'] if c['seed']==seed)
        def values(key, n=3):
            i = cmd.index(key)+1
            return list(map(float, cmd[i:i+n]))
        cases.append(dict(seed=seed, energy_GeV=manifest['energy_GeV'],
                          position_m=values('--position-m'), direction=values('--direction'),
                          max_primary_flight_m=974.6874593467251))
    config = dict(scene=str(ROOT/'scenes/screen.yaml'), cases=cases)
    save(WORK/'scan_input.yaml', config)
    env = os.environ.copy()
    env.update(manifest['runtime_environment'])
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    start = time.monotonic()
    with (WORK/'scan.log').open('w') as log:
        subprocess.run([str(WORK/'first_vertex'), str(WORK/'scan_input.yaml'),
                        str(WORK/'scan_output.yaml')], env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    wall = time.monotonic()-start
    output = yaml.safe_load((WORK/'scan_output.yaml').read_text())
    assert len(output['events'])==len(cases)
    rows = []
    for event in output['events']:
        candidate = bool(event.get('vertex_in_rock') and event.get('vertex_in_requested_interval')
                         and event.get('current')=='CC' and event['tau_energy_GeV']>=1e6
                         and event['hadronic_energy_GeV']>=1e6)
        vertex = event['neutrino'].get('interactions', [])
        assert len(vertex) <= 1 and not event['neutrino']['force_interaction_called']
        rows.append(dict(seed=event['seed'], candidate=candidate,
                         current=event.get('current', 'none'),
                         rock_to_exit_m=config['cases'][0]['max_primary_flight_m']-event.get('primary_flight_m', math.inf),
                         tau_energy_PeV=event.get('tau_energy_GeV', 0)/1e6,
                         hadronic_energy_PeV=event.get('hadronic_energy_GeV', 0)/1e6,
                         elapsed_s=event['elapsed_s']))
    selected = [r for r in rows if r['candidate']]
    # Keep the original depth ranking; no analytic decay is substituted for transport.
    summary = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   stage='first_vertex_only', confirmed_doublebangs=0,
                   seed_count=len(rows), cc_count=sum(r['current']=='CC' for r in rows),
                   nc_count=sum(r['current']=='NC' for r in rows),
                   candidate_count=len(selected), candidates=selected, all_seeds=rows,
                   total_wall_s=wall, vertex_only_s=sum(r['elapsed_s'] for r in rows),
                   selection='Natural rock CC; tau energy >=1 PeV; hadronic energy >=1 PeV; first vertex within first rock chord.',
                   limitation='No tau propagation, decay, shower or radio calculated; selected seeds are biased candidates, not an event-rate sample.')
    save(REPORT/'selection.json', summary)
    with (REPORT/'candidates.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for name in ['scan_input.yaml','scan_output.yaml','regression_input.yaml','regression_output.yaml','commands.json']:
        (REPORT/name).write_bytes((WORK/name).read_bytes())
    text = ['# 第一次中微子反应快筛', '',
            '最早的百万种子扫描只判断首次反应深度，得到的是几何候选。这里继续调用生产程序相同的自然 CC+NC 反应生成器，得到真实的首个反应末态；首个顶点之后立即结束。', '',
            f'扫描 {len(rows)} 个已通过深度预筛的种子：CC {summary["cc_count"]} 个，NC {summary["nc_count"]} 个；{len(selected)} 个满足岩石 CC、τ 和首个强子系统能量均 ≥1 PeV。', '',
            f'总耗时 {wall:.2f} s（含地形加载和校验）；逐种子反应生成累计 {summary["vertex_only_s"]:.2f} s。没有计算 shower，因而不需要 thinning。', '',
            '核对：与已有完整模拟的首个反应记录比较，检查反应类型、坐标/时间、靶粒子、全部次级粒子的种类/能量、四动量审计等全部字段。结果见 [regression.json](regression.json)。', '',
            '**这些是首顶点候选，不是已确认 double bang。** 初筛通过后直接以 emthin=1e-3 运行一次完整 shower＋80 站 CoREAS/ZHS，从同一次计算验收 τ 出山、空气衰变、两次 shower 和射电。没有单独的无射电完整输运复核。这里没有人为强制 CC、位置或衰变。', '',
            '| Seed | Current | Rock to exit (m) | Tau energy (PeV) | Hadronic energy (PeV) | Candidate |',
            '|---:|:---:|---:|---:|---:|:---:|']
    for r in rows:
        text.append(f'| {r["seed"]} | {r["current"]} | {r["rock_to_exit_m"]:.2f} | {r["tau_energy_PeV"]:.2f} | {r["hadronic_energy_PeV"]:.2f} | {"Yes" if r["candidate"] else "No"} |')
    text += ['', '[完整首顶点记录](scan_output.yaml) · [候选清单及时间](selection.json)']
    (REPORT/'README_CN.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['all_seeds','candidates']},indent=2),flush=True)


if __name__ == '__main__':
    main()
