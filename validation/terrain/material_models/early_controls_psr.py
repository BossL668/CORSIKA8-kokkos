#!/usr/bin/env python3
"""Finish independent warm-cache controls while the granite table is built.

The main drivers must be paused while this helper owns these output directories.
Their normal reuse gates verify the completed commands and binary hashes later.
"""
import importlib.util,json,pathlib
ROOT=pathlib.Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_material_models_20260913')
spec=importlib.util.spec_from_file_location('material_driver',ROOT/'source/validation/terrain/material_models/run_psr.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
for backend in ['openmp','cuda']:
    for label in ['silica','limestone']:
        d.select_cache(label)
        cmd=json.loads((ROOT/'runs'/(backend+'_'+label+'_off')/'command.json').read_text())
        for k,v in {'--primary':'proton','--energy-GeV':100,'--seed':1703}.items(): d.h.setarg(cmd,k,v)
        folder,s=d.h.run(backend+'_'+label+'_proton',cmd,backend,1800,reuse_complete=True)
        d.h.checks(folder,s,False)
d.select_cache('limestone')
cmd=json.loads((ROOT/'runs/openmp_limestone_off/command.json').read_text())
d.h.setarg(cmd,'--em-backend','proposal')
folder,s=d.h.run('cpu_limestone',cmd,'openmp',1800,reuse_complete=True)
assert s['complete'] and s['diagnostics']['material_mismatches']==0
assert abs(s['energy_ledger']['unexplained_over_initial'])<1.e-8
(ROOT/'EARLY_CONTROLS_PASSED').touch()
