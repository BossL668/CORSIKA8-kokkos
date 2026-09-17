#!/usr/bin/env python3
"""Build an output-only terrain variant against the accepted PSR libraries."""
import argparse
import hashlib
import json
import pathlib
import shlex
import shutil
import socket
import subprocess

BASE = pathlib.Path('/data/yhlu/CorsikaData/corsika_validation_results')
OLD = BASE / 'beta5_material_models_20260913'


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=pathlib.Path, required=True)
    root = p.parse_args().root
    assert socket.gethostname() == 'psrpku2025'
    source = root / 'source'
    source.mkdir()
    (root / 'build').mkdir()
    shutil.copytree(OLD / 'source/applications/detail/mountain', source / 'applications/detail/mountain')
    shutil.copy2(root / 'code/TerrainCsvStream.hpp', source / 'applications/detail/mountain/TerrainCsvStream.hpp')
    app = (OLD / 'source/applications/c8_terrain_cascade.cpp').read_text()
    app = replace_once(app, 'bool radioEnabled = false;', 'bool radioEnabled = false;\n  bool compressTerrainCsv = false;')
    app = replace_once(app, '  app.add_option("--primary", primary)',
        '  app.add_flag("--compress-terrain-csv",compressTerrainCsv,"Lossless gzip terrain CSV");\n  app.add_option("--primary", primary)')
    app = replace_once(app, 'requireNeutrinoCoverage, transportStepLimit);',
                       'requireNeutrinoCoverage, transportStepLimit,compressTerrainCsv);')
    (source / 'applications/c8_terrain_cascade.cpp').write_text(app)
    header = source / 'applications/detail/mountain/TerrainShowerOutput.hpp'
    text = header.read_text()
    for a, b in [('#include "TerrainEnergyLedger.hpp"', '#include "TerrainEnergyLedger.hpp"\n#include "TerrainCsvStream.hpp"'),
                 ('std::uint64_t stepLimit=2000000)', 'std::uint64_t stepLimit=2000000,bool compressCsv=false)'),
                 ('neutrinoDomain_(requireNeutrinoCoverage){}', 'compressCsv_(compressCsv),neutrinoDomain_(requireNeutrinoCoverage){}'),
                 ('tracks_.open((directory/"tracks.csv").string());', 'tracks_.open((directory/"tracks.csv").string(),compressCsv_);'),
                 ('survivors_.open((directory/"window_survivors.csv").string());', 'survivors_.open((directory/"window_survivors.csv").string(),compressCsv_);'),
                 ('deposits_.open((directory/"deposits.csv").string());', 'deposits_.open((directory/"deposits.csv").string(),compressCsv_);'),
                 ('std::ofstream tracks_;', 'TerrainCsvStream tracks_;'),
                 ('std::ofstream survivors_,deposits_;', 'TerrainCsvStream survivors_,deposits_;\n  bool compressCsv_{};'),
                 ('YAML::Node getConfig()const override {', 'YAML::Node getConfig()const override {'),
                 ('n["energy_ledger"]="deposition only;', 'n["csv_compression"]=compressCsv_?"gzip":"none";\n    n["energy_ledger"]="deposition only;')]:
        text = replace_once(text, a, b)
    header.write_text(text)
    build = OLD / 'build-openmp/applications'
    flags = {}
    for line in (build / 'CMakeFiles/c8_terrain_cascade.dir/flags.make').read_text().splitlines():
        if ' = ' in line:
            k, v = line.split(' = ', 1)
            flags[k] = shlex.split(v)
    link = shlex.split((build / 'CMakeFiles/c8_terrain_cascade.dir/link.txt').read_text())
    obj, binary = root / 'build/c8_terrain_cascade.o', root / 'build/c8_terrain_cascade'
    command = [link[0]] + flags['CXX_DEFINES'] + ['-I' + str(source)] + flags['CXX_INCLUDES'] + flags['CXX_FLAGS']
    command += ['-c', str(source / 'applications/c8_terrain_cascade.cpp'), '-o', str(obj)]
    link[next(i for i, x in enumerate(link) if x.endswith('c8_terrain_cascade.cpp.o'))] = str(obj)
    link[link.index('-o') + 1] = str(binary)
    # Cheap approximate interaction-depth prefilter. Final states and tau
    # decays remain unforced and must be verified by the complete simulation.
    scan = (OLD / 'source/validation/terrain/scan_natural_nutau_seeds.cpp').read_text()
    scan = replace_once(scan, 'if(argc!=2) return 2;', 'if(argc!=3) return 2;')
    scan = replace_once(scan, 'constexpr double energy=1e8, length=1387.4258149368304, rho=2.65;',
                        'const double energy=std::stod(argv[2]), length=1387.4258149368304, rho=2.65;')
    (source / 'scan_seeds.cpp').write_text(scan)
    scanner = [link[0]] + flags['CXX_DEFINES'] + flags['CXX_INCLUDES'] + flags['CXX_FLAGS']
    scanner += [str(source / 'scan_seeds.cpp'), '-o', str(root / 'build/scan_seeds')]
    (root / 'build/commands.json').write_text(json.dumps([command, link, scanner], indent=2))
    with (root / 'build/build.log').open('w') as log:
        for cmd in [command, link, scanner]:
            subprocess.run(cmd, cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
        for energy in [300000000, 1000000000]:
            subprocess.run([str(root / 'build/scan_seeds'), str(root / 'build' / ('seeds_%d.csv' % energy)), str(energy)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    (root / 'build/BUILT').touch()
    print('Built compressed-output variant', hashlib.sha256(binary.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
