#!/usr/bin/env python3
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
# Distributed under the 3-clause BSD license; see LICENSE.
"""Check installer command contracts without downloading or compiling packages."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2]


class InstallHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='c8 installer ')
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source with spaces'
        for name in ['conan-install.sh', 'tools/build_kokkos.sh',
                     'tools/build_kokkos_dual.sh', 'tools/prepare_data.sh',
                     'modules/fluka/CMakeLists.txt',
                     'resources/GeoMag/IGRF14.COF']:
            target = self.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / name, target)
        (self.source / 'conanfile.py').touch()
        for name in ['modules/data/GeoMag/IGRF13.COF',
                     'modules/conex/cxroot/CMakeLists.txt']:
            target = self.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
        for name in ['openmp', 'cuda-volta70', 'cuda-openmp-volta70']:
            target = self.source / 'externals/conan/profiles' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('include(default)\n')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.log = self.root / 'commands.jsonl'
        for name in ['conan', 'cmake']:
            command = self.bin / name
            command.write_text('#!' + sys.executable + '\n' +
                'import json,os,sys\n' +
                'with open(os.environ["COMMAND_LOG"], "a") as f:\n' +
                ' f.write(json.dumps([os.path.basename(sys.argv[0]), *sys.argv[1:]])+"\\n")\n')
            command.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(('C8_', 'CONAN_'))}
        self.env.update(PATH=str(self.bin) + ':' + os.environ['PATH'],
                        COMMAND_LOG=str(self.log), C8_WORKSPACE=str(self.root / 'workspace'),
                        C8_KOKKOS_PROFILE='cuda-volta70')

    def tearDown(self):
        self.tmp.cleanup()

    def run_helper(self, script, *args, expected=0):
        result = subprocess.run(['bash', str(self.source / script), *args],
                                env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, expected, result.stderr)
        return [json.loads(line) for line in self.log.read_text().splitlines()] \
            if self.log.exists() else []

    def test_independent_graphs_export_and_install(self):
        rows = self.run_helper('tools/build_kokkos.sh', 'openmp,cuda', '-DWITH_FLUKA=ON')
        installs = [r for r in rows if r[:2] == ['conan', 'install']]
        self.assertEqual(len(installs), 2)
        for backend, row in zip(['openmp', 'cuda'], installs):
            self.assertIn('with_proposal_export=True', row)
            self.assertIn('compiler.cppstd=17', row)
            self.assertIn('tools.build:jobs=1', row)
            self.assertIn('--output-folder=' + str(self.root/'workspace/build'/backend/'deps'), row)
        exports = [Path(r[2]).name for r in rows if r[:2] == ['conan', 'export']]
        self.assertEqual(exports, ['cubicinterpolation', 'proposal', 'kokkos'] * 2)
        configure = [r for r in rows if r[:2] == ['cmake', '-S']]
        self.assertEqual(len(configure), 2)
        for backend, row in zip(['OPENMP', 'CUDA'], configure):
            self.assertIn('-DCORSIKA_KOKKOS_BACKEND=' + backend, row)
            self.assertIn('-DCORSIKA_ENABLE_PROPOSAL_EXPORT=ON', row)
            self.assertIn('-DWITH_FLUKA=ON', row)
        self.assertEqual(len([r for r in rows if r[:2] == ['cmake', '--install']]), 2)
        self.assertFalse(any(r[:3] == ['conan', 'profile', 'detect'] for r in rows))
        self.assertEqual((self.source/'resources/GeoMag/IGRF14.COF').read_bytes(),
                         (self.source/'modules/data/GeoMag/IGRF14.COF').read_bytes())

    def test_invalid_request_has_no_dependency_side_effect(self):
        self.run_helper('tools/build_kokkos.sh', 'openmp,unknown', expected=2)
        self.assertFalse(self.log.exists())
        self.env['C8_BUILD_JOBS'] = '0'
        self.run_helper('tools/build_kokkos.sh', 'openmp', expected=2)
        self.assertFalse(self.log.exists())

    def test_help_does_not_prepare_dependencies(self):
        for script in ['conan-install.sh', 'tools/build_kokkos.sh', 'tools/build_kokkos_dual.sh']:
            self.run_helper(script, '--help')
        self.assertFalse(self.log.exists())

    def test_different_user_data_are_not_overwritten(self):
        target = self.source / 'modules/data/GeoMag/IGRF14.COF'
        target.write_text('user-provided data')
        self.run_helper('tools/prepare_data.sh', expected=1)
        self.assertEqual(target.read_text(), 'user-provided data')

    @unittest.skipUnless(shutil.which('cmake'), 'CMake is needed for the configuration check')
    def test_missing_requested_fluka_fails_during_configuration(self):
        project = self.root / 'fluka probe'
        project.mkdir()
        (project / 'CMakeLists.txt').write_text(
            'cmake_minimum_required (VERSION 3.24)\n'
            'project (FlukaConfiguration LANGUAGES NONE)\n'
            'set (WITH_FLUKA ON)\n'
            f'include ("{self.source / "modules/fluka/CMakeLists.txt"}")\n')
        env = dict(os.environ, FLUPRO=str(self.root / 'missing fluka'))
        result = subprocess.run(
            [shutil.which('cmake'), '-S', str(project), '-B', str(project / 'build')],
            env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('WITH_FLUKA=ON requires libflukahp.a', result.stderr)


if __name__ == '__main__':
    unittest.main()
