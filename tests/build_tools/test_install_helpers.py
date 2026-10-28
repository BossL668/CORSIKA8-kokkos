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
        for original in (SOURCE / 'externals/conan/profiles').iterdir():
            name = original.name
            target = self.source / 'externals/conan/profiles' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.log = self.root / 'commands.jsonl'
        stub = r"""import json,os,sys
name = os.path.basename(sys.argv[0])
args = sys.argv[1:]
with open(os.environ['COMMAND_LOG'], 'a') as stream:
    stream.write(json.dumps([name, *args]) + '\n')
if name == 'cmake' and args == ['--version']:
    print('cmake version ' + os.environ.get('CMAKE_TEST_VERSION', '3.31.6'))
elif name == 'conan':
    if args == ['--version']:
        print('Conan version 2.11.0')
    elif args == ['profile', 'path', 'default']:
        marker = Path(os.environ['COMMAND_LOG']).with_suffix('.profile')
        if os.environ.get('MISSING_PROFILE') == '1' and not marker.exists():
            sys.exit(1)
    elif args == ['profile', 'detect']:
        Path(os.environ['COMMAND_LOG']).with_suffix('.profile').touch()
elif name == 'nvidia-smi':
    if os.environ.get('GPU_QUERY_FAIL') == '1':
        sys.exit(1)
    print(os.environ.get('GPU_CAP', '7.0'))
elif name == 'nvcc':
    print(os.environ.get('GPU_CODES', 'sm_70\nsm_75\nsm_80\nsm_86\nsm_89\nsm_90\nsm_120'))
elif name == 'build-python' and os.environ.get('MISSING_PYTHON_MODULE') == '1':
    sys.exit(1)
"""
        for name in ['conan', 'cmake', 'nvidia-smi', 'nvcc', 'build-python']:
            command = self.bin / name
            command.write_text('#!' + sys.executable + '\nfrom pathlib import Path\n' + stub)
            command.chmod(0o755)
        fluka = self.root / 'licensed fluka'
        fluka.mkdir()
        member = fluka / 'ndmhep.o'
        member.touch()
        subprocess.run(['ar', 'rcs', str(fluka / 'libflukahp.a'), str(member)], check=True)
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(('C8_', 'CONAN_'))}
        self.env.update(PATH=str(self.bin) + ':' + os.environ['PATH'],
                        COMMAND_LOG=str(self.log), C8_WORKSPACE=str(self.root / 'workspace'),
                        C8_KOKKOS_PROFILE='cuda-volta70', FLUPRO=str(fluka),
                        C8_PYTHON=str(self.bin / 'build-python'),
                        CC='gcc', CXX='g++', FC='gfortran')

    def tearDown(self):
        self.tmp.cleanup()

    def run_helper(self, script, *args, expected=0):
        result = subprocess.run(['bash', str(self.source / script), *args],
                                env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        self.result = result
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

    def assert_no_build(self, rows):
        self.assertFalse(any(row[:2] in (['conan', 'export'], ['conan', 'install'],
                                        ['cmake', '-S'], ['cmake', '--build'],
                                        ['cmake', '--install']) for row in rows))

    def test_fluka_is_on_by_default(self):
        rows = self.run_helper('tools/build_kokkos.sh', 'openmp')
        row = next(r for r in rows if r[:2] == ['cmake', '-S'])
        self.assertIn('-DWITH_FLUKA=ON', row)
        self.assertIn('-DC8_FLUKALIB=' + self.env['FLUPRO'] + '/libflukahp.a', row)
        self.assertIn('-DPYTHON_EXECUTABLE=' + self.env['C8_PYTHON'], row)

    def test_missing_fluka_fails_before_dependencies(self):
        del self.env['FLUPRO']
        rows = self.run_helper('tools/build_kokkos.sh', 'openmp', expected=2)
        self.assertIn('FLUKA is ON', self.result.stderr)
        self.assert_no_build(rows)

    def test_incompatible_fluka_archive_is_rejected(self):
        archive = Path(self.env['FLUPRO']) / 'libflukahp.a'
        archive.unlink()
        subprocess.run(['ar', 'rcs', str(archive)], check=True)
        self.run_helper('tools/build_kokkos.sh', 'openmp', expected=2)
        self.assertIn('ndmhep.o', self.result.stderr)
        self.assertFalse(self.log.exists())

    def test_explicit_fluka_library(self):
        library = Path(self.env.pop('FLUPRO')) / 'libflukahp.a'
        rows = self.run_helper('tools/build_kokkos.sh', 'openmp',
                               '-DC8_FLUKALIB:FILEPATH=' + str(library))
        row = next(r for r in rows if r[:2] == ['cmake', '-S'])
        self.assertIn('-DC8_FLUKALIB=' + str(library), row)

    def test_explicit_fluka_off_allows_no_library(self):
        del self.env['FLUPRO']
        rows = self.run_helper('tools/build_kokkos.sh', 'openmp',
                               '-D', 'WITH_FLUKA:BOOL=OFF')
        row = next(r for r in rows if r[:2] == ['cmake', '-S'])
        self.assertIn('-DWITH_FLUKA=OFF', row)
        self.assertNotIn('-DWITH_FLUKA=ON', row)

    def test_cuda_architecture_detection(self):
        del self.env['C8_KOKKOS_PROFILE']
        for capability, target in [('7.0', 'volta70'), ('7.5', 'turing75'),
                                   ('8.0', 'ampere80'), ('8.6', 'ampere86'),
                                   ('8.9', 'ada89'), ('9.0', 'hopper90'),
                                   ('12.0', 'blackwell120')]:
            with self.subTest(capability=capability):
                self.env['GPU_CAP'] = capability
                rows = self.run_helper('tools/build_kokkos.sh', 'cuda', '--check')
                self.assertIn('cuda-' + target, self.result.stdout)
                self.assert_no_build(rows)

    def test_repeated_identical_gpus(self):
        del self.env['C8_KOKKOS_PROFILE']
        self.env['GPU_CAP'] = ' 8.9 \n8.9\r\n'
        self.run_helper('tools/build_kokkos.sh', 'cuda', '--check')
        self.assertIn('cuda-ada89', self.result.stdout)

    def test_mixed_and_unknown_gpus_are_not_guessed(self):
        del self.env['C8_KOKKOS_PROFILE']
        for cap in ['7.0\n8.9', '99.0', '']:
            self.env['GPU_CAP'] = cap
            rows = self.run_helper('tools/build_kokkos.sh', 'cuda', '--check', expected=2)
            self.assertIn('Select C8_CUDA_DEVICE', self.result.stderr)
            self.assert_no_build(rows)

    def test_query_failure_is_explained(self):
        del self.env['C8_KOKKOS_PROFILE']
        self.env['GPU_QUERY_FAIL'] = '1'
        rows = self.run_helper('tools/build_kokkos.sh', 'cuda', expected=2)
        self.assertIn('GPU query failed', self.result.stderr)
        self.assert_no_build(rows)

    def test_target_device_selection_is_forwarded(self):
        del self.env['C8_KOKKOS_PROFILE']
        self.env['C8_CUDA_DEVICE'] = 'GPU-test-uuid'
        rows = self.run_helper('tools/build_kokkos.sh', 'cuda', '--check')
        row = next(r for r in rows if r[0] == 'nvidia-smi')
        self.assertIn('--id=GPU-test-uuid', row)

    def test_explicit_profile_bypasses_gpu_query(self):
        self.env['GPU_QUERY_FAIL'] = '1'
        rows = self.run_helper('tools/build_kokkos.sh', 'cuda', '--check')
        self.assertFalse(any(row[0] == 'nvidia-smi' for row in rows))
        self.assert_no_build(rows)

    def test_nvcc_must_support_target(self):
        self.env['GPU_CODES'] = 'sm_80\nsm_89'
        rows = self.run_helper('tools/build_kokkos.sh', 'cuda', expected=2)
        self.assertIn('cannot compile sm_70', self.result.stderr)
        self.assert_no_build(rows)

    def test_combined_backend_detects_its_own_profile(self):
        del self.env['C8_KOKKOS_PROFILE']
        rows = self.run_helper('tools/build_kokkos_dual.sh', '--check')
        self.assertIn('cuda-openmp-volta70', self.result.stdout)
        self.assert_no_build(rows)

    def test_missing_python_module_stops_early(self):
        self.env['MISSING_PYTHON_MODULE'] = '1'
        rows = self.run_helper('tools/build_kokkos.sh', 'openmp', expected=2)
        self.assertIn('Build Python needs', self.result.stderr)
        self.assert_no_build(rows)

    def test_unsupported_cmake_stops_before_dependencies(self):
        for version in ['3.18.0', '4.0.0']:
            self.env['CMAKE_TEST_VERSION'] = version
            rows = self.run_helper('tools/build_kokkos.sh', 'openmp', expected=2)
            self.assertIn('CMake 3.24 through 3.x', self.result.stderr)
            self.assert_no_build(rows)

    def test_missing_default_profile_created_only_for_build(self):
        self.env['MISSING_PROFILE'] = '1'
        rows = self.run_helper('tools/build_kokkos.sh', 'openmp', '--check')
        self.assertFalse(any(row[:3] == ['conan', 'profile', 'detect'] for row in rows))
        self.assert_no_build(rows)
        rows = self.run_helper('tools/build_kokkos.sh', 'openmp')
        self.assertEqual(sum(row[:3] == ['conan', 'profile', 'detect'] for row in rows), 1)

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
    @unittest.skipUnless(shutil.which('cmake'), 'CMake is needed for the configuration check')
    def test_explicit_fluka_off_ignores_available_library(self):
        project = self.root / 'fluka off probe'
        project.mkdir()
        (project / 'CMakeLists.txt').write_text(
            'cmake_minimum_required (VERSION 3.24)\n'
            'project (FlukaConfiguration LANGUAGES NONE)\n'
            'set (WITH_FLUKA OFF)\n'
            'set (C8_FLUKA_ENABLED enabled CACHE INTERNAL "")\n'
            f'include ("{self.source / "modules/fluka/CMakeLists.txt"}")\n'
            'if (C8_FLUKA_ENABLED OR TARGET fluka)\n'
            '  message (FATAL_ERROR "FLUKA must be disabled")\n'
            'endif ()\n')
        env = dict(os.environ, FLUPRO=self.env['FLUPRO'])
        result = subprocess.run(
            [shutil.which('cmake'), '-S', str(project), '-B', str(project / 'build')],
            env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('FLUKA explicitly disabled', result.stdout)


if __name__ == '__main__':
    unittest.main()
