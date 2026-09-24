"""Build/install command-contract tests; no compiler, GPU or network is used.

Run: python -m unittest discover -s validation -p test_build_kokkos_helpers.py -v
These mocks do not substitute for an actual clean-machine dependency build.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1]


class BuildHelpers(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="c8-build-helper-test-")
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.source = self.project / "source"
        (self.source / "tools").mkdir(parents=True)
        self.profiles = self.source / "dependencies/kokkos/profiles"
        self.profiles.mkdir(parents=True)
        for name in ("build_kokkos.sh", "build_kokkos_dual.sh"):
            shutil.copyfile(SOURCE / "tools" / name, self.source / "tools" / name)
        for name in ("openmp", "cuda-ada89", "cuda-openmp-turing75",
                     "cuda-openmp-ampere80", "cuda-openmp-ampere86",
                     "cuda-openmp-ada89", "cuda-openmp-hopper90"):
            (self.profiles / name).write_text("include(default)\n")
        self.bin = self.project / "mock-bin"
        self.bin.mkdir()
        self.log = self.project / "commands.jsonl"
        mock = self.bin / "mock-tool"
        mock.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "name = pathlib.Path(sys.argv[0]).name\n"
            "args = sys.argv[1:]\n"
            "with open(os.environ['C8_TEST_COMMAND_LOG'], 'a') as out:\n"
            "    out.write(json.dumps([name, *args]) + '\\n')\n"
            "if name == 'nvidia-smi': print(os.environ.get('C8_TEST_COMPUTE_CAP', '8.9'))\n"
            "if os.environ.get('C8_TEST_FAIL') == ' '.join([name, *args[:1]]):\n"
            "    sys.exit(42)\n"
        )
        mock.chmod(0o755)
        for name in ("conan", "cmake", "nvcc", "nvidia-smi"):
            (self.bin / name).symlink_to(mock)
        self.env = dict(os.environ)
        for name in list(self.env):
            if name.startswith(("C8_KOKKOS_", "C8_TEST_")) or name == "C8_BUILD_JOBS":
                self.env.pop(name)
        self.env.update(
            PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
            C8_TEST_COMMAND_LOG=str(self.log),
        )

    def run_helper(self, name, *args, **env):
        result = subprocess.run(
            ["bash", str(self.source / "tools" / name), *args],
            env={**self.env, **env}, text=True, capture_output=True, timeout=15,
        )
        calls = [json.loads(row) for row in self.log.read_text().splitlines()] if self.log.exists() else []
        return result, calls

    def assert_complete_build_before_install(self, calls, jobs):
        builds = [(i, call) for i, call in enumerate(calls) if call[:2] == ["cmake", "--build"]]
        installs = [(i, call) for i, call in enumerate(calls) if call[:2] == ["cmake", "--install"]]
        self.assertEqual(len(builds), 1)
        self.assertEqual(len(installs), 1)
        index, build = builds[0]
        self.assertLess(index, installs[0][0])
        self.assertNotIn("--target", build, "global install must follow the full configured build")
        self.assertEqual(build[build.index("--parallel") + 1], jobs)

    def test_dual_clean_install_builds_all_configured_targets(self):
        result, calls = self.run_helper("build_kokkos_dual.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_complete_build_before_install(calls, "1")
        configure = next(c for c in calls if c[:2] == ["cmake", "-S"])
        self.assertIn("-DCORSIKA_KOKKOS_BACKEND=CUDA_OPENMP", configure)
        install = next(c for c in calls if c[:2] == ["cmake", "--install"])
        self.assertEqual(install[-1], str(self.project / "build/cuda-openmp"))
        conan = next(c for c in calls if c[:2] == ["conan", "install"])
        self.assertEqual(Path(conan[conan.index("-pr:h") + 1]).name, "cuda-openmp-ada89")

    def test_dual_auto_detects_supported_architectures(self):
        for cap, name in (("7.5", "turing75"), ("8.0", "ampere80"),
                          ("8.6", "ampere86"), ("8.9", "ada89"),
                          ("9.0", "hopper90")):
            with self.subTest(capability=cap):
                self.log.unlink(missing_ok=True)
                result, calls = self.run_helper("build_kokkos_dual.sh", C8_TEST_COMPUTE_CAP=cap)
                self.assertEqual(result.returncode, 0, result.stderr)
                conan = next(c for c in calls if c[:2] == ["conan", "install"])
                self.assertEqual(Path(conan[conan.index("-pr:h") + 1]).name,
                                 "cuda-openmp-" + name)

    def test_dual_rejects_mixed_or_unsupported_architectures(self):
        for cap in ("8.0\n8.9", "10.0"):
            with self.subTest(capability=cap):
                self.log.unlink(missing_ok=True)
                result, calls = self.run_helper("build_kokkos_dual.sh", C8_TEST_COMPUTE_CAP=cap)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("C8_KOKKOS_DUAL_PROFILE", result.stderr)
                self.assertFalse(any(c[:2] == ["conan", "install"] for c in calls))

    def test_dual_forwards_mountain_and_profile_options(self):
        custom = str(self.profiles / "cuda-openmp-ada89")
        result, calls = self.run_helper(
            "build_kokkos_dual.sh", "-DWITH_FLUKA=ON", "-DCORSIKA_BUILD_MOUNTAIN_APPLICATION=ON",
            "-DC8_INTERFACE_EXECUTION_SPACE=OPENMP", C8_BUILD_JOBS="2", C8_KOKKOS_DUAL_PROFILE=custom,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_complete_build_before_install(calls, "2")
        configure = next(c for c in calls if c[:2] == ["cmake", "-S"])
        for arg in ("-DWITH_FLUKA=ON", "-DCORSIKA_BUILD_MOUNTAIN_APPLICATION=ON", "-DC8_INTERFACE_EXECUTION_SPACE=OPENMP"):
            self.assertIn(arg, configure)
        conan = next(c for c in calls if c[:2] == ["conan", "install"])
        self.assertEqual(conan[conan.index("-pr:h") + 1], custom)
        self.assertIn("tools.build:jobs=2", conan)
        self.assertFalse(any(c[0] == "nvidia-smi" for c in calls))

    def test_dual_build_failure_prevents_install(self):
        result, calls = self.run_helper("build_kokkos_dual.sh", C8_TEST_FAIL="cmake --build")
        self.assertEqual(result.returncode, 42)
        self.assertFalse(any(c[:2] == ["cmake", "--install"] for c in calls))

    def test_dual_dependency_failure_prevents_configure(self):
        result, calls = self.run_helper("build_kokkos_dual.sh", C8_TEST_FAIL="conan install")
        self.assertEqual(result.returncode, 42)
        self.assertFalse(any(c[0] == "cmake" for c in calls))

    def test_invalid_jobs_rejected_before_external_commands(self):
        result, calls = self.run_helper("build_kokkos_dual.sh", C8_BUILD_JOBS="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])

    def test_independent_openmp_remains_gpu_free(self):
        result, calls = self.run_helper("build_kokkos.sh", "openmp", "-DWITH_FLUKA=OFF")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_complete_build_before_install(calls, "1")
        self.assertFalse(any(c[0] in ("nvcc", "nvidia-smi") for c in calls))

    def test_independent_cuda_architecture_discovery(self):
        result, calls = self.run_helper("build_kokkos.sh", "cuda", "-DWITH_FLUKA=ON")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_complete_build_before_install(calls, "1")
        conan = next(c for c in calls if c[:2] == ["conan", "install"])
        self.assertEqual(Path(conan[conan.index("-pr:h") + 1]).name, "cuda-ada89")

    def test_multiple_gpu_toolchains_rejected(self):
        result, calls = self.run_helper("build_kokkos.sh", "cuda,hip")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
