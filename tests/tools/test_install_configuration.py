"""Offline installation-helper regression tests; no GPU or downloads required."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
CMAKE = shutil.which("cmake")


class InstallationConfiguration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="c8-install-test-")
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def run_helper(self, dual=False, capability="12.0", profile=None, backend="cuda"):
        # Exercise the real scripts, stopping external tools at the CLI boundary.
        # Copying into an owned directory prevents changes to the caller's builds.
        source = self.work / "source"
        source.mkdir()
        shutil.copytree(ROOT / "tools", source / "tools")
        shutil.copytree(ROOT / "dependencies/kokkos/profiles",
                        source / "dependencies/kokkos/profiles")
        bin_dir = self.work / "bin"
        bin_dir.mkdir()
        for name in ("cmake", "conan", "nvcc"):
            tool = bin_dir / name
            tool.write_text('#!/bin/bash\nprintf "%s\\n" "$0 $*" >> "$C8_TEST_LOG"\n')
            tool.chmod(0o755)
        smi = bin_dir / "nvidia-smi"
        smi.write_text('#!/bin/bash\nprintf "%s\\n" "$C8_TEST_CAP"\n')
        smi.chmod(0o755)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("C8_KOKKOS_")}
        env.update(PATH=f"{bin_dir}:{os.environ['PATH']}",
                   C8_TEST_CAP=capability, C8_TEST_LOG=str(self.work / "calls"),
                   C8_BUILD_JOBS="1")
        if profile:
            env["C8_KOKKOS_DUAL_PROFILE" if dual else "C8_KOKKOS_PROFILE"] = profile
        args = ["bash", str(source / "tools" /
                           ("build_kokkos_dual.sh" if dual else "build_kokkos.sh"))]
        if not dual:
            args.append(backend)
        result = subprocess.run(args, env=env, capture_output=True, text=True)
        log = self.work / "calls"
        return result, log.read_text() if log.exists() else ""

    def test_blackwell_auto_profile(self):
        result, calls = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("profiles/cuda-blackwell120 -pr:b", calls)
        self.assertIn("-DCORSIKA_KOKKOS_BACKEND=CUDA", calls)

    def test_blackwell_dual_auto_profile(self):
        result, calls = self.run_helper(dual=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("profiles/cuda-openmp-blackwell120 -pr:b", calls)
        self.assertIn("-DCORSIKA_KOKKOS_BACKEND=CUDA_OPENMP", calls)

    def test_ada_auto_profile_unchanged(self):
        result, calls = self.run_helper(capability="8.9")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("profiles/cuda-ada89 -pr:b", calls)

    def test_mixed_devices_not_guessed(self):
        result, calls = self.run_helper(capability="8.9\n12.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mixed", result.stderr)
        self.assertEqual(calls, "")

    def test_unknown_blackwell_not_guessed(self):
        result, calls = self.run_helper(capability="10.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, "")

    def test_explicit_profile_without_device_discovery(self):
        result, calls = self.run_helper(capability="unknown", profile="cuda-blackwell120")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("profiles/cuda-blackwell120 -pr:b", calls)

    def test_dual_profile_short_name(self):
        result, calls = self.run_helper(
            dual=True, capability="unknown", profile="cuda-openmp-blackwell120")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("profiles/cuda-openmp-blackwell120 -pr:b", calls)

    def test_openmp_needs_no_gpu(self):
        result, calls = self.run_helper(backend="openmp", capability="unknown")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("profiles/openmp -pr:b", calls)
        self.assertIn("-DCORSIKA_KOKKOS_BACKEND=OPENMP", calls)

    @unittest.skipUnless(CMAKE, "cmake not installed")
    def test_toolchain_metadata_and_manifest(self):
        # This is the same direct-toolchain route as the shell helpers, not a preset.
        for backend, architecture, cuda in (
                ("CUDA", "BLACKWELL120", "120"),
                ("CUDA_OPENMP", "BLACKWELL120", "120"),
                ("CUDA", "ADA89", "89"),
                ("OPENMP", "NONE", "")):
            with self.subTest(backend=backend, architecture=architecture):
                case = self.work / f"{backend}-{architecture}"
                case.mkdir()
                toolchain = case / "toolchain.cmake"
                toolchain.write_text(
                    f'set(C8_CONAN_KOKKOS_BACKEND "{backend}")\n'
                    f'set(C8_CONAN_KOKKOS_ARCHITECTURE "{architecture}")\n'
                    + (f'set(C8_CONAN_KOKKOS_CUDA_ARCHITECTURES "{cuda}")\n'
                       if cuda else ""))
                (case / "CMakeLists.txt").write_text(
                    'cmake_minimum_required(VERSION 3.18)\nproject(InstallTest NONE)\n'
                    f'include("{ROOT}/cmake/CorsikaConanKokkos.cmake")\n'
                    'set(Kokkos_VERSION 4.7.03)\nset(CORSIKA_ENABLE_KOKKOS ON)\n'
                    f'add_subdirectory("{ROOT}/tools" tools)\n')
                args = [CMAKE, "-S", str(case), "-B", str(case / "build"),
                        f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
                        f"-DCORSIKA_KOKKOS_BACKEND={backend}",
                        # Upgrade the legacy unspecified cache values, too.
                        "-DCORSIKA_KOKKOS_ARCHITECTURE=NONE",
                        f"-DCMAKE_INSTALL_PREFIX={case}/install/{backend.lower()}",
                        # Old build caches must not recreate the Python entry point.
                        f"-DCORSIKA_LAUNCHER_PREFIX={case}/install"]
                if cuda:
                    args.append("-DCORSIKA_KOKKOS_CUDA_ARCHITECTURES=native")
                result = subprocess.run(args, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("CORSIKA_LAUNCHER_PREFIX is obsolete", result.stderr)
                result = subprocess.run([CMAKE, "--install", str(case / "build")],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                prefix = case / "install" / backend.lower()
                manifest = json.loads((prefix / "share/corsika/backend.json").read_text())
                self.assertEqual(manifest["architecture"], architecture)
                self.assertEqual(manifest["backend"], backend.lower())
                self.assertEqual(manifest["host_execution"],
                                 "openmp" if backend in ("OPENMP", "CUDA_OPENMP") else "serial")
                self.assertFalse((case / "install/bin/c8_air_shower").exists())
                # Every installed file stays within the selected backend prefix.
                paths = (case / "build/install_manifest.txt").read_text().splitlines()
                self.assertTrue(paths)
                self.assertTrue(all(Path(path).is_relative_to(prefix) for path in paths))
                cache = (case / "build/CMakeCache.txt").read_text()
                self.assertIn(f"CORSIKA_KOKKOS_ARCHITECTURE:STRING={architecture}", cache)
                if cuda:
                    self.assertIn(f"CORSIKA_KOKKOS_CUDA_ARCHITECTURES:STRING={cuda}", cache)

    def test_readme_commands_use_explicit_native_binary_backends(self):
        for name in ("README.md", "README_CN.md"):
            with self.subTest(readme=name):
                content = (ROOT / name).read_text().replace("\\\n", " ")
                count = 0
                for line in content.splitlines():
                    if not line.startswith("install/") or "/c8_air_shower " not in line:
                        continue
                    args = shlex.split(line)
                    self.assertNotEqual(args[0], "install/bin/c8_air_shower")
                    self.assertNotIn("--backend", args)
                    if "--help" not in args:
                        self.assertIn("--em-backend", args)
                        self.assertIn("--radio-backend", args)
                    count += 1
                self.assertGreaterEqual(count, 8)

    def test_python_air_shower_launcher_removed(self):
        self.assertFalse((ROOT / "tools/c8_air_shower_launcher.py").exists())

    @unittest.skipUnless(CMAKE, "cmake not installed")
    def test_explicit_metadata_conflicts_rejected(self):
        for option, chosen, package in (
                ("BACKEND", "OPENMP", "CUDA"),
                ("ARCHITECTURE", "ADA89", "BLACKWELL120"),
                ("CUDA_ARCHITECTURES", "89", "120")):
            with self.subTest(option=option):
                script = self.work / "conflict.cmake"
                script.write_text(
                    f'set(CORSIKA_KOKKOS_{option} "{chosen}")\n'
                    f'set(C8_CONAN_KOKKOS_{option} "{package}")\n'
                    f'include("{ROOT}/cmake/CorsikaConanKokkos.cmake")\n')
                result = subprocess.run([CMAKE, "-P", str(script)],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("conflicts with the Conan profile", result.stderr)

    def test_conan_generator_exports_architecture_to_toolchain(self):
        try:
            import conan  # noqa: F401
        except ImportError:
            self.skipTest("Conan not installed in this Python environment")
        spec = importlib.util.spec_from_file_location("c8_recipe", ROOT / "conanfile.py")
        recipe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recipe)
        for backend, architecture, cuda in (
                ("cuda", "BLACKWELL120", "120"),
                ("cuda_openmp", "BLACKWELL120", "120"),
                ("cuda", "ADA89", "89"), ("openmp", "NONE", None)):
            with self.subTest(backend=backend, architecture=architecture):
                toolchain = SimpleNamespace(cache_variables={}, variables={},
                                            generate=lambda: None)
                options = SimpleNamespace(with_kokkos=True, kokkos_backend=backend,
                                          kokkos_architecture=architecture)
                with patch.object(recipe, "CMakeToolchain", return_value=toolchain):
                    recipe.Pkg.generate(SimpleNamespace(options=options))
                for option, expected in (("BACKEND", backend.upper()),
                                         ("ARCHITECTURE", architecture)):
                    self.assertEqual(toolchain.variables[f"C8_CONAN_KOKKOS_{option}"],
                                     expected)
                    self.assertEqual(toolchain.cache_variables[f"CORSIKA_KOKKOS_{option}"],
                                     expected)
                self.assertEqual(
                    toolchain.variables.get("C8_CONAN_KOKKOS_CUDA_ARCHITECTURES"), cuda)


if __name__ == "__main__":
    unittest.main()
