"""Launcher policy tests. Fake backends do not count as HIP/SYCL validation."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[2]
LAUNCHER = SOURCE / "tools/c8_air_shower_launcher.py"
spec = importlib.util.spec_from_file_location("c8_launcher", LAUNCHER)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="c8-launcher-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "install with spaces"
        (self.root / "bin").mkdir(parents=True)
        self.entry = self.root / "bin/c8_air_shower"
        shutil.copyfile(LAUNCHER, self.entry)
        self.events = self.root / "events.jsonl"
        self.environment = dict(os.environ, C8_TEST_EVENTS=str(self.events))

    def add_backend(self, backend, probe_code=0, app_code=0, delay=0):
        prefix = self.root / backend
        (prefix / "bin").mkdir(parents=True)
        (prefix / "share/corsika").mkdir(parents=True)
        (prefix / "share/corsika/backend.json").write_text(json.dumps({
            "schema": 1, "backend": backend, "physics_source": "proposal-native"}))
        common = ("#!/usr/bin/env python3\nimport json, os, sys, time\n"
                  f"backend={backend!r}\n"
                  "with open(os.environ['C8_TEST_EVENTS'], 'a') as f:\n"
                  " f.write(json.dumps([backend, os.path.basename(sys.argv[0])])+'\\n')\n")
        fields = {"backend": backend, "gpu": str(backend != "openmp").lower(),
                  "openmp": str(backend == "openmp").lower(), "scan_valid": "true",
                  "queue_stable_order": "true", "queue_roundtrip_exact": "true",
                  "memory_query_valid": "true"}
        probe = prefix / "bin/kokkos_backend_probe"
        probe.write_text(common + f"time.sleep({delay!r})\n"
                         f"print({''.join(k + '=' + v + chr(10) for k, v in fields.items())!r})\n"
                         f"sys.exit({probe_code})\n")
        app = prefix / "bin/c8_air_shower"
        app.write_text(common + "print(json.dumps({'argv':sys.argv[1:], 'pid':os.getpid(), "
                       "'omp':os.environ.get('OMP_NUM_THREADS')}), flush=True)\n"
                       "if '--test-sleep' in sys.argv: time.sleep(30)\n"
                       f"sys.exit({app_code})\n")
        probe.chmod(0o755)
        app.chmod(0o755)

    def run_entry(self, *arguments):
        return subprocess.run([sys.executable, str(self.entry), *arguments],
                              env=self.environment, text=True, capture_output=True, timeout=10)

    def calls(self):
        return [json.loads(line) for line in self.events.read_text().splitlines()] \
            if self.events.exists() else []

    def test_inventory_and_help_never_probe(self):
        self.add_backend("cuda")
        for flag in ("--list-backends", "--help"):
            result = self.run_entry(flag)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), [])

    def test_unified_device_parser(self):
        for arguments, expected in [
                (["--device", "0"], ["0"]),
                (["--device=0,1,2,3"], ["0", "1", "2", "3"]),
                (["--device", "0", "1", "-E", "100"], ["0", "1"]),
                (["--devices", "0,1"], ["0", "1"]),
                (["--device", "GPU-test-0"], ["GPU-test-0"])]:
            self.assertEqual(launcher.selected_devices(arguments), expected)
        for arguments in [
                ["--device"], ["--device", "0,"], ["--device", "0,,1"],
                ["--device", "0,00"], ["--device", "-1"],
                ["--device", "0", "--devices", "1"],
                ["--device", "0", "--kokkos-device", "0"]]:
            with self.subTest(arguments=arguments), self.assertRaises(launcher.LaunchError):
                launcher.selected_devices(arguments)

    def test_public_device_selects_cuda_and_preserves_list(self):
        self.add_backend("cuda")
        self.add_backend("openmp")
        smi = self.root / "bin/nvidia-smi"
        smi.write_text("#!/bin/sh\nprintf '0, GPU-test-0\\n1, GPU-test-1\\n'\n")
        smi.chmod(0o755)
        self.environment["PATH"] = str(smi.parent) + os.pathsep + os.environ["PATH"]
        for selector in (["--device", "0"], ["--device", "0", "1"],
                         ["--devices", "0,1"]):
            result = self.run_entry("--dry-run", *selector)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(data["selected"]["backend"], "cuda")
            self.assertTrue(all(value in data["argv"] for value in selector))

    def test_public_device_rejects_cpu_and_unknown_gpu(self):
        self.add_backend("openmp")
        self.add_backend("cuda")
        for arguments in [
                ["--backend", "openmp", "--device", "0"],
                ["--em-backend", "proposal", "--device", "0"]]:
            result = self.run_entry(*arguments)
            self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(self.calls(), [])
        with patch.object(launcher.shutil, "which", return_value="nvidia-smi"), \
                patch.object(launcher.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], 0, "0, GPU-test-0\n1, GPU-test-1\n", "")):
            self.assertEqual(launcher.resolve_requested_gpus(["1"]), ["GPU-test-1"])
            for selection in (["5"], ["0", "GPU-test-0"]):
                with self.assertRaises(launcher.LaunchError):
                    launcher.resolve_requested_gpus(selection)

    def test_openmp_does_not_probe_installed_gpus(self):
        self.add_backend("openmp")
        self.add_backend("cuda")
        result = self.run_entry("--backend", "openmp", "--kokkos-num-threads", "8", "-E", "1000")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["omp"], "8")
        self.assertTrue(all(call[0] == "openmp" for call in self.calls()))

    def test_auto_cpu_only(self):
        self.add_backend("openmp")
        result = self.run_entry("--dry-run", "-E", "1000")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["selected"]["backend"], "openmp")

    def test_auto_single_gpu_and_serial_host(self):
        self.add_backend("openmp")
        self.add_backend("cuda")
        self.environment["OMP_NUM_THREADS"] = "128"
        result = self.run_entry("--dry-run", "-E", "1000")
        data = json.loads(result.stdout)
        self.assertEqual(data["selected"]["backend"], "cuda")
        self.assertEqual(data["thread_environment"]["OMP_NUM_THREADS"], "1")
        self.assertEqual(self.calls(), [["cuda", "kokkos_backend_probe"]])

    def test_hip_and_sycl_dispatch_protocol(self):
        for backend in ("hip", "sycl"):
            self.add_backend(backend)
            result = self.run_entry("--backend", backend, "--dry-run")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["selected"]["backend"], backend)

    def test_failed_auto_gpu_reports_cpu_selection(self):
        self.add_backend("cuda", probe_code=4)
        self.add_backend("openmp")
        result = self.run_entry("--dry-run", "-E", "1000")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("selecting OpenMP before startup", result.stderr)

    def test_explicit_failed_gpu_never_falls_back(self):
        self.add_backend("cuda", probe_code=4)
        self.add_backend("openmp")
        result = self.run_entry("--backend=cuda", "-E", "1000")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.calls(), [["cuda", "kokkos_backend_probe"]])

    def test_no_backend(self):
        result = self.run_entry("--dry-run", "-E", "1000")
        self.assertEqual(result.returncode, 2)
        self.assertIn("no usable installed backend", result.stderr)

    def test_ambiguous_auto(self):
        self.add_backend("cuda")
        self.add_backend("sycl")
        result = self.run_entry("--dry-run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("multiple GPU", result.stderr)

    def test_auto_threads_select_openmp(self):
        self.add_backend("cuda")
        self.add_backend("openmp")
        result = self.run_entry("--dry-run", "--kokkos-num-threads=4")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), [["openmp", "kokkos_backend_probe"]])

    def test_runtime_failure_preserves_exit_code_no_retry(self):
        self.add_backend("cuda", app_code=7)
        self.add_backend("openmp")
        result = self.run_entry("--backend", "cuda", "-E", "1000")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(self.calls(), [["cuda", "kokkos_backend_probe"],
                                       ["cuda", "c8_air_shower"]])

    def test_arguments_with_spaces_and_shell_characters(self):
        self.add_backend("openmp")
        path = "some directory/$(false); 'quoted' output"
        result = self.run_entry("--backend=openmp", "-f", path, "--seed", "42")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)["argv"]
        self.assertEqual(args, ["--em-backend", "kokkos-proposal", "--radio-backend", "kokkos",
                                "-f", path, "--seed", "42"])

    def test_scalar_reference_preserved(self):
        self.add_backend("openmp")
        self.add_backend("cuda")
        result = self.run_entry("--em-backend", "proposal", "-E", "1000")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cpu", json.loads(result.stdout)["argv"])
        self.assertTrue(all(call[0] == "openmp" for call in self.calls()))

    def test_named_kokkos_physics_backends(self):
        self.add_backend("openmp")
        self.add_backend("cuda")
        for em in ("kokkos-proposal", "kokkos-egs4"):
            for execution in ("openmp", "cuda"):
                with self.subTest(em=em, execution=execution):
                    result = self.run_entry("--backend", execution, "--em-backend", em,
                                            "--kokkos-execution", execution, "--dry-run")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    data = json.loads(result.stdout)
                    self.assertEqual(data["selected"]["backend"], execution)
                    args = data["argv"]
                    self.assertEqual(args[args.index("--em-backend") + 1], em)
                    self.assertEqual(args[args.index("--radio-backend") + 1], "kokkos")
                    self.assertEqual(args[args.index("--kokkos-execution") + 1], execution)

    def test_egs4_cpu_radio_still_allowed(self):
        args, scalar = launcher.application_arguments(
            ["--em-backend", "kokkos-egs4", "--radio-backend", "cpu"])
        self.assertFalse(scalar)
        self.assertEqual(args, ["--em-backend", "kokkos-egs4", "--radio-backend", "cpu"])

    def test_old_em_names_are_not_canonical_options(self):
        for em in ("kokkos", "egs4"):
            with self.subTest(em=em), self.assertRaises(launcher.LaunchError):
                launcher.application_arguments(["--em-backend", em])

    def test_mismatched_manifest(self):
        self.add_backend("openmp")
        path = self.root / "openmp/share/corsika/backend.json"
        path.write_text('{"schema":1,"backend":"cuda"}')
        result = self.run_entry("--backend", "openmp", "-E", "1000")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_wrong_probe_identity(self):
        self.add_backend("cuda")
        record = launcher.inventory(self.root)["cuda"]
        with patch.object(launcher.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, "backend=openmp\n", "")):
            self.assertFalse(launcher.probe(record, [])["available"])

    def test_probe_timeout(self):
        self.add_backend("cuda", delay=1)
        with patch.dict(os.environ, self.environment):
            result = launcher.probe(launcher.inventory(self.root)["cuda"], [], timeout=.05)
        self.assertFalse(result["available"])
        self.assertIn("timed out", result["reason"])

    def test_rejected_safety_options(self):
        self.add_backend("cuda")
        self.add_backend("openmp")
        cases = [("--backend=cuda", "--kokkos-num-threads", "2"),
                 ("--backend=openmp", "--kokkos-device", "0"),
                 ("--em-backend=kokkos-proposal", "--radio-backend=cpu"),
                 ("--gpu-physics-source=c8emrt",), ("--hadronic-workers=12",),
                 ("--backend=cuda", "--backend=openmp"),
                 ("--kokkos-num-threads=oops",), ("--kokkos-device=-1",),
                 ("--em-backend=kokkos-proposal", "--em-backend=proposal")]
        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(self.run_entry(*case).returncode, 2)

    def test_application_help_forwarded(self):
        self.add_backend("openmp")
        result = self.run_entry("--backend", "openmp", "--", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--help", json.loads(result.stdout)["argv"])

    def test_exec_pid_and_sigterm(self):
        self.add_backend("openmp")
        child = subprocess.Popen([sys.executable, str(self.entry), "--backend", "openmp",
                                  "--test-sleep"], env=self.environment,
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            data = json.loads(child.stdout.readline())
            self.assertEqual(data["pid"], child.pid)
            child.send_signal(signal.SIGTERM)
            child.communicate(timeout=5)
            self.assertEqual(child.returncode, -signal.SIGTERM)
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()


class BuildHelperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="c8-build-helper-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.log = self.root / "commands.jsonl"
        for command in ("conan", "cmake", "nvidia-smi"):
            path = self.root / command
            path.write_text(
                "#!/usr/bin/env python3\nimport json,os,sys\n"
                f"with open({str(self.log)!r},'a') as f: f.write(json.dumps(sys.argv)+'\\n')\n"
                "if os.path.basename(sys.argv[0])=='nvidia-smi': print('8.9')\n")
            path.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ["PATH"])
        for key in ("C8_KOKKOS_PROFILE", "C8_KOKKOS_OPENMP_PROFILE", "C8_BUILD_JOBS"):
            self.env.pop(key, None)

    def run_helper(self, *args):
        return subprocess.run(["bash", str(SOURCE / "tools/build_kokkos.sh"), *args],
                              env=self.env, text=True, capture_output=True, timeout=10)

    def test_paired_install_uses_distinct_paths_and_serial_builds(self):
        result = self.run_helper("openmp,cuda", "-DWITH_FLUKA=ON")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [json.loads(line) for line in self.log.read_text().splitlines()]
        installs = [c for c in commands if "--install" in c]
        self.assertEqual(len(installs), 2)
        self.assertTrue(installs[0][-1].endswith("/build/openmp"))
        self.assertTrue(installs[1][-1].endswith("/build/cuda"))
        conan = [c for c in commands if Path(c[0]).name == "conan"]
        self.assertTrue(all("tools.build:jobs=1" in c for c in conan))
        configure = [c for c in commands if "-S" in c]
        self.assertTrue(all(any(a.startswith("-DCORSIKA_LAUNCHER_PREFIX=") for a in c)
                            for c in configure))

    def test_invalid_lists_fail_before_build(self):
        for value in ("openmp,openmp", "cuda,hip", "openmp,", "unknown"):
            self.assertEqual(self.run_helper(value).returncode, 2)
        self.assertFalse(self.log.exists())

    def test_hip_requires_deliberate_target_profile(self):
        self.assertEqual(self.run_helper("openmp,hip").returncode, 2)
        self.assertFalse(self.log.exists())

    def test_explicit_profile_does_not_need_gpu_discovery(self):
        self.env["C8_KOKKOS_PROFILE"] = "hip-vega90a"
        result = self.run_helper("openmp,hip")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("nvidia-smi", self.log.read_text())


if __name__ == "__main__":
    unittest.main()
