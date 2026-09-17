"""Safety tests: no configure, build, production output or simulation is run."""
from pathlib import Path
import tempfile
import unittest

import build_cpu_priority_candidate as candidate


class CandidateStageTests(unittest.TestCase):
    def test_existing_candidate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / "already-created"
            stage.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                candidate.require_new_stage(stage, root / "frozen", root / "live")

    def test_nested_candidate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "frozen"
            frozen.mkdir()
            with self.assertRaisesRegex(ValueError, "must not contain or be inside"):
                candidate.require_new_stage(frozen / "new", frozen, root / "live")

    def test_non_data_source_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "file.cpp").symlink_to("/dev/null")
            with self.assertRaisesRegex(ValueError, "unapproved source symlink"):
                candidate.inventory(root)

    def test_allowlist_excludes_physics_and_build_changes(self):
        self.assertEqual(len(candidate.OVERLAY_ALLOWLIST), 6)
        for path in candidate.OVERLAY_ALLOWLIST:
            self.assertFalse(path.endswith((".cu", ".cmake", "CMakeLists.txt")))
            self.assertNotIn("mountain", path)
            self.assertNotIn("ResidentLeptonCascade", path)
        self.assertNotIn("install", candidate.TARGET_ALLOWLIST)

    def test_blocking_wait_overlay_requires_explicit_opt_in(self):
        self.assertEqual(candidate.selected_overlay_files(candidate.OVERLAY_ALLOWLIST),
                         candidate.OVERLAY_ALLOWLIST)
        selected = candidate.selected_overlay_files(candidate.OVERLAY_ALLOWLIST, True)
        self.assertEqual(len(selected), 11)
        self.assertEqual(selected[-5:], candidate.BLOCKING_WAIT_OVERLAY_ALLOWLIST)
        with self.assertRaises(ValueError):
            candidate.selected_overlay_files([candidate.BLOCKING_WAIT_OVERLAY_ALLOWLIST[0]])
        self.assertFalse(any("mountain" in name or name.endswith(".cu") for name in selected))

    def test_common_types_exception_is_only_three_statistics_fields(self):
        before = "struct Counters {\n};\n"
        after = before.replace("};", candidate.BLOCKING_WAIT_STATISTICS_ADDITION + "};")
        self.assertIn('exact three', candidate.validate_statistics_overlay(before, after))
        self.assertEqual(candidate.validate_statistics_overlay(before, before), 'unchanged')
        with self.assertRaises(ValueError):
            candidate.validate_statistics_overlay(before, after + "double changed_physical_constant;\n")

    def test_checkpoint_wait_opt_in_is_narrow_and_requires_blocking(self):
        with self.assertRaisesRegex(ValueError, 'require'):
            candidate.selected_overlay_files(candidate.OVERLAY_ALLOWLIST, False, True)
        selected = candidate.selected_overlay_files(candidate.OVERLAY_ALLOWLIST, True, True)
        self.assertEqual(len(selected), 14)
        self.assertEqual(selected[-3:], candidate.CHECKPOINT_WAIT_OVERLAY_ALLOWLIST)
        with self.assertRaises(ValueError):
            candidate.selected_overlay_files([candidate.CHECKPOINT_WAIT_OVERLAY_ALLOWLIST[0]])

    def test_checkpoint_wait_refuses_physics_or_owning_stream_changes(self):
        for species, name in [('photons', 'KokkosResidentPhotonCascade.hpp'),
                              ('leptons', 'KokkosResidentLeptonCascade.hpp')]:
            before = f'      result.remaining_{species} = queue.download(execution);\n'
            after = before.replace('execution)', 'execution, cooperative_wait)')
            self.assertIn('exact blocking', candidate.validate_checkpoint_overlay(name, before, after))
            self.assertEqual(candidate.validate_checkpoint_overlay(name, before, before), 'unchanged')
            with self.assertRaises(ValueError):
                candidate.validate_checkpoint_overlay(name, before, after+'energy *= .99;\n')
        name = candidate.CHECKPOINT_WAIT_OVERLAY_ALLOWLIST[0]
        before = ('#include <corsika/accelerator/em/kokkos/KokkosResidentMemoryBudget.hpp>\n'
                  '    std::vector<gpu::em::EmParticleState> download(\n'
                  '        ExecutionSpace const& execution) const {\n'
                  '      ordered_execution.fence("download Kokkos EM wavefront");\n'
                  '      execution_synchronized_ = true;\n')
        after = ('#include <corsika/accelerator/em/kokkos/KokkosResidentMemoryBudget.hpp>\n'
                 '#include <corsika/accelerator/em/kokkos/ResidentExecutionWait.hpp>\n'
                 '    std::vector<gpu::em::EmParticleState> download(\n'
                 '        ExecutionSpace const& execution,\n'
                 '        ResidentExecutionWait<ExecutionSpace>* const blocking_wait = nullptr) const {\n'
                 '      // Only CPU-primary\'s explicit blocking mode changes this boundary.\n'
                 '      // In particular, never introduce a progress callback/reentrant host-work\n'
                 '      // point here for GPU-primary or other existing callers. The queue\'s\n'
                 '      // owning stream, not the compatibility argument, orders pack and copy.\n'
                 '      auto* const waiter = blocking_wait && blocking_wait->blockingEnabled()\n'
                 '                               ? blocking_wait : nullptr;\n'
                 '      waitResidentExecution(\n'
                 '          ordered_execution, "download Kokkos EM wavefront", waiter);\n'
                 '      execution_synchronized_ = true;\n')
        self.assertIn('exact blocking', candidate.validate_checkpoint_overlay(name, before, after))
        with self.assertRaises(ValueError):
            candidate.validate_checkpoint_overlay(name, before, after.replace(
                'ordered_execution, "download Kokkos EM wavefront", waiter',
                'execution, "download Kokkos EM wavefront", waiter'))


if __name__ == "__main__":
    unittest.main()
