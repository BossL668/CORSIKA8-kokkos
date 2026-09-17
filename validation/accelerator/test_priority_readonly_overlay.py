"""A read-only frozen source is never made writable by candidate preparation."""
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from build_cpu_priority_candidate import copy_overlay


class ReadOnlyOverlayTest(unittest.TestCase):
    def test_private_copy_accepts_patch_and_preserves_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); frozen = root/'frozen'; frozen.mkdir()
            old = frozen/'test.hpp'; old.write_text('old'); old.chmod(0o444)
            candidate = root/'candidate'; shutil.copytree(frozen, candidate)
            patch = root/'patch.hpp'; patch.write_text('new')
            copy_overlay(patch, candidate/'test.hpp', candidate)
            self.assertEqual((candidate/'test.hpp').read_text(), 'new')
            self.assertEqual(old.read_text(), 'old')
            self.assertEqual(old.stat().st_mode & 0o777, 0o444)

    def test_links_are_rejected_without_changing_target(self):
        for kind in ('symlink', 'hardlink', 'parent'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); candidate = root/'candidate'; candidate.mkdir()
                target = root/'target'; target.write_text('old'); target.chmod(0o444)
                patch = root/'patch'; patch.write_text('new')
                dest = candidate/'test.hpp'
                if kind == 'symlink':
                    dest.symlink_to(target)
                elif kind == 'hardlink':
                    os.link(target, dest)
                else:
                    (candidate/'link').symlink_to(root, target_is_directory=True)
                    dest = candidate/'link/target'
                with self.assertRaises(ValueError):
                    copy_overlay(patch, dest, candidate)
                self.assertEqual(target.read_text(), 'old')
                self.assertEqual(target.stat().st_mode & 0o777, 0o444)


if __name__ == '__main__':
    unittest.main()
