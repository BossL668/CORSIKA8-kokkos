import unittest

from validation.gpu_em.run_final_test_acceptance import (
    parse_ctest_count,
    parse_unittest_count,
)


class FinalTestAcceptanceTest(unittest.TestCase):
    def test_counts_are_parsed_fail_closed(self):
        self.assertEqual(
            parse_ctest_count("100% tests passed, 0 failed out of 32"),
            32,
        )
        self.assertEqual(
            parse_unittest_count("Ran 108 tests in 1.25s"),
            108,
        )
        self.assertIsNone(parse_ctest_count("no summary"))
        self.assertIsNone(parse_unittest_count("OK"))


if __name__ == "__main__":
    unittest.main()
