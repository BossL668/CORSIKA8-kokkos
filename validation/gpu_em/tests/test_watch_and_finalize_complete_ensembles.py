#!/usr/bin/env python3

import unittest

from validation.gpu_em.watch_and_finalize_complete_ensembles import completion_state


def distributed(completed=0, staged=0, failed=0, status="running"):
    return {
        "status": "monitoring",
        "remote": [
            {
                "remote_completed": completed,
                "staged_events": staged,
                "remote_failed": failed,
                "remote_status": status,
            }
        ],
    }


def local(completed=0, target=500, status="running", failed=False):
    return [
        {
            "completed_events": completed,
            "target_events": target,
            "status": status,
            "failed": failed,
        }
    ]


class CompleteEnsembleWatcherTest(unittest.TestCase):
    def test_exact_terminal_ensembles_are_ready(self):
        state = completion_state(
            distributed(500, 500, status="complete"),
            local(500, status="complete"),
            500,
        )
        self.assertTrue(state["ready"])
        self.assertFalse(state["terminal_failure"])

    def test_partial_ensembles_wait_without_becoming_acceptance_sample(self):
        state = completion_state(distributed(123, 123), local(100), 500)
        self.assertFalse(state["ready"])
        self.assertFalse(state["terminal_failure"])
        self.assertEqual(state["reason"], "waiting for exact ensembles")

    def test_runner_failure_is_terminal(self):
        state = completion_state(distributed(20, 20, failed=1), local(20), 500)
        self.assertFalse(state["ready"])
        self.assertTrue(state["terminal_failure"])

    def test_wrong_cuda_target_is_terminal(self):
        state = completion_state(distributed(), local(target=499), 500)
        self.assertTrue(state["terminal_failure"])

    def test_transient_distributed_poll_failure_waits(self):
        state = completion_state(
            {"status": "poll_failed", "error": "temporary SSH failure"},
            local(),
            500,
        )
        self.assertFalse(state["ready"])
        self.assertFalse(state["terminal_failure"])


if __name__ == "__main__":
    unittest.main()
