"""Invariant tests for per-worker running-task accounting.

The nodes page must always show the exact number of tasks a worker's executor
is actually running — never zero while tasks run, never inflated, never
negative — under retries, speculation, duplicate/out-of-order reports and
concurrent submissions.

The accounting model is set-based: workers publish the full set of running task
ids (with a monotonic sequence number); the Master replaces, never increments
or decrements a counter.
"""

import shutil
import tempfile
import time
import unittest

from backend.common.config import ClusterConfig
from backend.common.logbus import LogBus
from backend.common.storage import Storage
from backend.master.registry import WorkerRegistry
from backend.worker.executor import Executor


def _registry(tmp: str, timeout: float = 80.0) -> WorkerRegistry:
    return WorkerRegistry(
        Storage(tmp), ClusterConfig(heartbeat_timeout_sec=timeout),
    )


def _reg(registry: WorkerRegistry, wid: str = "w1", port: int = 9001):
    return registry.register({
        "worker_id": wid, "name": wid, "host": "127.0.0.1", "port": port,
        "cpu_cores": 4, "mem_total_mb": 1024, "exec_mode": "thread",
    })


def _hb(registry: WorkerRegistry, wid: str, ids, seq, running=None):
    payload = {
        "worker_id": wid, "cpu_percent": 1.0, "mem_percent": 2.0, "load1": 0.5,
        "running_task_ids": list(ids), "state_seq": seq,
        "queued_tasks": 0,
    }
    if running is not None:
        payload["running_tasks"] = running
    else:
        payload["running_tasks"] = len(ids)
    return registry.heartbeat(payload)


class TestRegistryAccounting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.registry = _registry(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- basics --------------------------------------------------------
    def test_fresh_register_has_zero(self):
        _reg(self.registry)
        w = self.registry.get("w1")
        self.assertEqual(w.running_tasks, 0)
        self.assertEqual(w.running_task_ids, [])

    def test_heartbeat_snapshot_sets_count(self):
        _reg(self.registry)
        _hb(self.registry, "w1", ["a", "b", "c"], 1)
        self.assertEqual(self.registry.get("w1").running_tasks, 3)
        _hb(self.registry, "w1", ["a"], 2)
        self.assertEqual(self.registry.get("w1").running_tasks, 1)
        _hb(self.registry, "w1", [], 3)
        self.assertEqual(self.registry.get("w1").running_tasks, 0)

    # -- out-of-order delivery (the original "larger than reality" bug) --
    def test_stale_snapshot_is_ignored(self):
        _reg(self.registry)
        _hb(self.registry, "w1", ["a", "b"], 5)
        self.assertEqual(self.registry.get("w1").running_tasks, 2)
        # A delayed heartbeat carrying an older, larger set must not roll back.
        _hb(self.registry, "w1", ["a", "b", "c", "d"], 4)
        self.assertEqual(self.registry.get("w1").running_tasks, 2)
        self.assertEqual(self.registry.get("w1").state_seq, 5)
        # A fresher snapshot applies normally.
        _hb(self.registry, "w1", ["a"], 6)
        self.assertEqual(self.registry.get("w1").running_tasks, 1)

    def test_task_report_completion_beats_inflight_heartbeat(self):
        """Complete(seq2) arriving before heartbeat(seq1) must leave count 0."""
        _reg(self.registry)
        _hb(self.registry, "w1", ["t1"], 1)
        self.assertEqual(self.registry.get("w1").running_tasks, 1)
        # The task finishes; completion carries the post-removal snapshot.
        self.registry.apply_task_report({
            "worker_id": "w1", "task_id": "t1", "state_seq": 2,
            "running_task_ids": [],
        })
        self.assertEqual(self.registry.get("w1").running_tasks, 0)
        # The stale in-flight heartbeat (seq1) lands afterwards — must stay 0.
        _hb(self.registry, "w1", ["t1"], 1)
        self.assertEqual(self.registry.get("w1").running_tasks, 0)

    # -- dispatch window (the original "shows zero while running" bug) --
    def test_dispatched_count_immediately(self):
        _reg(self.registry)
        self.registry.note_dispatched("w1", "t1")
        self.assertEqual(self.registry.get("w1").running_tasks, 1)
        # Second concurrent dispatch on the same node.
        self.registry.note_dispatched("w1", "t2")
        self.assertEqual(self.registry.get("w1").running_tasks, 2)
        # First worker snapshot arrives confirming t1 only; t2 still launching.
        _hb(self.registry, "w1", ["t1"], 1)
        self.assertEqual(self.registry.get("w1").running_tasks, 2)
        # Next snapshot confirms both — no double counting.
        _hb(self.registry, "w1", ["t1", "t2"], 2)
        self.assertEqual(self.registry.get("w1").running_tasks, 2)

    def test_completion_drops_launching_even_without_progress_report(self):
        """An ultra-fast task that never sent progress still clears at finish."""
        _reg(self.registry)
        self.registry.note_dispatched("w1", "t1")
        self.assertEqual(self.registry.get("w1").running_tasks, 1)
        self.registry.apply_task_report({
            "worker_id": "w1", "task_id": "t1", "state_seq": 2,
            "running_task_ids": [],
        })
        self.assertEqual(self.registry.get("w1").running_tasks, 0)

    def test_launching_union_dedupes(self):
        _reg(self.registry)
        self.registry.note_dispatched("w1", "t1")
        # Snapshot already contains the launching id -> union, not sum.
        _hb(self.registry, "w1", ["t1", "t2"], 3)
        self.assertEqual(self.registry.get("w1").running_tasks, 2)

    # -- retries / speculation / duplicates (the "negative" family) -----
    def test_duplicate_completion_idempotent(self):
        _reg(self.registry)
        _hb(self.registry, "w1", ["t1"], 1)
        finish = {
            "worker_id": "w1", "task_id": "t1", "state_seq": 2,
            "running_task_ids": [],
        }
        for _ in range(5):  # HTTP client retries deliver it repeatedly
            self.registry.apply_task_report(dict(finish))
        self.assertEqual(self.registry.get("w1").running_tasks, 0)

    def test_never_negative_under_any_report_sequence(self):
        _reg(self.registry)
        # Fuzz a storm of snapshot/completion/launch operations; the count must
        # always equal len(observed ∪ launching) and stay non-negative.
        ids = ["a", "b", "c", "d"]
        seq = 0
        import random
        rng = random.Random(0)
        for _ in range(2000):
            action = rng.randrange(3)
            if action == 0:
                seq += rng.randrange(1, 3)
                _hb(self.registry, "w1", rng.sample(ids, rng.randrange(len(ids) + 1)), seq)
            elif action == 1:
                tid = rng.choice(ids)
                self.registry.note_dispatched("w1", tid)
            else:
                self.registry.apply_task_report({
                    "worker_id": "w1", "task_id": rng.choice(ids),
                    "state_seq": seq, "running_task_ids": rng.sample(ids, rng.randrange(len(ids) + 1)),
                })
            w = self.registry.get("w1")
            self.assertGreaterEqual(w.running_tasks, 0)
            launching = self.registry._launching.get("w1", {})
            self.assertEqual(w.running_tasks, len(set(w.running_task_ids) | set(launching)))

    def test_retry_on_another_worker_counts_independently(self):
        _reg(self.registry, "w1")
        _reg(self.registry, "w2", port=9002)
        # t1 running on w1, fails; the retry lands on w2.
        _hb(self.registry, "w1", ["t1"], 1)
        self.registry.apply_task_report({
            "worker_id": "w1", "task_id": "t1", "state_seq": 2,
            "running_task_ids": [],
        })
        self.assertEqual(self.registry.get("w1").running_tasks, 0)
        self.registry.note_dispatched("w2", "t1")
        self.assertEqual(self.registry.get("w2").running_tasks, 1)
        _hb(self.registry, "w2", ["t1"], 1)
        self.assertEqual(self.registry.get("w2").running_tasks, 1)
        # A late failure/complete from the OLD attempt on w1 must not decrement w2.
        self.registry.apply_task_report({
            "worker_id": "w1", "task_id": "t1", "state_seq": 9,
            "running_task_ids": [],
        })
        self.assertEqual(self.registry.get("w1").running_tasks, 0)
        self.assertEqual(self.registry.get("w2").running_tasks, 1)

    def test_speculative_copies_count_per_worker(self):
        _reg(self.registry, "w1")
        _reg(self.registry, "w2", port=9002)
        # Same task id physically running on two workers during speculation.
        _hb(self.registry, "w1", ["t1"], 1)
        _hb(self.registry, "w2", ["t1"], 1)
        self.assertEqual(self.registry.get("w1").running_tasks, 1)
        self.assertEqual(self.registry.get("w2").running_tasks, 1)
        # Winner on w1 finishes; loser on w2 is canceled and reports empty.
        self.registry.apply_task_report({
            "worker_id": "w1", "task_id": "t1", "state_seq": 2,
            "running_task_ids": [],
        })
        self.registry.apply_task_report({
            "worker_id": "w2", "task_id": "t1", "state_seq": 2,
            "running_task_ids": [],
        })
        self.assertEqual(self.registry.get("w1").running_tasks, 0)
        self.assertEqual(self.registry.get("w2").running_tasks, 0)

    # -- lifecycle ------------------------------------------------------
    def test_reregister_resets_stale_state(self):
        _reg(self.registry)
        _hb(self.registry, "w1", ["a", "b", "c"], 5)
        self.assertEqual(self.registry.get("w1").running_tasks, 3)
        # Simulate a restart reusing the same worker id (persisted on disk).
        _reg(self.registry)
        w = self.registry.get("w1")
        self.assertEqual(w.running_tasks, 0)
        self.assertEqual(w.running_task_ids, [])
        self.assertEqual(w.state_seq, 0)
        self.assertNotIn("w1", self.registry._launching)

    def test_reap_zeros_dead_worker(self):
        _reg(self.registry)
        _hb(self.registry, "w1", ["a"], 1)
        # Force the heartbeat far into the past, then reap with a short timeout.
        self.registry.get("w1").last_heartbeat_ms = 0
        dead = self.registry.reap()
        self.assertEqual([w.worker_id for w in dead], ["w1"])
        self.assertEqual(self.registry.get("w1").running_tasks, 0)
        self.assertEqual(self.registry.get("w1").status, "dead")

    def test_launching_ttl_pruned_for_lost_dispatch(self):
        _reg(self.registry)
        self.registry.note_dispatched("w1", "ghost")
        self.registry._launching["w1"]["ghost"] = time.time() - 120
        self.assertEqual(self.registry.get("w1").running_tasks, 1)
        self.registry.prune_launching(ttl_sec=60)
        self.assertEqual(self.registry.get("w1").running_tasks, 0)

    def test_outcome_counters_independent_of_running_set(self):
        _reg(self.registry)
        self.registry.record_task_outcome("w1", success=True)
        self.registry.record_task_outcome("w1", success=True)
        self.registry.record_task_outcome("w1", success=False)
        w = self.registry.get("w1")
        self.assertEqual(w.total_tasks_completed, 2)
        self.assertEqual(w.total_tasks_failed, 1)
        self.assertEqual(w.running_tasks, 0)

    def test_persistence_roundtrip(self):
        _reg(self.registry)
        _hb(self.registry, "w1", ["a", "b"], 7)
        # A brand-new registry over the same storage must restore the snapshot.
        reloaded = _registry(self.tmp)
        w = reloaded.get("w1")
        self.assertEqual(w.running_task_ids, ["a", "b"])
        self.assertEqual(w.running_tasks, 2)
        self.assertEqual(w.state_seq, 7)


class TestExecutorSnapshot(unittest.TestCase):
    """The worker side must publish an accurate, self-consistent snapshot."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _executor(self) -> Executor:
        return Executor(
            "w-x", self.tmp, "http://127.0.0.1:1",
            ClusterConfig(), exec_mode="thread",
        )

    def test_snapshot_add_and_remove(self):
        ex = self._executor()
        accepted = []
        for i in range(3):
            ok = ex.start_task({
                "task_id": f"t{i}", "job_id": "j", "kind": "map",
                "mapper": "wordcount_mapper", "params": {},
                "records": [], "partition_count": 1,
            })
            accepted.append(ok)
        self.assertTrue(all(accepted))
        time.sleep(0.3)  # empty map tasks finish quickly
        seq, ids = ex.snapshot()
        self.assertIsInstance(seq, int)
        # Finished tasks are removed promptly; count is never negative.
        self.assertGreaterEqual(seq, 3)
        self.assertGreaterEqual(ex.running_count, 0)
        self.assertEqual(ex.running_count, len(ex.running_task_ids()))

    def test_duplicate_task_id_not_double_counted(self):
        ex = self._executor()
        spec = {
            "task_id": "dup", "job_id": "j", "kind": "map",
            "mapper": "wordcount_mapper", "params": {},
            "records": [], "partition_count": 1,
        }
        self.assertTrue(ex.start_task(spec))
        self.assertFalse(ex.start_task(dict(spec)))
        self.assertLessEqual(ex.running_count, 1)


if __name__ == "__main__":
    unittest.main()
