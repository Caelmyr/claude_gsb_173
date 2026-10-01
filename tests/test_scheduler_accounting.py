"""Scheduler-level tests for running-task accounting under retries/speculation.

These exercise ``on_task_status`` / ``on_task_complete`` against the real
JobManager + WorkerRegistry + FaultTolerance, with only the network, metrics,
shuffle and log sides stubbed (no Flask required).
"""

import shutil
import tempfile
import unittest

from backend.common import constants as C
from backend.common.config import ClusterConfig
from backend.common.logbus import LogBus
from backend.common.storage import Storage
from backend.master.fault_tolerance import FaultTolerance
from backend.master.job_manager import JobManager
from backend.master.registry import WorkerRegistry
from backend.master.scheduler import Scheduler


class _StubShuffle:
    def mark_partition_done(self, *a, **k):
        pass


class _StubMetrics:
    def record_task(self, *a, **k):
        pass


class _StubLog:
    def info(self, *a, **k):
        pass

    warn = info
    error = info


class _StubClient:
    def post(self, *a, **k):
        raise AssertionError("scheduler must not perform HTTP in these tests")


class TestSchedulerCompletionAccounting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        storage = Storage(self.tmp)
        self.config = ClusterConfig(max_attempts=3, retry_backoff_base_sec=0.01)
        self.logbus = LogBus(storage)
        self.jm = JobManager(storage, self.config, self.logbus)
        self.registry = WorkerRegistry(storage, self.config)
        self.ft = FaultTolerance(storage, self.jm, self.config, self.logbus)
        self.scheduler = Scheduler(
            storage, self.jm, self.registry, _StubShuffle(), self.ft,
            _StubMetrics(), self.config, _StubLog(),
        )
        self.scheduler.client = _StubClient()
        self.job = self.jm.submit({
            "name": "t", "mapper": "wordcount_mapper", "reducer": "count_reducer",
            "num_map_tasks": 2, "num_reduce_tasks": 1, "input_rows": 40, "params": {},
        })
        self.task = self.jm.tasks_for(self.job.job_id, C.TASK_MAP)[0]
        self.registry.register({
            "worker_id": "w1", "name": "w1", "host": "127.0.0.1", "port": 1,
            "cpu_cores": 4, "mem_total_mb": 1024, "exec_mode": "thread",
        })

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _status(self, wid="w1", attempt=0, ids=None):
        self.scheduler.on_task_status({
            "worker_id": wid, "job_id": self.job.job_id, "task_id": self.task.task_id,
            "state_seq": 1, "running_task_ids": ids if ids is not None else [self.task.task_id],
            "progress": 0.5, "records_processed": 1, "records_emitted": 1,
            "attempt": attempt,
        })

    def _fail(self, wid="w1", attempt=0, speculative=False, ids=None):
        self.scheduler.on_task_complete({
            "worker_id": wid, "job_id": self.job.job_id, "task_id": self.task.task_id,
            "status": C.TASK_FAILED, "error": "boom", "attempt": attempt,
            "speculative": speculative,
            "state_seq": 2, "running_task_ids": ids if ids is not None else [],
        })

    def _succeed(self, wid="w1", attempt=0):
        self.scheduler.on_task_complete({
            "worker_id": wid, "job_id": self.job.job_id, "task_id": self.task.task_id,
            "kind": C.TASK_MAP, "status": C.TASK_SUCCEEDED, "attempt": attempt,
            "records_processed": 10, "records_emitted": 10, "duration_ms": 5,
            "state_seq": 2, "running_task_ids": [],
        })

    def test_running_then_success_zeroes_and_tallies(self):
        self.registry.note_dispatched("w1", self.task.task_id)
        self.assertEqual(self.registry.get("w1").running_tasks, 1)
        self._status()
        self.assertEqual(self.registry.get("w1").running_tasks, 1)
        self.assertEqual(self.jm.get_task(self.job.job_id, self.task.task_id).status, C.TASK_RUNNING)
        self._succeed()
        w = self.registry.get("w1")
        self.assertEqual(w.running_tasks, 0)
        self.assertEqual(w.total_tasks_completed, 1)
        self.assertEqual(w.total_tasks_failed, 0)

    def test_duplicate_failure_reports_counted_once(self):
        self._status()
        self._fail(attempt=0)
        task = self.jm.get_task(self.job.job_id, self.task.task_id)
        self.assertEqual(task.status, C.TASK_RETRYING)
        self.assertEqual(task.attempts, 1)
        self.assertEqual(self.registry.get("w1").total_tasks_failed, 1)
        # The HTTP client retries the same failed POST — must be a no-op.
        self._fail(attempt=0)
        task = self.jm.get_task(self.job.job_id, self.task.task_id)
        self.assertEqual(task.status, C.TASK_RETRYING)
        self.assertEqual(task.attempts, 1)
        self.assertEqual(self.registry.get("w1").total_tasks_failed, 1)

    def test_late_failure_from_old_attempt_ignored_after_retry_succeeds(self):
        self._status(attempt=0)
        self._fail(attempt=0)  # attempt 0 fails -> retrying (attempts=1)
        # Attempt 1 redispatches and succeeds.
        self.registry.note_dispatched("w1", self.task.task_id)
        self._status(attempt=1)
        self._succeed(attempt=1)
        self.assertEqual(
            self.jm.get_task(self.job.job_id, self.task.task_id).status,
            C.TASK_SUCCEEDED,
        )
        # A delayed failure report for the old attempt arrives afterwards.
        self._fail(attempt=0)
        task = self.jm.get_task(self.job.job_id, self.task.task_id)
        self.assertEqual(task.status, C.TASK_SUCCEEDED)
        self.assertEqual(self.registry.get("w1").running_tasks, 0)

    def test_failing_speculative_copy_does_not_retry_task(self):
        self._status()
        # A speculative backup on w2 fails while the primary on w1 still runs.
        self.registry.register({
            "worker_id": "w2", "name": "w2", "host": "127.0.0.1", "port": 2,
            "cpu_cores": 4, "mem_total_mb": 1024, "exec_mode": "thread",
        })
        self._fail(wid="w2", speculative=True)
        task = self.jm.get_task(self.job.job_id, self.task.task_id)
        self.assertEqual(task.status, C.TASK_RUNNING)  # not RETRYING
        self.assertEqual(task.attempts, 0)
        self.assertEqual(self.registry.get("w1").running_tasks, 1)
        self.assertEqual(self.registry.get("w2").running_tasks, 0)
        self.assertEqual(self.registry.get("w2").total_tasks_failed, 1)
        # Primary still succeeds normally.
        self._succeed(wid="w1")
        self.assertEqual(self.registry.get("w1").running_tasks, 0)

    def test_duplicate_success_after_failure_does_not_revive(self):
        self._status(attempt=0)
        self._fail(attempt=0)
        # A stray success for the same dead attempt must not flip to SUCCEEDED.
        self._succeed(attempt=0)
        task = self.jm.get_task(self.job.job_id, self.task.task_id)
        self.assertEqual(task.status, C.TASK_RETRYING)


if __name__ == "__main__":
    unittest.main()
