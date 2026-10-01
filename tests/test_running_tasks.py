"""Regression tests for the nodes-page 'running tasks' count.

The count shown per node must always equal the number of tasks actually
assigned to / running on that node — never zero while tasks run, never
stale-high after completion, never negative — across heartbeats, retries,
worker death and concurrent jobs.  It is derived from the Master's task
table (the same state the monitor page renders), not from a counter shared
between heartbeats and completion callbacks.
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
from backend.master.metrics import Metrics
from backend.master.registry import WorkerRegistry
from backend.master.scheduler import Scheduler
from backend.master.shuffle import ShuffleCoordinator


class TestRunningTasksCount(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = Storage(self.tmp)
        self.config = ClusterConfig()
        self.logbus = LogBus(self.storage)
        self.jm = JobManager(self.storage, self.config, self.logbus)
        self.registry = WorkerRegistry(self.storage, self.config)
        self.metrics = Metrics(self.storage)
        self.shuffle = ShuffleCoordinator(self.storage, self.jm, self.registry, self.logbus)
        self.ft = FaultTolerance(self.storage, self.jm, self.config, self.logbus)
        self.scheduler = Scheduler(
            self.storage, self.jm, self.registry, self.shuffle,
            self.ft, self.metrics, self.config, self.logbus,
        )
        # Wire exactly the way Master.__init__ does.
        self.registry.set_running_tasks_provider(self.scheduler.count_running_tasks)
        for wid, port in (("w1", 9001), ("w2", 9002)):
            self.registry.register({
                "worker_id": wid, "name": wid, "host": "127.0.0.1", "port": port,
            })
        self.job = self.jm.submit({
            "name": "t", "mapper": "wordcount_mapper", "reducer": "count_reducer",
            "num_map_tasks": 4, "num_reduce_tasks": 2, "input_rows": 100, "params": {},
        })

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- helpers ------------------------------------------------------
    def count(self, worker_id: str) -> int:
        """The value the nodes page would render for this worker."""
        for view in self.registry.summary()["workers"]:
            if view["worker_id"] == worker_id:
                return view["running_tasks"]
        raise AssertionError(f"worker {worker_id} missing from summary")

    def dispatch(self, task, worker_id: str, status: str = C.TASK_ASSIGNED):
        """Mimic Scheduler._dispatch's bookkeeping after a worker accepts."""
        self.jm.update_task(self.job.job_id, task.task_id,
                            status=status, worker_id=worker_id)

    def map_task(self, index: int):
        return self.jm.tasks_for(self.job.job_id, C.TASK_MAP)[index]

    # -- the reported symptoms ----------------------------------------
    def test_running_task_is_counted_immediately(self):
        """A task must show up the moment it is dispatched — not after a heartbeat."""
        self.dispatch(self.map_task(0), "w1")
        self.assertEqual(self.count("w1"), 1)
        self.assertEqual(self.count("w2"), 0)

    def test_assigned_and_running_both_count(self):
        self.dispatch(self.map_task(0), "w1", C.TASK_ASSIGNED)
        self.dispatch(self.map_task(1), "w1", C.TASK_RUNNING)
        self.dispatch(self.map_task(2), "w2", C.TASK_RUNNING)
        self.assertEqual(self.count("w1"), 2)
        self.assertEqual(self.count("w2"), 1)

    def test_completion_drops_count_to_zero_and_never_negative(self):
        task = self.map_task(0)
        self.dispatch(task, "w1", C.TASK_RUNNING)
        self.scheduler.on_task_complete({
            "worker_id": "w1", "job_id": self.job.job_id, "task_id": task.task_id,
            "status": C.TASK_SUCCEEDED, "records_processed": 10, "duration_ms": 5,
        })
        self.assertEqual(self.count("w1"), 0)
        # Extra completion bookkeeping (duplicates, late callbacks) must not
        # push the count below zero.
        for _ in range(5):
            self.registry.task_finished("w1", success=True)
            self.registry.task_finished("w1", success=False)
        self.assertEqual(self.count("w1"), 0)
        self.assertGreaterEqual(self.registry.get("w1").running_tasks, 0)

    def test_heartbeat_payload_cannot_corrupt_count(self):
        """In-flight heartbeats carry stale samples; they must be ignored."""
        self.dispatch(self.map_task(0), "w1", C.TASK_RUNNING)
        # Stale heartbeat sampled before the task started.
        self.registry.heartbeat({"worker_id": "w1", "running_tasks": 0, "queued_tasks": 0})
        self.assertEqual(self.count("w1"), 1)
        # Bogus / malicious values are not adopted either.
        self.registry.heartbeat({"worker_id": "w1", "running_tasks": 99, "queued_tasks": 0})
        self.assertEqual(self.count("w1"), 1)
        self.registry.heartbeat({"worker_id": "w1", "running_tasks": -3, "queued_tasks": 0})
        self.assertEqual(self.count("w1"), 1)

    def test_failure_and_retry_cycle(self):
        task = self.map_task(0)
        self.dispatch(task, "w1", C.TASK_RUNNING)
        self.assertEqual(self.count("w1"), 1)
        # The attempt fails: the task leaves the worker (RETRYING, no owner).
        self.scheduler.on_task_complete({
            "worker_id": "w1", "job_id": self.job.job_id, "task_id": task.task_id,
            "status": C.TASK_FAILED, "error": "boom",
        })
        self.assertEqual(self.count("w1"), 0)
        # Re-dispatched (possibly to another worker): counted there, once.
        self.dispatch(task, "w2", C.TASK_ASSIGNED)
        self.assertEqual(self.count("w1"), 0)
        self.assertEqual(self.count("w2"), 1)

    def test_retrying_task_with_stale_owner_is_not_counted(self):
        task = self.map_task(0)
        self.jm.update_task(self.job.job_id, task.task_id,
                            status=C.TASK_RETRYING, worker_id="w1")
        self.assertEqual(self.count("w1"), 0)

    def test_worker_death_reassignment_clears_count(self):
        self.dispatch(self.map_task(0), "w1", C.TASK_RUNNING)
        self.dispatch(self.map_task(1), "w1", C.TASK_ASSIGNED)
        self.assertEqual(self.count("w1"), 2)
        self.ft.handle_worker_death(self.registry.get("w1"))
        self.assertEqual(self.count("w1"), 0)

    def test_concurrent_jobs_sum_per_worker(self):
        other = self.jm.submit({
            "name": "t2", "mapper": "wordcount_mapper", "reducer": "count_reducer",
            "num_map_tasks": 3, "num_reduce_tasks": 1, "input_rows": 50, "params": {},
        })
        self.dispatch(self.map_task(0), "w1", C.TASK_RUNNING)
        self.dispatch(self.map_task(1), "w1", C.TASK_RUNNING)
        self.jm.update_task(other.job_id, "m-0000", status=C.TASK_RUNNING, worker_id="w1")
        self.jm.update_task(other.job_id, "m-0001", status=C.TASK_ASSIGNED, worker_id="w2")
        self.assertEqual(self.count("w1"), 3)
        self.assertEqual(self.count("w2"), 1)
        # A terminal job contributes nothing.
        self.jm.update_task(other.job_id, "m-0001", status=C.TASK_SUCCEEDED)
        self.jm.set_job_status(self.jm.get_job(other.job_id), C.JOB_SUCCEEDED)
        self.assertEqual(self.count("w2"), 0)

    def test_cumulative_counters_still_tracked(self):
        """task_finished still owns the Done/Failed columns (and nothing else)."""
        self.registry.task_finished("w1", success=True)
        self.registry.task_finished("w1", success=True)
        self.registry.task_finished("w1", success=False)
        worker = self.registry.get("w1")
        self.assertEqual(worker.total_tasks_completed, 2)
        self.assertEqual(worker.total_tasks_failed, 1)
        self.assertEqual(self.count("w1"), 0)

    def test_fallback_without_provider(self):
        """A registry with no provider falls back to the stored field."""
        standalone = WorkerRegistry(Storage(self.tmp + "/standalone"), self.config)
        self.assertEqual(standalone.running_count("wx"), 0)
        self.assertEqual(standalone.running_count("ghost"), 0)


if __name__ == "__main__":
    unittest.main()
