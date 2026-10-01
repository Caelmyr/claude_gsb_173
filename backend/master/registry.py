"""Worker registry: registration, heartbeat bookkeeping and death reaping.

The Master keeps one JSON record per worker under ``registry/workers/`` and
mirrors it in memory.  A background sweep (driven by the scheduler tick) marks
any worker whose heartbeat is older than ``heartbeat_timeout_sec`` as dead and
invokes the fault-tolerance callback so its in-flight tasks are reassigned —
this is the "distributed coordination and fault tolerance" backbone.

Running-task accounting
-----------------------
``running_tasks`` has exactly one writer per source and is *set-based*, never a
delta counter:

* the worker is the ground truth — every heartbeat / task-status / task-complete
  message carries the full set ``running_task_ids`` plus a monotonic
  ``state_seq``; ``apply_task_snapshot`` replaces the worker's set when the
  snapshot is not older than the one already applied;
* between dispatch and the worker's first report the Master records the task in
  an in-memory ``_launching`` set (``note_dispatched`` / ``note_observed``),
  closing the "dispatched but not yet reported" window.

The displayed count is always ``len(observed | launching)``: a set length, so
it can never be negative, and duplicate / out-of-order / retried reports are
inherently idempotent.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from backend.common import constants as C
from backend.common.jsonutil import now_ms
from backend.common.models import WorkerRecord, new_worker
from backend.common.storage import Storage, list_files, read_json

# A dispatched task must show up in one of the worker's snapshots (heartbeat or
# task-status) well within this many seconds; otherwise the dispatch was lost
# (worker crashed while accepting, etc.) and the optimistic entry is pruned.
_LAUNCHING_TTL_SEC = 60.0


class WorkerRegistry:
    def __init__(
        self,
        storage: Storage,
        config,
        on_death: Optional[Callable[[WorkerRecord], None]] = None,
    ) -> None:
        self.storage = storage
        self.config = config
        self.on_death = on_death
        self._workers: dict[str, WorkerRecord] = {}
        # worker_id -> {task_id: dispatched_epoch_seconds}; Master-side
        # optimistic view only (not persisted); a worker snapshot confirms or
        # the TTL sweep evicts each entry.
        self._launching: dict[str, dict[str, float]] = {}
        self._lock = threading.RLock()
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        root = self.storage.path("registry", "workers")
        for path in list_files(root, suffix=".json"):
            doc = read_json(path)
            if doc:
                w = WorkerRecord.from_dict(doc)
                self._workers[w.worker_id] = w

    def _save(self, worker: WorkerRecord) -> None:
        self.storage.write(worker.to_dict(), "registry", "workers", f"{worker.worker_id}.json")

    # ------------------------------------------------------------------
    def register(self, payload: dict) -> WorkerRecord:
        wid = payload["worker_id"]
        with self._lock:
            existing = self._workers.get(wid)
            if existing:
                existing.name = payload.get("name", existing.name)
                existing.host = payload.get("host", existing.host)
                existing.port = int(payload.get("port", existing.port))
                existing.status = C.WORKER_ALIVE
                existing.last_heartbeat_ms = now_ms()
                existing.cpu_cores = int(payload.get("cpu_cores", existing.cpu_cores))
                existing.mem_total_mb = int(payload.get("mem_total_mb", existing.mem_total_mb))
                existing.exec_mode = payload.get("exec_mode", existing.exec_mode)
                # A (re)registration is a fresh process: it cannot be running any
                # tasks yet, so discard every stale counter/snapshot/launching.
                self._reset_running(existing)
                worker = existing
            else:
                worker = new_worker(
                    wid,
                    payload.get("name", wid),
                    payload.get("host", "127.0.0.1"),
                    int(payload.get("port", 0)),
                    int(payload.get("cpu_cores", 0)),
                    int(payload.get("mem_total_mb", 0)),
                    payload.get("exec_mode", "process"),
                )
                self._workers[wid] = worker
            self._save(worker)
            return worker

    def heartbeat(self, payload: dict) -> Optional[WorkerRecord]:
        wid = payload.get("worker_id")
        with self._lock:
            worker = self._workers.get(wid)
            if worker is None:
                return None
            worker.status = C.WORKER_ALIVE
            worker.last_heartbeat_ms = now_ms()
            worker.cpu_percent = float(payload.get("cpu_percent", worker.cpu_percent))
            worker.mem_percent = float(payload.get("mem_percent", worker.mem_percent))
            worker.load1 = float(payload.get("load1", worker.load1))
            worker.queued_tasks = int(payload.get("queued_tasks", worker.queued_tasks))
            # Running set: updated workers always carry the authoritative
            # ``running_task_ids`` snapshot.  A legacy worker (ids absent) only
            # contributes resource data here; its count is reconciled via any
            # task-status/complete report and the master-side launching set.
            ids = payload.get("running_task_ids")
            if isinstance(ids, list):
                self._apply_snapshot(worker, [str(t) for t in ids],
                                     payload.get("state_seq"))
            self._recompute(worker)
            self._save(worker)
            return worker

    # ------------------------------------------------------------------
    # Running-task accounting (set-based, single source of truth)
    # ------------------------------------------------------------------
    def _reset_running(self, worker: WorkerRecord) -> None:
        worker.running_tasks = 0
        worker.running_task_ids = []
        worker.state_seq = 0
        self._launching.pop(worker.worker_id, None)

    def _apply_snapshot(self, worker: WorkerRecord, task_ids: list[str],
                        seq: Optional[object]) -> None:
        """Replace the worker's observed running set with a fresh snapshot.

        Out-of-order snapshots (older ``state_seq``) are ignored so a delayed
        heartbeat can never roll back state just learned from task-complete.
        Snapshots without a usable seq are accepted (they are still complete
        sets, just not orderable).
        """
        if seq is not None:
            try:
                seq_int = int(seq)
            except (TypeError, ValueError):
                seq_int = None
            if seq_int is not None and seq_int < worker.state_seq:
                return
            if seq_int is not None:
                worker.state_seq = seq_int
        worker.running_task_ids = list(task_ids)
        # The worker has now spoken authoritatively about every task it is
        # running, so those launches are observed.
        launching = self._launching.get(worker.worker_id)
        if launching:
            for tid in task_ids:
                launching.pop(tid, None)

    def _recompute(self, worker: WorkerRecord) -> None:
        """running_tasks = observed set ∪ pending (dispatched, unobserved)."""
        observed = set(worker.running_task_ids)
        launching = self._launching.get(worker.worker_id, {})
        worker.running_tasks = len(observed | launching.keys())

    def note_dispatched(self, worker_id: str, task_id: str) -> None:
        """Optimistically count a task the instant dispatch is accepted."""
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return
            self._launching.setdefault(worker_id, {})[task_id] = time.time()
            self._recompute(worker)
            self._save(worker)

    def note_observed(self, worker_id: str, task_id: str) -> None:
        """A task-status report confirms the worker is running the task."""
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return
            launching = self._launching.get(worker_id)
            if launching:
                launching.pop(task_id, None)
            self._recompute(worker)
            self._save(worker)

    def apply_task_report(self, payload: dict) -> Optional[WorkerRecord]:
        """Apply the running-set snapshot carried by status/complete reports."""
        wid = payload.get("worker_id")
        if not wid:
            return None
        with self._lock:
            worker = self._workers.get(wid)
            if worker is None:
                return None
            ids = payload.get("running_task_ids")
            if isinstance(ids, list):
                self._apply_snapshot(worker, [str(t) for t in ids],
                                     payload.get("state_seq"))
            tid = payload.get("task_id")
            if tid:
                # Once a terminal report arrives the task is definitely no
                # longer merely "launching" on this worker — whether the
                # snapshot carried ids or not (e.g. an ultra-fast task that
                # finished before sending any progress report).
                launching = self._launching.get(wid)
                if launching:
                    launching.pop(tid, None)
            self._recompute(worker)
            self._save(worker)
            return worker

    def prune_launching(self, ttl_sec: float = _LAUNCHING_TTL_SEC) -> None:
        """Drop dispatched entries the worker never confirmed (lost dispatch)."""
        deadline = time.time() - ttl_sec
        with self._lock:
            changed = []
            for wid, launching in list(self._launching.items()):
                stale = [tid for tid, ts in launching.items() if ts < deadline]
                for tid in stale:
                    launching.pop(tid, None)
                if stale:
                    worker = self._workers.get(wid)
                    if worker is not None:
                        self._recompute(worker)
                        changed.append(worker)
                if not launching:
                    self._launching.pop(wid, None)
            for worker in changed:
                self._save(worker)

    def record_task_outcome(self, worker_id: str, success: bool) -> None:
        """Tally a finished task (independent of the running-set accounting)."""
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return
            if success:
                worker.total_tasks_completed += 1
            else:
                worker.total_tasks_failed += 1
            self._save(worker)

    # ------------------------------------------------------------------
    def get(self, worker_id: str) -> Optional[WorkerRecord]:
        with self._lock:
            return self._workers.get(worker_id)

    def all(self) -> list[WorkerRecord]:
        with self._lock:
            return list(self._workers.values())

    def alive(self) -> list[WorkerRecord]:
        return [w for w in self.all() if w.is_alive]

    def least_loaded(self) -> list[WorkerRecord]:
        """Alive workers sorted by ascending load score."""
        return sorted(self.alive(), key=lambda w: w.load_score)

    def reap(self) -> list[WorkerRecord]:
        """Mark timed-out workers dead and return the newly-dead list."""
        timeout_ms = int(self.config.heartbeat_timeout_sec * 1000)
        now = now_ms()
        newly_dead: list[WorkerRecord] = []
        with self._lock:
            for worker in self._workers.values():
                if worker.is_alive and (now - worker.last_heartbeat_ms) > timeout_ms:
                    worker.status = C.WORKER_DEAD
                    # A dead worker is running nothing; its tasks are reassigned
                    # by the fault-tolerance callback, so drop stale bookkeeping.
                    self._reset_running(worker)
                    self._save(worker)
                    newly_dead.append(worker)
        for worker in newly_dead:
            if self.on_death is not None:
                self.on_death(worker)
        return newly_dead

    def summary(self) -> dict:
        workers = self.all()
        alive = [w for w in workers if w.is_alive]
        return {
            "total": len(workers),
            "alive": len(alive),
            "dead": len(workers) - len(alive),
            "workers": [w.to_dict() for w in sorted(workers, key=lambda w: w.worker_id)],
        }
