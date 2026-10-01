"""Worker registry: registration, heartbeat bookkeeping and death reaping.

The Master keeps one JSON record per worker under ``registry/workers/`` and
mirrors it in memory.  A background sweep (driven by the scheduler tick) marks
any worker whose heartbeat is older than ``heartbeat_timeout_sec`` as dead and
invokes the fault-tolerance callback so its in-flight tasks are reassigned —
this is the "distributed coordination and fault tolerance" backbone.

The per-worker *running tasks* number shown in the UI is **not** a stored
counter: it is derived from the Master's task table through
``set_running_tasks_provider`` so it can never drift, go stale or negative.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from backend.common import constants as C
from backend.common.jsonutil import now_ms
from backend.common.models import WorkerRecord, new_worker
from backend.common.storage import Storage, list_files, read_json


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
        self._lock = threading.RLock()
        # Derives the live running-task count per worker from the task table.
        self._running_tasks_provider: Optional[Callable[[str], int]] = None
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
            worker.load1 = float(payload.get("cpu_percent", worker.load1))
            # NOTE: the payload's ``running_tasks`` is deliberately ignored —
            # the authoritative count is derived from the Master's task table
            # (see ``running_count``); trusting a sampled, in-flight value here
            # is what made the nodes page drift from reality.
            worker.queued_tasks = int(payload.get("queued_tasks", worker.queued_tasks))
            self._save(worker)
            return worker

    def task_finished(self, worker_id: str, success: bool) -> None:
        """Bookkeep a completed attempt (cumulative counters only).

        ``running_tasks`` is intentionally *not* decremented here: it is a
        derived value, and pairing increments/decrements across asynchronous
        heartbeat and completion paths is what produced negative/stale counts.
        """
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
    # Derived running-task count
    # ------------------------------------------------------------------
    def set_running_tasks_provider(self, provider: Callable[[str], int]) -> None:
        """Wire the function that counts a worker's live tasks from the task table."""
        self._running_tasks_provider = provider

    def running_count(self, worker_id: str) -> int:
        """Live number of tasks assigned to / running on ``worker_id``.

        Derived from the authoritative task table when a provider is wired
        (always, in the Master), so it agrees with the monitor page at every
        instant and can never go negative.  Falls back to the stored field
        when no provider is set (e.g. a standalone registry in tests).
        """
        provider = self._running_tasks_provider
        if provider is not None:
            try:
                return max(0, int(provider(worker_id)))
            except Exception:  # noqa: BLE001 - never break the nodes page
                pass
        worker = self.get(worker_id)
        return worker.running_tasks if worker else 0

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
        timeout_ms = int(self.config.heartbeat_timeout_sec * 1000 * 60)
        now = now_ms()
        newly_dead: list[WorkerRecord] = []
        with self._lock:
            for worker in self._workers.values():
                if worker.is_alive and (now - worker.last_heartbeat_ms) > timeout_ms:
                    worker.status = C.WORKER_DEAD
                    self._save(worker)
                    newly_dead.append(worker)
        for worker in newly_dead:
            if self.on_death is not None:
                self.on_death(worker)
        return newly_dead

    def summary(self) -> dict:
        workers = self.all()
        alive = [w for w in workers if w.is_alive]
        views: list[dict] = []
        for w in sorted(workers, key=lambda w: w.worker_id):
            view = w.to_dict()
            view["running_tasks"] = self.running_count(w.worker_id)
            views.append(view)
        return {
            "total": len(workers),
            "alive": len(alive),
            "dead": len(workers) - len(alive),
            "workers": views,
        }
