"""Central task-tracking store — human-meaningful work items, not the peer
message inbox (see teams_server/inbox.py for that).

Central (not per-agent) so the dashboard can query across every agent's work
in one place. Progress and status are always explicit — a subtask completing
NEVER auto-flips or auto-advances its parent's progress/status; that would
produce surprising jumps whenever subtasks are added or removed later.
"""

import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("teams.tasks")

STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_BLOCKED = "blocked"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
VALID_STATUSES = (STATUS_PENDING, STATUS_IN_PROGRESS, STATUS_BLOCKED, STATUS_DONE, STATUS_FAILED)
TERMINAL_STATUSES = (STATUS_DONE, STATUS_FAILED)

MIN_PRIORITY, MAX_PRIORITY = 0, 3


class TasksDB:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS tasks (
        id              TEXT PRIMARY KEY,
        team_id         TEXT,
        parent_task_id  TEXT REFERENCES tasks(id) ON DELETE CASCADE,
        title           TEXT NOT NULL,
        description     TEXT,
        status          TEXT NOT NULL DEFAULT 'pending',
        priority        INTEGER NOT NULL DEFAULT 1,
        progress        INTEGER NOT NULL DEFAULT 0,
        assigned_to     TEXT,
        created_by      TEXT NOT NULL,
        created_at      REAL NOT NULL,
        updated_at      REAL NOT NULL,
        completed_at    REAL,
        blocked_reason  TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_tasks_assigned ON tasks(assigned_to, status);
    CREATE INDEX IF NOT EXISTS idx_tasks_team     ON tasks(team_id, status);
    CREATE INDEX IF NOT EXISTS idx_tasks_parent   ON tasks(parent_task_id);
    CREATE INDEX IF NOT EXISTS idx_tasks_created  ON tasks(created_at DESC);
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA synchronous=NORMAL")
        # SQLite defaults foreign key enforcement OFF per connection; without
        # this, "ON DELETE CASCADE" above silently does nothing and deleting a
        # parent task would orphan its subtasks instead of removing them.
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(self.SCHEMA)
            conn.commit()

    def _row(self, conn, task_id: str) -> Optional[dict]:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _clamp_priority(priority: Any) -> int:
        try:
            p = int(priority)
        except (TypeError, ValueError):
            p = 1
        return max(MIN_PRIORITY, min(MAX_PRIORITY, p))

    def create_task(
        self,
        title: str,
        created_by: str,
        assigned_to: Optional[str] = None,
        description: str = "",
        team_id: Optional[str] = None,
        priority: int = 1,
        parent_task_id: Optional[str] = None,
    ) -> dict:
        title = (title or "").strip()
        if not title:
            raise ValueError("title is required")
        task_id = str(uuid.uuid4())
        now = time.time()
        with self._lock, self._conn() as conn:
            if parent_task_id is not None and self._row(conn, parent_task_id) is None:
                raise ValueError(f"parent_task_id '{parent_task_id}' does not exist")
            conn.execute(
                "INSERT INTO tasks (id, team_id, parent_task_id, title, description, "
                "status, priority, progress, assigned_to, created_by, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (task_id, team_id, parent_task_id, title, description or "",
                 STATUS_PENDING, self._clamp_priority(priority), 0,
                 assigned_to, created_by, now, now),
            )
            conn.commit()
            row = self._row(conn, task_id)
        log.info("[TasksDB] Created task %s '%s' (assigned_to=%s)", task_id[:8], title, assigned_to)
        return row

    def get_task(self, task_id: str) -> Optional[dict]:
        with self._conn() as conn:
            return self._row(conn, task_id)

    def get_subtasks(self, parent_task_id: str) -> List[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM tasks WHERE parent_task_id=? ORDER BY created_at ASC",
                (parent_task_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_tasks(
        self,
        team_id: Optional[str] = None,
        assigned_to: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 200,
    ) -> List[dict]:
        sql = "SELECT * FROM tasks WHERE 1=1"
        params: list = []
        if team_id:
            sql += " AND team_id = ?"
            params.append(team_id)
        if assigned_to:
            sql += " AND assigned_to = ?"
            params.append(assigned_to)
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, tuple(params)).fetchall()
            return [dict(r) for r in rows]

    def edit_task(
        self,
        task_id: str,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        priority: Optional[int] = None,
    ) -> Optional[dict]:
        """Partial update of descriptive fields only — never status/progress/assignee."""
        with self._lock, self._conn() as conn:
            existing = self._row(conn, task_id)
            if existing is None:
                return None
            sets = []
            params: list = []
            if title is not None:
                title = title.strip()
                if not title:
                    raise ValueError("title cannot be blank")
                sets.append("title=?")
                params.append(title)
            if description is not None:
                sets.append("description=?")
                params.append(description)
            if priority is not None:
                sets.append("priority=?")
                params.append(self._clamp_priority(priority))
            if not sets:
                return existing
            sets.append("updated_at=?")
            params.append(time.time())
            params.append(task_id)
            conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id=?", params)
            conn.commit()
            return self._row(conn, task_id)

    def reassign_task(self, task_id: str, new_assignee: str) -> Optional[dict]:
        with self._lock, self._conn() as conn:
            if self._row(conn, task_id) is None:
                return None
            conn.execute(
                "UPDATE tasks SET assigned_to=?, updated_at=? WHERE id=?",
                (new_assignee, time.time(), task_id),
            )
            conn.commit()
            return self._row(conn, task_id)

    def update_progress(self, task_id: str, progress: int) -> Optional[dict]:
        """Clamp to 0-100. A pending task moving off 0% implicitly starts."""
        with self._lock, self._conn() as conn:
            existing = self._row(conn, task_id)
            if existing is None:
                return None
            try:
                p = int(progress)
            except (TypeError, ValueError):
                raise ValueError("progress must be an integer")
            p = max(0, min(100, p))
            now = time.time()
            conn.execute(
                "UPDATE tasks SET progress=?, updated_at=?, "
                "status = CASE WHEN status = ? THEN ? ELSE status END "
                "WHERE id=?",
                (p, now, STATUS_PENDING, STATUS_IN_PROGRESS, task_id),
            )
            conn.commit()
            return self._row(conn, task_id)

    def set_status(
        self, task_id: str, status: str, *, blocked_reason: Optional[str] = None,
    ) -> Optional[dict]:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status '{status}' — must be one of {VALID_STATUSES}")
        with self._lock, self._conn() as conn:
            existing = self._row(conn, task_id)
            if existing is None:
                return None
            now = time.time()
            sets = ["status=?", "updated_at=?"]
            params: list = [status, now]
            if status == STATUS_BLOCKED:
                sets.append("blocked_reason=?")
                params.append(blocked_reason or "")
            else:
                sets.append("blocked_reason=NULL")
            if status in TERMINAL_STATUSES:
                sets.append("completed_at=?")
                params.append(now)
                if status == STATUS_DONE:
                    sets.append("progress=100")
            else:
                # Re-opening a terminal task clears the old completion stamp.
                sets.append("completed_at=NULL")
            params.append(task_id)
            conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id=?", params)
            conn.commit()
            return self._row(conn, task_id)

    def delete_task(self, task_id: str) -> Optional[List[str]]:
        """Delete a task and cascade to its subtasks. Returns the list of
        deleted subtask ids (empty list if none), or None if task_id didn't exist."""
        with self._lock, self._conn() as conn:
            if self._row(conn, task_id) is None:
                return None
            conn.row_factory = sqlite3.Row
            subtasks = conn.execute(
                "SELECT id FROM tasks WHERE parent_task_id=?", (task_id,)
            ).fetchall()
            sub_ids = [r["id"] for r in subtasks]
            conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))
            conn.commit()
            return sub_ids


# Global singleton instance
from teams_server.config import TASKS_DB  # noqa: E402

task_db = TasksDB(TASKS_DB)
