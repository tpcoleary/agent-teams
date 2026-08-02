#!/usr/bin/env python3
"""Unit tests for the central task-tracking store (teams_server/tasks_db.py).

Covers, with no LLM and no Hermes required:
  1. CREATE / GET      — basic lifecycle, blank-title rejection
  2. PARENT REFERENCE   — optional flat parent link, missing-parent rejection,
                         no roll-up onto the parent
  3. LIST FILTERS       — team_id / assigned_to / status filters
  4. EDIT               — partial update of descriptive fields only
  5. REASSIGN           — assignee change
  6. PROGRESS           — clamping, pending->in_progress auto-flip
  7. SET STATUS         — enum validation, blocked_reason, completed_at,
                         reopening a terminal task
  8. DELETE             — children survive as top-level (ON DELETE SET NULL)

Run:  pytest tests/test_tasks_db.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teams_server.tasks_db import TasksDB  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    return TasksDB(tmp_path / "tasks.db")


# ---------------------------------------------------------------------------
# 1. Create / get
# ---------------------------------------------------------------------------
def test_create_and_get(db):
    task = db.create_task("Write docs", created_by="alice", assigned_to="bob",
                           description="the readme", team_id="t1", priority=2)
    assert task["title"] == "Write docs"
    assert task["status"] == "pending"
    assert task["progress"] == 0
    assert task["priority"] == 2
    assert task["assigned_to"] == "bob"
    assert task["created_by"] == "alice"
    assert task["parent_task_id"] is None

    fetched = db.get_task(task["id"])
    assert fetched == task


def test_create_blank_title_rejected(db):
    with pytest.raises(ValueError):
        db.create_task("   ", created_by="alice", assigned_to="bob")


def test_get_missing_task_returns_none(db):
    assert db.get_task("nonexistent") is None


def test_priority_clamped_to_range(db):
    lo = db.create_task("a", created_by="alice", assigned_to="bob", priority=-5)
    hi = db.create_task("b", created_by="alice", assigned_to="bob", priority=99)
    bad = db.create_task("c", created_by="alice", assigned_to="bob", priority="oops")
    assert lo["priority"] == 0
    assert hi["priority"] == 3
    assert bad["priority"] == 1


# ---------------------------------------------------------------------------
# 1b. Creation is idempotent on identical OPEN work
# ---------------------------------------------------------------------------
def test_duplicate_open_task_returns_existing_row(db):
    """The wake payload embeds the task id, so a duplicate insert would produce
    a byte-different payload and slip past the inbox's identical-pending dedup,
    waking the assignee twice for one piece of work."""
    first = db.create_task("Check deploy", created_by="alice", assigned_to="bob")
    second = db.create_task("Check deploy", created_by="alice", assigned_to="bob")
    assert second["id"] == first["id"]
    assert len(db.list_tasks()) == 1


def test_dedup_ignores_terminal_tasks(db):
    """Re-creating work after the previous one finished is genuinely new."""
    first = db.create_task("Check deploy", created_by="alice", assigned_to="bob")
    db.set_status(first["id"], "done")
    second = db.create_task("Check deploy", created_by="alice", assigned_to="bob")
    assert second["id"] != first["id"]
    assert len(db.list_tasks()) == 2


def test_dedup_is_scoped_to_creator(db):
    """created_by is part of a task's identity because completion reports back
    to it — collapsing two creators' tasks would leave one never hearing back."""
    a = db.create_task("Check deploy", created_by="alice", assigned_to="bob")
    c = db.create_task("Check deploy", created_by="carol", assigned_to="bob")
    assert a["id"] != c["id"]


def test_dedup_is_scoped_to_assignee_and_parent(db):
    a = db.create_task("Write tests", created_by="alice", assigned_to="bob")
    b = db.create_task("Write tests", created_by="alice", assigned_to="carol")
    assert a["id"] != b["id"]
    # same title+assignee but filed under a different parent is distinct work
    parent = db.create_task("Epic", created_by="alice", assigned_to="alice")
    nested = db.create_task("Write tests", created_by="alice", assigned_to="bob",
                             parent_task_id=parent["id"])
    assert nested["id"] != a["id"]


def test_dedup_matches_on_null_parent(db):
    """`IS` not `=`, so two parentless tasks actually compare equal."""
    a = db.create_task("t", created_by="alice", assigned_to="bob")
    b = db.create_task("t", created_by="alice", assigned_to="bob")
    assert a["id"] == b["id"]
    assert a["parent_task_id"] is None


def test_dedup_reuses_blocked_and_in_progress_tasks(db):
    """Only terminal states are exempt — an in-flight or blocked task is still
    the same open work."""
    for status in ("in_progress", "blocked"):
        d = TasksDB(db.db_path.parent / f"dedup_{status}.db")
        first = d.create_task("t", created_by="alice", assigned_to="bob")
        d.set_status(first["id"], status, blocked_reason="x")
        second = d.create_task("t", created_by="alice", assigned_to="bob")
        assert second["id"] == first["id"], status


# ---------------------------------------------------------------------------
# 2. Parent reference (flat, optional — not a subtask tree)
# ---------------------------------------------------------------------------
def test_create_task_with_parent_reference(db):
    parent = db.create_task("Ship feature", created_by="alice", assigned_to="alice")
    child = db.create_task("Write tests", created_by="alice", assigned_to="bob",
                            parent_task_id=parent["id"])
    assert child["parent_task_id"] == parent["id"]


def test_create_task_missing_parent_rejected(db):
    with pytest.raises(ValueError):
        db.create_task("orphan", created_by="alice", assigned_to="bob",
                        parent_task_id="does-not-exist")


def test_children_are_listed_like_any_other_task(db):
    """A parent reference is metadata, not containment: children show up in
    list_tasks alongside everything else."""
    parent = db.create_task("parent", created_by="alice", assigned_to="bob")
    db.create_task("child", created_by="alice", assigned_to="bob",
                    parent_task_id=parent["id"])
    assert len(db.list_tasks()) == 2


def test_parent_progress_is_not_affected_by_child(db):
    parent = db.create_task("parent", created_by="alice", assigned_to="bob")
    child = db.create_task("child", created_by="alice", assigned_to="bob",
                            parent_task_id=parent["id"])
    db.set_status(child["id"], "done")
    refreshed = db.get_task(parent["id"])
    assert refreshed["status"] == "pending"
    assert refreshed["progress"] == 0


# ---------------------------------------------------------------------------
# 3. List filters
# ---------------------------------------------------------------------------
def test_list_tasks_filters(db):
    db.create_task("t1", created_by="alice", assigned_to="bob", team_id="teamA")
    t2 = db.create_task("t2", created_by="alice", assigned_to="carol", team_id="teamA")
    db.create_task("t3", created_by="alice", assigned_to="bob", team_id="teamB")
    db.set_status(t2["id"], "done")

    assert len(db.list_tasks(team_id="teamA")) == 2
    assert len(db.list_tasks(assigned_to="bob")) == 2
    assert len(db.list_tasks(team_id="teamA", assigned_to="bob")) == 1
    assert len(db.list_tasks(status="done")) == 1
    assert len(db.list_tasks()) == 3


def test_list_tasks_respects_limit_and_order(db):
    for i in range(5):
        db.create_task(f"task-{i}", created_by="alice", assigned_to="bob")
    limited = db.list_tasks(limit=2)
    assert len(limited) == 2
    # newest first
    assert limited[0]["title"] == "task-4"


def test_list_tasks_filters_by_creator(db):
    """Powers the "you assigned" half of the live-context block."""
    db.create_task("from alice", created_by="alice", assigned_to="bob")
    db.create_task("from carol", created_by="carol", assigned_to="bob")
    mine = db.list_tasks(created_by="alice")
    assert [t["title"] for t in mine] == ["from alice"]


def test_list_tasks_open_only_excludes_terminal(db):
    """The live context must show what's outstanding, not a growing history."""
    pending = db.create_task("pending", created_by="alice", assigned_to="bob")
    prog = db.create_task("working", created_by="alice", assigned_to="bob")
    blocked = db.create_task("stuck", created_by="alice", assigned_to="bob")
    done = db.create_task("finished", created_by="alice", assigned_to="bob")
    failed = db.create_task("dead", created_by="alice", assigned_to="bob")
    db.update_progress(prog["id"], 40)
    db.set_status(blocked["id"], "blocked", blocked_reason="waiting")
    db.set_status(done["id"], "done")
    db.set_status(failed["id"], "failed")

    open_titles = {t["title"] for t in db.list_tasks(open_only=True)}
    assert open_titles == {"pending", "working", "stuck"}
    assert pending["id"] in {t["id"] for t in db.list_tasks(open_only=True)}
    # without the flag, everything is still visible
    assert len(db.list_tasks()) == 5


def test_open_only_composes_with_other_filters(db):
    db.create_task("a", created_by="alice", assigned_to="bob", team_id="t1")
    done = db.create_task("b", created_by="alice", assigned_to="bob", team_id="t1")
    db.create_task("c", created_by="alice", assigned_to="carol", team_id="t1")
    db.set_status(done["id"], "done")
    got = db.list_tasks(team_id="t1", assigned_to="bob", open_only=True)
    assert [t["title"] for t in got] == ["a"]


# ---------------------------------------------------------------------------
# 4. Edit
# ---------------------------------------------------------------------------
def test_edit_task_partial_update(db):
    task = db.create_task("Original", created_by="alice", assigned_to="bob",
                           description="d1", priority=1)
    updated = db.edit_task(task["id"], title="New title")
    assert updated["title"] == "New title"
    assert updated["description"] == "d1"  # untouched
    assert updated["priority"] == 1  # untouched


def test_edit_task_does_not_touch_status_progress_assignee(db):
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    db.update_progress(task["id"], 40)
    updated = db.edit_task(task["id"], description="new desc")
    assert updated["progress"] == 40
    assert updated["status"] == "in_progress"
    assert updated["assigned_to"] == "bob"


def test_edit_task_blank_title_rejected(db):
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    with pytest.raises(ValueError):
        db.edit_task(task["id"], title="   ")


def test_edit_missing_task_returns_none(db):
    assert db.edit_task("nonexistent", title="x") is None


def test_edit_task_no_fields_returns_existing_unchanged(db):
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    result = db.edit_task(task["id"])
    assert result == task


# ---------------------------------------------------------------------------
# 5. Reassign
# ---------------------------------------------------------------------------
def test_reassign_task(db):
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    updated = db.reassign_task(task["id"], "carol")
    assert updated["assigned_to"] == "carol"


def test_reassign_missing_task_returns_none(db):
    assert db.reassign_task("nonexistent", "carol") is None


# ---------------------------------------------------------------------------
# 6. Progress
# ---------------------------------------------------------------------------
def test_update_progress_clamps_range(db):
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    over = db.update_progress(task["id"], 500)
    assert over["progress"] == 100
    under = db.update_progress(task["id"], -20)
    assert under["progress"] == 0


def test_update_progress_flips_pending_to_in_progress(db):
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    assert task["status"] == "pending"
    updated = db.update_progress(task["id"], 10)
    assert updated["status"] == "in_progress"


def test_update_progress_does_not_reflip_other_statuses(db):
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    db.set_status(task["id"], "blocked", blocked_reason="waiting")
    updated = db.update_progress(task["id"], 50)
    assert updated["status"] == "blocked"


def test_update_progress_rejects_non_integer(db):
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    with pytest.raises(ValueError):
        db.update_progress(task["id"], "not-a-number")


def test_update_progress_missing_task_returns_none(db):
    assert db.update_progress("nonexistent", 50) is None


# ---------------------------------------------------------------------------
# 7. Set status
# ---------------------------------------------------------------------------
def test_set_status_invalid_rejected(db):
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    with pytest.raises(ValueError):
        db.set_status(task["id"], "not-a-real-status")


def test_set_status_done_sets_progress_and_completed_at(db):
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    updated = db.set_status(task["id"], "done")
    assert updated["status"] == "done"
    assert updated["progress"] == 100
    assert updated["completed_at"] is not None


def test_set_status_failed_sets_completed_at_not_progress(db):
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    updated = db.set_status(task["id"], "failed")
    assert updated["status"] == "failed"
    assert updated["completed_at"] is not None
    assert updated["progress"] == 0  # failed doesn't force 100


def test_set_status_blocked_sets_reason_and_clears_on_unblock(db):
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    blocked = db.set_status(task["id"], "blocked", blocked_reason="need creds")
    assert blocked["status"] == "blocked"
    assert blocked["blocked_reason"] == "need creds"

    resumed = db.set_status(task["id"], "in_progress")
    assert resumed["blocked_reason"] is None


def test_reopening_terminal_task_clears_completed_at(db):
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    db.set_status(task["id"], "done")
    reopened = db.set_status(task["id"], "pending")
    assert reopened["completed_at"] is None


def test_set_status_missing_task_returns_none(db):
    assert db.set_status("nonexistent", "done") is None


# ---------------------------------------------------------------------------
# 8. Delete (no cascade — children survive as top-level)
# ---------------------------------------------------------------------------
def test_delete_task_returns_deleted_row(db):
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    deleted = db.delete_task(task["id"])
    assert deleted["id"] == task["id"]
    assert db.get_task(task["id"]) is None


def test_delete_parent_keeps_children_as_top_level(db):
    """Deleting a parent must NOT destroy work assigned to someone else — the
    FK is ON DELETE SET NULL, so children survive and become top-level."""
    parent = db.create_task("parent", created_by="alice", assigned_to="bob")
    child = db.create_task("child", created_by="alice", assigned_to="carol",
                            parent_task_id=parent["id"])

    db.delete_task(parent["id"])
    survivor = db.get_task(child["id"])
    assert survivor is not None
    assert survivor["parent_task_id"] is None
    assert survivor["title"] == "child"


def test_delete_missing_task_returns_none(db):
    assert db.delete_task("nonexistent") is None


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
