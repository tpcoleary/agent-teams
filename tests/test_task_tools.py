#!/usr/bin/env python3
"""Unit tests for the agent-facing task tools (teams_server/task_tools.py).

Covers, with no LLM and no Hermes required:
  1. CREATE_TASK        — self/peer assignment, non-peer rejection, validation,
                         optional flat parent_task_id
  2. WAKE ON ASSIGN      — creating/reassigning a task delivers it into the
                         assignee's inbox so they actually start work
  3. REASSIGN_TASK       — creator/assignee may reassign, third parties may not,
                          new assignee must be self-or-peer
  4. EDIT_TASK           — creator/assignee may edit; never touches status/progress
  5. UPDATE_PROGRESS     — assignee-only self-report
  6. MARK_COMPLETE       — assignee-only
  7. MARK_BLOCKED        — assignee-only, reason required
  8. LIST_MY_TASKS       — scoped to caller, status filter

Uses a real TasksDB against tmp_path (cheap, exercises real SQL) and a fake
agents config for peer authorization, monkeypatching module-level singletons
so no server/Hermes wiring is needed.

Run:  pytest tests/test_task_tools.py -v
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import teams_server.task_tools as task_tools  # noqa: E402
from teams_server.tasks_db import TasksDB  # noqa: E402

CFG = {
    "agents": {
        "alice": {"team_id": "t1", "allowed_peers": ["bob"]},
        "bob": {"team_id": "t1", "allowed_peers": []},
        "carol": {"team_id": "t1", "allowed_peers": []},  # not linked to alice/bob
        "dave": {"team_id": "t2", "allowed_peers": []},   # different team entirely
    }
}


def _kwargs(caller):
    return {"task_id": f"agent_name:{caller}"}


@pytest.fixture(autouse=True)
def _wire(tmp_path, monkeypatch):
    """Point the module's singletons at a throwaway DB / no-op broadcast, and
    a fixed config, so handlers run without any server/Hermes context."""
    db = TasksDB(tmp_path / "tasks.db")
    monkeypatch.setattr(task_tools, "task_db", db)
    monkeypatch.setattr(task_tools, "_broadcast", lambda *a, **k: None)
    monkeypatch.setattr(task_tools.monitor_db, "log_event", lambda *a, **k: None)
    import teams_server.config as config_mod

    monkeypatch.setattr(config_mod, "load_agents_config", lambda: CFG)
    return db


def _ok(raw):
    body = json.loads(raw)
    assert body["success"] is True, body
    return body


def _fail(raw):
    body = json.loads(raw)
    assert body["success"] is False, body
    return body


# ---------------------------------------------------------------------------
# 1. create_task
# ---------------------------------------------------------------------------
def test_create_task_self_assign(_wire):
    out = _ok(task_tools.create_task_handler(
        {"title": "Write docs", "assigned_to": "alice"}, **_kwargs("alice")))
    assert out["task"]["assigned_to"] == "alice"
    assert out["task"]["created_by"] == "alice"


def test_create_task_assign_to_linked_peer(_wire):
    out = _ok(task_tools.create_task_handler(
        {"title": "Review PR", "assigned_to": "bob"}, **_kwargs("alice")))
    assert out["task"]["assigned_to"] == "bob"


def test_create_task_assign_to_non_peer_rejected(_wire):
    out = _fail(task_tools.create_task_handler(
        {"title": "Nope", "assigned_to": "carol"}, **_kwargs("alice")))
    assert "not a linked peer" in out["error"]


def test_create_task_assign_to_other_team_rejected(_wire):
    out = _fail(task_tools.create_task_handler(
        {"title": "Nope", "assigned_to": "dave"}, **_kwargs("alice")))
    assert "not a linked peer" in out["error"]


def test_create_task_blank_title_rejected(_wire):
    out = _fail(task_tools.create_task_handler(
        {"title": "   ", "assigned_to": "alice"}, **_kwargs("alice")))
    assert "title" in out["error"]


def test_create_task_missing_assignee_rejected(_wire):
    out = _fail(task_tools.create_task_handler(
        {"title": "x", "assigned_to": ""}, **_kwargs("alice")))
    assert "assigned_to" in out["error"]


def test_create_task_unknown_caller_rejected(_wire):
    out = _fail(task_tools.create_task_handler(
        {"title": "x", "assigned_to": "alice"}, task_id="agent_name:ghost"))
    assert "not found" in out["error"]


def test_create_task_carries_priority_and_description(_wire):
    out = _ok(task_tools.create_task_handler(
        {"title": "x", "assigned_to": "alice", "description": "d", "priority": 3},
        **_kwargs("alice")))
    assert out["task"]["priority"] == 3
    assert out["task"]["description"] == "d"


def test_create_task_with_parent_reference(_wire):
    parent = _ok(task_tools.create_task_handler(
        {"title": "Parent", "assigned_to": "alice"}, **_kwargs("alice")))["task"]
    out = _ok(task_tools.create_task_handler(
        {"title": "Child", "assigned_to": "bob", "parent_task_id": parent["id"]},
        **_kwargs("alice")))
    assert out["task"]["parent_task_id"] == parent["id"]


def test_create_task_missing_parent_rejected(_wire):
    out = _fail(task_tools.create_task_handler(
        {"title": "Child", "assigned_to": "alice", "parent_task_id": "nonexistent"},
        **_kwargs("alice")))
    assert "not found" in out["error"]


def test_create_task_without_parent_is_top_level(_wire):
    out = _ok(task_tools.create_task_handler(
        {"title": "x", "assigned_to": "alice"}, **_kwargs("alice")))
    assert out["task"]["parent_task_id"] is None


# ---------------------------------------------------------------------------
# 2. Wake on assign
# ---------------------------------------------------------------------------
class FakeDaemon:
    """Records what got delivered into an agent's inbox."""

    def __init__(self):
        self.ingested = []

    def ingest_task(self, from_agent, payload):
        self.ingested.append((from_agent, payload))
        return "queued-1"


def test_create_task_wakes_assignee(_wire, monkeypatch):
    """A task nobody is woken for is just a row in a table — assigning must
    deliver it into the assignee's inbox."""
    from teams_server.tools import _daemon_registry

    bob = FakeDaemon()
    monkeypatch.setitem(_daemon_registry, "bob", bob)

    out = _ok(task_tools.create_task_handler(
        {"title": "Review PR", "assigned_to": "bob", "description": "check the auth diff"},
        **_kwargs("alice")))
    assert out["assignee_woken"] is True
    assert len(bob.ingested) == 1
    from_agent, payload = bob.ingested[0]
    assert from_agent == "alice"
    assert "Review PR" in payload
    assert "check the auth diff" in payload
    # the payload must tell the woken agent how to close the loop
    assert "mark_task_complete" in payload
    assert out["task"]["id"][:8] in payload


def test_create_task_self_assign_also_wakes(_wire, monkeypatch):
    from teams_server.tools import _daemon_registry

    alice = FakeDaemon()
    monkeypatch.setitem(_daemon_registry, "alice", alice)
    out = _ok(task_tools.create_task_handler(
        {"title": "my own todo", "assigned_to": "alice"}, **_kwargs("alice")))
    assert out["assignee_woken"] is True
    assert len(alice.ingested) == 1


def test_create_task_records_even_when_assignee_has_no_daemon(_wire):
    """No running daemon must not lose the task — it's recorded, and the caller
    is told plainly that nothing will start yet."""
    out = _ok(task_tools.create_task_handler(
        {"title": "x", "assigned_to": "bob"}, **_kwargs("alice")))
    assert out["assignee_woken"] is False
    assert "no running daemon" in out["message"]
    assert out["task"]["id"]  # still persisted


def test_wake_failure_does_not_fail_the_task(_wire, monkeypatch):
    """A daemon that throws on ingest must not roll back or error the create —
    the task is already the durable record."""
    from teams_server.tools import _daemon_registry

    class Boom:
        def ingest_task(self, from_agent, payload):
            raise RuntimeError("inbox exploded")

    monkeypatch.setitem(_daemon_registry, "bob", Boom())
    out = _ok(task_tools.create_task_handler(
        {"title": "x", "assigned_to": "bob"}, **_kwargs("alice")))
    assert out["assignee_woken"] is False
    assert out["task"]["title"] == "x"


def test_reassign_wakes_new_assignee(_wire, monkeypatch):
    from teams_server.tools import _daemon_registry

    bob = FakeDaemon()
    monkeypatch.setitem(_daemon_registry, "bob", bob)
    task = _ok(task_tools.create_task_handler(
        {"title": "t", "assigned_to": "alice"}, **_kwargs("alice")))["task"]
    bob.ingested.clear()

    out = _ok(task_tools.reassign_task_handler(
        {"task_id": task["id"], "assigned_to": "bob"}, **_kwargs("alice")))
    assert out["assignee_woken"] is True
    assert len(bob.ingested) == 1
    assert "t" in bob.ingested[0][1]


# ---------------------------------------------------------------------------
# 3. reassign_task
# ---------------------------------------------------------------------------
def test_reassign_by_creator(_wire):
    task = _ok(task_tools.create_task_handler(
        {"title": "t", "assigned_to": "bob"}, **_kwargs("alice")))["task"]
    out = _ok(task_tools.reassign_task_handler(
        {"task_id": task["id"], "assigned_to": "alice"}, **_kwargs("alice")))
    assert out["task"]["assigned_to"] == "alice"


def test_reassign_by_current_assignee(_wire):
    task = _ok(task_tools.create_task_handler(
        {"title": "t", "assigned_to": "bob"}, **_kwargs("alice")))["task"]
    out = _ok(task_tools.reassign_task_handler(
        {"task_id": task["id"], "assigned_to": "bob"}, **_kwargs("bob")))
    # bob reassigning to himself is a no-op but must succeed (assignee is authorized)
    assert out["task"]["assigned_to"] == "bob"


def test_reassign_by_third_party_rejected(_wire):
    task = _ok(task_tools.create_task_handler(
        {"title": "t", "assigned_to": "bob"}, **_kwargs("alice")))["task"]
    out = _fail(task_tools.reassign_task_handler(
        {"task_id": task["id"], "assigned_to": "alice"}, **_kwargs("carol")))
    assert "creator or assignee" in out["error"]


def test_reassign_new_assignee_must_be_peer(_wire):
    task = _ok(task_tools.create_task_handler(
        {"title": "t", "assigned_to": "bob"}, **_kwargs("alice")))["task"]
    out = _fail(task_tools.reassign_task_handler(
        {"task_id": task["id"], "assigned_to": "carol"}, **_kwargs("alice")))
    assert "not a linked peer" in out["error"]


def test_reassign_missing_task_rejected(_wire):
    out = _fail(task_tools.reassign_task_handler(
        {"task_id": "nonexistent", "assigned_to": "alice"}, **_kwargs("alice")))
    assert "not found" in out["error"]


# ---------------------------------------------------------------------------
# 4. edit_task
# ---------------------------------------------------------------------------
def test_edit_by_creator(_wire):
    task = _ok(task_tools.create_task_handler(
        {"title": "old", "assigned_to": "bob"}, **_kwargs("alice")))["task"]
    out = _ok(task_tools.edit_task_handler(
        {"task_id": task["id"], "title": "new"}, **_kwargs("alice")))
    assert out["task"]["title"] == "new"


def test_edit_by_assignee(_wire):
    task = _ok(task_tools.create_task_handler(
        {"title": "old", "assigned_to": "bob"}, **_kwargs("alice")))["task"]
    out = _ok(task_tools.edit_task_handler(
        {"task_id": task["id"], "priority": 3}, **_kwargs("bob")))
    assert out["task"]["priority"] == 3


def test_edit_by_third_party_rejected(_wire):
    task = _ok(task_tools.create_task_handler(
        {"title": "old", "assigned_to": "bob"}, **_kwargs("alice")))["task"]
    out = _fail(task_tools.edit_task_handler(
        {"task_id": task["id"], "title": "new"}, **_kwargs("carol")))
    assert "creator or assignee" in out["error"]


def test_edit_no_fields_rejected(_wire):
    task = _ok(task_tools.create_task_handler(
        {"title": "t", "assigned_to": "alice"}, **_kwargs("alice")))["task"]
    out = _fail(task_tools.edit_task_handler({"task_id": task["id"]}, **_kwargs("alice")))
    assert "at least one" in out["error"]


def test_edit_never_touches_status_or_assignee(_wire):
    task = _ok(task_tools.create_task_handler(
        {"title": "t", "assigned_to": "bob"}, **_kwargs("alice")))["task"]
    task_tools.update_task_progress_handler(
        {"task_id": task["id"], "progress": 30}, **_kwargs("bob"))
    out = _ok(task_tools.edit_task_handler(
        {"task_id": task["id"], "description": "d"}, **_kwargs("alice")))
    assert out["task"]["status"] == "in_progress"
    assert out["task"]["assigned_to"] == "bob"


# ---------------------------------------------------------------------------
# 5. update_task_progress
# ---------------------------------------------------------------------------
def test_progress_by_assignee(_wire):
    task = _ok(task_tools.create_task_handler(
        {"title": "t", "assigned_to": "bob"}, **_kwargs("alice")))["task"]
    out = _ok(task_tools.update_task_progress_handler(
        {"task_id": task["id"], "progress": 55}, **_kwargs("bob")))
    assert out["task"]["progress"] == 55
    assert out["task"]["status"] == "in_progress"


def test_progress_by_non_assignee_rejected(_wire):
    task = _ok(task_tools.create_task_handler(
        {"title": "t", "assigned_to": "bob"}, **_kwargs("alice")))["task"]
    out = _fail(task_tools.update_task_progress_handler(
        {"task_id": task["id"], "progress": 55}, **_kwargs("alice")))
    assert "assignee" in out["error"]


def test_progress_missing_task_rejected(_wire):
    out = _fail(task_tools.update_task_progress_handler(
        {"task_id": "nonexistent", "progress": 55}, **_kwargs("alice")))
    assert "not found" in out["error"]


# ---------------------------------------------------------------------------
# 6. mark_task_complete
# ---------------------------------------------------------------------------
def test_complete_by_assignee(_wire):
    task = _ok(task_tools.create_task_handler(
        {"title": "t", "assigned_to": "bob"}, **_kwargs("alice")))["task"]
    out = _ok(task_tools.mark_task_complete_handler(
        {"task_id": task["id"]}, **_kwargs("bob")))
    assert out["task"]["status"] == "done"
    assert out["task"]["progress"] == 100
    assert "message" not in out


def test_complete_by_non_assignee_rejected(_wire):
    task = _ok(task_tools.create_task_handler(
        {"title": "t", "assigned_to": "bob"}, **_kwargs("alice")))["task"]
    out = _fail(task_tools.mark_task_complete_handler(
        {"task_id": task["id"]}, **_kwargs("alice")))
    assert "assignee" in out["error"]


# ---------------------------------------------------------------------------
# 7. mark_task_blocked
# ---------------------------------------------------------------------------
def test_blocked_by_assignee_with_reason(_wire):
    task = _ok(task_tools.create_task_handler(
        {"title": "t", "assigned_to": "bob"}, **_kwargs("alice")))["task"]
    out = _ok(task_tools.mark_task_blocked_handler(
        {"task_id": task["id"], "reason": "need creds"}, **_kwargs("bob")))
    assert out["task"]["status"] == "blocked"
    assert out["task"]["blocked_reason"] == "need creds"


def test_blocked_requires_reason(_wire):
    task = _ok(task_tools.create_task_handler(
        {"title": "t", "assigned_to": "bob"}, **_kwargs("alice")))["task"]
    out = _fail(task_tools.mark_task_blocked_handler(
        {"task_id": task["id"], "reason": ""}, **_kwargs("bob")))
    assert "reason" in out["error"]


def test_blocked_by_non_assignee_rejected(_wire):
    task = _ok(task_tools.create_task_handler(
        {"title": "t", "assigned_to": "bob"}, **_kwargs("alice")))["task"]
    out = _fail(task_tools.mark_task_blocked_handler(
        {"task_id": task["id"], "reason": "x"}, **_kwargs("alice")))
    assert "assignee" in out["error"]


# ---------------------------------------------------------------------------
# 8. list_my_tasks
# ---------------------------------------------------------------------------
def test_list_my_tasks_scoped_to_caller(_wire):
    task_tools.create_task_handler({"title": "for bob", "assigned_to": "bob"}, **_kwargs("alice"))
    task_tools.create_task_handler({"title": "for alice", "assigned_to": "alice"}, **_kwargs("alice"))
    out = _ok(task_tools.list_my_tasks_handler({}, **_kwargs("bob")))
    assert len(out["tasks"]) == 1
    assert out["tasks"][0]["title"] == "for bob"


def test_list_my_tasks_status_filter(_wire):
    t1 = task_tools.create_task_handler({"title": "a", "assigned_to": "bob"}, **_kwargs("alice"))
    task_tools.create_task_handler({"title": "b", "assigned_to": "bob"}, **_kwargs("alice"))
    t1_id = json.loads(t1)["task"]["id"]
    task_tools.mark_task_complete_handler({"task_id": t1_id}, **_kwargs("bob"))

    out = _ok(task_tools.list_my_tasks_handler({"status": "done"}, **_kwargs("bob")))
    assert len(out["tasks"]) == 1
    assert out["tasks"][0]["title"] == "a"


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
