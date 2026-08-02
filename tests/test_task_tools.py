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
import types
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
        # A 3-level chain for the report-upward tests, mutually linked so
        # assignment is allowed at every level: lead -> mid -> worker.
        "lead": {"team_id": "t1", "allowed_peers": ["mid", "worker"]},
        "mid": {"team_id": "t1", "allowed_peers": ["worker"]},
        "worker": {"team_id": "t1", "allowed_peers": []},
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


def test_duplicate_create_reports_reuse(_wire):
    first = _ok(task_tools.create_task_handler(
        {"title": "Check deploy", "assigned_to": "bob"}, **_kwargs("alice")))
    second = _ok(task_tools.create_task_handler(
        {"title": "Check deploy", "assigned_to": "bob"}, **_kwargs("alice")))
    assert "already_existed" not in first
    assert second["already_existed"] is True
    assert second["task"]["id"] == first["task"]["id"]
    assert "already open" in second["message"]


def test_duplicate_create_produces_an_identical_wake_payload(_wire, monkeypatch):
    """The two dedup layers have to compose. create_task returning the SAME row
    means the wake payload is byte-identical, which is exactly what the inbox's
    identical-pending dedup keys on — so the assignee is woken once, not twice.

    (A fresh uuid per call would defeat that, which is the bug the InboxQueue
    dedup comment documents: 'a random id was embedded in the header, so two
    identical TASK sends produced different payloads and slipped past the
    queue's byte-identical pending-dedup — waking the recipient twice'.)
    """
    from teams_server.tools import _daemon_registry

    bob = FakeDaemon()
    monkeypatch.setitem(_daemon_registry, "bob", bob)
    for _ in range(2):
        task_tools.create_task_handler(
            {"title": "Check deploy", "assigned_to": "bob", "description": "prod"},
            **_kwargs("alice"))

    assert len(bob.ingested) == 2, "handler wakes on both calls"
    assert bob.ingested[0] == bob.ingested[1], (
        "payloads must be byte-identical for the inbox dedup to absorb the second")


def test_real_inbox_absorbs_the_duplicate_wake(_wire, monkeypatch, tmp_path):
    """End-to-end of the composition, against the real InboxQueue rather than a
    fake: two identical create_task calls must cost the assignee ONE turn."""
    from teams_server.inbox import InboxQueue
    from teams_server.tools import _daemon_registry

    inbox = InboxQueue(tmp_path / "bob_inbox.db")
    monkeypatch.setitem(
        _daemon_registry, "bob",
        types.SimpleNamespace(ingest_task=lambda from_agent, payload:
                              inbox.enqueue(from_agent, payload)))

    for _ in range(2):
        task_tools.create_task_handler(
            {"title": "Check deploy", "assigned_to": "bob"}, **_kwargs("alice"))

    assert inbox.get_pending_count() == 1


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
# 7b. Setting a terminal status IS reporting it
# ---------------------------------------------------------------------------
def test_complete_reports_to_creator(_wire, monkeypatch):
    """Completing delegated work must deliver the report itself. Previously an
    agent could mark a task done and still leave its delegator waiting, because
    reporting was a separate call it had to remember."""
    from teams_server.tools import _daemon_registry

    alice = FakeDaemon()
    monkeypatch.setitem(_daemon_registry, "alice", alice)
    task = _ok(task_tools.create_task_handler(
        {"title": "Review PR", "assigned_to": "bob"}, **_kwargs("alice")))["task"]
    alice.ingested.clear()  # drop nothing; alice assigned to bob, wasn't woken

    out = _ok(task_tools.mark_task_complete_handler(
        {"task_id": task["id"], "summary": "looks good, merged"}, **_kwargs("bob")))
    assert out["reported_to"] == "alice"
    assert len(alice.ingested) == 1
    from_agent, payload = alice.ingested[0]
    assert from_agent == "bob"
    assert "RESULT" in payload
    assert "looks good, merged" in payload
    assert "Review PR" in payload
    # and the agent is told not to double-report
    assert "do NOT also send" in out["message"] or "do NOT" in out["message"]


def test_complete_does_not_report_on_self_assigned_task(_wire, monkeypatch):
    """No delegator is waiting on work you gave yourself — reporting there is
    what produced the self-message link_violation."""
    from teams_server.tools import _daemon_registry

    alice = FakeDaemon()
    monkeypatch.setitem(_daemon_registry, "alice", alice)
    task = _ok(task_tools.create_task_handler(
        {"title": "my own todo", "assigned_to": "alice"}, **_kwargs("alice")))["task"]
    alice.ingested.clear()  # discard the wake

    out = _ok(task_tools.mark_task_complete_handler(
        {"task_id": task["id"]}, **_kwargs("alice")))
    assert out["reported_to"] is None
    assert alice.ingested == []


def test_completion_is_idempotent_and_reports_once(_wire, monkeypatch):
    """The turn-guard could drive a second completion call; the delegator must
    not be woken twice for one piece of work (observed at 12:52 as two RESULTs
    for a single completion)."""
    from teams_server.tools import _daemon_registry

    alice = FakeDaemon()
    monkeypatch.setitem(_daemon_registry, "alice", alice)
    task = _ok(task_tools.create_task_handler(
        {"title": "t", "assigned_to": "bob"}, **_kwargs("alice")))["task"]
    alice.ingested.clear()

    first = _ok(task_tools.mark_task_complete_handler(
        {"task_id": task["id"], "summary": "done"}, **_kwargs("bob")))
    second = _ok(task_tools.mark_task_complete_handler(
        {"task_id": task["id"], "summary": "done again"}, **_kwargs("bob")))

    assert first["reported_to"] == "alice"
    assert second.get("already_complete") is True
    assert second["reported_to"] is None
    assert len(alice.ingested) == 1, "one completion must produce exactly one report"


def test_blocked_reports_to_creator(_wire, monkeypatch):
    """Blocked is when the delegator most needs to hear: silence is
    indistinguishable from 'still working'."""
    from teams_server.tools import _daemon_registry

    alice = FakeDaemon()
    monkeypatch.setitem(_daemon_registry, "alice", alice)
    task = _ok(task_tools.create_task_handler(
        {"title": "Deploy", "assigned_to": "bob"}, **_kwargs("alice")))["task"]
    alice.ingested.clear()

    out = _ok(task_tools.mark_task_blocked_handler(
        {"task_id": task["id"], "reason": "need prod credentials"}, **_kwargs("bob")))
    assert out["reported_to"] == "alice"
    assert "need prod credentials" in alice.ingested[0][1]
    assert "BLOCKED" in alice.ingested[0][1]


def test_report_delivery_failure_does_not_fail_completion(_wire, monkeypatch):
    """The task row is the durable record — a delivery problem must not roll
    back or error the completion."""
    from teams_server.tools import _daemon_registry

    class Boom:
        def ingest_task(self, from_agent, payload):
            raise RuntimeError("inbox exploded")

    monkeypatch.setitem(_daemon_registry, "alice", Boom())
    task = _ok(task_tools.create_task_handler(
        {"title": "t", "assigned_to": "bob"}, **_kwargs("alice")))["task"]

    out = _ok(task_tools.mark_task_complete_handler(
        {"task_id": task["id"]}, **_kwargs("bob")))
    assert out["task"]["status"] == "done"
    assert out["reported_to"] is None


def test_complete_with_no_creator_daemon_still_completes(_wire):
    task = _ok(task_tools.create_task_handler(
        {"title": "t", "assigned_to": "bob"}, **_kwargs("alice")))["task"]
    out = _ok(task_tools.mark_task_complete_handler(
        {"task_id": task["id"]}, **_kwargs("bob")))
    assert out["task"]["status"] == "done"
    assert out["reported_to"] is None


# ---------------------------------------------------------------------------
# 7c. The report walks the WORK chain, one level at a time
# ---------------------------------------------------------------------------
def _chain(monkeypatch):
    """lead --T1--> mid --T2(child of T1)--> worker, with inboxes to inspect."""
    from teams_server.tools import _daemon_registry

    boxes = {}
    for name in ("lead", "mid", "worker"):
        boxes[name] = FakeDaemon()
        monkeypatch.setitem(_daemon_registry, name, boxes[name])

    t1 = _ok(task_tools.create_task_handler(
        {"title": "ship the feature", "assigned_to": "mid"}, **_kwargs("lead")))["task"]
    t2 = _ok(task_tools.create_task_handler(
        {"title": "write the tests", "assigned_to": "worker",
         "parent_task_id": t1["id"]}, **_kwargs("mid")))["task"]
    for b in boxes.values():
        b.ingested.clear()  # discard the assignment wakes
    return boxes, t1, t2


def test_child_reports_to_parents_assignee_not_creator(_wire, monkeypatch):
    """The agent working the parent is the one blocked on this result."""
    boxes, _t1, t2 = _chain(monkeypatch)

    out = _ok(task_tools.mark_task_complete_handler(
        {"task_id": t2["id"], "summary": "12 tests, all green"}, **_kwargs("worker")))
    assert out["reported_to"] == "mid"
    assert len(boxes["mid"].ingested) == 1
    assert "12 tests, all green" in boxes["mid"].ingested[0][1]
    # the top of the chain is NOT told about a grandchild finishing
    assert boxes["lead"].ingested == []


def test_report_target_prefers_parent_assignee_over_filer(_wire, monkeypatch):
    """The case that distinguishes "parent's assignee" from "creator": when the
    LEAD files a child of the task MID is executing, mid is the one who needs
    the answer — reporting to lead would leave mid waiting forever."""
    from teams_server.tools import _daemon_registry

    boxes = {}
    for name in ("lead", "mid", "worker"):
        boxes[name] = FakeDaemon()
        monkeypatch.setitem(_daemon_registry, name, boxes[name])

    t1 = _ok(task_tools.create_task_handler(
        {"title": "ship it", "assigned_to": "mid"}, **_kwargs("lead")))["task"]
    # lead — NOT mid — files the child
    child = _ok(task_tools.create_task_handler(
        {"title": "subtask", "assigned_to": "worker",
         "parent_task_id": t1["id"]}, **_kwargs("lead")))["task"]
    assert child["created_by"] == "lead"
    for b in boxes.values():
        b.ingested.clear()

    out = _ok(task_tools.mark_task_complete_handler(
        {"task_id": child["id"]}, **_kwargs("worker")))
    assert out["reported_to"] == "mid", "must follow the work, not the paperwork"
    assert len(boxes["mid"].ingested) == 1
    assert boxes["lead"].ingested == []


def test_top_of_chain_reports_to_creator(_wire, monkeypatch):
    """A task with no parent has nothing above it but whoever asked for it."""
    boxes, t1, _t2 = _chain(monkeypatch)
    out = _ok(task_tools.mark_task_complete_handler(
        {"task_id": t1["id"], "summary": "shipped"}, **_kwargs("mid")))
    assert out["reported_to"] == "lead"
    assert len(boxes["lead"].ingested) == 1


def test_each_level_reports_exactly_one_level_up(_wire, monkeypatch):
    """Walking the full chain bottom-to-top: no level skips or doubles."""
    boxes, t1, t2 = _chain(monkeypatch)

    task_tools.mark_task_complete_handler(
        {"task_id": t2["id"], "summary": "inner done"}, **_kwargs("worker"))
    task_tools.mark_task_complete_handler(
        {"task_id": t1["id"], "summary": "outer done"}, **_kwargs("mid"))

    assert len(boxes["mid"].ingested) == 1, "mid hears only about its child"
    assert len(boxes["lead"].ingested) == 1, "lead hears only about its own task"
    assert "inner done" in boxes["mid"].ingested[0][1]
    assert "outer done" in boxes["lead"].ingested[0][1]
    assert boxes["worker"].ingested == [], "nothing reports downward"


def test_blocked_also_reports_up_the_chain(_wire, monkeypatch):
    boxes, _t1, t2 = _chain(monkeypatch)
    out = _ok(task_tools.mark_task_blocked_handler(
        {"task_id": t2["id"], "reason": "need a staging DB"}, **_kwargs("worker")))
    assert out["reported_to"] == "mid"
    assert "need a staging DB" in boxes["mid"].ingested[0][1]
    assert boxes["lead"].ingested == []


def test_falls_back_to_creator_when_parent_was_deleted(_wire, monkeypatch):
    """Deleting a parent nulls the child's link (ON DELETE SET NULL), so the
    child must still have somewhere to report rather than going silent."""
    boxes, t1, t2 = _chain(monkeypatch)
    task_tools.task_db.delete_task(t1["id"])

    out = _ok(task_tools.mark_task_complete_handler(
        {"task_id": t2["id"]}, **_kwargs("worker")))
    assert out["reported_to"] == "mid", "mid created it, so mid still hears"
    assert len(boxes["mid"].ingested) == 1


def test_falls_back_to_creator_when_parent_has_no_assignee(_wire, monkeypatch):
    """A parent can be unassigned (e.g. created via the REST API)."""
    from teams_server.tools import _daemon_registry

    mid = FakeDaemon()
    monkeypatch.setitem(_daemon_registry, "mid", mid)
    orphan_parent = task_tools.task_db.create_task(
        "unassigned epic", created_by="mid", assigned_to=None, team_id="t1")
    child = _ok(task_tools.create_task_handler(
        {"title": "child", "assigned_to": "worker",
         "parent_task_id": orphan_parent["id"]}, **_kwargs("mid")))["task"]
    mid.ingested.clear()

    out = _ok(task_tools.mark_task_complete_handler(
        {"task_id": child["id"]}, **_kwargs("worker")))
    assert out["reported_to"] == "mid"


def test_no_report_when_the_chain_ends_at_you(_wire, monkeypatch):
    """If you are the parent's assignee AND the one finishing, nobody above is
    waiting — reporting to yourself is what caused the link_violation."""
    from teams_server.tools import _daemon_registry

    mid = FakeDaemon()
    monkeypatch.setitem(_daemon_registry, "mid", mid)
    parent = _ok(task_tools.create_task_handler(
        {"title": "epic", "assigned_to": "mid"}, **_kwargs("mid")))["task"]
    child = _ok(task_tools.create_task_handler(
        {"title": "own subtask", "assigned_to": "mid",
         "parent_task_id": parent["id"]}, **_kwargs("mid")))["task"]
    mid.ingested.clear()

    out = _ok(task_tools.mark_task_complete_handler(
        {"task_id": child["id"]}, **_kwargs("mid")))
    assert out["reported_to"] is None
    assert mid.ingested == []


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
