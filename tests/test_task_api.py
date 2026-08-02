#!/usr/bin/env python3
"""Tests for the /tasks REST namespace (teams_server/server.py).

Covers the human/dashboard-facing surface for the task tracker:
  * GET  /tasks                      — list with filters
  * GET  /tasks/agent/{name}         — scoped to one assignee
  * GET  /tasks/team/{team_id}       — scoped to one team
  * GET  /tasks/{id}                 — single task
  * PATCH /tasks/{id}                — human edit (more permissive than the
                                        agent tools: any field, incl. status/
                                        assigned_to)
  * DELETE /tasks/{id}               — removes one task; anything referencing it
                                        as parent survives as top-level

Uses FastAPI TestClient against the real app, with server_mod.task_db swapped
for a throwaway TasksDB (tmp_path) so tests never touch the real tasks.db.

Run:  pytest tests/test_task_api.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import teams_server.server as server_mod  # noqa: E402
from teams_server.tasks_db import TasksDB  # noqa: E402


@pytest.fixture()
def db(tmp_path, monkeypatch):
    d = TasksDB(tmp_path / "tasks.db")
    monkeypatch.setattr(server_mod, "task_db", d)
    monkeypatch.setattr(server_mod.monitor_db, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "TEAMS_API_KEY", "")
    return d


@pytest.fixture()
def client():
    return TestClient(server_mod.app)


# ---------------------------------------------------------------------------
# GET /tasks
# ---------------------------------------------------------------------------
def test_list_tasks_empty(client, db):
    r = client.get("/tasks")
    assert r.status_code == 200
    assert r.json()["tasks"] == []


def test_list_tasks_returns_created(client, db):
    db.create_task("t1", created_by="alice", assigned_to="bob", team_id="teamA")
    db.create_task("t2", created_by="alice", assigned_to="carol", team_id="teamB")
    r = client.get("/tasks")
    assert len(r.json()["tasks"]) == 2


def test_list_tasks_filters_by_team(client, db):
    db.create_task("t1", created_by="alice", assigned_to="bob", team_id="teamA")
    db.create_task("t2", created_by="alice", assigned_to="carol", team_id="teamB")
    r = client.get("/tasks?team_id=teamA")
    tasks = r.json()["tasks"]
    assert len(tasks) == 1
    assert tasks[0]["title"] == "t1"


def test_list_tasks_filters_by_status(client, db):
    t1 = db.create_task("t1", created_by="alice", assigned_to="bob")
    db.create_task("t2", created_by="alice", assigned_to="bob")
    db.set_status(t1["id"], "done")
    r = client.get("/tasks?status=done")
    tasks = r.json()["tasks"]
    assert len(tasks) == 1
    assert tasks[0]["title"] == "t1"


# ---------------------------------------------------------------------------
# GET /tasks/agent/{name}, /tasks/team/{id}
# ---------------------------------------------------------------------------
def test_list_tasks_for_agent(client, db):
    db.create_task("for bob", created_by="alice", assigned_to="bob")
    db.create_task("for carol", created_by="alice", assigned_to="carol")
    r = client.get("/tasks/agent/bob")
    assert r.status_code == 200
    tasks = r.json()["tasks"]
    assert len(tasks) == 1
    assert tasks[0]["title"] == "for bob"


def test_list_tasks_for_team(client, db):
    db.create_task("t1", created_by="alice", assigned_to="bob", team_id="teamA")
    db.create_task("t2", created_by="alice", assigned_to="bob", team_id="teamB")
    r = client.get("/tasks/team/teamA")
    tasks = r.json()["tasks"]
    assert len(tasks) == 1
    assert tasks[0]["title"] == "t1"


# ---------------------------------------------------------------------------
# GET /tasks/{id}
# ---------------------------------------------------------------------------
def test_get_task(client, db):
    task = db.create_task("parent", created_by="alice", assigned_to="bob")
    r = client.get(f"/tasks/{task['id']}")
    assert r.status_code == 200
    assert r.json()["task"]["title"] == "parent"


def test_get_task_exposes_parent_reference(client, db):
    parent = db.create_task("parent", created_by="alice", assigned_to="bob")
    child = db.create_task("child", created_by="alice", assigned_to="bob",
                            parent_task_id=parent["id"])
    r = client.get(f"/tasks/{child['id']}")
    assert r.json()["task"]["parent_task_id"] == parent["id"]


def test_get_task_404_when_missing(client, db):
    r = client.get("/tasks/nonexistent")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /tasks/{id}
# ---------------------------------------------------------------------------
def test_patch_task_descriptive_fields(client, db):
    task = db.create_task("old", created_by="alice", assigned_to="bob")
    r = client.patch(f"/tasks/{task['id']}", json={"title": "new", "priority": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["task"]["title"] == "new"
    assert body["task"]["priority"] == 3


def test_patch_task_can_change_status_and_assignee(client, db):
    """Human PATCH is intentionally more permissive than the agent tools —
    it can change status/assignee directly, which no single agent tool allows."""
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    r = client.patch(f"/tasks/{task['id']}", json={"status": "done", "assigned_to": "carol"})
    assert r.status_code == 200
    body = r.json()["task"]
    assert body["status"] == "done"
    assert body["assigned_to"] == "carol"


def test_patch_task_can_reach_failed_status(client, db):
    """failed has no agent tool by design — only the human PATCH can set it."""
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    r = client.patch(f"/tasks/{task['id']}", json={"status": "failed"})
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "failed"


def test_patch_task_progress(client, db):
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    r = client.patch(f"/tasks/{task['id']}", json={"progress": 42})
    assert r.status_code == 200
    assert r.json()["task"]["progress"] == 42


def test_patch_task_invalid_status_rejected(client, db):
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    r = client.patch(f"/tasks/{task['id']}", json={"status": "not-a-status"})
    assert r.status_code == 400


def test_patch_task_404_when_missing(client, db):
    r = client.patch("/tasks/nonexistent", json={"title": "x"})
    assert r.status_code == 404


def test_patch_task_broadcasts_update(client, db, monkeypatch):
    broadcasts = []
    monkeypatch.setattr("teams_server.websocket._broadcast",
                        lambda ev, data: broadcasts.append((ev, data)))
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    client.patch(f"/tasks/{task['id']}", json={"title": "new"})
    assert [ev for ev, _ in broadcasts] == ["task_updated"]
    assert broadcasts[0][1]["title"] == "new"


# ---------------------------------------------------------------------------
# DELETE /tasks/{id}
# ---------------------------------------------------------------------------
def test_delete_task_keeps_children_as_top_level(client, db):
    parent = db.create_task("parent", created_by="alice", assigned_to="bob")
    child = db.create_task("child", created_by="alice", assigned_to="carol",
                            parent_task_id=parent["id"])
    r = client.delete(f"/tasks/{parent['id']}")
    assert r.status_code == 200
    assert db.get_task(parent["id"]) is None
    survivor = db.get_task(child["id"])
    assert survivor is not None
    assert survivor["parent_task_id"] is None


def test_delete_task_404_when_missing(client, db):
    r = client.delete("/tasks/nonexistent")
    assert r.status_code == 404


def test_delete_task_broadcasts(client, db, monkeypatch):
    broadcasts = []
    monkeypatch.setattr("teams_server.websocket._broadcast",
                        lambda ev, data: broadcasts.append((ev, data)))
    task = db.create_task("t", created_by="alice", assigned_to="bob")
    client.delete(f"/tasks/{task['id']}")
    assert [ev for ev, _ in broadcasts] == ["task_deleted"]


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
