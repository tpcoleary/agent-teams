#!/usr/bin/env python3
"""Tests for slash-command dispatch at the HTTP boundary.

Covers the four intercept points and, more importantly, the invariants that make
dispatching at the boundary safe:

  * a command never becomes an LLM turn (never reaches ingest_task)
  * ordinary text is completely unaffected
  * only humans can run commands -- an agent cannot drive another agent's
    control plane through send_peer_message
  * inbox replies are intercepted before _finalize_human_answer wraps them
  * repeating a command runs it twice (the queue's dedup does not apply)

Uses FastAPI TestClient with stub daemons injected into server_mod.daemons --
TestClient does not run lifespan, so that dict is otherwise empty.

Run:  pytest tests/test_commands_endpoints.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import teams_server.server as server_mod  # noqa: E402

AGENT = "researcher"


class StubInbox:
    def __init__(self):
        self.pending = []

    def get_pending_count(self):
        return len(self.pending)

    def get_all_tasks(self, limit=50):
        return list(self.pending)


class StubAI:
    session_total_tokens = 1234
    session_input_tokens = 1000
    session_output_tokens = 234
    session_cache_read_tokens = 0
    session_api_calls = 3
    session_estimated_cost_usd = 0.0012

    def __init__(self):
        self.steered = []

    def steer(self, text):
        self.steered.append(text)
        return True


class StubDaemon:
    """Minimal stand-in for AgentDaemon, recording what was called."""

    def __init__(self, name=AGENT, state="busy"):
        self.name = name
        self.state = state
        self._paused = False
        self.cfg = {"team_id": "acme", "model": "gpt-5", "session_id": "s1"}
        self.inbox = StubInbox()
        self._ai_agent = StubAI()
        self.ingested = []
        self.stopped = 0
        self.directives = []

    # --- the call that must NOT happen for a command ---
    def ingest_task(self, from_agent, payload):
        self.ingested.append((from_agent, payload))
        return f"task-{len(self.ingested)}"

    async def stop_execution(self):
        self.stopped += 1
        self.state = "idle"

    def pause_execution(self, reason="", by=""):
        self._paused = True

    def resume_execution(self, by=""):
        self._paused = False

    def set_directive(self, payload, minutes, from_agent=""):
        self.directives.append((payload, minutes))


@pytest.fixture()
def daemon(monkeypatch):
    d = StubDaemon()
    monkeypatch.setitem(server_mod.daemons, AGENT, d)
    return d


@pytest.fixture()
def client(monkeypatch):
    # Auth off so these tests exercise dispatch, not the middleware
    # (tests/test_auth.py owns the guard itself).
    monkeypatch.setattr(server_mod, "TEAMS_API_KEY", "")
    return TestClient(server_mod.app)


def _task(client, payload, from_agent="human_operator"):
    return client.post(f"/agent/{AGENT}/task",
                       json={"from_agent": from_agent, "payload": payload})


# ---------------------------------------------------------------------------
# GET /commands
# ---------------------------------------------------------------------------
def test_commands_catalog_endpoint(client):
    r = client.get("/commands")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["commands"]
    assert isinstance(body["hermes"], bool)


def test_commands_catalog_respects_scope(client):
    agent = {c["name"] for c in client.get("/commands?scope=agent").json()["commands"]}
    architect = {c["name"] for c in
                 client.get("/commands?scope=architect").json()["commands"]}
    assert "steer" in agent
    assert "steer" not in architect


# ---------------------------------------------------------------------------
# POST /agent/{name}/task -- the main door
# ---------------------------------------------------------------------------
def test_command_does_not_become_a_task(client, daemon):
    r = _task(client, "/steer focus on the API docs")
    assert r.status_code == 200
    assert r.json()["command"] is True
    # THE invariant: nothing was enqueued, so no LLM turn is burned.
    assert daemon.ingested == []
    assert daemon._ai_agent.steered == ["focus on the API docs"]


def test_ordinary_text_still_enqueues(client, daemon):
    r = _task(client, "please summarize the competitor research")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "queued"
    assert "command" not in body
    assert daemon.ingested == [("human_operator", "please summarize the competitor research")]


def test_pasted_path_is_ordinary_text(client, daemon):
    """The path-paste regression, end to end."""
    payload = "/Users/pradhyun/Movies/agent-teams/README.md"
    r = _task(client, payload)
    assert r.json()["status"] == "queued"
    assert daemon.ingested == [("human_operator", payload)]


def test_stop_command_interrupts_without_enqueueing(client, daemon):
    r = _task(client, "/stop")
    assert r.status_code == 200
    assert daemon.stopped == 1
    assert daemon.ingested == []


def test_pause_and_resume_commands(client, daemon):
    _task(client, "/pause")
    assert daemon._paused is True
    _task(client, "/resume")
    assert daemon._paused is False
    assert daemon.ingested == []


def test_unknown_command_is_rejected_not_forwarded(client, daemon):
    r = _task(client, "/modle gpt-5")
    assert r.status_code == 400
    body = r.json()
    assert body["ok"] is False
    assert "/model" in body["message"]      # did-you-mean
    # Never handed to the model as prose.
    assert daemon.ingested == []


def test_repeated_command_runs_every_time(client, daemon):
    """The inbox dedups byte-identical pending payloads (inbox.py:57-85).

    Commands must not inherit that: /usage twice has to answer twice.
    """
    first = _task(client, "/usage")
    second = _task(client, "/usage")
    assert first.status_code == second.status_code == 200
    assert first.json()["message"] == second.json()["message"]
    assert daemon.ingested == []


def test_repeated_stop_acts_twice(client, daemon):
    """Same dedup concern, but observable through a side effect."""
    _task(client, "/stop")
    daemon.state = "busy"          # agent picked up work again
    _task(client, "/stop")
    assert daemon.stopped == 2


def test_command_bypasses_directive_promotion(client, daemon):
    """duration_minutes must not turn a command into a standing directive."""
    r = client.post(f"/agent/{AGENT}/task",
                    json={"from_agent": "human_operator", "payload": "/usage",
                          "duration_minutes": 60})
    assert r.json()["command"] is True
    assert daemon.directives == []
    assert daemon.ingested == []


def test_model_command_patches_config(client, daemon, monkeypatch):
    applied = {}
    monkeypatch.setattr(server_mod, "_apply_agent_config_patch",
                        lambda name, fields: applied.setdefault(name, fields))
    r = _task(client, "/model gpt-4o")
    assert r.status_code == 200
    assert applied == {AGENT: {"model": "gpt-4o"}}
    assert daemon.ingested == []


# ---------------------------------------------------------------------------
# The security boundary: agents must not reach the control plane
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sender", ["researcher", "supervisor", "autonomous", "cron",
                                    "unknown", "turn-guard"])
def test_non_human_senders_cannot_run_commands(client, daemon, sender):
    """send_peer_message routes through ingest_task, so an agent could otherwise
    pause or stop its teammates by messaging them "/pause"."""
    r = _task(client, "/stop", from_agent=sender)
    assert r.json()["status"] == "queued"      # treated as ordinary text
    assert daemon.stopped == 0                 # command did NOT run
    assert daemon.ingested == [(sender, "/stop")]


def test_human_sender_variants_do_run_commands(client, daemon):
    for sender in ("human", "human_operator"):
        daemon.ingested.clear()
        r = _task(client, "/usage", from_agent=sender)
        assert r.json().get("command") is True, f"{sender} should be trusted"
        assert daemon.ingested == []


# ---------------------------------------------------------------------------
# Inbox: intercept BEFORE the text is template-wrapped
# ---------------------------------------------------------------------------
def test_inbox_respond_intercepts_command(client, daemon, monkeypatch):
    """_finalize_human_answer prefixes the human's text with an emoji template
    before enqueueing, so the leading "/" is only visible at the endpoint."""
    called = []

    async def _boom(*a, **k):
        called.append(a)
        return {"ok": True, "delivery": "task", "question_id": "q1"}

    monkeypatch.setattr(server_mod, "_finalize_human_answer", _boom)

    r = client.post(f"/inbox/{AGENT}/respond", json={"response": "/usage"})
    assert r.status_code == 200
    assert r.json()["command"] is True
    assert called == [], "command must not be delivered as a question answer"
    assert daemon.ingested == []


def test_human_response_endpoint_intercepts_command(client, daemon, monkeypatch):
    called = []

    async def _boom(*a, **k):
        called.append(a)
        return {"ok": True, "delivery": "task", "question_id": "q1"}

    monkeypatch.setattr(server_mod, "_finalize_human_answer", _boom)

    r = client.post(f"/agent/{AGENT}/human_response", json={"response": "/stop"})
    assert r.status_code == 200
    assert daemon.stopped == 1
    assert called == []


def test_inbox_ordinary_answer_still_delivered(client, daemon, monkeypatch):
    """No regression: a normal answer must still reach the delivery path."""
    called = []

    async def _ok(daemon_arg, qid, text):
        called.append(text)
        return {"ok": True, "delivery": "in_turn", "question_id": qid}

    monkeypatch.setattr(server_mod, "_finalize_human_answer", _ok)
    monkeypatch.setattr(server_mod, "_resolve_pending_qid", lambda n, d: "q1")

    r = client.post(f"/agent/{AGENT}/human_response",
                    json={"response": "use the staging credentials"})
    assert r.status_code == 200
    assert called == ["use the staging credentials"]


# ---------------------------------------------------------------------------
# Architect chat -- separate runtime, own intercept
# ---------------------------------------------------------------------------
def test_master_chat_command_bypasses_submit(client, monkeypatch):
    submitted = []
    broadcasts = []

    class FakeMaster:
        def is_configured(self):
            return True

        def submit(self, msg):
            submitted.append(msg)
            return True

    monkeypatch.setattr("teams_server.master.get_master", lambda: FakeMaster())
    monkeypatch.setattr("teams_server.websocket._broadcast",
                        lambda ev, data: broadcasts.append((ev, data)))

    r = client.post("/master/chat", json={"message": "/help"})
    assert r.status_code == 200
    assert r.json()["command"] is True
    assert submitted == [], "command must not reach the Architect's runtime"
    # Echoed on the same event its normal replies use, so the transcript is coherent.
    assert [ev for ev, _ in broadcasts] == ["master_message"]
    assert "/help" in broadcasts[0][1]["content"] or "commands" in broadcasts[0][1]["content"]


def test_master_chat_ordinary_message_still_submits(client, monkeypatch):
    submitted = []

    class FakeMaster:
        def is_configured(self):
            return True

        def submit(self, msg):
            submitted.append(msg)
            return True

    monkeypatch.setattr("teams_server.master.get_master", lambda: FakeMaster())

    r = client.post("/master/chat", json={"message": "build me a marketing team"})
    assert r.status_code == 200
    assert submitted == ["build me a marketing team"]


def test_agent_scoped_command_rejected_in_architect(client, monkeypatch):
    submitted = []

    class FakeMaster:
        def is_configured(self):
            return True

        def submit(self, msg):
            submitted.append(msg)
            return True

    monkeypatch.setattr("teams_server.master.get_master", lambda: FakeMaster())
    monkeypatch.setattr("teams_server.websocket._broadcast", lambda ev, data: None)

    r = client.post("/master/chat", json={"message": "/steer do the thing"})
    assert r.status_code == 400
    assert "not available here" in r.json()["message"]
    assert submitted == []


# ---------------------------------------------------------------------------
# Cron instructions are prompts, not commands
# ---------------------------------------------------------------------------
def test_cron_instruction_rejects_slash_command(client, monkeypatch):
    monkeypatch.setattr(server_mod, "load_agents_config",
                        lambda: {"agents": {AGENT: {}}})
    r = client.post(f"/agent/{AGENT}/crons",
                    json={"schedule": "0 9 * * *", "instruction": "/stop"})
    assert r.status_code == 400
    assert "slash command" in r.json()["error"]


def test_cron_instruction_accepts_a_pasted_path(client, monkeypatch):
    """The guard must use the same heuristic as dispatch, not a bare startswith."""
    monkeypatch.setattr(server_mod, "load_agents_config",
                        lambda: {"agents": {AGENT: {}}})
    captured = {}

    def _add(cfg, name, **kw):
        captured.update(kw)
        return {"id": "c1", **kw}

    monkeypatch.setattr(server_mod, "add_agent_cron", _add)
    monkeypatch.setattr(server_mod, "_refresh_daemon_crons", lambda n: None)

    r = client.post(f"/agent/{AGENT}/crons",
                    json={"schedule": "0 9 * * *",
                          "instruction": "/Users/me/brief.md — read and summarize"})
    assert r.status_code == 200
    assert captured["instruction"].startswith("/Users/me/brief.md")


# ---------------------------------------------------------------------------
# Missing agent
# ---------------------------------------------------------------------------
def test_command_for_unknown_agent_404s(client):
    r = client.post("/agent/ghost/task",
                    json={"from_agent": "human_operator", "payload": "/usage"})
    assert r.status_code == 404
