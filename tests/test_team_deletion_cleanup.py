import pytest
import time
from unittest.mock import MagicMock, patch

from teams_server.tasks_db import TasksDB
from teams_server.monitoring import MonitoringDB
from teams_server.tools import (
    _pending_human_questions,
    _pending_config_proposals,
    _pending_lock,
    _proposals_lock,
    add_pending_question,
    add_config_proposal,
    clear_pending_for_team,
    clear_pending_for_agent,
)
from teams_server.config import delete_team, delete_agent
from teams_server.browser_pool import TeamBrowserManager


def test_tasks_db_delete_tasks_for_team(tmp_path):
    db_file = tmp_path / "tasks.db"
    db = TasksDB(db_file)

    t1 = db.create_task("Team Alpha Task 1", "human", team_id="alpha", assigned_to="scout")
    t2 = db.create_task("Team Alpha Task 2", "human", team_id="alpha", assigned_to="researcher")
    t3 = db.create_task("Team Beta Task 1", "human", team_id="beta", assigned_to="other_agent")

    assert len(db.list_tasks(team_id="alpha")) == 2
    assert len(db.list_tasks(team_id="beta")) == 1

    deleted_count = db.delete_tasks_for_team("alpha", ["scout", "researcher"])
    assert deleted_count == 2

    assert len(db.list_tasks(team_id="alpha")) == 0
    assert len(db.list_tasks(team_id="beta")) == 1


def test_tasks_db_delete_tasks_for_agent(tmp_path):
    db_file = tmp_path / "tasks.db"
    db = TasksDB(db_file)

    t1 = db.create_task("Task for Alice", "human", team_id="alpha", assigned_to="alice")
    t2 = db.create_task("Task for Bob", "human", team_id="alpha", assigned_to="bob")

    deleted_count = db.delete_tasks_for_agent("alice")
    assert deleted_count == 1
    assert db.get_task(t1["id"]) is None
    assert db.get_task(t2["id"]) is not None


def test_monitoring_db_delete_team_records(tmp_path):
    db_file = tmp_path / "monitoring.db"
    db = MonitoringDB(db_file)

    db.log_event("agent_alpha", "turn_start", team_id="team_alpha", data={"k": "v"})
    db.log_message("agent_alpha", "user", "hello", team_id="team_alpha")
    db.log_decision("agent_alpha", "decided something", team_id="team_alpha")
    db.open_delegation("msg_1", "agent_alpha", "agent_beta", "TASK", "do work", team_id="team_alpha")

    db.log_event("agent_gamma", "turn_start", team_id="team_gamma", data={"k": "v"})

    res = db.delete_team_records("team_alpha", ["agent_alpha"])
    assert res["events"] >= 1
    assert res["messages"] >= 1
    assert res["decisions"] >= 1
    assert res["delegations"] >= 1

    with db._conn() as conn:
        alpha_events = conn.execute("SELECT count(*) FROM events WHERE team_id='team_alpha'").fetchone()[0]
        gamma_events = conn.execute("SELECT count(*) FROM events WHERE team_id='team_gamma'").fetchone()[0]
        assert alpha_events == 0
        assert gamma_events == 1


def test_clear_pending_in_memory():
    with _pending_lock:
        _pending_human_questions.clear()
    with _proposals_lock:
        _pending_config_proposals.clear()

    q1 = add_pending_question("agent1", "What is your name?")
    with _pending_lock:
        _pending_human_questions[q1]["team_id"] = "team_alpha"

    q2 = add_pending_question("agent2", "Other team question?")
    with _pending_lock:
        _pending_human_questions[q2]["team_id"] = "team_beta"

    p1 = add_config_proposal("agent1", {"model": "gpt-4"}, "need more power")
    with _proposals_lock:
        _pending_config_proposals[p1]["team_id"] = "team_alpha"

    p2 = add_config_proposal("agent2", {"model": "claude-3"}, "speed")
    with _proposals_lock:
        _pending_config_proposals[p2]["team_id"] = "team_beta"

    clear_pending_for_team("team_alpha", ["agent1"])

    with _pending_lock:
        assert q1 not in _pending_human_questions
        assert q2 in _pending_human_questions

    with _proposals_lock:
        assert p1 not in _pending_config_proposals
        assert p2 in _pending_config_proposals


def test_browser_manager_stop_team_browser():
    mgr = TeamBrowserManager()
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None

    mgr._browsers["team_x"] = {"proc": fake_proc, "port": 9335, "profile": "/tmp/fake", "headful": False}
    mgr._ports["team_x"] = 9335

    with patch.object(mgr, "_quit_browser") as mock_quit:
        mgr.stop_team_browser("team_x")
        mock_quit.assert_called_once()
        assert "team_x" not in mgr._browsers
        assert "team_x" not in mgr._ports


def test_delete_team_scrubs_allowed_peers_of_other_teams(tmp_path):
    cfg = {
        "teams": {
            "alpha": {"name": "Alpha"},
            "beta": {"name": "Beta"},
        },
        "agents": {
            "a1": {"team_id": "alpha", "allowed_peers": ["a2", "b1"]},
            "a2": {"team_id": "alpha", "allowed_peers": ["a1"]},
            "b1": {"team_id": "beta", "allowed_peers": ["a1", "b2"]},
            "b2": {"team_id": "beta", "allowed_peers": ["b1"]},
        },
    }

    with patch("teams_server.config._save_full_config"):
        success = delete_team(cfg, "alpha")
        assert success is True
        assert "alpha" not in cfg["teams"]
        assert "a1" not in cfg["agents"]
        assert "a2" not in cfg["agents"]
        # b1 should have had 'a1' removed from allowed_peers
        assert cfg["agents"]["b1"]["allowed_peers"] == ["b2"]
        assert cfg["agents"]["b2"]["allowed_peers"] == ["b1"]
