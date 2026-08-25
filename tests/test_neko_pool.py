#!/usr/bin/env python3
"""Tests for the neko viewer pool (teams_server/neko_pool).

Covers the pure/deterministic parts — config rendering, secrets persistence,
port allocation, and the mode-resolution seam — without touching Docker. The
container lifecycle itself needs a live Docker daemon and is exercised manually.

Run:  pytest tests/test_neko_pool.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from teams_server import neko_pool as np  # noqa: E402


# ---------------------------------------------------------------------------
# render_supervisord_conf / render_chrome_policy
# ---------------------------------------------------------------------------
def test_supervisord_conf_exposes_cdp():
    conf = np.render_supervisord_conf()
    assert "--remote-debugging-port=9222" in conf
    assert "--remote-debugging-address=0.0.0.0" in conf
    assert "--remote-allow-origins=*" in conf


def test_supervisord_conf_keeps_profile_path():
    # Must stay /home/neko/.config/chromium — that's where we bind-mount the
    # team profile, so cookies persist across container rebuilds.
    assert "--user-data-dir=/home/neko/.config/chromium" in np.render_supervisord_conf()


def test_chrome_policy_allows_persistent_cookies():
    import json
    policy = json.loads(np.render_chrome_policy())
    assert policy["DefaultCookiesSetting"] == 1   # allow
    assert policy["RestoreOnStartup"] == 1


def test_supervisord_conf_defines_both_programs():
    conf = np.render_supervisord_conf()
    assert "[program:chromium]" in conf
    assert "[program:openbox]" in conf


# ---------------------------------------------------------------------------
# load_or_create_secrets — stable per-team room passwords
# ---------------------------------------------------------------------------
def test_secrets_created_and_stable(tmp_path):
    s1 = np.load_or_create_secrets(tmp_path)
    assert s1["admin_pwd"] and s1["user_pwd"]
    assert s1["admin_pwd"] != s1["user_pwd"]
    s2 = np.load_or_create_secrets(tmp_path)
    assert s2 == s1   # no regeneration on restart


def test_secrets_regen_when_corrupt(tmp_path):
    (tmp_path / ".neko.json").write_text("{not json")
    s = np.load_or_create_secrets(tmp_path)
    assert s["admin_pwd"] and s["user_pwd"]


def test_secrets_regen_when_partial(tmp_path):
    (tmp_path / ".neko.json").write_text('{"admin_pwd": "x"}')
    s = np.load_or_create_secrets(tmp_path)
    assert s["admin_pwd"] and s["user_pwd"]


# ---------------------------------------------------------------------------
# Port allocation — never hands out the same port twice while in use
# ---------------------------------------------------------------------------
def test_alloc_ports_unique_across_teams():
    mgr = np.NekoManager()
    a = mgr._alloc_ports("team-a")
    b = mgr._alloc_ports("team-b")
    assert set(a.values()).isdisjoint(set(b.values()))
    for ports in (a, b):
        assert {"http", "cdp", "udp", "tcp"} == set(ports.keys())


def test_release_ports_allows_reuse():
    mgr = np.NekoManager()
    a = mgr._alloc_ports("team-a")
    mgr._release_ports(a)
    b = mgr._alloc_ports("team-b")
    assert set(a.values()) & set(b.values())   # freed block can come back


# ---------------------------------------------------------------------------
# Container naming — docker-safe for odd team ids
# ---------------------------------------------------------------------------
def test_container_name_sanitized():
    assert np.NekoManager._container_name('team/one two') == "agent-neko-team-one-two"
    assert np.NekoManager._container_name("ok-team") == "agent-neko-ok-team"


# ---------------------------------------------------------------------------
# resolve_team_cdp_url — the seam between viewing modes
# ---------------------------------------------------------------------------
class _FakeManager:
    def __init__(self):
        self.calls = []

    def enabled_for(self, team_id):
        return team_id != "local-only"

    def ensure_neko(self, team_id):
        self.calls.append(team_id)
        return {"cdp_url": "http://127.0.0.1:9641"}


@pytest.fixture()
def fake_neko(monkeypatch):
    fake = _FakeManager()
    monkeypatch.setattr(np, "neko_manager", fake)
    yield fake


def test_resolve_prefers_neko_when_enabled(fake_neko, monkeypatch):
    class _Pool:
        @staticmethod
        def ensure_team_browser(team_id):
            raise AssertionError("should not be called when neko is up")

    import teams_server.browser_pool as bp
    monkeypatch.setattr(bp, "team_browser_manager", _Pool)
    assert np.resolve_team_cdp_url("devs") == "http://127.0.0.1:9641"
    assert fake_neko.calls == ["devs"]


def test_resolve_falls_back_to_local(fake_neko, monkeypatch):
    class _Pool:
        @staticmethod
        def ensure_team_browser(team_id):
            return "http://127.0.0.1:9333"

    import teams_server.browser_pool as bp
    monkeypatch.setattr(bp, "team_browser_manager", _Pool)
    assert np.resolve_team_cdp_url("local-only") == "http://127.0.0.1:9333"


def test_resolve_falls_back_when_neko_errors(fake_neko, monkeypatch):
    def boom(team_id):
        raise RuntimeError("docker exploded")

    fake_neko.ensure_neko = boom

    class _Pool:
        @staticmethod
        def ensure_team_browser(team_id):
            return "http://127.0.0.1:9333"

    import teams_server.browser_pool as bp
    monkeypatch.setattr(bp, "team_browser_manager", _Pool)
    assert np.resolve_team_cdp_url("devs") == "http://127.0.0.1:9333"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
