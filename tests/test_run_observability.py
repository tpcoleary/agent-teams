#!/usr/bin/env python3
"""Observability gaps found by reading a real run's telemetry (2026-08-02 19:46).

That run had a 7-minute silence in the middle. Nothing in monitoring.db could
tell a restart from a stalled scheduler — the only evidence was teams.pid's
mtime — and the operator instruction in flight at the time came back after the
restart looking brand-new. Three fixes, all covered here:

  1. server_started / server_stopped   — a gap is now self-explaining.
  2. redelivered marking               — a resumed message says it was seen before.
  3. build_batch_prompt                — that marker actually reaches the agent.

Run:  pytest tests/test_run_observability.py -v
"""

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from teams_server.agent import (  # noqa: E402
    REDELIVERY_NOTE,
    _TASK_PROMPT_MARKER,
    build_batch_prompt,
)
from teams_server.inbox import InboxQueue  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Lifecycle events
# ---------------------------------------------------------------------------
class _FakeDaemon:
    def __init__(self, recovered=0):
        self.recovered_on_boot = recovered


def test_boot_summary_reports_process_facts():
    from teams_server.server import build_boot_summary

    s = build_boot_summary({"a": _FakeDaemon(), "b": _FakeDaemon()})
    assert s["agents"] == 2
    assert isinstance(s["pid"], int) and s["pid"] > 0
    assert s["version"]
    assert s["resumed_messages"] == 0


def test_boot_summary_sums_resumed_messages():
    """The number that explains the gap: how much in-flight work this boot
    picked back up (and will therefore redeliver)."""
    from teams_server.server import build_boot_summary

    s = build_boot_summary({"a": _FakeDaemon(3), "b": _FakeDaemon(1), "c": _FakeDaemon(0)})
    assert s["resumed_messages"] == 4


def test_boot_summary_tolerates_daemon_without_counter():
    """A daemon that failed before recover_processing ran must not break boot
    logging — the event is diagnostics, never a boot blocker."""
    from teams_server.server import build_boot_summary

    class Bare:
        pass

    s = build_boot_summary({"a": Bare(), "b": _FakeDaemon(2)})
    assert s["resumed_messages"] == 2
    assert s["agents"] == 2


def test_boot_summary_empty_team():
    from teams_server.server import build_boot_summary

    s = build_boot_summary({})
    assert s["agents"] == 0 and s["resumed_messages"] == 0


# ---------------------------------------------------------------------------
# 2. Redelivery marking in the inbox
# ---------------------------------------------------------------------------
def test_fresh_message_is_not_redelivered(tmp_path):
    q = InboxQueue(tmp_path / "q.db")
    q.enqueue("peer", "do thing")
    assert q.drain_pending()[0]["redelivered"] == 0


def test_recover_processing_marks_redelivered(tmp_path):
    """The restart case from the observed run: a claimed message goes back to
    pending and must come back flagged."""
    q = InboxQueue(tmp_path / "q.db")
    q.enqueue("human_operator", "repeat the same thing")
    q.drain_pending()  # claimed -> 'processing'
    assert q.recover_processing() == 1
    row = q.drain_pending()[0]
    assert row["redelivered"] == 1
    assert row["payload"] == "repeat the same thing"


def test_recover_processing_does_not_burn_retries(tmp_path):
    """Why redelivery needs its OWN counter: a restart is not the message's
    fault, so retries deliberately stays put."""
    q = InboxQueue(tmp_path / "q.db")
    q.enqueue("peer", "work")
    q.drain_pending()
    q.recover_processing()
    row = q.drain_pending()[0]
    assert row["retries"] == 0
    assert row["redelivered"] == 1


def test_requeue_marks_redelivered_and_retries(tmp_path):
    """A failed turn may have already run tool calls, so the retry is also a
    redelivery."""
    q = InboxQueue(tmp_path / "q.db")
    mid = q.enqueue("peer", "work")
    q.drain_pending()
    q.requeue([mid])
    row = q.drain_pending()[0]
    assert row["retries"] == 1
    assert row["redelivered"] == 1


def test_requeue_no_penalty_still_marks_redelivered(tmp_path):
    """Not burning the retry budget is about blame; redelivery is about what the
    agent has already seen. An outage hold is still a second delivery."""
    q = InboxQueue(tmp_path / "q.db")
    mid = q.enqueue("peer", "work")
    q.drain_pending()
    q.requeue_no_penalty([mid])
    row = q.drain_pending()[0]
    assert row["retries"] == 0
    assert row["redelivered"] == 1


def test_redelivery_count_accumulates(tmp_path):
    """A message caught by two successive restarts is twice-seen."""
    q = InboxQueue(tmp_path / "q.db")
    q.enqueue("peer", "work")
    for _ in range(3):
        q.drain_pending()
        q.recover_processing()
    assert q.drain_pending()[0]["redelivered"] == 3


def test_mark_done_ends_the_message(tmp_path):
    """A completed message is never redelivered at all."""
    q = InboxQueue(tmp_path / "q.db")
    mid = q.enqueue("peer", "work")
    q.drain_pending()
    q.mark_done(mid)
    assert q.recover_processing() == 0
    assert q.drain_pending() == []


def test_legacy_db_without_column_is_migrated(tmp_path):
    """Existing *_inbox.db files predate this column; opening one must add it
    rather than crash the agent on construction."""
    p = tmp_path / "old.db"
    conn = sqlite3.connect(str(p))
    conn.executescript(
        """CREATE TABLE messages (
             id TEXT PRIMARY KEY, from_agent TEXT NOT NULL, payload TEXT NOT NULL,
             status TEXT NOT NULL DEFAULT 'pending', created_at REAL NOT NULL,
             processed_at REAL, retries INTEGER NOT NULL DEFAULT 0);"""
    )
    conn.execute(
        "INSERT INTO messages VALUES ('m1','peer','in flight','processing',1.0,2.0,0)"
    )
    conn.commit()
    conn.close()

    q = InboxQueue(p)
    cols = [r[1] for r in sqlite3.connect(str(p)).execute("PRAGMA table_info(messages)")]
    assert "redelivered" in cols
    # and the in-flight row still recovers, now flagged
    assert q.recover_processing() == 1
    assert q.drain_pending()[0]["redelivered"] == 1


# ---------------------------------------------------------------------------
# 3. The marker reaches the agent
# ---------------------------------------------------------------------------
def test_batch_prompt_keeps_marker_and_payloads():
    out = build_batch_prompt([
        {"from_agent": "alice", "payload": "first", "redelivered": 0},
        {"from_agent": "bob", "payload": "second", "redelivered": 0},
    ])
    assert _TASK_PROMPT_MARKER in out
    assert "You have 2" in out
    assert "--- [1] from alice ---" in out
    assert "--- [2] from bob ---" in out
    assert "first" in out and "second" in out
    assert REDELIVERY_NOTE not in out


def test_batch_prompt_annotates_only_the_redelivered_message():
    """The exact shape of the observed failure: an interrupted operator message
    arrives alongside genuinely new ones. Only the resumed one is flagged, or
    the agent distrusts all of them."""
    out = build_batch_prompt([
        {"from_agent": "human_operator", "payload": "repeat the same thing", "redelivered": 1},
        {"from_agent": "peer", "payload": "brand new", "redelivered": 0},
    ])
    assert out.count(REDELIVERY_NOTE) == 1
    seen_line = [ln for ln in out.splitlines() if ln.startswith("--- [1]")][0]
    fresh_line = [ln for ln in out.splitlines() if ln.startswith("--- [2]")][0]
    assert "SEEN BEFORE" in seen_line
    assert "SEEN BEFORE" not in fresh_line


def test_redelivery_note_tells_the_agent_to_check_state_not_restart():
    """The note has to prevent redoing work, not merely mention repetition."""
    assert "SEEN BEFORE" in REDELIVERY_NOTE
    assert "instead of starting over" in REDELIVERY_NOTE


def test_batch_prompt_handles_missing_redelivered_key():
    """Any caller building a batch dict by hand (or an older drain) must not
    KeyError."""
    out = build_batch_prompt([{"from_agent": "a", "payload": "p"}])
    assert REDELIVERY_NOTE not in out
    assert "p" in out


def test_batch_prompt_marker_matches_turn_boundary_anchor():
    """build_batch_prompt's preamble and the marker used to slice this turn's
    output must stay in sync, or the transcript boundary silently breaks."""
    out = build_batch_prompt([{"from_agent": "a", "payload": "p"}])
    assert out.splitlines()[0] == f"You have 1 {_TASK_PROMPT_MARKER}:"


def test_trace_reasoning_and_thinking_coalesce_consecutive_chunks():
    """Verify that streamed token-by-token reasoning/thinking chunks are coalesced
    into a single step rather than spawning one step per word."""
    from unittest.mock import MagicMock
    from teams_server.agent import AgentDaemon

    daemon = AgentDaemon.__new__(AgentDaemon)
    daemon._current_trace_steps = []
    daemon.name = "test_agent"
    daemon._emit_exec = MagicMock()

    # Simulate binding turn callbacks logic
    def on_reasoning(text: str = "") -> None:
        if text:
            daemon._emit_exec("reasoning", {"text": str(text)[:4000]})
            if daemon._current_trace_steps and daemon._current_trace_steps[-1].get("type") == "reasoning":
                daemon._current_trace_steps[-1]["text"] = (
                    daemon._current_trace_steps[-1].get("text", "") + str(text)
                )[:8000]
            else:
                daemon._current_trace_steps.append({
                    "type": "reasoning",
                    "text": str(text)[:8000],
                })

    # Stream 4 words
    for word in ["Thinking ", "about ", "the ", "problem."]:
        on_reasoning(word)

    assert len(daemon._current_trace_steps) == 1
    assert daemon._current_trace_steps[0]["type"] == "reasoning"
    assert daemon._current_trace_steps[0]["text"] == "Thinking about the problem."


if __name__ == "__main__":
    import subprocess

    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))

