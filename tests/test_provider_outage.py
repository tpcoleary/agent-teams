#!/usr/bin/env python3
"""Provider-outage handling: circuit breaker + visible degraded state.

From the 2026-08-02 default-team run, where a provider outage produced 23
identical "Connection error." events between 01:11 and 10:07 — every one saying
the same thing, none of them a single actionable signal — while the agent sat in
state `idle`, which is indistinguishable from "finished its work and waiting".
It only stopped because a human happened to notice and paused the agent by hand
at 10:11.

Covers:
  1. CIRCUIT BREAKER  — consecutive infra failures open the circuit; the opening
                        is reported ONCE and further attempts stay quiet.
  2. CLASSIFICATION   — task-caused failures never count toward the circuit and
                        are always reported.
  3. BACKOFF          — exponential while the outage might be a blip, fixed slow
                        probe once the circuit is open (so recovery is still
                        noticed promptly hours in, rather than after an
                        ever-growing sleep).
  4. RESTING STATE    — degraded is visible and never masquerades as idle;
                        an explicit human pause outranks it.

Run:  pytest tests/test_provider_outage.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from teams_server.agent import (  # noqa: E402
    AGENT_STATE_DEGRADED,
    AGENT_STATE_IDLE,
    AGENT_STATE_PAUSED,
    INFRA_CIRCUIT_OPEN_AFTER,
    INFRA_CIRCUIT_PROBE_SECONDS,
    INFRA_RETRY_BACKOFF_BASE_SECONDS,
    INFRA_RETRY_BACKOFF_MAX_SECONDS,
    AgentDaemon,
)

# The exact error text from the observed outage.
OUTAGE_ERR = "Connection error."
# A deterministic, task-caused failure — says nothing about provider health.
TASK_ERR = "maximum context length exceeded"


class BreakerFake:
    """Minimal stand-in exposing exactly what the outage bookkeeping uses."""

    # staticmethod() matters: class access unwraps the descriptor to a plain
    # function, so a bare assignment would rebind it as an instance method and
    # pass `self` in the `err` slot.
    _is_infra_failure = staticmethod(AgentDaemon._is_infra_failure)
    _note_turn_failure = AgentDaemon._note_turn_failure
    _infra_backoff_seconds = AgentDaemon._infra_backoff_seconds
    _resting_state = AgentDaemon._resting_state

    def __init__(self):
        self.name = "w1"
        self._paused = False
        self._infra_misses = 0
        self._infra_circuit_open = False
        self._infra_outage_started_at = 0.0


# ---------------------------------------------------------------------------
# 1. Circuit breaker
# ---------------------------------------------------------------------------
def test_circuit_opens_after_threshold_consecutive_infra_failures():
    f = BreakerFake()
    for i in range(1, INFRA_CIRCUIT_OPEN_AFTER):
        out = f._note_turn_failure(OUTAGE_ERR)
        assert out["newly_opened"] is False, i
        assert f._infra_circuit_open is False, i
        assert out["report"] is True, "pre-threshold failures are still news"

    out = f._note_turn_failure(OUTAGE_ERR)
    assert out["newly_opened"] is True
    assert f._infra_circuit_open is True
    assert out["report"] is True, "the opening itself must be reported"


def test_open_circuit_goes_quiet():
    """The headline fix: a sustained outage stops re-reporting itself."""
    f = BreakerFake()
    for _ in range(INFRA_CIRCUIT_OPEN_AFTER):
        f._note_turn_failure(OUTAGE_ERR)
    for _ in range(20):
        out = f._note_turn_failure(OUTAGE_ERR)
        assert out["report"] is False
        assert out["newly_opened"] is False, "opens exactly once per outage"


def test_long_outage_reports_a_handful_not_one_per_attempt():
    """Encodes the actual regression: 23 attempts previously meant 23 error
    events. Now the count is bounded by the threshold plus the one opening."""
    f = BreakerFake()
    reports = sum(f._note_turn_failure(OUTAGE_ERR)["report"] for _ in range(23))
    assert reports == INFRA_CIRCUIT_OPEN_AFTER
    assert reports < 23


def test_outage_start_is_stamped_once():
    f = BreakerFake()
    f._note_turn_failure(OUTAGE_ERR)
    first = f._infra_outage_started_at
    assert first > 0
    for _ in range(5):
        f._note_turn_failure(OUTAGE_ERR)
    assert f._infra_outage_started_at == first, "stamp marks the outage, not the attempt"


# ---------------------------------------------------------------------------
# 2. Classification
# ---------------------------------------------------------------------------
def test_task_caused_failure_never_opens_the_circuit():
    """A poison task must not be mistaken for a provider outage — otherwise it
    would hold the whole agent instead of dead-lettering."""
    f = BreakerFake()
    for _ in range(INFRA_CIRCUIT_OPEN_AFTER + 5):
        out = f._note_turn_failure(TASK_ERR)
        assert out["infra"] is False
        assert out["report"] is True, "task failures are always news"
    assert f._infra_circuit_open is False


def test_task_failure_resets_the_infra_streak():
    """The circuit is about CONSECUTIVE provider failures. A turn that reached
    the provider and failed on its own merits proves the provider is up."""
    f = BreakerFake()
    for _ in range(INFRA_CIRCUIT_OPEN_AFTER - 1):
        f._note_turn_failure(OUTAGE_ERR)
    assert f._infra_misses == INFRA_CIRCUIT_OPEN_AFTER - 1

    f._note_turn_failure(TASK_ERR)
    assert f._infra_misses == 0
    # so the next infra failure starts a fresh streak rather than tipping it over
    out = f._note_turn_failure(OUTAGE_ERR)
    assert out["newly_opened"] is False
    assert f._infra_circuit_open is False


def test_non_infra_failures_are_always_reported_even_with_circuit_open():
    f = BreakerFake()
    for _ in range(INFRA_CIRCUIT_OPEN_AFTER):
        f._note_turn_failure(OUTAGE_ERR)
    assert f._infra_circuit_open is True
    out = f._note_turn_failure(TASK_ERR)
    assert out["report"] is True


# ---------------------------------------------------------------------------
# 3. Backoff
# ---------------------------------------------------------------------------
def test_backoff_is_exponential_before_the_circuit_opens():
    f = BreakerFake()
    f._note_turn_failure(OUTAGE_ERR)
    assert f._infra_backoff_seconds() == INFRA_RETRY_BACKOFF_BASE_SECONDS
    f._note_turn_failure(OUTAGE_ERR)
    assert f._infra_backoff_seconds() == INFRA_RETRY_BACKOFF_BASE_SECONDS * 2


def test_backoff_settles_on_a_fixed_probe_once_open():
    """Not an ever-growing sleep: during a long outage an unbounded backoff
    means recovery goes unnoticed for however long the last sleep was."""
    f = BreakerFake()
    for _ in range(INFRA_CIRCUIT_OPEN_AFTER + 10):
        f._note_turn_failure(OUTAGE_ERR)
    assert f._infra_backoff_seconds() == INFRA_CIRCUIT_PROBE_SECONDS


def test_backoff_never_exceeds_the_cap():
    f = BreakerFake()
    f._infra_misses = 99
    f._infra_circuit_open = False
    assert f._infra_backoff_seconds() == INFRA_RETRY_BACKOFF_MAX_SECONDS


# ---------------------------------------------------------------------------
# 4. Resting state
# ---------------------------------------------------------------------------
def test_degraded_is_not_idle():
    """The ambiguity that hid a 9-hour outage: held-for-outage looked exactly
    like finished-and-waiting."""
    f = BreakerFake()
    assert f._resting_state() == AGENT_STATE_IDLE
    for _ in range(INFRA_CIRCUIT_OPEN_AFTER):
        f._note_turn_failure(OUTAGE_ERR)
    assert f._resting_state() == AGENT_STATE_DEGRADED
    assert f._resting_state() != AGENT_STATE_IDLE


def test_explicit_pause_outranks_degraded():
    """The operator's intent is the more important fact to show."""
    f = BreakerFake()
    for _ in range(INFRA_CIRCUIT_OPEN_AFTER):
        f._note_turn_failure(OUTAGE_ERR)
    f._paused = True
    assert f._resting_state() == AGENT_STATE_PAUSED


def test_below_threshold_still_reads_as_idle():
    """A single blip shouldn't paint the agent red."""
    f = BreakerFake()
    f._note_turn_failure(OUTAGE_ERR)
    assert f._resting_state() == AGENT_STATE_IDLE


def test_recovery_returns_to_idle():
    """Mirrors what the success path in _process_tasks_batch resets."""
    f = BreakerFake()
    for _ in range(INFRA_CIRCUIT_OPEN_AFTER):
        f._note_turn_failure(OUTAGE_ERR)
    assert f._resting_state() == AGENT_STATE_DEGRADED

    f._infra_misses = 0
    f._infra_circuit_open = False
    f._infra_outage_started_at = 0.0
    assert f._resting_state() == AGENT_STATE_IDLE


# ---------------------------------------------------------------------------
# The infra/task split these all rest on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("err,is_infra", [
    ("Connection error.", True),
    ("connection reset by peer", True),
    ("Read timed out", True),
    ("maximum context length exceeded", False),
    ("LLM call failed after 3 attempts: maximum context length exceeded", False),
    ("invalid_request: bad model", False),
    ("content policy violation", False),
])
def test_infra_classification(err, is_infra):
    assert AgentDaemon._is_infra_failure(err) is is_infra


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
