#!/usr/bin/env python3
"""Ping-pong loop detection over WAKING peer messages.

Exists because STATUS now wakes its recipient. STATUS used to be passive, and
that passivity WAS the anti-ping-pong guarantee — "an agent literally cannot
wake a peer just to confirm a status". Waking it reopens that door, and this
detector is what has to catch an A<->B acknowledgement loop instead. It had no
tests at all, which is a bad place to put a mitigation.

Covers:
  1. STATUS is loop-capable      — the newly-reopened risk is actually caught.
  2. FYI is not                  — the one passive kind can't be a loop.
  3. Shape guards                — needs volume AND both directions AND repetition,
                                   so normal collaboration isn't flagged.

Run:  pytest tests/test_loop_detector.py -v
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from teams_server.config import LOOP_PAIR_THRESHOLD  # noqa: E402
from teams_server.loop_detector import (  # noqa: E402
    _detect_pair_pingpong,
    _norm,
)


def _msgs(n, kind, preview="status: still working on it", a="w1", b="w2",
          both_ways=True, previews=None):
    """n messages between a and b, alternating direction by default."""
    out = []
    for i in range(n):
        frm, to = (a, b) if (i % 2 == 0 or not both_ways) else (b, a)
        out.append({
            "ts": 1000 + i,
            "from": frm,
            "to": to,
            "kind": kind,
            "preview": previews[i] if previews else preview,
        })
    return out


# ---------------------------------------------------------------------------
# 1. STATUS is loop-capable
# ---------------------------------------------------------------------------
def test_status_pingpong_is_detected():
    """The whole reason this file exists: STATUS wakes now, so a STATUS
    acknowledgement loop is a real loop and must be caught."""
    hit = _detect_pair_pingpong(_msgs(LOOP_PAIR_THRESHOLD, "STATUS"))
    assert hit is not None
    assert hit["pair"] == ["w1", "w2"]
    assert hit["count"] == LOOP_PAIR_THRESHOLD
    assert hit["dup_ratio"] > 0.4


def test_question_and_result_pingpong_still_detected():
    for kind in ("QUESTION", "RESULT"):
        assert _detect_pair_pingpong(_msgs(LOOP_PAIR_THRESHOLD, kind)) is not None, kind


def test_legacy_task_kind_still_detected():
    """Historical rows predate the move of assignment to create_task."""
    assert _detect_pair_pingpong(_msgs(LOOP_PAIR_THRESHOLD, "TASK")) is not None


def test_mixed_waking_kinds_count_together():
    """A loop dressed up by alternating kinds is still one loop."""
    msgs = _msgs(LOOP_PAIR_THRESHOLD, "STATUS")
    for i, m in enumerate(msgs):
        m["kind"] = ("STATUS", "QUESTION", "RESULT")[i % 3]
    assert _detect_pair_pingpong(msgs) is not None


# ---------------------------------------------------------------------------
# 2. FYI is not
# ---------------------------------------------------------------------------
def test_fyi_flood_is_not_a_loop():
    """FYI is the deliberate passive kind — it costs the recipient no turn, so
    volume alone is not a loop."""
    assert _detect_pair_pingpong(_msgs(LOOP_PAIR_THRESHOLD * 3, "FYI")) is None


def test_fyi_does_not_pad_a_real_loop_over_the_threshold():
    """Passive traffic must not be counted toward the waking threshold."""
    msgs = _msgs(LOOP_PAIR_THRESHOLD - 1, "STATUS")
    msgs += _msgs(6, "FYI", a="w1", b="w2")
    assert _detect_pair_pingpong(msgs) is None


# ---------------------------------------------------------------------------
# 3. Shape guards — don't flag healthy collaboration
# ---------------------------------------------------------------------------
def test_below_threshold_is_not_a_loop():
    assert _detect_pair_pingpong(_msgs(LOOP_PAIR_THRESHOLD - 1, "STATUS")) is None


def test_one_way_fanout_is_not_a_loop():
    """A ping-pong needs both directions; a broadcaster is not looping."""
    msgs = _msgs(LOOP_PAIR_THRESHOLD * 2, "STATUS", both_ways=False)
    assert _detect_pair_pingpong(msgs) is None


def test_distinct_messages_are_not_a_loop():
    """High volume with genuinely different content is real work."""
    previews = [f"finished module {i} and moving to the next one" for i in range(8)]
    msgs = _msgs(8, "STATUS", previews=previews)
    assert _detect_pair_pingpong(msgs) is None


def test_near_duplicates_are_treated_as_repetition():
    """Politely reworded re-acknowledgements are the actual failure mode, so
    normalisation has to collapse them."""
    previews = [
        "Acknowledged, still working on it.",
        "acknowledged - still working on it",
        "Acknowledged, still working on it!",
        "ACKNOWLEDGED: still working on it",
        "Acknowledged, still working on it.",
        "acknowledged   still working on it",
    ]
    hit = _detect_pair_pingpong(_msgs(6, "STATUS", previews=previews))
    assert hit is not None


def test_messages_without_previews_are_not_flagged():
    """No text means nothing to judge repetition by — don't guess."""
    assert _detect_pair_pingpong(_msgs(LOOP_PAIR_THRESHOLD, "STATUS", preview="")) is None


def test_worst_pair_is_reported_when_several_loop():
    a = _msgs(LOOP_PAIR_THRESHOLD, "STATUS", a="w1", b="w2")
    b = _msgs(LOOP_PAIR_THRESHOLD + 4, "STATUS", a="w3", b="w4")
    hit = _detect_pair_pingpong(a + b)
    assert hit["pair"] == ["w3", "w4"]


def test_latest_sender_is_identified():
    """Who to steer: the one who spoke last."""
    hit = _detect_pair_pingpong(_msgs(LOOP_PAIR_THRESHOLD, "STATUS"))
    # LOOP_PAIR_THRESHOLD messages alternate w1,w2,... so the last index decides
    expected = "w1" if (LOOP_PAIR_THRESHOLD - 1) % 2 == 0 else "w2"
    assert hit["latest_from"] == expected


# ---------------------------------------------------------------------------
# _norm, which the repetition ratio rests on
# ---------------------------------------------------------------------------
def test_norm_collapses_case_and_punctuation():
    assert _norm("Still Working On It!") == _norm("still working on it")


def test_norm_drops_tool_result_suffix():
    """A tool RESULT arrow makes two identical intents look different."""
    assert _norm("checking the deploy → HTTP 200") == _norm("checking the deploy")


def test_norm_handles_empty_and_none():
    assert _norm("") == ""
    assert _norm(None) == ""


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
