#!/usr/bin/env python3
"""Tests for slash-command parsing.

The load-bearing case is the path-paste guard: operators paste absolute paths
into the task box constantly, and treating `/Users/me/notes.md` as a failed
command would be worse than useless. Mirrors Hermes' own heuristic
(cli.py:_looks_like_slash_command).

Run:  pytest tests/test_commands_parse.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from teams_server import commands as C  # noqa: E402


# ---------------------------------------------------------------------------
# looks_like_slash_command -- "does this text INTEND to be a command?"
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "/stop",
    "/model gpt-5",
    "  /stop  ",          # leading whitespace tolerated
    "/STOP",              # case-insensitive
    "/steer focus on the API docs",
    "/stop\nnote",        # newline is a word boundary, same as split()
])
def test_recognized_as_command(text):
    assert C.looks_like_slash_command(text) is True


@pytest.mark.parametrize("text", [
    "/Users/pradhyun/notes.md",           # THE regression: pasted absolute path
    "/etc/hosts",
    "/usr/local/bin/python",
    "see /tmp/x then /model",             # slash mid-sentence, not leading
    "read the file /a/b and summarize",
    "/",                                  # bare slash: no command name
    "//",                                 # empty name after the slash
    "///",
    "",
    "   ",
    None,
    "not a command",
    "http://example.com",
    "50/50 split",
])
def test_not_recognized_as_command(text):
    assert C.looks_like_slash_command(text) is False


def test_path_paste_is_ordinary_text_not_a_failed_command():
    """A pasted path must flow through to the agent untouched.

    parse() returning None is the signal for 'ordinary prose'; ("", args) would
    mean 'command-shaped but unknown' and would get the operator a rejection.
    """
    assert C.parse("/Users/pradhyun/Movies/agent-teams/README.md") is None


# ---------------------------------------------------------------------------
# parse -- (canonical, args) | ("", args) for unknown | None for prose
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("/stop", ("stop", "")),
    ("/stop   ", ("stop", "")),
    ("/STOP", ("stop", "")),
    ("/Stop", ("stop", "")),
    ("/model gpt-5", ("model", "gpt-5")),
    ("/steer focus on the API docs", ("steer", "focus on the API docs")),
    ("  /usage  ", ("usage", "")),
    ("/q", ("queue", "")),                     # alias
    ("/steer   padded   args  ", ("steer", "padded   args")),
])
def test_parse_resolves(text, expected):
    assert C.parse(text) == expected


def test_parse_preserves_inner_whitespace_of_args():
    """/steer text is injected verbatim; collapsing whitespace would corrupt it."""
    name, args = C.parse("/steer line one\nline two")
    assert name == "steer"
    assert args == "line one\nline two"


def test_parse_unknown_command_signals_rejection():
    assert C.parse("/modle gpt-5") == ("", "gpt-5")
    assert C.parse("/definitelynotacommand") == ("", "")


def test_parse_returns_none_for_prose():
    assert C.parse("just do the thing") is None
    assert C.parse("") is None


# ---------------------------------------------------------------------------
# Whitespace agreement between the two functions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["/stop\nnote", "/stop\tnote", "/stop note"])
def test_parse_agrees_with_looks_like_on_any_whitespace(text):
    """Both must treat any whitespace as the boundary.

    They disagreed at first -- looks_like used str.split() (any whitespace) while
    parse used partition(" ") (literal space only), so "/stop\\nnote" was
    recognized as a command and then parsed with the newline glued to the name.
    """
    assert C.looks_like_slash_command(text) is True
    assert C.parse(text) == ("stop", "note")


# ---------------------------------------------------------------------------
# resolve / suggest
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["stop", "/stop", "STOP", " /Stop "])
def test_resolve_normalizes_input(name):
    assert C.resolve(name).name == "stop"


def test_resolve_rejects_unknown():
    assert C.resolve("nope") is None
    assert C.resolve("") is None
    assert C.resolve("/") is None
    assert C.resolve(None) is None


def test_suggest_finds_near_miss():
    assert C.suggest("/modle") == "model"
    assert C.suggest("/staus") == "status"
    assert C.suggest("/steeer") == "steer"


def test_suggest_gives_up_on_nonsense():
    assert C.suggest("/zzzzzzzz") is None
    assert C.suggest("") is None


def test_suggest_respects_scope():
    """/steer is agent-scoped, so a near-miss in the Architect must not offer it."""
    assert C.suggest("/steerr", "agent") == "steer"
    assert C.suggest("/steerr", "architect") is None


# ---------------------------------------------------------------------------
# unknown_result -- reject, never silently swallow
# ---------------------------------------------------------------------------
def test_unknown_result_is_handled_but_not_ok_and_not_forwarded():
    r = C.unknown_result("/modle gpt-5")
    assert r.handled is True      # we own it; caller must not also enqueue
    assert r.ok is False
    assert r.forwarded is False   # nothing reached the agent
    assert "/modle" in r.text
    assert "/model" in r.text     # did-you-mean


def test_unknown_result_falls_back_to_help_hint():
    r = C.unknown_result("/zzzzzzzz")
    assert r.ok is False
    assert "/help" in r.text


def test_unknown_result_survives_degenerate_input():
    # Never raises, whatever it is handed.
    for text in ("", "   ", "/", None):
        r = C.unknown_result(text)
        assert r.handled is True
        assert r.ok is False
