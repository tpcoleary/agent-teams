#!/usr/bin/env python3
"""Tests for the slash-command catalog.

The catalog is the dashboard's autocomplete data source, so these tests mostly
guard against *lying to the operator*: advertising a command we do not
implement, or inheriting Hermes wording/flags that do not apply to Teams.

Runs against the REAL installed Hermes, matching tests/test_hermes_compat.py --
a Hermes release that reworks a description should surface here.

Run:  pytest tests/test_commands_catalog.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from teams_server import commands as C  # noqa: E402


# Commands deliberately absent. Not "not yet" -- each is infeasible or would be
# dishonest on this architecture; see the block comment under _SPEC in
# teams_server/commands.py for the per-command reason.
DEFERRED = ["compress", "new", "soul"]
DROPPED = ["undo", "retry", "title", "background", "reasoning"]


def test_catalog_non_empty_and_well_formed():
    cat = C.catalog()
    assert cat, "catalog is empty"
    for c in cat:
        assert c.name and not c.name.startswith("/"), f"bad name {c.name!r}"
        assert c.description.strip(), f"/{c.name} has no description"
        assert c.scope in ("agent", "architect", "both"), f"/{c.name} scope={c.scope!r}"
        assert c.category.strip(), f"/{c.name} has no category"


def test_every_catalog_entry_is_implemented():
    """No advertised-but-unimplemented commands: catalog ⊆ _SPEC."""
    spec_names = set(C._SPEC)
    for c in C.catalog():
        assert c.name in spec_names, f"/{c.name} advertised but not in _SPEC"


@pytest.mark.parametrize("name", DEFERRED + DROPPED)
def test_unimplemented_commands_absent(name):
    assert name not in {c.name for c in C.catalog()}, f"/{name} should not be advertised"
    assert C.resolve(name) is None, f"/{name} should not resolve"


def test_hermes_aliases_for_unimplemented_commands_do_not_resolve():
    """`/compact` is a Hermes alias for `/compress`, which we defer.

    It must resolve to nothing rather than half-resolving to a command with no
    handler -- the operator gets a clear 'unknown command' instead of a 500.
    """
    assert C.resolve("compact") is None
    assert C.parse("/compact") == ("", "")


def test_scope_partitions_by_surface():
    agent = {c.name for c in C.catalog("agent")}
    architect = {c.name for c in C.catalog("architect")}

    # Agent-only controls must not appear in the Architect chat: the Architect is
    # not a team member and has no turn to steer or interrupt.
    assert "steer" not in architect
    assert "stop" not in architect
    assert "pause" not in architect

    # Shared informational commands appear on both.
    assert "help" in agent and "help" in architect
    assert "status" in agent and "status" in architect

    assert "steer" in agent


def test_unknown_scope_falls_back_to_both():
    assert {c.name for c in C.catalog("nonsense")} == {c.name for c in C.catalog("both")}


def test_alias_round_trip():
    # /q is Hermes' alias for /queue and we keep it.
    assert C.resolve("q").name == "queue"
    assert C.parse("/q") == ("queue", "")


# ---------------------------------------------------------------------------
# Anti-drift: Teams semantics must override Hermes wording where they diverge.
# If Hermes rewords one of these, the override keeps the dashboard honest.
# ---------------------------------------------------------------------------
COLLIDING = ["stop", "status", "model", "resume", "queue", "usage", "goal", "cron"]


def _require_hermes():
    """Skip from inside the test body, not via @skipif.

    A skipif decorator is evaluated at collection time, which would force the
    Hermes import back onto the module-import path -- the exact thing
    commands.py now defers (see its _load_hermes docstring).
    """
    if not C.hermes_available():
        pytest.skip("Hermes registry unavailable")


@pytest.mark.parametrize("name", COLLIDING)
def test_colliding_descriptions_are_overridden(name):
    """These names exist in both systems with *different* meanings."""
    _require_hermes()
    entry = C._hermes_entry(name)
    assert entry is not None, f"expected /{name} in the Hermes registry"
    ours = C.by_name()[name].description
    assert ours != entry.description, (
        f"/{name} inherits Hermes' description but the semantics differ — "
        f"Teams must override it"
    )


def test_colliding_arg_hints_do_not_advertise_hermes_flags():
    """Overriding the description is not enough -- the arg hint documents flags.

    Hermes' /model offers --global|--session scoping, /usage offers
    `reset [--force]` for a banked Codex limit reset, and /queue takes <prompt>
    because it *queues* one. None of that exists in Teams.
    """
    _require_hermes()
    assert "--global" not in C.by_name()["model"].args_hint
    assert "--session" not in C.by_name()["model"].args_hint
    assert "reset" not in C.by_name()["usage"].args_hint
    # Teams' /queue lists the queue; it takes no prompt.
    assert "prompt" not in C.by_name()["queue"].args_hint


def test_inherited_descriptions_match_hermes():
    """/steer and /help mean the same thing in both systems, so inherit.

    This is the counterweight to the override tests: it proves we are decorating
    from the registry rather than hardcoding everything locally.
    """
    _require_hermes()
    for name in ("steer", "help"):
        entry = C._hermes_entry(name)
        assert entry is not None
        assert C.by_name()[name].description == entry.description


def test_native_commands_have_local_metadata():
    """/pause has no Hermes registry entry, so _SPEC must supply everything."""
    assert C._hermes_entry("pause") is None
    pause = C.by_name()["pause"]
    assert pause.description.strip()
    assert pause.category.strip()


# ---------------------------------------------------------------------------
# Wire shape for GET /commands
# ---------------------------------------------------------------------------
def test_catalog_payload_shape():
    payload = C.catalog_payload("agent")
    assert payload["ok"] is True
    assert isinstance(payload["hermes"], bool)
    assert payload["commands"]
    for entry in payload["commands"]:
        assert set(entry) == {
            "name", "description", "scope", "aliases",
            "args_hint", "subcommands", "category",
        }
        # JSON-serializable: lists, not tuples.
        assert isinstance(entry["aliases"], list)
        assert isinstance(entry["subcommands"], list)


def test_catalog_payload_is_json_serializable():
    import json
    json.dumps(C.catalog_payload("both"))
