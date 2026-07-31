#!/usr/bin/env python3
"""Tests for graceful degradation when Hermes' command registry is unavailable.

The teams rides Hermes without an upper version bound (see hermes_compat.py), so
`hermes_cli.commands` can vanish or change shape under us. When it does, every
command Teams implements must still work -- dispatch keys off the local _SPEC,
not the registry. What degrades is cosmetic: inherited wording and alias breadth.

Because commands.py resolves Hermes lazily, these tests just swap sys.modules and
call reset_cache() -- no module reload, so no risk of leaking a half-initialized
module into other test files.

Run:  pytest tests/test_commands_degradation.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from teams_server import commands as C  # noqa: E402

MODULE = "hermes_cli.commands"

# Every command that must survive Hermes going away.
V1 = ["steer", "stop", "usage", "status", "queue", "model", "pause", "resume",
      "help", "goal", "cron"]


@pytest.fixture()
def sabotage():
    """Point commands.py at a fake registry; always restore.

    Yields a callable: pass None to make the import fail outright (Python raises
    ImportError when sys.modules holds None), or a stub module object to simulate
    upstream shape drift.
    """
    saved = sys.modules.get(MODULE)
    had = MODULE in sys.modules

    def _apply(registry_module):
        sys.modules[MODULE] = registry_module
        C.reset_cache()
        return C

    yield _apply

    if had:
        sys.modules[MODULE] = saved
    else:
        sys.modules.pop(MODULE, None)
    C.reset_cache()


# ---------------------------------------------------------------------------
# Registry entirely missing
# ---------------------------------------------------------------------------
def test_catalog_survives_missing_registry(sabotage):
    mod = sabotage(None)
    assert mod.hermes_available() is False
    assert mod.catalog(), "catalog empty without Hermes"
    assert {c.name for c in mod.catalog()} == set(V1)


def test_all_v1_commands_still_resolve_without_hermes(sabotage):
    mod = sabotage(None)
    for name in V1:
        assert mod.resolve(name) is not None, f"/{name} stopped resolving"
        assert mod.parse(f"/{name}") == (name, "")


def test_descriptions_stay_human_readable_without_hermes(sabotage):
    """Inherited commands fall back to a local string, not the bare name.

    Without the 'fallback' rung in _SPEC, /steer would render as "steer — steer"
    in the autocomplete.
    """
    mod = sabotage(None)
    for c in mod.catalog():
        assert c.description != c.name, f"/{c.name} degraded to a bare name"
        assert len(c.description) > len(c.name), f"/{c.name} description too thin"


def test_parsing_unaffected_by_missing_registry(sabotage):
    mod = sabotage(None)
    assert mod.looks_like_slash_command("/model gpt-5") is True
    assert mod.looks_like_slash_command("/Users/me/notes.md") is False
    assert mod.parse("/steer do the thing") == ("steer", "do the thing")
    assert mod.suggest("/modle") == "model"


def test_payload_reports_degraded_state(sabotage):
    """The dashboard uses this flag to hint that wording/aliases are reduced."""
    mod = sabotage(None)
    payload = mod.catalog_payload("agent")
    assert payload["ok"] is True
    assert payload["hermes"] is False
    assert payload["commands"]


def test_dispatch_still_works_without_hermes(sabotage):
    """The point of the whole design: behavior does not depend on Hermes.

    Uses asyncio.run rather than @pytest.mark.asyncio -- pytest-asyncio is not a
    declared dependency of this project and no other test needs it, so relying on
    it would make this test silently no-op on a fresh clone.
    """
    import asyncio

    mod = sabotage(None)

    class _Q:
        def get_pending_count(self):
            return 0

    class _D:
        state = "idle"
        _paused = False
        cfg = {"model": "gpt-5"}
        queue = _Q()

    ctx = mod.CommandContext(scope="agent", agent_name="a", daemon=_D())

    r = asyncio.run(mod.dispatch("/status", ctx))
    assert r.handled is True and r.ok is True

    r = asyncio.run(mod.dispatch("/help", ctx))
    assert r.handled is True and r.ok is True
    # And the degraded-mode hint reaches the operator.
    assert "registry unavailable" in r.text


def test_hermes_only_aliases_are_the_documented_casualty(sabotage):
    """`/q` comes from Hermes' registry, so it is lost -- by design.

    Asserted rather than glossed over: this is the exact scope of the
    degradation. The canonical /queue keeps working, which is what matters.
    """
    assert C.resolve("q").name == "queue"      # with Hermes
    mod = sabotage(None)
    assert mod.resolve("q") is None            # without
    assert mod.resolve("queue") is not None    # canonical still fine


# ---------------------------------------------------------------------------
# Registry present but wrong shape
# ---------------------------------------------------------------------------
class _EmptyRegistry:
    COMMAND_REGISTRY = ()

    @staticmethod
    def resolve_command(name):
        return None


class _Partial:
    """A CommandDef that lost fields we read -- the realistic drift case."""

    def __init__(self, name, description):
        self.name = name
        self.description = description
        # no aliases, args_hint, subcommands, category


class _DriftedRegistry:
    COMMAND_REGISTRY = (_Partial("steer", "Drifted steer text"),)

    @staticmethod
    def resolve_command(name):
        return None


class _ExplodingRegistry:
    COMMAND_REGISTRY = (_Partial("steer", "text"),)

    @staticmethod
    def resolve_command(name):
        raise RuntimeError("registry blew up")


def test_empty_registry_reports_degraded(sabotage):
    mod = sabotage(_EmptyRegistry)
    assert mod.hermes_available() is False
    assert {c.name for c in mod.catalog()} == set(V1)


def test_commanddef_missing_fields_does_not_break_catalog(sabotage):
    mod = sabotage(_DriftedRegistry)
    assert mod.catalog()
    steer = mod.by_name()["steer"]
    assert steer.aliases == ()
    assert steer.subcommands == ()
    assert steer.category  # defaulted, not blank
    # Inheritance still works off the surviving field.
    assert steer.description == "Drifted steer text"


def test_resolve_command_raising_is_contained(sabotage):
    """A registry that raises must not take down resolution of our own commands."""
    mod = sabotage(_ExplodingRegistry)
    assert mod.resolve("stop") is not None          # local table, never touches Hermes
    assert mod.resolve("totally-unknown") is None   # exception swallowed, no raise


def test_overrides_win_even_if_hermes_rewords(sabotage):
    """Teams semantics are not at the mercy of upstream wording."""

    class Reworded:
        COMMAND_REGISTRY = (_Partial("stop", "Some new upstream meaning"),)

        @staticmethod
        def resolve_command(name):
            return None

    mod = sabotage(Reworded)
    assert mod.by_name()["stop"].description != "Some new upstream meaning"
    assert "nterrupt" in mod.by_name()["stop"].description


# ---------------------------------------------------------------------------
# The fixture's own guarantee
# ---------------------------------------------------------------------------
def test_registry_is_restored_after_sabotage():
    """Proves the sabotage above did not leak into the rest of the session."""
    assert C.hermes_available() is True
    assert C.resolve("q").name == "queue"
