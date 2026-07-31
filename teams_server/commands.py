"""Slash commands for human operators.

Hermes' CLI accepts ``/`` commands; this module brings the same vocabulary to the
Agent Teams dashboard so an operator can read usage, steer a turn, or interrupt a
stuck agent without burning an LLM turn on it.

**Hermes supplies names, not behavior.** ``hermes_cli.commands.COMMAND_REGISTRY``
is UI-free and importable (its ``prompt_toolkit`` import is optional), but every
Hermes front-end -- ``cli.py``, ``gateway/slash_commands.py``, ``tui_gateway`` --
reimplements the handlers. There is no dispatch to import. So we borrow the
registry for *metadata* (aliases, arg hints, wording) and own the behavior, the
same way ``tui_gateway`` does.

``_SPEC`` below is authoritative about what exists. The registry only decorates
it, and ``_SPEC`` overrides the registry wherever Teams semantics diverge -- see
the ``# COLLISION`` markers. Getting that backwards would make the dashboard's
autocomplete advertise flags Teams does not have and behavior it does not do.

Losing Hermes entirely costs inherited wording and alias breadth, never
functionality: dispatch keys off ``_SPEC``, so every command still works.
"""

from __future__ import annotations

import difflib
import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("teams.commands")


# --------------------------------------------------------------------------- #
# Hermes registry -- optional AND lazily imported.
#
# Lazy is deliberate, not incidental. The server keeps Hermes imports off the
# module-import path by convention (tests/test_auth.py: "server import keeps
# Hermes lazy"), and tests/test_compaction.py pushes a second Hermes checkout
# onto sys.path during collection. Importing at module load bound this catalog to
# whichever copy happened to be first and shifted import order for everything
# after it. Resolving on first use keeps that out of the import graph.
#
# See hermes_compat.py's slash_command_registry probe, which turns a silent shape
# change here into a loud `doctor` warning.
# --------------------------------------------------------------------------- #
_registry: Tuple = ()
_resolver = None
_hermes_ok: Optional[bool] = None
_loaded = False


def _load_hermes() -> None:
    """Resolve Hermes' registry once, on first use. Never raises."""
    global _registry, _resolver, _hermes_ok, _loaded
    if _loaded:
        return
    _loaded = True
    try:
        from hermes_cli.commands import COMMAND_REGISTRY, resolve_command
        _registry = tuple(COMMAND_REGISTRY or ())
        _resolver = resolve_command
        _hermes_ok = bool(_registry)
        if not _hermes_ok:
            log.info("Hermes COMMAND_REGISTRY is empty — using built-in catalog")
    except Exception as e:
        log.info("Hermes command registry unavailable (%s) — using built-in catalog", e)
        _registry, _resolver, _hermes_ok = (), None, False


def hermes_available() -> bool:
    """True iff Hermes' registry supplied the catalog's metadata."""
    _load_hermes()
    return bool(_hermes_ok)


def reset_cache() -> None:
    """Drop cached Hermes state and the built catalog. For tests."""
    global _registry, _resolver, _hermes_ok, _loaded, _CATALOG, _BY_NAME, _BY_ALIAS
    _registry, _resolver, _hermes_ok, _loaded = (), None, None, False
    _CATALOG, _BY_NAME, _BY_ALIAS = None, None, None


# --------------------------------------------------------------------------- #
# What Teams actually implements.
#
# Keys present here override the Hermes registry entry of the same name; keys
# absent are inherited. ``native=True`` means Hermes has no such command, so
# every field must be supplied locally.
#
# Scope partitions the catalog per surface: the task box and inbox reply serve
# {agent, both}; the Architect chat serves {architect, both}.
# --------------------------------------------------------------------------- #
_SPEC: Dict[str, dict] = {
    # ---- inherited wording: Hermes' description is already accurate for us.
    # "fallback" is used ONLY when the registry is unavailable, so inheritance
    # still works normally but a Hermes-less install does not show "steer —
    # steer" in the autocomplete.
    "steer": {
        "scope": "agent",
        "fallback": "Inject a message after the next tool call without interrupting",
    },
    "help": {"scope": "both", "fallback": "Show available commands"},

    # ---- COLLISION: Hermes "kill background processes" -> Teams interrupt turn.
    "stop": {
        "scope": "agent",
        "description": "Interrupt the agent's in-flight turn and drain pending tasks",
        "args_hint": "",
    },
    # ---- COLLISION: Hermes shows *session* info -> Teams shows agent/team health.
    "status": {
        "scope": "both",
        "description": "Show agent state, queue depth, model, and cost",
        "args_hint": "",
    },
    # ---- COLLISION: Hermes switches the live session model with
    # --global/--session scoping. Teams patches the agent's config instead
    # (switch_model rebuilds clients — unsafe mid-turn), so those flags do not
    # exist here and must not be advertised.
    "model": {
        "scope": "agent",
        "description": "Show or change the agent's model",
        "args_hint": "[model]",
    },
    # ---- COLLISION: Hermes *queues a prompt*; Teams *lists the queue*. Opposite
    # meanings, so both the wording and the arg hint have to be replaced.
    "queue": {
        "scope": "agent",
        "description": "List the agent's pending tasks",
        "args_hint": "",
    },
    # ---- COLLISION: Hermes resumes a saved session -> Teams un-pauses an agent.
    "resume": {
        "scope": "agent",
        "description": "Resume a paused agent",
        "args_hint": "",
    },
    # ---- COLLISION: Hermes' hint advertises `reset [--force]` for redeeming a
    # banked Codex rate-limit reset. Teams has no such concept.
    "usage": {
        "scope": "both",
        "description": "Show this agent's token usage and cost",
        "args_hint": "",
    },

    # ---- Teams-native: no Hermes registry entry exists for these ----
    "pause": {
        "scope": "agent",
        "native": True,
        "category": "Session",
        "description": "Pause the agent (in-flight work is held, not lost)",
        "args_hint": "",
    },

    # ---- COLLISION: Hermes' /goal is a whole subsystem (hermes_cli/goals.py,
    # ~80KB, its own persistence). Teams already has the equivalent primitive --
    # a time-boxed directive that idle heartbeats re-present -- so /goal maps
    # onto that instead of grafting the upstream one.
    "goal": {
        "scope": "agent",
        "description": "Set a standing directive the agent keeps working on",
        "args_hint": "[show | clear | <text> [--hours N]]",
        "subcommands": ("show", "clear"),
    },
    # ---- /cron is cli_only upstream, which is exactly why _SPEC (not the
    # cli_only flag) decides membership. Read-only here: creating a schedule
    # needs validation and a real form, which the agent's config panel already
    # provides.
    "cron": {
        "scope": "agent",
        "description": "List the agent's scheduled wake-ups",
        "args_hint": "",
        "subcommands": (),
    },
}

# Commands whose handlers are not implemented are deliberately absent from
# _SPEC: the catalog *is* the autocomplete source, and offering a completion that
# is guaranteed to fail is worse than omitting it.
#
# Dropped as infeasible or dishonest on this architecture:
#   /undo, /retry  - Hermes implements these against the gateway's own session
#                    store (rewind_session / rewrite_transcript). Nothing
#                    equivalent is exposed on AIAgent and Teams has no
#                    transcript-truncation primitive.
#   /title         - Teams agents are long-lived named daemons; there is no
#                    per-session title to set.
#   /background    - the gateway spawns a second AIAgent on a thread; Teams
#                    already covers this with peer delegation and one-shot crons.
#   /compress      - real compaction is Hermes' private _compress_context, which
#                    needs (messages, system_message), performs an LLM call, and
#                    is unsafe to touch while the executor thread is inside
#                    run_conversation. Teams' summarizer.py is NOT a substitute:
#                    it writes a rolling dashboard digest from monitor_db and
#                    does not shrink the agent's live context at all, so wiring
#                    /compress to it would report success while changing nothing.
#   /new           - reset_session_state() exists but mutates session_* and
#                    rebinds the compressor; it needs to be deferred onto the
#                    agent's executor thread rather than run from a request.
#   /soul          - rewriting an agent's charter is a deliberate, reviewable
#                    edit; a one-line command is the wrong affordance for it
#                    (the config panel already does this well).

_SCOPES = ("agent", "architect", "both")


@dataclass(frozen=True)
class TeamsCommand:
    """One dashboard-completable command."""

    name: str
    description: str
    scope: str
    aliases: Tuple[str, ...] = ()
    args_hint: str = ""
    subcommands: Tuple[str, ...] = ()
    category: str = "Session"

    def matches_scope(self, scope: str) -> bool:
        return self.scope == "both" or scope == "both" or self.scope == scope


@dataclass
class CommandResult:
    """Outcome of a dispatch attempt.

    ``handled`` distinguishes "this was a command" from "this was ordinary text",
    so callers know whether to fall through to their normal enqueue path.
    ``forwarded`` says whether anything actually reached the agent, which the
    dashboard uses to decide if it should expect a reply.
    """

    handled: bool
    ok: bool = True
    text: str = ""
    forwarded: bool = False


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #
def _hermes_entry(name: str):
    _load_hermes()
    for c in _registry:
        if getattr(c, "name", None) == name:
            return c
    return None


def build_catalog() -> List[TeamsCommand]:
    """Build the command catalog: _SPEC decorated by the Hermes registry.

    Note we deliberately do NOT filter on ``cli_only``/``gateway_only``. Those
    flags describe which *Hermes* front-end a command suits, and some we want
    are flagged for surfaces we are not (``/cron`` is ``cli_only``). _SPEC
    decides membership; the flags are metadata.
    """
    out: List[TeamsCommand] = []
    for name, spec in _SPEC.items():
        entry = None if spec.get("native") else _hermes_entry(name)

        if entry is None and not spec.get("native") and hermes_available():
            # In _SPEC, not native, but the registry has no such name: Hermes
            # renamed or removed it. We still own the behavior, so ship it with
            # local metadata and let the compat probe raise the alarm.
            log.warning("Command /%s not found in Hermes registry — using local metadata", name)

        # Precedence: explicit override > inherited from Hermes > degraded-mode
        # fallback > bare name. The fallback rung exists so a Hermes-less install
        # does not render "steer — steer" in the autocomplete.
        description = (
            spec.get("description")
            or getattr(entry, "description", "")
            or spec.get("fallback")
            or name
        )
        out.append(TeamsCommand(
            name=name,
            description=description,
            scope=spec.get("scope", "agent"),
            aliases=tuple(spec.get("aliases", getattr(entry, "aliases", ()) or ())),
            args_hint=spec.get("args_hint", getattr(entry, "args_hint", "") or ""),
            subcommands=tuple(spec.get("subcommands", getattr(entry, "subcommands", ()) or ())),
            category=spec.get("category", getattr(entry, "category", "Session") or "Session"),
        ))
    out.sort(key=lambda c: c.name)
    return out


_CATALOG: Optional[List[TeamsCommand]] = None
_BY_NAME: Optional[Dict[str, TeamsCommand]] = None
_BY_ALIAS: Optional[Dict[str, str]] = None


def _ensure_catalog() -> None:
    """Build the catalog and lookup tables once, on first use."""
    global _CATALOG, _BY_NAME, _BY_ALIAS
    if _CATALOG is not None:
        return
    _CATALOG = build_catalog()
    _BY_NAME = {c.name: c for c in _CATALOG}
    # Teams-native aliases resolve before any Hermes alias, so a future Hermes
    # release that adds an alias colliding with one of our names cannot hijack it.
    _BY_ALIAS = {a: c.name for c in _CATALOG for a in c.aliases}


def by_name() -> Dict[str, TeamsCommand]:
    _ensure_catalog()
    return _BY_NAME


def catalog(scope: str = "both") -> List[TeamsCommand]:
    """The catalog, optionally narrowed to one surface."""
    if scope not in _SCOPES:
        scope = "both"
    _ensure_catalog()
    return [c for c in _CATALOG if c.matches_scope(scope)]


def catalog_payload(scope: str = "both") -> dict:
    """JSON shape for ``GET /commands``.

    ``hermes`` lets the dashboard hint that it is running on the built-in
    catalog with Hermes' wording and aliases unavailable.
    """
    return {
        "ok": True,
        "hermes": hermes_available(),
        "commands": [
            {
                "name": c.name,
                "description": c.description,
                "scope": c.scope,
                "aliases": list(c.aliases),
                "args_hint": c.args_hint,
                "subcommands": list(c.subcommands),
                "category": c.category,
            }
            for c in catalog(scope)
        ],
    }


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def looks_like_slash_command(text: str) -> bool:
    """Does this text *intend* to be a command?

    Mirrors Hermes' own heuristic (``cli.py:_looks_like_slash_command``): the
    first word starts with ``/`` and contains no further ``/``. That is what
    separates ``/model gpt-5`` from a pasted ``/Users/me/notes.md`` -- operators
    paste absolute paths into the task box constantly, and treating one as a
    failed command would be worse than useless.

    Deliberately narrow: false here means "ordinary text", and ordinary text
    flows through to the agent untouched.
    """
    if not text:
        return False
    stripped = text.lstrip()
    if not stripped.startswith("/"):
        return False
    first = stripped.split()[0]
    body = first[1:]
    return bool(body) and "/" not in body


def resolve(name: str) -> Optional[TeamsCommand]:
    """Resolve a name or alias to a command Teams implements.

    Order matters: our own names and aliases win, then Hermes' alias table (so
    ``/compact`` reaches ``/compress`` and ``/q`` reaches ``/queue``). A Hermes
    alias pointing at something we do not implement resolves to None.
    """
    key = (name or "").strip().lower().lstrip("/")
    if not key:
        return None
    _ensure_catalog()
    if key in _BY_NAME:
        return _BY_NAME[key]
    if key in _BY_ALIAS:
        return _BY_NAME[_BY_ALIAS[key]]
    _load_hermes()
    if _resolver is not None:
        try:
            entry = _resolver(key)
        except Exception:
            entry = None
        if entry is not None:
            return _BY_NAME.get(getattr(entry, "name", ""))
    return None


def parse(text: str) -> Optional[Tuple[str, str]]:
    """Split command text into ``(canonical_name, args)``.

    Returns None when the text is not a command at all, and ``("", args)`` when
    it looks like one but resolves to nothing -- callers distinguish "pass this
    through as prose" from "tell the operator this command does not exist".
    """
    if not looks_like_slash_command(text):
        return None
    # split(None, 1) not partition(" ") — looks_like_slash_command() uses split()
    # and so treats any whitespace as the boundary. Splitting on a literal space
    # here would disagree with it for "/stop\nnote" and reject a valid command.
    parts = text.strip().split(None, 1)
    head = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    cmd = resolve(head)
    if cmd is None:
        return ("", rest.strip())
    return (cmd.name, rest.strip())


def suggest(text: str, scope: str = "both") -> Optional[str]:
    """Nearest catalog name for an unknown command, for a 'did you mean' hint."""
    parts = (text or "").strip().lstrip("/").lower().split()
    if not parts:
        return None
    key = parts[0]
    names = [c.name for c in catalog(scope)]
    close = difflib.get_close_matches(key, names, n=1, cutoff=0.6)
    return close[0] if close else None


def unknown_result(text: str, scope: str = "both") -> CommandResult:
    """Rejection for a command-shaped token we do not implement.

    We reject rather than pass through to the model. A typo'd ``/modle gpt-5``
    sent as prose is the worst outcome: the operator believes they reconfigured
    the agent while the agent burns a turn puzzling over it.

    This is not the trap Hermes documents in ``should_bypass_active_session`` --
    that bug silently *discarded* a recognized command. Nothing is swallowed
    here; the operator always gets told.
    """
    # split() not split(" "): keeps the echoed name clean for "/modle\nargs".
    parts = (text or "").strip().split()
    head = parts[0] if parts else "/"
    hint = suggest(head, scope)
    msg = f"Unknown command `{head}`."
    if hint:
        msg += f" Did you mean `/{hint}`?"
    else:
        msg += " Type `/help` to see available commands."
    return CommandResult(handled=True, ok=False, text=msg, forwarded=False)


# --------------------------------------------------------------------------- #
# Dispatch
#
# Handlers get their server-side dependencies through CommandContext rather than
# importing them, so this module stays free of FastAPI/server imports (no import
# cycle, and it unit-tests against a stub daemon).
# --------------------------------------------------------------------------- #
@dataclass
class CommandContext:
    """Everything a handler may need, injected by the calling endpoint."""

    scope: str = "agent"
    agent_name: str = ""
    daemon: object = None
    # Injected by server.py: (agent_name, body_dict) -> None. Routes /model
    # through the same path as PATCH /agent/{name}/config so both normalize
    # identically. We never call AIAgent.switch_model directly -- it rebuilds
    # clients and is unsafe while the executor thread is inside run_conversation.
    apply_config: object = None
    budget: object = None


def _fmt_int(n) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "?"


def _live_agent(daemon):
    """Re-read ``_ai_agent`` at call time.

    It is set to None on the stop/reinit paths (agent.py:389/663/1084/1134, with
    an explicit warning at 656 about nulling it mid-turn from another thread), so
    a cached reference can go stale between dispatch and use.
    """
    return getattr(daemon, "_ai_agent", None)


def _h_help(args: str, ctx: CommandContext) -> CommandResult:
    lines = ["**Available commands**", ""]
    by_cat: Dict[str, List[TeamsCommand]] = {}
    for c in catalog(ctx.scope):
        by_cat.setdefault(c.category, []).append(c)
    for cat in sorted(by_cat):
        lines.append(f"*{cat}*")
        for c in sorted(by_cat[cat], key=lambda x: x.name):
            usage = f"/{c.name}" + (f" {c.args_hint}" if c.args_hint else "")
            alias = f" (also /{', /'.join(c.aliases)})" if c.aliases else ""
            lines.append(f"- `{usage}` — {c.description}{alias}")
        lines.append("")
    if not hermes_available():
        lines.append("_Hermes command registry unavailable — descriptions and "
                     "aliases are reduced._")
    return CommandResult(handled=True, text="\n".join(lines).strip())


def _h_status(args: str, ctx: CommandContext) -> CommandResult:
    d = ctx.daemon
    if d is None:
        return CommandResult(handled=True, ok=False, text="No agent selected.")
    cfg = getattr(d, "cfg", {}) or {}
    paused = bool(getattr(d, "_paused", False))
    lines = [
        f"**{ctx.agent_name}** — `{getattr(d, 'state', '?')}`" + (" (paused)" if paused else ""),
        f"- Pending tasks: {getattr(getattr(d, 'queue', None), 'get_pending_count', lambda: '?')()}",
        f"- Model: `{cfg.get('model') or 'teams default'}`",
    ]
    if cfg.get("team_id"):
        lines.append(f"- Team: `{cfg['team_id']}`")
    if ctx.budget is not None and cfg.get("team_id"):
        try:
            st = ctx.budget.status(cfg["team_id"])
            spent, limit = st.get("spent_usd"), st.get("daily_usd")
            if limit:
                lines.append(f"- Budget today: ${spent:.4f} / ${limit:.2f}"
                             + (" — **blocked**" if st.get("blocked") else ""))
            elif spent is not None:
                lines.append(f"- Spent today: ${spent:.4f} (no cap)")
        except Exception as e:
            log.debug("budget status failed: %s", e)
    return CommandResult(handled=True, text="\n".join(lines))


def _h_usage(args: str, ctx: CommandContext) -> CommandResult:
    a = _live_agent(ctx.daemon)
    if a is None:
        return CommandResult(
            handled=True,
            text=f"**{ctx.agent_name}** has no live session yet — "
                 f"token usage starts accruing on its first turn.",
        )
    lines = [f"**{ctx.agent_name}** — session usage"]
    for label, attr in (
        ("Total tokens", "session_total_tokens"),
        ("Input", "session_input_tokens"),
        ("Output", "session_output_tokens"),
        ("Cache read", "session_cache_read_tokens"),
        ("API calls", "session_api_calls"),
    ):
        val = getattr(a, attr, None)
        if val is not None:
            lines.append(f"- {label}: {_fmt_int(val)}")
    cost = getattr(a, "session_estimated_cost_usd", None)
    if isinstance(cost, (int, float)):
        lines.append(f"- Estimated cost: ${cost:.4f}")
    return CommandResult(handled=True, text="\n".join(lines))


def _h_queue(args: str, ctx: CommandContext) -> CommandResult:
    q = getattr(ctx.daemon, "queue", None)
    if q is None:
        return CommandResult(handled=True, ok=False, text="No queue for this agent.")
    try:
        tasks = [t for t in q.get_all_tasks(limit=50) if t.get("status") == "pending"]
    except Exception as e:
        return CommandResult(handled=True, ok=False, text=f"Could not read the queue: {e}")
    if not tasks:
        return CommandResult(handled=True, text=f"**{ctx.agent_name}** has no pending tasks.")
    lines = [f"**{ctx.agent_name}** — {len(tasks)} pending task(s)"]
    for t in tasks[:20]:
        preview = " ".join((t.get("payload") or "").split())[:90]
        lines.append(f"- from `{t.get('from_agent', '?')}`: {preview}")
    if len(tasks) > 20:
        lines.append(f"- …and {len(tasks) - 20} more")
    return CommandResult(handled=True, text="\n".join(lines))


def _h_steer(args: str, ctx: CommandContext) -> CommandResult:
    if not args:
        return CommandResult(handled=True, ok=False,
                             text="Usage: `/steer <message>` — injected after the "
                                  "agent's next tool call, without interrupting it.")
    d = ctx.daemon
    a = _live_agent(d)
    if a is None or not hasattr(a, "steer"):
        return CommandResult(handled=True, ok=False,
                             text=f"**{ctx.agent_name}** has no live session to steer. "
                                  f"Send it a task instead.")

    # Hermes' steer() returns False ONLY for empty text -- it does not check
    # whether a turn is in flight (run_agent.py:2899). Steering an idle agent
    # therefore stashes the text silently until some future turn drains it, which
    # could be hours later and reads as the command having been lost. So gate on
    # the daemon's own state (values from agent.py:76-79; string literals here to
    # keep this module import-light and server-free).
    #
    # asking_human counts as steerable: that turn is parked inside ask_human, so
    # the steer lands on the tool result once the human answers.
    state = getattr(d, "state", None)
    if state not in ("busy", "asking_human"):
        detail = "paused" if getattr(d, "_paused", False) else (state or "not running")
        return CommandResult(handled=True, ok=False,
                             text=f"**{ctx.agent_name}** is {detail}, so there is no "
                                  f"turn to steer — the message would sit unseen until "
                                  f"its next turn. Send it a task instead.")
    try:
        # Inherently racy: the turn can end between the check above and here.
        # Accepted -- the check makes the common case honest, and a late steer is
        # drained into the following turn rather than lost.
        accepted = bool(a.steer(args))
    except Exception as e:
        return CommandResult(handled=True, ok=False, text=f"Steer failed: {e}")
    if not accepted:
        return CommandResult(handled=True, ok=False, text="Nothing to steer with.")
    return CommandResult(handled=True, forwarded=True,
                         text=f"Steering **{ctx.agent_name}** — your message lands "
                              f"after its next tool call.")


async def _h_stop(args: str, ctx: CommandContext) -> CommandResult:
    d = ctx.daemon
    if d is None:
        return CommandResult(handled=True, ok=False, text="No agent selected.")
    if getattr(d, "state", None) == "idle":
        return CommandResult(handled=True, text=f"**{ctx.agent_name}** is already idle.")
    # stop_execution() is async and already calls Hermes' interrupt() internally
    # (see its comment at agent.py:1005-1011) plus releases a turn parked in
    # ask_human -- so never call interrupt() ourselves.
    await d.stop_execution()
    return CommandResult(handled=True,
                         text=f"Stopped **{ctx.agent_name}** — in-flight turn "
                              f"interrupted and pending tasks drained.")


def _h_pause(args: str, ctx: CommandContext) -> CommandResult:
    d = ctx.daemon
    if d is None:
        return CommandResult(handled=True, ok=False, text="No agent selected.")
    if getattr(d, "_paused", False):
        return CommandResult(handled=True, text=f"**{ctx.agent_name}** is already paused.")
    d.pause_execution(reason=args or "Paused by operator", by="human")
    return CommandResult(handled=True,
                         text=f"Paused **{ctx.agent_name}** — its queue is held, "
                              f"not lost. Resume with `/resume`.")


def _h_resume(args: str, ctx: CommandContext) -> CommandResult:
    d = ctx.daemon
    if d is None:
        return CommandResult(handled=True, ok=False, text="No agent selected.")
    if not getattr(d, "_paused", False):
        return CommandResult(handled=True, text=f"**{ctx.agent_name}** is not paused.")
    d.resume_execution(by="human")
    return CommandResult(handled=True, text=f"Resumed **{ctx.agent_name}**.")


def _h_model(args: str, ctx: CommandContext) -> CommandResult:
    d = ctx.daemon
    if d is None:
        return CommandResult(handled=True, ok=False, text="No agent selected.")
    cfg = getattr(d, "cfg", {}) or {}
    if not args:
        current = cfg.get("model") or "teams default"
        return CommandResult(handled=True,
                             text=f"**{ctx.agent_name}** uses `{current}`.\n\n"
                                  f"Change it with `/model <name>`.")
    if ctx.apply_config is None:
        return CommandResult(handled=True, ok=False,
                             text="Model changes are unavailable on this surface.")
    name = args.split()[0]
    try:
        ctx.apply_config(ctx.agent_name, {"model": name})
    except Exception as e:
        return CommandResult(handled=True, ok=False, text=f"Could not set the model: {e}")
    return CommandResult(handled=True,
                         text=f"**{ctx.agent_name}** now uses `{name}`. "
                              f"It takes effect on the agent's next turn.")


def _h_goal(args: str, ctx: CommandContext) -> CommandResult:
    """Teams' analogue of Hermes' /goal: a time-boxed directive.

    While a directive is armed, the agent's idle heartbeats re-present it instead
    of a generic check-in (and the heartbeat backoff is suspended), so a
    multi-hour push survives the queue going empty.
    """
    d = ctx.daemon
    if d is None:
        return CommandResult(handled=True, ok=False, text="No agent selected.")

    directive = getattr(d, "_directive", None)
    sub = args.split()[0].lower() if args else "show"

    if not args or sub == "show":
        if not directive:
            return CommandResult(
                handled=True,
                text=f"**{ctx.agent_name}** has no standing directive.\n\n"
                     f"Set one with `/goal <text> [--hours N]`.")
        left = (float(directive.get("until_ts", 0)) - time.time()) / 60.0
        when = f"{left:.0f} min" if left < 90 else f"{left / 60.0:.1f} h"
        return CommandResult(
            handled=True,
            text=f"**{ctx.agent_name}** directive (expires in {when}):\n\n"
                 f"> {directive.get('payload', '')}")

    if sub == "clear":
        if not directive:
            return CommandResult(handled=True,
                                 text=f"**{ctx.agent_name}** has no directive to clear.")
        # Direct assignment matches how the daemon itself retires an expired
        # directive (agent.py:1346); there is no public clear method.
        d._directive = None
        return CommandResult(handled=True,
                             text=f"Cleared **{ctx.agent_name}**'s directive — back to "
                                  f"normal idle cadence.")

    # Anything else is the directive text, with an optional --hours override.
    # (?:^|\s) not \s: parse() already stripped args, so a flag-only argument
    # like "/goal --hours 3" has no leading whitespace to match. Without the
    # ^ alternative it fell through and armed a directive whose text was
    # literally "--hours 3".
    hours = 1.0
    text = args
    m = re.search(r"(?:^|\s)--hours\s+([0-9]+(?:\.[0-9]+)?)\s*$", args)
    if m:
        hours = float(m.group(1))
        text = args[:m.start()].strip()
    if not text:
        return CommandResult(handled=True, ok=False,
                             text="Usage: `/goal <text> [--hours N]`")
    if hours <= 0:
        return CommandResult(handled=True, ok=False, text="`--hours` must be greater than 0.")
    d.set_directive(text, hours * 60.0, from_agent="human")
    return CommandResult(handled=True, forwarded=True,
                         text=f"**{ctx.agent_name}** directive armed for {hours:g}h:\n\n"
                              f"> {text}\n\nIts idle heartbeats will keep pushing on this. "
                              f"`/goal clear` to stop.")


def _h_cron(args: str, ctx: CommandContext) -> CommandResult:
    """Read-only listing. Creating a schedule needs validation and a real form,
    which the agent's config panel already provides."""
    d = ctx.daemon
    if d is None:
        return CommandResult(handled=True, ok=False, text="No agent selected.")
    try:
        entries = d.crons_runtime()
    except Exception as e:
        return CommandResult(handled=True, ok=False, text=f"Could not read schedules: {e}")
    if not entries:
        return CommandResult(
            handled=True,
            text=f"**{ctx.agent_name}** has no scheduled wake-ups.\n\n"
                 f"Add one from the agent's ⚙ panel.")

    try:
        from teams_server.cron import cron_describe
    except Exception:
        cron_describe = None

    lines = [f"**{ctx.agent_name}** — {len(entries)} schedule(s)"]
    for c in entries:
        sched = c.get("schedule", "?")
        desc = ""
        if cron_describe:
            try:
                desc = cron_describe(sched) or ""
            except Exception:
                desc = ""
        # cron_describe echoes the raw expression when it has no plain-English
        # form for it, which would render as "`0 9 * * *` — 0 9 * * *".
        if desc.strip() == str(sched).strip():
            desc = ""
        state = "" if c.get("enabled", True) else " *(disabled)*"
        instr = " ".join(str(c.get("instruction") or "").split())[:70]
        lines.append(f"- `{sched}`{f' — {desc}' if desc else ''}{state}: {instr}")
        nxt = c.get("next_fire_at")
        if nxt:
            mins = (float(nxt) - time.time()) / 60.0
            if mins >= 0:
                lines.append(f"  next in {mins:.0f} min · {c.get('runs', 0)} run(s)")
    return CommandResult(handled=True, text="\n".join(lines))


_HANDLERS = {
    "help": _h_help,
    "status": _h_status,
    "usage": _h_usage,
    "queue": _h_queue,
    "steer": _h_steer,
    "stop": _h_stop,
    "pause": _h_pause,
    "resume": _h_resume,
    "model": _h_model,
    "goal": _h_goal,
    "cron": _h_cron,
}

# Every advertised command must have a handler, or the catalog is lying. Checked
# at import so a half-finished command cannot reach the autocomplete.
_missing_handlers = sorted(set(_SPEC) - set(_HANDLERS))
if _missing_handlers:  # pragma: no cover - guards against developer error
    raise RuntimeError(
        f"commands.py: _SPEC advertises {_missing_handlers} with no handler. "
        f"Either implement them or remove them from _SPEC."
    )


async def dispatch(text: str, ctx: CommandContext) -> CommandResult:
    """Run ``text`` as a command.

    Returns ``handled=False`` for ordinary prose, which is the caller's signal to
    fall through to its normal enqueue path. Anything command-shaped is either
    executed or explicitly rejected -- never silently swallowed.
    """
    parsed = parse(text)
    if parsed is None:
        return CommandResult(handled=False)

    name, args = parsed
    if not name:
        return unknown_result(text, ctx.scope)

    cmd = by_name().get(name)
    if cmd is None or not cmd.matches_scope(ctx.scope):
        # Real case: /steer typed into the Architect chat. Rejecting beats
        # running it against a surface it was not designed for.
        return CommandResult(
            handled=True, ok=False,
            text=f"`/{name}` is not available here.",
        )

    handler = _HANDLERS.get(name)
    if handler is None:  # pragma: no cover - import-time guard prevents this
        return CommandResult(handled=True, ok=False, text=f"`/{name}` is not implemented.")

    try:
        result = handler(args, ctx)
        if hasattr(result, "__await__"):
            result = await result
        return result
    except Exception as e:
        log.exception("Command /%s failed", name)
        return CommandResult(handled=True, ok=False, text=f"`/{name}` failed: {e}")
