"""Agent-facing task-tracking tools — create/assign/report on human-meaningful
work items (teams_server/tasks_db.py), distinct from the peer-message inbox.

Authorization model: an assignee must be the caller itself or a peer the
caller is linked to (the same `peer_allowed` predicate send_peer_message
already uses). Self-report actions (progress/complete/blocked) require the
caller to BE the current assignee — a delegator cannot fake a worker's status.

Assigning a task WAKES the assignee (it lands in their inbox via the same
ingest_task path send_peer_message uses), so create_task is a real hand-off on
its own and needs no accompanying message. `parent_task_id` is an optional flat
reference for grouping, not a tree: no roll-up, no cascade.

`failed` has no tool here by design — it's reachable only via the dashboard's
human PATCH endpoint (see server.py). Agents signal trouble with
mark_task_blocked; a human makes the final call that something is dead.
"""

import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

from teams_server.monitoring import monitor_db
from teams_server.tasks_db import task_db
from teams_server.websocket import _broadcast

log = logging.getLogger("teams.task_tools")


def _err(message: str) -> str:
    return json.dumps({"success": False, "error": message})


def _resolve_caller(kwargs: dict) -> Tuple[str, Optional[dict], Optional[str]]:
    """Return (caller, cfg, error). error is set (and the other two unusable)
    when the caller can't be identified. Local imports avoid a circular import
    with teams_server.tools, which imports this module's schemas."""
    from teams_server.tools import _caller_from_kwargs
    from teams_server.config import load_agents_config

    caller = _caller_from_kwargs(kwargs)
    if caller == "unknown":
        return caller, None, "Could not identify the calling agent."
    cfg = load_agents_config()
    if caller not in cfg.get("agents", {}):
        return caller, cfg, f"Calling agent '{caller}' not found in config."
    return caller, cfg, None


def _check_assignee(cfg: dict, caller: str, assignee: str) -> Optional[str]:
    """Return an error string, or None if `assignee` is a valid target for
    `caller` (self, or a peer caller is linked to)."""
    from teams_server.config import peer_allowed

    if not assignee:
        return "assigned_to is required."
    if assignee == caller:
        return None  # self-assignment always allowed; peer_allowed(x, x) is False by definition
    if assignee not in cfg.get("agents", {}):
        return f"Unknown agent '{assignee}'."
    if not peer_allowed(cfg, caller, assignee):
        return (f"'{assignee}' is not a linked peer of yours. You are linked to: "
                f"{cfg['agents'].get(caller, {}).get('allowed_peers', [])}")
    return None


def _load_owned_task(task_id: str, caller: str, *, assignee_only: bool) -> Tuple[Optional[dict], Optional[str]]:
    """Return (task, error). Enforces ownership: assignee-only for self-report
    actions, else creator-or-assignee."""
    task = task_db.get_task(task_id)
    if task is None:
        return None, f"Task '{task_id}' not found."
    if assignee_only:
        if task.get("assigned_to") != caller:
            return None, f"Only the assignee ('{task.get('assigned_to')}') may do this."
    else:
        if caller not in (task.get("created_by"), task.get("assigned_to")):
            return None, "Only the task's creator or assignee may do this."
    return task, None


def _emit(event: str, task: dict, caller: str, **extra) -> None:
    monitor_db.log_event(caller, f"task_{event}", data={"task_id": task["id"], **extra})
    _broadcast(f"task_{event}", {**task, "timestamp": time.time(), **extra})


def _wake_assignee(assignee: str, caller: str, task: dict) -> bool:
    """Deliver the task into the assignee's inbox so it actually WAKES and works
    on it — a task nobody is woken for is just a row in a table. Returns False
    if that agent has no running daemon (the task is still recorded).

    Self-assignment is delivered too (the caller is mid-turn now, so this lands
    as its next turn's work) but deliberately does NOT use the "[TASK · from X]"
    header: that header means "a delegator is waiting for a RESULT", and the
    turn-guard nudges on it. On a self-assigned task there is no delegator, so
    that header made agents try to message themselves — which surfaced as a
    bogus link_violation, since peer_allowed(x, x) is False by definition.
    """
    from teams_server.tools import _daemon_registry

    target = _daemon_registry.get(assignee)
    if target is None:
        return False
    body = task["title"]
    if task.get("description"):
        body += f"\n{task['description']}"
    close_out = (
        f"When you're done, call mark_task_complete(task_id=\"{task['id']}\") — "
        f"that reports back to whoever created the task, so you do not need to "
        f"message them separately. Report partial progress with "
        f"update_task_progress, or mark_task_blocked(reason=…) if you can't proceed."
    )
    if assignee == caller:
        payload = f"[OWN TASK · id={task['id'][:8]}]\n{body}\n\n{close_out}"
    else:
        payload = f"[TASK · id={task['id'][:8]} · from {caller}]\n{body}\n\n{close_out}"
    try:
        target.ingest_task(from_agent=caller, payload=payload)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("[task] could not wake '%s': %s", assignee, exc)
        return False


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
CREATE_TASK_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_task",
        "description": (
            "Create a task and assign it to yourself or a linked peer. This is "
            "the durable, human-visible record of work AND the way to hand work "
            "off: the assignee is woken and receives the task in their inbox, so "
            "you do not need a separate send_peer_message to kick them off. "
            "Optionally set parent_task_id to file this under a broader task."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short task title."},
                "description": {"type": "string", "description": "Optional detail."},
                "assigned_to": {"type": "string", "description": "Yourself or a linked peer."},
                "priority": {
                    "type": "integer",
                    "description": "0=low, 1=normal (default), 2=high, 3=urgent.",
                },
                "parent_task_id": {
                    "type": "string",
                    "description": (
                        "Optional id of a broader task this belongs under. A "
                        "reference only — the parent's own status and progress "
                        "are unaffected by this task."
                    ),
                },
            },
            "required": ["title", "assigned_to"],
        },
    },
}

REASSIGN_TASK_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "reassign_task",
        "description": (
            "Reassign a task to a different agent, waking them with it. You must "
            "be the task's creator or its current assignee, and the new assignee "
            "must be yourself or a linked peer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "assigned_to": {"type": "string", "description": "New assignee."},
            },
            "required": ["task_id", "assigned_to"],
        },
    },
}

EDIT_TASK_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit_task",
        "description": (
            "Edit a task's title/description/priority. You must be the creator "
            "or assignee. Does not change status, progress, or assignee — use "
            "the dedicated tools for those."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "priority": {"type": "integer", "description": "0=low .. 3=urgent."},
            },
            "required": ["task_id"],
        },
    },
}

UPDATE_TASK_PROGRESS_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "update_task_progress",
        "description": (
            "Report your own progress (0-100) on a task assigned to you. Only "
            "the assignee may report progress — this is a self-report, not "
            "something a delegator sets on your behalf. Moving progress off 0% "
            "on a pending task automatically starts it (in_progress)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "progress": {"type": "integer", "description": "0-100."},
            },
            "required": ["task_id", "progress"],
        },
    },
}

MARK_TASK_COMPLETE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "mark_task_complete",
        "description": "Mark a task assigned to you as done (sets progress to 100). Assignee-only.",
        "parameters": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
    },
}

MARK_TASK_BLOCKED_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "mark_task_blocked",
        "description": (
            "Mark a task assigned to you as blocked, with a required reason. "
            "Assignee-only. Use this — not a status update — when you cannot "
            "proceed without something external (a decision, a credential, "
            "another task finishing)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "reason": {"type": "string", "description": "Why you're blocked."},
            },
            "required": ["task_id", "reason"],
        },
    },
}

LIST_MY_TASKS_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_my_tasks",
        "description": "List tasks assigned to you, optionally filtered by status.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "blocked", "done", "failed"],
                },
            },
        },
    },
}

TASK_TOOL_SCHEMAS = (
    CREATE_TASK_TOOL_SCHEMA,
    REASSIGN_TASK_TOOL_SCHEMA,
    EDIT_TASK_TOOL_SCHEMA,
    UPDATE_TASK_PROGRESS_TOOL_SCHEMA,
    MARK_TASK_COMPLETE_TOOL_SCHEMA,
    MARK_TASK_BLOCKED_TOOL_SCHEMA,
    LIST_MY_TASKS_TOOL_SCHEMA,
)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def create_task_handler(args: dict, **kwargs) -> str:
    caller, cfg, error = _resolve_caller(kwargs)
    if error:
        return _err(error)
    title = (args.get("title") or "").strip()
    if not title:
        return _err("title is required.")
    assignee = args.get("assigned_to") or ""
    assign_error = _check_assignee(cfg, caller, assignee)
    if assign_error:
        return _err(assign_error)
    parent_task_id = (args.get("parent_task_id") or "").strip() or None
    if parent_task_id and task_db.get_task(parent_task_id) is None:
        return _err(f"Parent task '{parent_task_id}' not found.")
    team_id = cfg["agents"].get(caller, {}).get("team_id")
    try:
        task = task_db.create_task(
            title, created_by=caller, assigned_to=assignee,
            description=args.get("description") or "", team_id=team_id,
            priority=args.get("priority", 1), parent_task_id=parent_task_id,
        )
    except ValueError as e:
        return _err(str(e))
    log.info("[create_task] %s created '%s' -> %s", caller, title, assignee)
    _emit("created", task, caller)
    woken = _wake_assignee(assignee, caller, task)
    result: Dict[str, Any] = {"success": True, "task": task, "assignee_woken": woken}
    if not woken:
        result["message"] = (
            f"Task recorded, but '{assignee}' has no running daemon — it will not "
            "start until that agent is up."
        )
    return json.dumps(result)


def reassign_task_handler(args: dict, **kwargs) -> str:
    caller, cfg, error = _resolve_caller(kwargs)
    if error:
        return _err(error)
    task_id = args.get("task_id") or ""
    task, own_error = _load_owned_task(task_id, caller, assignee_only=False)
    if own_error:
        return _err(own_error)
    new_assignee = args.get("assigned_to") or ""
    assign_error = _check_assignee(cfg, caller, new_assignee)
    if assign_error:
        return _err(assign_error)
    if new_assignee == task.get("assigned_to"):
        return json.dumps({"success": True, "task": task, "message": "Already assigned there."})
    updated = task_db.reassign_task(task_id, new_assignee)
    log.info("[reassign_task] %s reassigned %s -> %s", caller, task_id[:8], new_assignee)
    _emit("reassigned", updated, caller, previous_assignee=task.get("assigned_to"))
    woken = _wake_assignee(new_assignee, caller, updated)
    return json.dumps({"success": True, "task": updated, "assignee_woken": woken})


def edit_task_handler(args: dict, **kwargs) -> str:
    caller, cfg, error = _resolve_caller(kwargs)
    if error:
        return _err(error)
    task_id = args.get("task_id") or ""
    task, own_error = _load_owned_task(task_id, caller, assignee_only=False)
    if own_error:
        return _err(own_error)
    fields = {k: args.get(k) for k in ("title", "description", "priority") if args.get(k) is not None}
    if not fields:
        return _err("Supply at least one of title/description/priority to edit.")
    try:
        updated = task_db.edit_task(task_id, **fields)
    except ValueError as e:
        return _err(str(e))
    log.info("[edit_task] %s edited %s: %s", caller, task_id[:8], list(fields))
    _emit("updated", updated, caller)
    return json.dumps({"success": True, "task": updated})


def update_task_progress_handler(args: dict, **kwargs) -> str:
    caller, cfg, error = _resolve_caller(kwargs)
    if error:
        return _err(error)
    task_id = args.get("task_id") or ""
    task, own_error = _load_owned_task(task_id, caller, assignee_only=True)
    if own_error:
        return _err(own_error)
    try:
        updated = task_db.update_progress(task_id, args.get("progress"))
    except ValueError as e:
        return _err(str(e))
    log.info("[update_task_progress] %s -> %s%% on %s", caller, updated["progress"], task_id[:8])
    _emit("progress", updated, caller)
    return json.dumps({"success": True, "task": updated})


def mark_task_complete_handler(args: dict, **kwargs) -> str:
    caller, cfg, error = _resolve_caller(kwargs)
    if error:
        return _err(error)
    task_id = args.get("task_id") or ""
    task, own_error = _load_owned_task(task_id, caller, assignee_only=True)
    if own_error:
        return _err(own_error)
    from teams_server.tasks_db import STATUS_DONE

    updated = task_db.set_status(task_id, STATUS_DONE)
    log.info("[mark_task_complete] %s completed %s", caller, task_id[:8])
    _emit("completed", updated, caller)
    return json.dumps({"success": True, "task": updated})


def mark_task_blocked_handler(args: dict, **kwargs) -> str:
    caller, cfg, error = _resolve_caller(kwargs)
    if error:
        return _err(error)
    task_id = args.get("task_id") or ""
    task, own_error = _load_owned_task(task_id, caller, assignee_only=True)
    if own_error:
        return _err(own_error)
    reason = (args.get("reason") or "").strip()
    if not reason:
        return _err("reason is required.")
    from teams_server.tasks_db import STATUS_BLOCKED

    updated = task_db.set_status(task_id, STATUS_BLOCKED, blocked_reason=reason)
    log.info("[mark_task_blocked] %s blocked %s: %s", caller, task_id[:8], reason[:80])
    _emit("blocked", updated, caller)
    return json.dumps({"success": True, "task": updated})


def list_my_tasks_handler(args: dict, **kwargs) -> str:
    caller, cfg, error = _resolve_caller(kwargs)
    if error:
        return _err(error)
    tasks = task_db.list_tasks(assigned_to=caller, status=args.get("status"))
    return json.dumps({"success": True, "tasks": tasks})


_HANDLERS = (
    ("create_task", CREATE_TASK_TOOL_SCHEMA, create_task_handler,
     "Create a task, assign it to yourself or a linked peer, and wake them with it."),
    ("reassign_task", REASSIGN_TASK_TOOL_SCHEMA, reassign_task_handler,
     "Reassign a task to a different agent, waking them with it."),
    ("edit_task", EDIT_TASK_TOOL_SCHEMA, edit_task_handler,
     "Edit a task's title/description/priority."),
    ("update_task_progress", UPDATE_TASK_PROGRESS_TOOL_SCHEMA, update_task_progress_handler,
     "Report your own progress on a task assigned to you."),
    ("mark_task_complete", MARK_TASK_COMPLETE_TOOL_SCHEMA, mark_task_complete_handler,
     "Mark a task assigned to you as done."),
    ("mark_task_blocked", MARK_TASK_BLOCKED_TOOL_SCHEMA, mark_task_blocked_handler,
     "Mark a task assigned to you as blocked, with a reason."),
    ("list_my_tasks", LIST_MY_TASKS_TOOL_SCHEMA, list_my_tasks_handler,
     "List tasks assigned to you."),
)


def register_task_tools(registry) -> None:
    """Idempotent registration of all task tools into the Hermes tool registry."""
    existing = registry.get_tool_to_toolset_map() or {}
    for name, schema, handler, description in _HANDLERS:
        if name in existing:
            continue
        registry.register(
            name=name, toolset="custom", schema=schema["function"],
            handler=handler, description=description,
        )
        log.info("[%s] Registered", name)
