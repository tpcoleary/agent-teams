"""Human takeover of a team's headless Chrome over one WebSocket.

CDP ``Page.startScreencast`` pushes JPEG frames to the dashboard's <canvas>;
dashboard input events come back and are translated into CDP ``Input.*`` /
``Page.navigate``. This is how a human completes a login/CAPTCHA on a
display-less VPS without touching the agent's session.

Three things keep it feeling like a real browser:
  * LATEST-WINS FRAMES — a slow client drops stale frames instead of queueing
    them (a backlog is exactly what "the browser keeps freezing" feels like).
  * STALL WATCHDOG — screencast only emits on repaint; when it goes quiet we
    force a ``Page.captureScreenshot`` so the view can't look frozen.
  * TAB DEATH SURVIVAL — if the streamed page closes under us (OAuth popups
    self-close), hop to another live page or mint a blank one.

Protocol server→client: frame, navigated, tabs, loading, stall, ping, error.
Protocol client→server: mouse, key, text, navigate, refresh, resize,
switch_tab, new_tab, close_tab, back, forward, ping.
"""

import asyncio
import json
import logging
import time
import urllib.request
from typing import Any, Dict, List, Optional

# Screencast tuning — standard 1440x900 desktop viewport at JPEG quality 60
# delivers crisp rendering and fast software compression (<8ms) for 30+ FPS.
SCREENCAST_PARAMS = {"format": "jpeg", "quality": 60,
                     "maxWidth": 1440, "maxHeight": 900, "everyNthFrame": 1}
DEVICE_SCALE = 1.0

_STALL_AFTER_S = 4.0      # screencast silence before nudging a repaint
_TICK_S = 2.5             # watchdog/pinger tick
_PING_EVERY_S = 15.0
_TABS_DEBOUNCE_S = 0.25   # let Target.* storms settle before refetching /json
_MAX_HOPS = 20            # give up after this many auto page-hops in a row

_INTERNAL_PREFIXES = ("about:", "chrome:", "devtools:", "chrome-extension:")


def _cdp_targets(base_url: str) -> List[Dict[str, Any]]:
    with urllib.request.urlopen(f"{base_url}/json", timeout=3) as r:
        return json.load(r)


def _cdp_http_json(url: str) -> Dict[str, Any]:
    """CDP HTTP control endpoints require PUT (Chrome rejects GET for
    /json/new and /json/close). Returns parsed body, {} when empty."""
    req = urllib.request.Request(url, method="PUT")
    with urllib.request.urlopen(req, timeout=3) as r:
        body = r.read().decode("utf-8", "replace").strip()
    return json.loads(body) if body else {}


def pick_page_target(targets: List[Dict[str, Any]],
                     exclude_target_id: Optional[str] = None
                     ) -> Optional[Dict[str, Any]]:
    """The page to stream: first real URL wins over blank/internal ones.
    ``exclude_target_id`` omits a dead/current target (fallback picking).
    Pure → unit-testable."""
    pages = [t for t in targets
             if t.get("type") == "page"
             and t.get("webSocketDebuggerUrl")
             and (exclude_target_id is None or t.get("id") != exclude_target_id)]
    for t in pages:
        u = t.get("url", "") or ""
        if u and not u.startswith(_INTERNAL_PREFIXES):
            return t
    return pages[0] if pages else None


def translate_client_message(msg: Dict[str, Any], next_id
                             ) -> Optional[Dict[str, Any]]:
    """Map one dashboard→server message to a CDP command dict (None = ignore).

    Coordinates pass through unchanged: screencast metadata (deviceWidth/
    Height) is DIPs == CSS pixels, exactly what CDP Input.* expects.

    Session-level messages (tabs management, history, refresh, resize) are
    handled by relay() itself.
    """
    t = msg.get("type")

    if t == "mouse":
        cdp_type = {"pressed": "mousePressed", "released": "mouseReleased",
                    "moved": "mouseMoved", "wheel": "mouseWheel"}.get(msg.get("action"))
        if not cdp_type:
            return None
        params: Dict[str, Any] = {
            "type": cdp_type,
            "x": float(msg.get("x", 0)), "y": float(msg.get("y", 0)),
            "modifiers": int(msg.get("modifiers", 0)),
        }
        if cdp_type in ("mousePressed", "mouseReleased"):
            params["button"] = msg.get("button", "left")
            params["clickCount"] = int(msg.get("clickCount", 1))
        if cdp_type == "mouseWheel":
            params["deltaX"] = float(msg.get("deltaX", 0))
            params["deltaY"] = float(msg.get("deltaY", 0))
        return {"id": next_id(), "method": "Input.dispatchMouseEvent", "params": params}

    if t == "key":
        cdp_type = {"down": "keyDown", "up": "keyUp", "char": "char"}.get(msg.get("action"))
        if not cdp_type:
            return None
        params = {"type": cdp_type, "modifiers": int(msg.get("modifiers", 0)),
                  "key": msg.get("key", ""), "code": msg.get("code", "")}
        if msg.get("text"):
            params["text"] = msg["text"]
        if msg.get("windowsVirtualKeyCode") is not None:
            params["windowsVirtualKeyCode"] = int(msg["windowsVirtualKeyCode"])
        return {"id": next_id(), "method": "Input.dispatchKeyEvent", "params": params}

    if t == "text":
        return {"id": next_id(), "method": "Input.insertText",
                "params": {"text": str(msg.get("text", ""))}}

    if t == "navigate":
        url = str(msg.get("url", "")).strip()
        if not url:
            return None
        if not url.startswith(("http://", "https://", "about:", "file:")):
            url = "https://" + url
        return {"id": next_id(), "method": "Page.navigate", "params": {"url": url}}

    return None


class _LatestSlot:
    """Single-slot latest-wins mailbox between the CDP pump and the sender."""

    def __init__(self) -> None:
        self.item: Optional[Dict[str, Any]] = None
        self._evt = asyncio.Event()

    def put(self, item: Dict[str, Any]) -> None:
        self.item = item
        self._evt.set()

    async def get(self) -> Optional[Dict[str, Any]]:
        await self._evt.wait()
        item, self.item = self.item, None
        self._evt.clear()
        # A put() racing the read above would strand its item until the NEXT
        # frame — re-arm so the last-ever frame always goes out promptly even
        # if the page goes static right after.
        if self.item is not None:
            self._evt.set()
        return item


async def _send_error(client_ws, message: str) -> None:
    try:
        await client_ws.send_text(
            json.dumps({"type": "error", "payload": {"message": message}}))
    except Exception:
        pass


async def relay(client_ws, team_id: str) -> None:
    """Bridge an accepted dashboard WebSocket to the team's headless Chrome.
    Runs until either side disconnects."""
    from websockets.asyncio.client import connect as ws_connect

    loop = asyncio.get_running_loop()
    from teams_server.neko_pool import resolve_team_cdp_url

    cdp_url = await loop.run_in_executor(None, resolve_team_cdp_url, team_id)
    if not cdp_url:
        await _send_error(client_ws, "No browser available on this host.")
        return

    try:
        targets = await loop.run_in_executor(None, _cdp_targets, cdp_url)
    except Exception as e:
        await _send_error(client_ws, f"Could not list browser tabs: {e}")
        return

    target = pick_page_target(targets)
    if not target:
        try:
            target = await loop.run_in_executor(
                None, _cdp_http_json, f"{cdp_url}/json/new?about:blank")
        except Exception as e:
            await _send_error(client_ws, f"No page open and could not create one: {e}")
            return

    _id = 0

    def next_id() -> int:
        nonlocal _id
        _id += 1
        return _id

    # An OAuth login often opens a popup as a NEW target, so tab switching
    # reconnects the CDP socket while keeping the SAME client socket.
    state = {
        "target": dict(target),
        "switch_to": None,                # debugger ws url of the next target
        "client_gone": False,
        "vw": 1440, "vh": 900,
        "pending_shot": set(),            # ids of forced screenshots in flight
        "hops": 0,
    }
    loading = False
    stall_notified = False
    last_frame_at = time.monotonic()
    sock_holder: Dict[str, Any] = {"cdp": None}
    frame_slot = _LatestSlot()
    tabs_dirty = asyncio.Event()

    async def send_client(payload: Dict[str, Any]) -> None:
        nonlocal state
        try:
            await client_ws.send_text(json.dumps(payload))
        except Exception:
            state["client_gone"] = True

    def push_loading(v: bool) -> None:
        nonlocal loading
        if loading != v:
            loading = v
            asyncio.create_task(send_client(
                {"type": "loading", "payload": {"loading": v}}))

    async def fetch_tabs() -> List[Dict[str, Any]]:
        try:
            return await loop.run_in_executor(None, _cdp_targets, cdp_url)
        except Exception:
            return []

    async def create_blank_tab() -> Optional[str]:
        try:
            info = await loop.run_in_executor(
                None, _cdp_http_json, f"{cdp_url}/json/new?about:blank")
        except Exception as e:
            log.debug("[%s] could not open a fresh blank tab: %s", team_id, e)
            return None
        if info.get("webSocketDebuggerUrl"):
            state["target"] = info
            return info["webSocketDebuggerUrl"]
        return None

    async def sender() -> None:
        nonlocal last_frame_at, stall_notified
        while True:
            payload = await frame_slot.get()
            if payload is not None:
                await send_client({"type": "frame", "payload": payload})
                last_frame_at = time.monotonic()
                stall_notified = False

    async def tabs_pusher() -> None:
        """Debounced /json refetches. Target events arrive in storms during
        page loads; fetching inline in the frame pump stalled every frame
        behind a synchronous HTTP round-trip. Never do that again."""
        while True:
            await tabs_dirty.wait()
            await asyncio.sleep(_TABS_DEBOUNCE_S)
            tabs_dirty.clear()
            tabs = await fetch_tabs()
            await send_client({"type": "tabs", "payload": {
                "current": state["target"].get("id"),
                "tabs": [{"targetId": t.get("id"), "title": t.get("title"),
                          "url": t.get("url")}
                         for t in tabs if t.get("type") == "page"]}})

    async def heartbeat() -> None:
        """Watchdog + keepalive in one tick loop. Screencast only emits on
        repaint, so silence alone isn't an error — past _STALL_AFTER_S we tell
        the client (it shows 'waiting') and force a screenshot as a repaint
        cure, capped at 3 consecutive nudges so static pages don't eat latency
        spikes forever. Pings let the client tell idle from dead."""
        nonlocal stall_notified, last_frame_at
        nudges = 0
        last_ping = time.monotonic()
        while True:
            await asyncio.sleep(_TICK_S)
            cdp = sock_holder["cdp"]
            if time.monotonic() - last_ping >= _PING_EVERY_S:
                last_ping = time.monotonic()
                await send_client({"type": "ping", "payload": {}})
                if cdp is not None:
                    try:
                        await cdp.send(json.dumps({"id": next_id(), "method": "Storage.flushStorage"}))
                    except Exception:
                        pass
            if cdp is None:
                continue
            idle = time.monotonic() - last_frame_at
            if idle < _STALL_AFTER_S:
                nudges = 0
                continue
            if not stall_notified:
                stall_notified = True
                await send_client({"type": "stall",
                                   "payload": {"seconds": round(idle, 1)}})
            if nudges >= 3:
                continue
            nudges += 1
            sid = next_id()
            state["pending_shot"].add(sid)
            try:
                await cdp.send(json.dumps({"id": sid,
                    "method": "Page.captureScreenshot",
                    "params": {"format": "jpeg", "quality": 68}}))
            except Exception:
                pass  # reconnect path handles a dead socket

    bg_tasks = [asyncio.create_task(sender()),
                asyncio.create_task(tabs_pusher()),
                asyncio.create_task(heartbeat())]

    async def pump_cdp_to_client(cdp) -> None:
        """Forward screencast frames and page events to the dashboard. Stays
        cheap: no blocking calls, no awaits on slow sockets."""
        from websockets.exceptions import ConnectionClosed

        try:
            async for raw in cdp:
                try:
                    evt = json.loads(raw)
                except Exception:
                    continue
                method = evt.get("method")

                if method is None:
                    mid = evt.get("id")
                    # Forced-screenshot replies double as fresh frames.
                    if mid in state["pending_shot"]:
                        state["pending_shot"].discard(mid)
                        data = (evt.get("result") or {}).get("data")
                        if data:
                            frame_slot.put({
                                "data": data,
                                # Same DIP-space dims as screencast metadata so
                                # the canvas never flips resolution mid-session.
                                "deviceWidth": state["vw"],
                                "deviceHeight": state["vh"],
                                "offsetTop": 0, "pageScaleFactor": 1})
                    continue

                p = evt.get("params", {})
                if method == "Page.screencastFrame":
                    md = p.get("metadata", {})
                    sid = p.get("sessionId")
                    if sid is not None:
                        try:
                            await cdp.send(json.dumps({"id": next_id(),
                                "method": "Page.screencastFrameAck",
                                "params": {"sessionId": sid}}))
                        except Exception:
                            pass
                    frame_slot.put({
                        "data": p.get("data", ""),
                        "deviceWidth": md.get("deviceWidth"),
                        "deviceHeight": md.get("deviceHeight"),
                        "offsetTop": md.get("offsetTop", 0),
                        "pageScaleFactor": md.get("pageScaleFactor", 1)})
                elif method in ("Page.frameNavigated", "Page.navigatedWithinDocument"):
                    frame = p.get("frame") or {}
                    if method == "Page.frameNavigated" and frame.get("parentId"):
                        continue
                    u = frame.get("url") or p.get("url", "")
                    if u:
                        await send_client({"type": "navigated", "payload": {
                            "url": u, "targetId": state["target"].get("id", "")}})
                elif method == "Page.frameStartedLoading":
                    if not p.get("frame", {}).get("parentId"):
                        push_loading(True)
                elif method == "Page.frameStoppedLoading":
                    if not p.get("frame", {}).get("parentId"):
                        push_loading(False)
                        try:
                            await cdp.send(json.dumps({"id": next_id(), "method": "Storage.flushStorage"}))
                        except Exception:
                            pass
                elif method.startswith("Target.target"):
                    tabs_dirty.set()  # debounced task does the HTTP work
        except ConnectionClosed:
            # Expected: tab closed / renderer died / browser quit. The relay
            # loop hops to another live page or ends the session.
            log.debug("[%s] CDP socket closed", team_id)

    async def pump_client_to_cdp(cdp) -> None:
        """Dashboard input → CDP; session-level commands handled locally.
        Returns when the client disconnects or asks to switch targets."""
        while True:
            try:
                raw = await client_ws.receive_text()
            except Exception:
                state["client_gone"] = True
                return
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            mtype = msg.get("type")

            if mtype in (None, "ping"):
                continue

            if mtype == "switch_tab":
                tid = msg.get("targetId", "")
                meta = next((t for t in await fetch_tabs()
                             if t.get("id") == tid), None)
                if meta and meta.get("webSocketDebuggerUrl"):
                    state["target"] = meta
                    state["switch_to"] = meta["webSocketDebuggerUrl"]
                    return
                tabs_dirty.set()
                continue

            if mtype == "new_tab":
                url = await create_blank_tab()
                if url:
                    state["switch_to"] = url
                    return
                tabs_dirty.set()
                continue

            if mtype == "close_tab":
                tid = msg.get("targetId", "")
                if not tid:
                    continue
                closing_current = tid == state["target"].get("id")
                try:
                    await loop.run_in_executor(
                        None, _cdp_http_json, f"{cdp_url}/json/close/{tid}")
                except Exception as e:
                    log.debug("[%s] close_tab %s failed: %s", team_id, tid, e)
                if closing_current:
                    # Last page closed: hop to another live one, else blank.
                    await asyncio.sleep(0.15)  # let Chrome tear the target down
                    fb = pick_page_target(await fetch_tabs(),
                                          exclude_target_id=tid)
                    if fb:
                        state["target"] = fb
                        state["switch_to"] = fb["webSocketDebuggerUrl"]
                    else:
                        url = await create_blank_tab()
                        if url:
                            state["switch_to"] = url
                            return
                        state["client_gone"] = True
                    return
                tabs_dirty.set()
                continue

            if mtype in ("back", "forward"):
                expr = "history.back()" if mtype == "back" else "history.forward()"
                await cdp.send(json.dumps({"id": next_id(),
                    "method": "Runtime.evaluate", "params": {"expression": expr}}))
                continue

            if mtype == "refresh":
                push_loading(True)
                await cdp.send(json.dumps(
                    {"id": next_id(), "method": "Page.reload"}))
                continue

            if mtype == "resize":
                w, h = int(msg.get("width") or 0), int(msg.get("height") or 0)
                if w > 400 and h > 300 and (abs(w - state["vw"]) > 50 or abs(h - state["vh"]) > 50):
                    state["vw"], state["vh"] = w, h
                    await cdp.send(json.dumps({"id": next_id(),
                        "method": "Emulation.setDeviceMetricsOverride",
                        "params": {"width": w, "height": h,
                                   "deviceScaleFactor": DEVICE_SCALE,
                                   "mobile": False}}))
                continue

            cmd = translate_client_message(msg, next_id)
            if cmd is not None:
                if cmd.get("method") == "Page.navigate":
                    push_loading(True)
                await cdp.send(json.dumps(cmd))

    try:
        while not state["client_gone"]:
            state["switch_to"] = None
            async with ws_connect(state["target"]["webSocketDebuggerUrl"],
                                  max_size=64 * 1024 * 1024,
                                  open_timeout=10) as cdp:
                sock_holder["cdp"] = cdp
                last_frame_at = time.monotonic()
                stall_notified = False
                for cmd in (
                    {"method": "Page.enable"},
                    {"method": "Target.setDiscoverTargets",
                     "params": {"discover": True}},
                    {"method": "Page.bringToFront"},
                    {"method": "Emulation.setDeviceMetricsOverride",
                     "params": {"width": state["vw"], "height": state["vh"],
                                "deviceScaleFactor": DEVICE_SCALE,
                                "mobile": False}},
                    {"method": "Page.startScreencast", "params": SCREENCAST_PARAMS},
                ):
                    try:
                        await cdp.send(json.dumps({"id": next_id(), **cmd}))
                    except Exception:
                        pass

                tgt = state["target"]
                await send_client({"type": "navigated", "payload": {
                    "url": tgt.get("url", ""), "title": tgt.get("title", ""),
                    "targetId": tgt.get("id", "")}})
                tabs_dirty.set()
                push_loading(False)

                done, pending = await asyncio.wait(
                    [asyncio.create_task(pump_cdp_to_client(cdp)),
                     asyncio.create_task(pump_client_to_cdp(cdp))],
                    return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                # Retrieve exceptions (incl. cancelled) — unretrieved ones make
                # asyncio log scary "Task exception was never retrieved" noise.
                for task in list(done) + list(pending):
                    try:
                        task.exception()
                    except Exception:
                        pass
                sock_holder["cdp"] = None
                try:
                    await cdp.send(json.dumps(
                        {"id": next_id(), "method": "Page.stopScreencast"}))
                except Exception:
                    pass

            if state["client_gone"]:
                break
            if state["switch_to"]:
                state["hops"] = 0
                continue

            # Streamed page died under us (closed popup/tab) — hop to another
            # live page, or mint a blank one, instead of killing the session.
            state["hops"] += 1
            if state["hops"] > _MAX_HOPS:
                await _send_error(client_ws,
                                  "Browser pages kept closing — stopped streaming.")
                break
            await asyncio.sleep(0.2)  # let Chrome settle its target list
            fb = pick_page_target(await fetch_tabs(),
                                  exclude_target_id=state["target"].get("id"))
            if fb:
                log.info("[%s] streamed page closed — switching to %s",
                         team_id, fb.get("url") or "blank tab")
                state["target"] = fb
                continue
            if not await create_blank_tab():
                await _send_error(
                    client_ws, "The streamed page closed and no other page is open.")
                break
    except Exception as e:
        log.warning("[%s] [browser-stream] relay ended: %s", team_id, e)
        await _send_error(client_ws, f"Browser stream error: {e}")
    finally:
        for task in bg_tasks:
            task.cancel()
