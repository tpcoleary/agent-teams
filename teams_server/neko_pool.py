"""neko-backed team browser — a real Chromium in Docker streamed to the
dashboard over WebRTC (60fps, audio, native input).

Agents drive the SAME browser over CDP: the supervisord override below adds
``--remote-debugging-port=9222``, published alongside the stream, with the
team profile bind-mounted at ``/home/neko/.config/chromium`` so logins persist
across container rebuilds.

Per team one container (``agent-neko-<team>``): its own HTTP viewer port, CDP
port, and single-port UDP/TCP ICE muxes, stable room passwords in
``data/teams/<team>/.neko.json``, and a cookies-preserving Chrome policy (the
stock policy wipes them on exit). :func:`resolve_team_cdp_url` is the seam:
neko's CDP when enabled, else browser_pool's local headless Chrome.
"""

import json
import logging
import os
import secrets
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Dict, Optional

from teams_server.config import WORKSPACE_ROOT

log = logging.getLogger("teams.browser.neko")

# Port blocks for per-team endpoints (kept clear of browser_pool's 9333+).
_HTTP_BASE = 9600     # viewer UI
_CDP_BASE = 9640      # published --remote-debugging-port
_MUX_BASE = 9680      # ICE UDP mux; TCP mux takes base+1

# The stock neko chromium image launches the browser from this supervisord
# fragment. Our override mirrors it verbatim plus remote-debugging flags so
# agents can drive the SAME browser the human watches.
_SUPERVISORD_CONF = """\
[program:chromium]
environment=HOME="/home/%(ENV_USER)s",USER="%(ENV_USER)s",DISPLAY="%(ENV_DISPLAY)s"
command=/usr/bin/chromium
  --no-sandbox
  --window-position=0,0
  --display=%(ENV_DISPLAY)s
  --user-data-dir=/home/neko/.config/chromium
  --no-first-run
  --start-maximized
  --bwsi
  --force-dark-mode
  --disable-file-system
  --disable-gpu
  --disable-software-rasterizer
  --disable-dev-shm-usage
  --remote-debugging-port=9222
  --remote-debugging-address=0.0.0.0
  --remote-allow-origins=*
stopsignal=INT
autorestart=true
priority=800
user=%(ENV_USER)s
stdout_logfile=/var/log/neko/chromium.log
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=10
redirect_stderr=true

[program:openbox]
environment=HOME="/home/%(ENV_USER)s",USER="%(ENV_USER)s",DISPLAY="%(ENV_DISPLAY)s"
command=/usr/bin/openbox --config-file /etc/neko/openbox.xml
autorestart=true
priority=300
user=%(ENV_USER)s
"""

# Stock policy clears cookies/history on shutdown — fatal for login sessions.
# This one keeps everything persistent.
_CHROME_POLICY = {
    "DefaultCookiesSetting": 1,
    "RestoreOnStartup": 1,
    "AllowFileSelectionDialogs": True,
    "DownloadRestrictions": 0,
}


def _free_port(start: int) -> int:
    port = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1


def _advertise_host() -> str:
    """IP humans should point their WebRTC/HTTP at: explicit env wins, then the
    primary LAN IP (route trick), then loopback for local-only setups."""
    explicit = os.environ.get("NEKO_ADVERTISE_HOST", "").strip()
    if explicit:
        return explicit
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))   # no packets sent; just picks an iface
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def render_supervisord_conf() -> str:
    """The supervisord fragment mounted into the container. Pure → testable."""
    return _SUPERVISORD_CONF


def render_chrome_policy() -> str:
    """Chrome enterprise policy JSON mounted into the container. Pure."""
    return json.dumps(_CHROME_POLICY)


def load_or_create_secrets(team_dir: Path) -> Dict[str, str]:
    """Stable per-team room passwords (admin controls, user view-only).
    Stored next to the profile so they survive restarts. Pure-ish → testable."""
    path = team_dir / ".neko.json"
    try:
        data = json.loads(path.read_text())
        if data.get("admin_pwd") and data.get("user_pwd"):
            return data
    except Exception:
        pass
    data = {"admin_pwd": secrets.token_urlsafe(12),
            "user_pwd": secrets.token_urlsafe(12)}
    try:
        path.write_text(json.dumps(data))
    except Exception as e:
        log.debug("could not persist neko secrets (%s)", e)
    return data


class NekoManager:
    """Launches/stops one neko container per team and tracks connection info."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._containers: Dict[str, dict] = {}
        self._docker_ok: Optional[bool] = None
        self._ports_used: set = set()
        # Teams explicitly stopped from the dashboard stay off until restart —
        # otherwise every CDP touch would silently resurrect the container.
        self._disabled_teams: set = set()
        # Last failure per team, surfaced to the dashboard so a silent
        # screencast fallback never hides WHY the WebRTC stream is missing.
        self._last_error: Dict[str, str] = {}

    def last_error_for(self, team_id: str) -> Optional[str]:
        return self._last_error.get(team_id)

    def _fail(self, team_id: str, reason: str) -> None:
        self._last_error[team_id] = reason
        log.warning("[neko] '%s': %s", team_id, reason)

    # -- environment --------------------------------------------------------
    def enabled_for(self, team_id: str) -> bool:
        """TEAMS_VIEWER=neko forces it on; auto (default) uses it when Docker
        answers; anything else keeps the pure-local screencast behavior."""
        if team_id in self._disabled_teams:
            return False
        mode = os.environ.get("TEAMS_VIEWER", "auto").strip().lower()
        if mode == "neko":
            return True
        if mode in ("off", "local"):
            return False
        return self.docker_available()

    def disable_team(self, team_id: str) -> None:
        self._disabled_teams.add(team_id)

    def docker_available(self) -> bool:
        if self._docker_ok is not None:
            return self._docker_ok
        try:
            r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                               capture_output=True, text=True, timeout=10)
            self._docker_ok = r.returncode == 0
        except Exception:
            self._docker_ok = False
        if not self._docker_ok:
            log.info("[neko] Docker not available — viewer falls back to "
                     "the built-in CDP screencast.")
        return self._docker_ok

    # -- ports --------------------------------------------------------------
    def _alloc_ports(self, team_id: str) -> Dict[str, int]:
        http_port = _free_port(_HTTP_BASE)
        cdp_port = _free_port(_CDP_BASE)
        udp_port = _free_port(_MUX_BASE)
        tcp_port = _free_port(_MUX_BASE + 50)
        used = {http_port, cdp_port, udp_port, tcp_port}
        while used & self._ports_used:
            http_port = _free_port(http_port + 1)
            cdp_port = _free_port(cdp_port + 1)
            udp_port = _free_port(udp_port + 1)
            tcp_port = _free_port(tcp_port + 1)
            used = {http_port, cdp_port, udp_port, tcp_port}
        self._ports_used |= used
        return {"http": http_port, "cdp": cdp_port, "udp": udp_port, "tcp": tcp_port}

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _container_name(team_id: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in team_id)
        return f"agent-neko-{safe}"

    @staticmethod
    def _container_running(name: str) -> bool:
        try:
            r = subprocess.run(["docker", "inspect", "-f",
                                "{{.State.Running}}", name],
                               capture_output=True, text=True, timeout=10)
            return r.returncode == 0 and "true" in r.stdout
        except Exception:
            return False

    def _http_healthy(self, port: int) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def _cdp_healthy(self, port: int) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version",
                                        timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def _ensure_image(self, image: str) -> bool:
        try:
            r = subprocess.run(["docker", "images", "-q", image],
                               capture_output=True, text=True, timeout=15)
            if r.returncode == 0 and r.stdout.strip():
                return True
            log.info("[neko] pulling %s (first run only)…", image)
            subprocess.run(["docker", "pull", image], timeout=900,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            r = subprocess.run(["docker", "images", "-q", image],
                               capture_output=True, text=True, timeout=15)
            return bool(r.stdout.strip())
        except Exception as e:
            log.error("[neko] image pull failed: %s", e)
            return False

    def _write_team_files(self, team_dir: Path) -> None:
        neko_dir = team_dir / ".neko"
        neko_dir.mkdir(parents=True, exist_ok=True)
        (neko_dir / "chromium.conf").write_text(render_supervisord_conf())
        (neko_dir / "policy.json").write_text(render_chrome_policy())
        # Container writes the profile as UID 1000; fix ownership where the OS
        # allows it (Linux VPS). macOS Docker Desktop bind mounts don't care.
        profile = team_dir / ".browser-profile"
        profile.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            for p in (profile, neko_dir):
                try:
                    os.chown(p, 1000, 1000)
                except (PermissionError, OSError):
                    pass

    # -- lifecycle ----------------------------------------------------------
    def ensure_neko(self, team_id: str) -> Optional[dict]:
        """Return viewer/cdp info for the team, starting the container when
        needed. Idempotent; returns None when neko can't be provided (reason
        available via last_error_for)."""
        if not self.docker_available():
            self._fail(team_id, "Docker daemon is not running — start Docker "
                                "Desktop, then reopen the browser panel.")
            return None
        with self._lock:
            info = self._containers.get(team_id)
            if info and self._http_healthy(info["http"]) \
                    and self._cdp_healthy(info["cdp"]):
                return info

            name = self._container_name(team_id)
            if self._container_running(name):
                # Running but our bookkeeping is stale (server restart): probe.
                subprocess.run(["docker", "rm", "-f", name],
                               capture_output=True, timeout=30)

            team_dir = WORKSPACE_ROOT / team_id
            self._write_team_files(team_dir)
            secrets_data = load_or_create_secrets(team_dir)

            image = os.environ.get(
                "NEKO_IMAGE", "ghcr.io/m1k1o/neko/chromium:latest").strip()
            if not self._ensure_image(image):
                self._fail(team_id, f"could not pull neko image '{image}' — "
                                    "check network/docker and try again")
                return None

            ports = self._alloc_ports(team_id)
            nat_ip = _advertise_host()
            abs_profile = (WORKSPACE_ROOT / team_id / ".browser-profile").resolve()
            abs_neko = (team_dir / ".neko").resolve()

            args = [
                "docker", "run", "-d", "--name", name,
                "--shm-size", "1g",
                "-p", f"{ports['http']}:8080",
                "-p", f"{ports['cdp']}:9222",
                "-p", f"{ports['udp']}:{ports['udp']}/udp",
                "-p", f"{ports['tcp']}:{ports['tcp']}/tcp",
                "-v", f"{abs_profile}:/home/neko/.config/chromium",
                "-v", f"{abs_neko / 'chromium.conf'}:/etc/neko/supervisord/chromium.conf:ro",
                "-v", f"{abs_neko / 'policy.json'}:/etc/chromium/policies/managed/policies.json:ro",
                "-e", f"NEKO_DESKTOP_SCREEN={os.environ.get('NEKO_SCREEN', '1920x1080@30')}",
                "-e", "NEKO_MEMBER_PROVIDER=multiuser",
                "-e", f"NEKO_MEMBER_MULTIUSER_ADMIN_PASSWORD={secrets_data['admin_pwd']}",
                "-e", f"NEKO_MEMBER_MULTIUSER_USER_PASSWORD={secrets_data['user_pwd']}",
                "-e", "NEKO_WEBRTC_ICELITE=true",
                "-e", f"NEKO_WEBRTC_UDPMUX={ports['udp']}",
                "-e", f"NEKO_WEBRTC_TCPMUX={ports['tcp']}",
                "-e", f"NEKO_WEBRTC_NAT1TO1={nat_ip}",
                "-e", "NEKO_SESSION_IMPLICIT_HOSTING=true",
                image,
            ]
            try:
                r = subprocess.run(args, capture_output=True, text=True, timeout=120)
                if r.returncode != 0:
                    self._fail(team_id, "container start failed: "
                               + (r.stderr or r.stdout or "unknown docker error").strip()[:300])
                    self._release_ports(ports)
                    return None
            except Exception as e:
                self._fail(team_id, f"docker run error: {e}")
                self._release_ports(ports)
                return None

            host = _advertise_host()
            info = {
                "mode": "neko",
                **ports,
                "host": host,
                "url": f"http://{host}:{ports['http']}/",
                # Auto-login embed (admin password ⇒ control). The FAQ-documented
                # params prefill credentials; embed=1 strips the surrounding UI.
                "embed_url": (f"http://{host}:{ports['http']}/"
                              f"?usr=Operator&pwd={secrets_data['admin_pwd']}&embed=1"),
                "cdp_url": f"http://127.0.0.1:{ports['cdp']}",
            }
            deadline = time.time() + 60
            while time.time() < deadline:
                if self._http_healthy(ports["http"]) and self._cdp_healthy(ports["cdp"]):
                    self._containers[team_id] = info
                    self._last_error.pop(team_id, None)
                    log.info("[neko] '%s' ready: ui=http://%s:%d cdp=:%d ice=%d/%d",
                             team_id, host, ports["http"], ports["cdp"],
                             ports["udp"], ports["tcp"])
                    return info
                time.sleep(1.0)

            # Grab the container's own logs so the reason is actionable.
            tail = ""
            try:
                tail = subprocess.run(["docker", "logs", "--tail", "25", name],
                                      capture_output=True, text=True,
                                      timeout=15).stdout[-800:]
            except Exception:
                pass
            self._fail(team_id,
                       f"container did not become healthy in 60s. Logs:\n{tail}")
            self.stop_neko(team_id)
            return None

    def _release_ports(self, ports: Dict[str, int]) -> None:
        self._ports_used -= set(ports.values())

    def stop_neko(self, team_id: str) -> None:
        with self._lock:
            info = self._containers.pop(team_id, None)
            if info:
                self._release_ports({k: v for k, v in info.items()
                                     if k in ("http", "cdp", "udp", "tcp")})
        name = self._container_name(team_id)
        try:
            subprocess.run(["docker", "rm", "-f", name],
                           capture_output=True, timeout=30)
        except Exception:
            pass

    def stop_all(self) -> None:
        for team_id in list(self._containers.keys()):
            self.stop_neko(team_id)


def resolve_team_cdp_url(team_id: str) -> Optional[str]:
    """THE seam between viewing mode and agent tooling: neko's Chromium when
    the viewer is enabled, else browser_pool's local headless Chrome. Agents
    and the dashboard relay both go through here."""
    if neko_manager.enabled_for(team_id):
        try:
            info = neko_manager.ensure_neko(team_id)
        except Exception as e:
            log.warning("[neko] ensure failed for '%s' (%s); falling back to "
                        "local headless browser", team_id, e)
            info = None
        if info:
            return info["cdp_url"]
    from teams_server.browser_pool import team_browser_manager
    return team_browser_manager.ensure_team_browser(team_id)


# Process-wide singletons.
neko_manager = NekoManager()
