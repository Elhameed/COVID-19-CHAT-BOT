"""Start the JupyterLab server the Jupyter MCP server connects to.

    python scripts/start_jupyter.py

Reads JUPYTER_TOKEN from the environment, or from .claude/settings.local.json,
and generates one on first run if neither has it. See docs/jupyter-mcp.md.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SETTINGS = PROJECT_ROOT / ".claude" / "settings.local.json"
TOKEN_FILE = PROJECT_ROOT / ".jupyter_token"

HOST = os.environ.get("JUPYTER_HOST", "127.0.0.1")
PORT = os.environ.get("JUPYTER_PORT", "8888")


def _read_local_settings() -> dict:
    if not LOCAL_SETTINGS.exists():
        return {}
    try:
        return json.loads(LOCAL_SETTINGS.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def resolve_token() -> str:
    """Find the shared token, generating and persisting one if needed.

    The same value has to reach two places: this server, and the MCP server that
    Claude Code spawns. `.mcp.json` is committed and holds no secret -- it reads
    ${JUPYTER_TOKEN} from the environment, which Claude Code populates from the
    gitignored .claude/settings.local.json.
    """
    if token := os.environ.get("JUPYTER_TOKEN"):
        return token

    settings = _read_local_settings()
    if token := settings.get("env", {}).get("JUPYTER_TOKEN"):
        return token

    token = secrets.token_hex(32)
    settings.setdefault("env", {})["JUPYTER_TOKEN"] = token
    LOCAL_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_SETTINGS.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"generated a new token and wrote it to {LOCAL_SETTINGS.relative_to(PROJECT_ROOT)}")
    print("restart Claude Code so it picks the value up.")
    return token


def main() -> int:
    token = resolve_token()
    # Mirrored to a gitignored file so shell one-liners (curl, scripts) can read
    # it without parsing the settings JSON.
    TOKEN_FILE.write_text(token, encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "jupyterlab",
        "--no-browser",
        f"--ip={HOST}",
        f"--port={PORT}",
        # IdentityProvider.token, not ServerApp.token -- the latter is
        # deprecated since Jupyter Server 2.0 and warns on every start.
        f"--IdentityProvider.token={token}",
        f"--ServerApp.root_dir={PROJECT_ROOT}",
        "--ServerApp.open_browser=False",
    ]

    print(f"JupyterLab   http://{HOST}:{PORT}")
    print(f"root_dir     {PROJECT_ROOT}")
    print(f"token        {token[:6]}...{token[-4:]}  (full value in .jupyter_token)")
    print("\nLeave this running; Claude Code's `jupyter` MCP server connects to it.\n")

    try:
        return subprocess.call(cmd, cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
