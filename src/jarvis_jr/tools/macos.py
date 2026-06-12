"""macOS control via `open` and AppleScript (osascript).

Deliberately a fixed verb set — open app, music transport, volume — rather
than arbitrary AppleScript execution, so a confused model can't do anything
worse than open the wrong app.
"""

from __future__ import annotations

import subprocess

_MUSIC_ACTIONS = {
    "play": "tell application \"Music\" to play",
    "pause": "tell application \"Music\" to pause",
    "next": "tell application \"Music\" to next track",
    "previous": "tell application \"Music\" to previous track",
}


def _run(cmd: list[str], timeout: float = 10.0) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        return f"ERROR: {(proc.stderr or proc.stdout).strip() or f'exit {proc.returncode}'}"
    return "ok"


class MacControl:
    def open_app(self, name: str) -> str:
        name = name.strip()
        if not name:
            return "ERROR: no app name given."
        result = _run(["open", "-a", name])
        return f"Opened {name}." if result == "ok" else result

    def control_music(self, action: str) -> str:
        script = _MUSIC_ACTIONS.get(action.strip().lower())
        if script is None:
            return f"ERROR: unknown action '{action}'. Use: {', '.join(_MUSIC_ACTIONS)}."
        result = _run(["osascript", "-e", script])
        return f"Music: {action}." if result == "ok" else result

    def set_volume(self, level: int) -> str:
        level = max(0, min(100, int(level)))
        result = _run(["osascript", "-e", f"set volume output volume {level}"])
        return f"Volume set to {level}." if result == "ok" else result
