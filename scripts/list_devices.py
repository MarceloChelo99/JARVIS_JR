"""List audio devices and cameras visible to the system."""

from __future__ import annotations

import sys

import sounddevice as sd

from jarvis_jr.camera import list_cameras


def main() -> int:
    print("=== Cameras ===")
    cams = list_cameras()
    if not cams:
        print("  (none detected via system_profiler)")
    for i, name in enumerate(cams):
        print(f"  [{i}] {name}")

    print("\n=== Audio devices ===")
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"  failed to query: {e}")
        return 1
    for i, d in enumerate(devices):
        kind = []
        if d["max_input_channels"] > 0:
            kind.append("in")
        if d["max_output_channels"] > 0:
            kind.append("out")
        print(f"  [{i}] {d['name']}  ({'/'.join(kind) or '—'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
