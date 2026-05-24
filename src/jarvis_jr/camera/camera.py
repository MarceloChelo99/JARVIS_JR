"""Camera abstraction with macOS Continuity Camera support."""

from __future__ import annotations

import platform
import subprocess
import time
from abc import ABC, abstractmethod
from datetime import datetime

import cv2

from jarvis_jr.types import CapturedFrame


class CameraError(RuntimeError):
    """Raised when camera open/read fails."""


class Camera(ABC):
    """Replaceable camera interface."""

    @abstractmethod
    def open(self) -> None:
        """Open the underlying device."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying device."""

    @abstractmethod
    def grab(self) -> CapturedFrame:
        """Read one frame from the camera."""

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class WebcamCamera(Camera):
    """OpenCV-backed camera, using AVFoundation on macOS.

    Per-turn lifecycle: each `grab()` opens the device, reads a frame, and
    releases it. Slower than a persistent capture (~500-800ms per grab on
    macOS), but reliable — keeping the camera open between turns lets the OS
    park it and subsequent reads block indefinitely waiting for a frame that
    isn't coming. Per-turn open sidesteps that entirely.
    """

    # Reads to attempt per grab until AVFoundation returns a real frame.
    WARMUP_FRAMES = 20
    WARMUP_GAP = 0.05  # seconds between warmup reads

    def __init__(self, index: int, width: int = 1280, height: int = 720, name: str | None = None):
        self.index = index
        self.width = width
        self.height = height
        self.name = name
        self._opened = False

    @staticmethod
    def _backend() -> int:
        return cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY

    def open(self) -> None:
        """Verify the device is accessible at all and warn loudly otherwise.
        The actual capture happens lazily in grab().
        """
        cap = cv2.VideoCapture(self.index, self._backend())
        if not cap.isOpened():
            raise CameraError(f"Failed to open camera at index {self.index}")
        ok = False
        for _ in range(self.WARMUP_FRAMES):
            ok, _ = cap.read()
            if ok:
                break
            time.sleep(self.WARMUP_GAP)
        cap.release()
        if not ok:
            print(
                f"[camera] WARNING: camera at index {self.index} ({self.name or '?'}) "
                "produced no frames during warmup. Most likely causes:\n"
                "  1. Camera permission not granted to your terminal. Fix:\n"
                "     System Settings → Privacy & Security → Camera → enable for your\n"
                "     terminal app (Terminal/iTerm/PyCharm), then fully quit and relaunch it.\n"
                "  2. Another process is holding the camera (Zoom, FaceTime, Photo Booth,\n"
                "     a previous crashed run). Close them or restart the Mac.\n"
                "  3. Continuity Camera mid-handshake. Wake the iPhone or disable it in\n"
                "     System Settings → General → AirDrop & Handoff.\n"
                "Vision-using turns will fall back to text-only until this is fixed."
            )
        self._opened = True

    def close(self) -> None:
        self._opened = False

    def grab(self) -> CapturedFrame:
        if not self._opened:
            raise CameraError("Camera is not open; call open() first")
        cap = cv2.VideoCapture(self.index, self._backend())
        if not cap.isOpened():
            raise CameraError(f"Failed to open camera at index {self.index}")
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            frame = None
            ok = False
            for _ in range(self.WARMUP_FRAMES):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                time.sleep(self.WARMUP_GAP)
            if not ok or frame is None:
                raise CameraError("Failed to read a frame on grab() — camera may have gone idle.")
            h, w = frame.shape[:2]
            return CapturedFrame(image=frame, timestamp=datetime.now(), width=w, height=h)
        finally:
            cap.release()


def list_cameras() -> list[str]:
    """Return camera names in the order macOS exposes them.

    On macOS the order tends to match the cv2 index space, so the i-th name
    corresponds to index i in cv2.VideoCapture. Not guaranteed across OS
    versions, but reliable enough for the iPhone Continuity Camera case.
    """
    if platform.system() != "Darwin":
        return []
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return []

    names: list[str] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.rstrip()
        if not line or not line.endswith(":"):
            continue
        stripped = line.strip().rstrip(":")
        # Skip the section header and any 'Model ID:'-style sub-keys (those
        # appear deeply indented; camera names are indented exactly 4 spaces).
        leading_spaces = len(line) - len(line.lstrip(" "))
        if leading_spaces == 4 and stripped.lower() != "camera":
            names.append(stripped)
    return names


def open_camera(
    name_contains: str | None = None,
    width: int = 1280,
    height: int = 720,
    index: int | None = None,
) -> WebcamCamera:
    """Factory: find a camera by substring match on its name, or by explicit index.

    If `index` is given it wins. Otherwise scans `list_cameras()` for a
    case-insensitive substring match on `name_contains`. Falls back to index 0
    if no match (with a warning printed).
    """
    if index is not None:
        cam = WebcamCamera(index=index, width=width, height=height)
        cam.open()
        return cam

    names = list_cameras()
    chosen_index = 0
    chosen_name: str | None = names[0] if names else None

    if name_contains:
        needle = name_contains.lower()
        for i, name in enumerate(names):
            if needle in name.lower():
                chosen_index = i
                chosen_name = name
                break
        else:
            if names:
                print(
                    f"[camera] No camera matched '{name_contains}'. "
                    f"Available: {names}. Falling back to index 0 ({names[0]})."
                )
            else:
                print("[camera] system_profiler returned no cameras; trying index 0 blindly.")

    cam = WebcamCamera(index=chosen_index, width=width, height=height, name=chosen_name)
    cam.open()
    return cam
