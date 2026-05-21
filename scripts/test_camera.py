"""Open the configured camera and show a live preview window.

Press 'q' to quit, 's' to save the current frame to /tmp/jarvis_jr_frame.jpg.
"""

from __future__ import annotations

import sys

import cv2

from jarvis_jr.camera import list_cameras, open_camera
from jarvis_jr.settings import load_settings


def main() -> int:
    settings = load_settings()
    print(f"Cameras detected: {list_cameras()}")
    print(f"Opening camera matching: {settings.camera.name_contains!r}")

    try:
        cam = open_camera(
            name_contains=settings.camera.name_contains,
            width=settings.camera.width,
            height=settings.camera.height,
        )
    except Exception as e:
        print(f"Failed to open camera: {e}")
        return 1

    print("Camera open. Press 'q' to quit, 's' to save a snapshot.")
    try:
        while True:
            frame = cam.grab()
            cv2.imshow("JARVIS Jr. — camera preview", frame.image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = "/tmp/jarvis_jr_frame.jpg"
                cv2.imwrite(path, frame.image)
                print(f"Saved {path} ({frame.width}x{frame.height})")
    finally:
        cam.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
