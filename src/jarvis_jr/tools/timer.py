"""Background timers that fire a callback when they expire."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class TimerHandle:
    id: str
    label: str
    duration_seconds: int


class TimerManager:
    """In-process timers. Each fires `on_fire(message)` when it expires."""

    def __init__(self, on_fire: Callable[[str], None] | None = None):
        self.on_fire = on_fire or self._default_on_fire
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _default_on_fire(message: str) -> None:
        print(f"\n⏰ {message}")

    def set_timer(self, duration_seconds: int, label: str) -> TimerHandle:
        timer_id = uuid.uuid4().hex[:8]
        handle = TimerHandle(id=timer_id, label=label, duration_seconds=duration_seconds)

        def fire() -> None:
            with self._lock:
                self._timers.pop(timer_id, None)
            self.on_fire(f"Timer '{label}' is up.")

        t = threading.Timer(duration_seconds, fire)
        t.daemon = True
        t.start()
        with self._lock:
            self._timers[timer_id] = t
        return handle

    def cancel_all(self) -> None:
        with self._lock:
            for t in self._timers.values():
                t.cancel()
            self._timers.clear()
