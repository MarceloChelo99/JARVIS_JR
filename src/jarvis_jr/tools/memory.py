"""Persistent memory: small JSON-backed fact store.

Lets the assistant remember facts across sessions ("my gym code is 4521",
"I'm allergic to penicillin") and recall them later. Searches are simple
case-insensitive substring matches — at assistant-memory scale that beats
embeddings for predictability and zero latency.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path


class MemoryStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._facts: list[dict[str, str]] = []
        if self.path.exists():
            try:
                self._facts = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                self._facts = []

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._facts, indent=2, ensure_ascii=False))
        tmp.replace(self.path)

    def remember(self, content: str) -> str:
        content = content.strip()
        if not content:
            return "ERROR: nothing to remember."
        with self._lock:
            fact = {
                "id": uuid.uuid4().hex[:8],
                "content": content,
                "created": datetime.now().isoformat(timespec="seconds"),
            }
            self._facts.append(fact)
            self._flush()
        return f"Remembered (id {fact['id']}): {content}"

    def recall(self, query: str = "", max_results: int = 10) -> str:
        q = (query or "").strip().lower()
        with self._lock:
            if q:
                hits = [f for f in self._facts if q in f["content"].lower()]
            else:
                hits = list(self._facts)
        if not hits:
            return "No matching memories." if q else "No memories stored yet."
        hits = hits[-max_results:]
        return json.dumps(
            [{"id": f["id"], "content": f["content"], "created": f["created"]} for f in hits],
            ensure_ascii=False,
        )

    def forget(self, id_or_query: str) -> str:
        needle = id_or_query.strip().lower()
        if not needle:
            return "ERROR: specify an id or text to forget."
        with self._lock:
            before = len(self._facts)
            self._facts = [
                f
                for f in self._facts
                if f["id"] != needle and needle not in f["content"].lower()
            ]
            removed = before - len(self._facts)
            if removed:
                self._flush()
        return f"Forgot {removed} memor{'y' if removed == 1 else 'ies'}."
