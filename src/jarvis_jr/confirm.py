"""Confirmation prompts for sensitive tool calls.

Two flavors live here:
- `stdin_confirmer`: reads y/n from the terminal. Used by `scripts/test_llm.py`.
- `VoiceConfirmer`:  speaks the proposal via TTS, listens for yes/no via STT,
                     re-prompts on ambiguity up to twice, defaults to no.
"""

from __future__ import annotations

import string
from collections.abc import Callable, Iterable

from jarvis_jr.audio.recorder import AudioRecorder
from jarvis_jr.stt.whisper import WhisperSTT


Confirmer = Callable[[str], bool]
Speaker = Callable[[str], None]


def stdin_confirmer(prompt: str) -> bool:
    """Print the proposal and read y/n from stdin. Default is no."""
    print(f"\n💬 {prompt}")
    while True:
        try:
            ans = input("Confirm? [y/N]: ").strip().lower()
        except EOFError:
            return False
        if ans in {"y", "yes"}:
            return True
        if ans in {"", "n", "no"}:
            return False
        print("Please answer y or n.")


def always_yes(prompt: str) -> bool:  # noqa: ARG001
    """Confirmer that auto-approves. Use only in tests."""
    return True


def _normalize(s: str) -> str:
    return s.lower().strip().translate(str.maketrans("", "", string.punctuation))


def parse_yes_no(text: str, yes_words: Iterable[str], no_words: Iterable[str]) -> bool | None:
    """Return True/False/None based on whether `text` reads as yes / no / ambiguous.

    Strategy: prefer prefix matches with the longest phrase wins. Falls back
    to whole-word membership. "No" is checked before "yes" so a denial doesn't
    get drowned out by a stray affirmative ("no, don't do it" → False).
    Input and word list are normalized the same way (lowercase + punctuation
    stripped) so e.g. "don't." matches the config word "don't".
    """
    cleaned = _normalize(text)
    if not cleaned:
        return None

    no_sorted = sorted({_normalize(w) for w in no_words if _normalize(w)}, key=len, reverse=True)
    yes_sorted = sorted({_normalize(w) for w in yes_words if _normalize(w)}, key=len, reverse=True)

    for w in no_sorted:
        if cleaned == w or cleaned.startswith(w + " "):
            return False
    for w in yes_sorted:
        if cleaned == w or cleaned.startswith(w + " "):
            return True

    tokens = cleaned.split()
    token_set = set(tokens)
    padded = " " + cleaned + " "
    for w in no_sorted:
        if " " in w:
            if f" {w} " in padded:
                return False
        elif w in token_set:
            return False
    for w in yes_sorted:
        if " " in w:
            if f" {w} " in padded:
                return True
        elif w in token_set:
            return True
    return None


class VoiceConfirmer:
    """Speak a proposal, listen for yes/no, retry on ambiguity, default to no.

    Implements the `Confirmer = Callable[[str], bool]` interface, so it can be
    handed directly to any LLMClient. `speak` is injected so the host (e.g.
    Session) can serialize all TTS through a single locked entry point.
    """

    MAX_RETRIES = 2

    def __init__(
        self,
        speak: Speaker,
        recorder: AudioRecorder,
        stt: WhisperSTT,
        yes_words: Iterable[str],
        no_words: Iterable[str],
        key: str = "space",
    ):
        self.speak = speak
        self.recorder = recorder
        self.stt = stt
        self.yes_words = list(yes_words)
        self.no_words = list(no_words)
        self.key = key

    def __call__(self, prompt: str) -> bool:
        self.speak(prompt)
        for attempt in range(self.MAX_RETRIES + 1):
            wav = self.recorder.record_while_held(key=self.key)
            transcription = self.stt.transcribe(wav)
            heard = transcription.text.strip()
            print(f"   (heard: {heard!r})")
            decision = parse_yes_no(heard, self.yes_words, self.no_words)
            if decision is not None:
                return decision
            if attempt < self.MAX_RETRIES:
                self.speak("Sorry, was that a yes or a no?")
        self.speak("I'll take that as a no.")
        return False
