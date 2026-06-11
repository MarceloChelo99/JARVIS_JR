"""Top-level orchestrator for one running assistant session."""

from __future__ import annotations

import re
import threading
import time
import traceback
from dataclasses import dataclass

import cv2
import numpy as np

from jarvis_jr.audio.player import AudioPlayer
from jarvis_jr.audio.recorder import AudioRecorder
from jarvis_jr.camera import Camera, open_camera
from jarvis_jr.confirm import VoiceConfirmer
from jarvis_jr.llm import LLMClient, build_llm_client
from jarvis_jr.llm.prompts import build_system_prompt
from jarvis_jr.settings import Settings, load_settings
from jarvis_jr.stt.whisper import WhisperSTT
from jarvis_jr.tools.calendar import CalendarError, GoogleCalendar
from jarvis_jr.tools.mcp_client import MCPManager
from jarvis_jr.tools.registry import ToolRegistry
from jarvis_jr.tools.timer import TimerManager
from jarvis_jr.tts.piper import PiperTTS
from jarvis_jr.types import Turn


# Words/phrases that suggest the user is asking about what they're seeing.
# Vision prefill is the dominant latency cost on local models, so we only
# attach the camera frame when the question sounds visual (attach_policy=auto).
_VISION_PHRASES = (
    "what is this", "what's this", "what is that", "what's that",
    "what am i looking", "what am i holding", "what do you see",
    "can you see", "in front of me", "on the screen", "on my screen",
    "this thing", "that thing", "over there", "right here",
)
_VISION_WORDS = re.compile(
    r"\b(see|look|looking|picture|image|photo|camera|color|colour|read|sign|"
    r"label|text|screen|wearing|holding|object|describe)\b"
)


def _looks_visual(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _VISION_PHRASES) or bool(_VISION_WORDS.search(t))


@dataclass
class _LatencyReport:
    stt_s: float
    capture_s: float
    llm_s: float
    tts_total_s: float
    tts_first_s: float | None = None

    def __str__(self) -> str:
        if self.tts_first_s is not None and self.tts_first_s < self.tts_total_s:
            tts = (
                f"tts {self.tts_total_s*1000:5.0f}ms "
                f"(first@{self.tts_first_s*1000:.0f}ms)"
            )
        else:
            tts = f"tts {self.tts_total_s*1000:5.0f}ms"
        return (
            f"stt {self.stt_s*1000:4.0f}ms  "
            f"capture {self.capture_s*1000:4.0f}ms  "
            f"llm {self.llm_s*1000:5.0f}ms  "
            f"{tts}"
        )


class Session:
    """Wires the camera, mic, STT, TTS, LLM, and tools into one assistant loop."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()

        print("[session] loading Whisper…")
        self.stt = WhisperSTT(
            model=self.settings.stt.model, compute_type=self.settings.stt.compute_type
        )

        print(f"[session] loading Piper voice ({self.settings.tts.voice})…")
        self.tts = PiperTTS(voice=self.settings.tts.voice, speed=self.settings.tts.speed)
        self.tts.synthesize("warm up")  # download + load now, not on first speak

        self.recorder = AudioRecorder(
            sample_rate=self.settings.audio.sample_rate,
            channels=self.settings.audio.channels,
            input_device=self.settings.audio.input_device,
        )
        self.player = AudioPlayer(output_device=self.settings.audio.output_device)

        self.camera: Camera | None = None
        if self.settings.camera.attach_policy != "never":
            print(f"[session] opening camera (matching {self.settings.camera.name_contains!r})…")
            self.camera = open_camera(
                name_contains=self.settings.camera.name_contains,
                width=self.settings.camera.width,
                height=self.settings.camera.height,
            )
        else:
            print("[session] vision disabled (attach_policy=never); camera not opened.")

        self.calendar: GoogleCalendar | None = None
        try:
            self.calendar = GoogleCalendar(
                calendar_id=self.settings.calendar.default_calendar_id,
                default_duration_minutes=self.settings.calendar.default_duration_minutes,
                default_timezone=self.settings.calendar.default_timezone,
            )
            print("[session] Google Calendar connected.")
        except CalendarError as e:
            print(f"[session] {e}")
            print("[session] Continuing without calendar tools.")

        self.mcp: MCPManager | None = None
        if self.settings.mcp.servers:
            print(f"[session] starting {len(self.settings.mcp.servers)} MCP server(s)…")
            self.mcp = MCPManager(
                servers=self.settings.mcp.servers,
                call_timeout_s=self.settings.mcp.call_timeout_s,
            )
            self.mcp.start()

        self._speak_lock = threading.Lock()
        self._last_ttfa: float | None = None
        self.timer_manager = TimerManager(on_fire=self.speak)
        self.registry = ToolRegistry(
            calendar=self.calendar, timer_manager=self.timer_manager, mcp=self.mcp
        )

        self.confirmer = VoiceConfirmer(
            speak=self.speak,
            recorder=self.recorder,
            stt=self.stt,
            yes_words=self.settings.confirmation.yes_words,
            no_words=self.settings.confirmation.no_words,
        )

        self.llm: LLMClient = self._build_llm()

    def _build_llm(self) -> LLMClient:
        provider = self.settings.llm.provider
        per_provider = getattr(self.settings.llm, provider, {}) or {}
        extras = {k: v for k, v in per_provider.items() if k != "model"}
        print(f"[session] LLM: {provider} / {self.settings.llm.model}")
        return build_llm_client(
            provider=provider,
            model=self.settings.llm.model,
            tools=self.registry,
            confirmer=self.confirmer,
            system_prompt=build_system_prompt(),
            require_confirmation_for=self.settings.confirmation.require_for,
            max_tokens=self.settings.llm.max_tokens,
            **extras,
        )

    def speak(self, text: str) -> None:
        """Synthesize and play `text`. Thread-safe (single lock around the
        whole synth+play). Currently non-streaming: synthesizes the full
        reply, then plays. Trades TTFA for guaranteed-clean audio.

        After returning, `self._last_ttfa` holds the seconds from call entry
        to start of first audio (== synth time, in the non-streaming path).
        """
        self._last_ttfa = None
        if not text or not text.strip():
            return
        with self._speak_lock:
            start = time.perf_counter()
            wav = self.tts.synthesize(text)
            self._last_ttfa = time.perf_counter() - start
            self.player.play(wav)

    def one_turn(self) -> Turn | None:
        """One full user turn: listen → see → LLM → speak. Returns None if no speech."""
        wav = self.recorder.record_while_held(key="space")

        t0 = time.perf_counter()
        transcription = self.stt.transcribe(wav, vad_filter=self.settings.stt.vad_filter)
        t_stt = time.perf_counter() - t0

        user_text = transcription.text.strip()
        if not user_text:
            print("(no speech detected)\n")
            return None
        print(f"📝 USER: {user_text}")

        t0 = time.perf_counter()
        frame_image: np.ndarray | None = None
        jpeg_bytes: bytes | None = None
        if self.camera is not None and self._should_attach_frame(user_text):
            try:
                frame = self.camera.grab()
                frame_image = frame.image
                jpeg_bytes = self._encode_jpeg(frame.image)
            except Exception as e:
                print(
                    f"[session] camera unavailable for this turn ({e}); "
                    "continuing without image."
                )
        t_capture = time.perf_counter() - t0

        t0 = time.perf_counter()
        reply = self.llm.submit(user_text, image_jpeg_bytes=jpeg_bytes)
        t_llm = time.perf_counter() - t0

        t_tts_total = 0.0
        t_tts_first: float | None = None
        if reply:
            print(f"🗣  ASSIST: {reply}")
            t0 = time.perf_counter()
            self.speak(reply)
            t_tts_total = time.perf_counter() - t0
            t_tts_first = self._last_ttfa

        print(
            f"   ⏱  {_LatencyReport(t_stt, t_capture, t_llm, t_tts_total, t_tts_first)}\n"
        )
        return Turn(user_text=user_text, user_image=frame_image, assistant_text=reply)

    def run_forever(self) -> None:
        """Main loop. Recovers from per-turn errors but bubbles KeyboardInterrupt."""
        while True:
            try:
                self.one_turn()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[session] turn failed: {type(e).__name__}: {e}")
                traceback.print_exc()
                # Brief pause so a fast-failing loop doesn't spin.
                time.sleep(0.5)

    def close(self) -> None:
        self.timer_manager.cancel_all()
        try:
            if self.camera is not None:
                self.camera.close()
        except Exception:
            pass
        try:
            self.player.close()
        except Exception:
            pass
        try:
            if self.mcp is not None:
                self.mcp.stop()
        except Exception:
            pass

    def _should_attach_frame(self, user_text: str) -> bool:
        policy = self.settings.camera.attach_policy
        if policy == "always":
            return True
        if policy == "never":
            return False
        return _looks_visual(user_text)

    def _encode_jpeg(self, image_bgr: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(
            ".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.settings.camera.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("Failed to JPEG-encode camera frame")
        return buf.tobytes()
