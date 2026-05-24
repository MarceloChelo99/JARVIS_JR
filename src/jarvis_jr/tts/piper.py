"""Local TTS via Piper."""

from __future__ import annotations

import io
import wave
from pathlib import Path

from piper.config import SynthesisConfig
from piper.download_voices import download_voice
from piper.voice import PiperVoice


CACHE_DIR = Path.home() / ".cache" / "jarvis_jr" / "piper"


class PiperTTS:
    """Synthesize speech to in-memory WAV bytes.

    On first use the requested voice is downloaded to ~/.cache/jarvis_jr/piper/.
    """

    def __init__(self, voice: str = "en_US-amy-medium", speed: float = 1.0):
        self.voice_name = voice
        self.speed = speed
        self._voice: PiperVoice | None = None

    def _ensure_voice(self) -> PiperVoice:
        if self._voice is not None:
            return self._voice
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        onnx_path = CACHE_DIR / f"{self.voice_name}.onnx"
        json_path = CACHE_DIR / f"{self.voice_name}.onnx.json"
        if not onnx_path.exists() or not json_path.exists():
            print(f"[tts] Downloading voice {self.voice_name} → {CACHE_DIR}")
            download_voice(self.voice_name, CACHE_DIR)
        self._voice = PiperVoice.load(onnx_path, config_path=json_path)
        return self._voice

    def _syn_config(self) -> SynthesisConfig:
        # length_scale is inverse of speed (shorter = faster).
        length_scale = None if self.speed == 1.0 else 1.0 / self.speed
        return SynthesisConfig(length_scale=length_scale)

    def synthesize(self, text: str) -> bytes:
        voice = self._ensure_voice()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file, syn_config=self._syn_config())
        return buf.getvalue()
