"""Local STT via faster-whisper."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from jarvis_jr.types import Transcription


class WhisperSTT:
    """Wrapper around faster-whisper that operates on in-memory WAV bytes."""

    TARGET_SAMPLE_RATE = 16000

    def __init__(self, model: str = "base.en", compute_type: str = "int8"):
        self.model_name = model
        self.compute_type = compute_type
        self.model = WhisperModel(model, compute_type=compute_type)

    def transcribe(self, wav_bytes: bytes, vad_filter: bool = True) -> Transcription:
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.TARGET_SAMPLE_RATE:
            audio = _resample_linear(audio, sr, self.TARGET_SAMPLE_RATE)

        segments, info = self.model.transcribe(audio, vad_filter=vad_filter)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return Transcription(
            text=text,
            language=info.language,
            duration_seconds=float(info.duration),
        )


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    duration = len(audio) / src_sr
    dst_len = int(round(duration * dst_sr))
    src_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)
