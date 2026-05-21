"""Synchronous audio playback."""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator

import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioPlayer:
    """Play WAV bytes through the system output device, blocking until done.

    Two modes:
    - `play(wav_bytes)`: one-shot — convenient for single utterances.
    - `play_stream(iter_of_chunks)`: open one OutputStream and feed many WAV
      chunks through it back-to-back. Eliminates the click/gap that `play()`
      produces between sequential calls (because each `play()` opens and
      closes its own stream).
    """

    def __init__(self, output_device: int | str | None = None):
        self.output_device = output_device

    # Pad each utterance with this much silence so PortAudio's tail-replay
    # behavior on macOS doesn't echo the last syllable ("ready" → "ready-dy-dy").
    TAIL_SILENCE_SECONDS = 0.2

    def play(self, wav_bytes: bytes) -> None:
        data, samplerate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = np.ascontiguousarray(data, dtype="float32")
        if self.TAIL_SILENCE_SECONDS > 0:
            tail = np.zeros(int(self.TAIL_SILENCE_SECONDS * samplerate), dtype="float32")
            data = np.concatenate([data, tail])

        # Explicit OutputStream so we control draining. `with stream:` calls
        # stop() (which blocks until the buffer is drained) and then close()
        # on exit — no chance of the device looping the tail.
        stream = sd.OutputStream(
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            device=self.output_device,
        )
        with stream:
            stream.write(data)

    def play_stream(
        self,
        chunks: Iterator[bytes],
        on_first_audio: Callable[[], None] | None = None,
    ) -> None:
        """Play a sequence of WAV-byte chunks through a single OutputStream.

        Sample rate is taken from the first chunk; subsequent chunks must
        match. `on_first_audio` is called exactly once, immediately after the
        first chunk's samples are written (so the caller can capture
        time-to-first-audio).
        """
        stream: sd.OutputStream | None = None
        try:
            for chunk_bytes in chunks:
                data, sr = sf.read(io.BytesIO(chunk_bytes), dtype="float32", always_2d=False)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                data = np.ascontiguousarray(data, dtype="float32")
                if stream is None:
                    stream = sd.OutputStream(
                        samplerate=sr,
                        channels=1,
                        dtype="float32",
                        device=self.output_device,
                    )
                    stream.start()
                stream.write(data)
                if on_first_audio is not None:
                    on_first_audio()
                    on_first_audio = None
        finally:
            if stream is not None:
                # Drain the buffer before closing so the tail of the last
                # chunk isn't cut off.
                stream.stop()
                stream.close()
