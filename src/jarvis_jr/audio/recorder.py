"""Push-to-talk audio recorder."""

from __future__ import annotations

import io
import sys
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf
from pynput import keyboard


class AudioRecorder:
    """Record 16-bit PCM mono audio while a key is held."""

    # Drop the first N seconds of audio after key-down to avoid the key click.
    LEAD_TRIM_SECONDS = 0.1

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        input_device: int | str | None = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.input_device = input_device

    def record_while_held(self, key: str = "space") -> bytes:
        """Block until `key` is pressed, record while held, return WAV bytes.

        On macOS, pynput needs Accessibility permission for the running terminal
        (System Settings → Privacy & Security → Accessibility). Without it, the
        listener never sees key events and this call blocks forever.
        """
        target_key = self._resolve_key(key)
        pressed = threading.Event()
        released = threading.Event()

        def on_press(k):
            if k == target_key and not pressed.is_set():
                pressed.set()

        def on_release(k):
            if k == target_key and pressed.is_set():
                released.set()
                return False  # stop listener

        print(f"Hold {key.upper()} to record. (Ctrl+C to quit.)")
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        try:
            pressed.wait()
        except KeyboardInterrupt:
            listener.stop()
            raise

        frames: list[np.ndarray] = []

        def audio_callback(indata, _frame_count, _time_info, status):
            if status:
                # Underruns/overruns happen; not fatal, just note them.
                print(f"[recorder] {status}", file=sys.stderr)
            frames.append(indata.copy())

        sys.stdout.write("🎙  recording…")
        sys.stdout.flush()

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            callback=audio_callback,
            device=self.input_device,
        )
        stream.start()
        try:
            released.wait()
        finally:
            stream.stop()
            stream.close()
            listener.join()

        sys.stdout.write(" done.\n")
        sys.stdout.flush()

        if not frames:
            return self._silence_wav()

        audio = np.concatenate(frames, axis=0)
        trim = int(self.LEAD_TRIM_SECONDS * self.sample_rate)
        if len(audio) > trim:
            audio = audio[trim:]
        return self._to_wav_bytes(audio)

    def _to_wav_bytes(self, audio: np.ndarray) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, subtype="PCM_16", format="WAV")
        return buf.getvalue()

    def _silence_wav(self) -> bytes:
        return self._to_wav_bytes(np.zeros((self.sample_rate // 10, self.channels), dtype=np.int16))

    @staticmethod
    def _resolve_key(name: str) -> keyboard.Key | keyboard.KeyCode:
        if hasattr(keyboard.Key, name):
            return getattr(keyboard.Key, name)
        return keyboard.KeyCode.from_char(name)
