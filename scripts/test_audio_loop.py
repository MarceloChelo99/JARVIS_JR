"""M3 verification: hold SPACE → record → Whisper → Piper → speakers.

No LLM involved. Verifies that the mic + STT + TTS pipeline works end-to-end
and gives you a feel for the round-trip latency.

macOS gotcha: pynput needs Accessibility permission for the terminal you're
running this from. If the recorder hangs after you press SPACE, grant access
under System Settings → Privacy & Security → Accessibility.
"""

from __future__ import annotations

import sys
import time

from jarvis_jr.audio.player import AudioPlayer
from jarvis_jr.audio.recorder import AudioRecorder
from jarvis_jr.settings import load_settings
from jarvis_jr.stt.whisper import WhisperSTT
from jarvis_jr.tts.piper import PiperTTS


def main() -> int:
    settings = load_settings()

    print(f"Loading Whisper ({settings.stt.model}, {settings.stt.compute_type})…")
    stt = WhisperSTT(model=settings.stt.model, compute_type=settings.stt.compute_type)

    print(f"Loading Piper voice ({settings.tts.voice})…")
    tts = PiperTTS(voice=settings.tts.voice, speed=settings.tts.speed)
    # Warm up: forces download + model load now, not at first synth.
    tts.synthesize("Ready.")

    recorder = AudioRecorder(
        sample_rate=settings.audio.sample_rate,
        channels=settings.audio.channels,
        input_device=settings.audio.input_device,
    )
    player = AudioPlayer(output_device=settings.audio.output_device)

    print("\nReady. Hold SPACE, speak, release. Ctrl+C to quit.\n")
    try:
        while True:
            wav = recorder.record_while_held(key="space")

            t0 = time.perf_counter()
            transcription = stt.transcribe(wav, vad_filter=settings.stt.vad_filter)
            t_stt = time.perf_counter() - t0

            text = transcription.text.strip()
            if not text:
                print("(no speech detected)\n")
                continue
            print(f"📝 {text}    [stt {t_stt:.2f}s]")

            t0 = time.perf_counter()
            spoken = tts.synthesize(text)
            t_tts = time.perf_counter() - t0
            print(f"🔊 speaking…    [tts {t_tts:.2f}s]\n")

            player.play(spoken)
    except KeyboardInterrupt:
        print("\nGoodbye.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
