# JARVIS Jr.

First-person wearable AI assistant. Push-to-talk; the assistant sees what you see
(via an iPhone Continuity Camera or other webcam) and can manage your calendar
and set timers. Local STT (Whisper) and TTS (Piper); pluggable LLM backend
(Gemini by default; Ollama for offline; Anthropic for paid).

This is the project formerly drafted as "lookout" — package name forced to
`jarvis_jr`.

## Setup

```bash
uv sync
cp .env.example .env  # then fill in the API key for the backend you picked
```

See [LLM backends](#llm-backends) for choosing/configuring a backend.

## Build milestones

The build proceeds in pinned milestones. Current status:

- [x] **M1** — package skeleton; `uv sync` succeeds; `import jarvis_jr` works.
- [x] **M2** — camera: `uv run python scripts/test_camera.py` shows a live preview
      from the iPhone Continuity Camera.
- [x] **M3** — push-to-talk audio loop (mic → Whisper → Piper → speakers).
- [x] **M4** — text-only LLM with calendar + timer tools, stdin confirmation
      for sensitive actions.
- [x] **M5** — full voice assistant: `Session` orchestrator + voice
      confirmation loop + `scripts/run_jarvis.py`.
- [x] **M6** — autocoder MVP: single-agent autonomous coding loop in
      `src/jarvis_jr/autocoder/`, runs via `scripts/autocoder.py`, defaults to
      Gemini Flash for $0 variable cost, sandboxed to the repo, commits onto a
      fresh `autocoder/<run-id>` branch.
- [ ] M7 — multi-agent (planner + coder + reviewer).
- [ ] M8 — JARVIS voice integration: tools to dispatch + query autocoder runs.
- [ ] M9 — always-on launchd service.

## Verify M1

```bash
uv sync
uv run python -c "import jarvis_jr; print('ok', jarvis_jr.__version__)"
```

## Verify M2

```bash
uv run python scripts/list_devices.py     # confirm the iPhone shows up
uv run python scripts/test_camera.py      # live preview window; 'q' quits
```

## Verify M3

```bash
uv run python scripts/test_audio_loop.py
```

Hold SPACE, speak, release. The transcription prints and Piper speaks it back.
On macOS you'll need to grant Accessibility permission to the terminal app the
first time (System Settings → Privacy & Security → Accessibility).

## Verify M4

One-time setup:

1. Pick a backend (see [LLM backends](#llm-backends) below). Add the matching
   key to `.env` (Gemini is the default and has a free tier).
2. Set up Google Calendar OAuth (see the docstring of `scripts/setup_gcal.py`):
   ```bash
   uv run python scripts/setup_gcal.py
   ```

Then:

```bash
# Read-only (no confirmation needed)
uv run python scripts/test_llm.py "what's on my calendar today?"

# Write — prompts for y/n confirmation in stdin
uv run python scripts/test_llm.py "make an event tomorrow at 3pm called dentist"

# Timer (no confirmation; default config only confirms calendar writes)
uv run python scripts/test_llm.py "set a 5 minute timer for laundry"

# Image — no calendar needed
uv run python scripts/test_llm.py --no-calendar "what's in this picture?" --image /tmp/jarvis_jr_frame.jpg

# A/B between providers without editing the config
uv run python scripts/test_llm.py --provider ollama "what's on my calendar today?"
uv run python scripts/test_llm.py --provider anthropic "what's on my calendar today?"
```

## Verify M5

The full assistant:

```bash
uv run python scripts/run_jarvis.py
```

Hold SPACE, speak, release. JARVIS Jr. captures a camera frame, runs your
question through the LLM (with vision + tools), voice-confirms any
sensitive tool calls ("I'll create an event... should I do it?" → hold
SPACE → "yes"), and speaks the reply.

Per-turn timing is printed (`stt ▍ms  capture ▍ms  llm ▍ms  tts ▍ms`) so
you can see where any latency is coming from.

## Verify M6 (autocoder)

```bash
# Make sure your worktree is clean first.
uv run python scripts/autocoder.py "Add a one-line ASCII banner to scripts/run_jarvis.py that prints 'JARVIS Jr.' at startup"
```

The autocoder will:
1. Create a fresh `autocoder/<run-id>` branch off your current HEAD.
2. Read/edit files + run bash commands until the spec is satisfied or
   `--max-iterations` (default 30) is hit.
3. Commit its work along the way.
4. Write a markdown summary to `notes/autocoder/<run-id>/summary.md`.

Inspect the diff with `git log autocoder/<run-id>`; ship via `gh pr create`;
discard with `git checkout main && git branch -D autocoder/<run-id>`.

Defaults to Gemini 2.5 Flash (free tier). Switch backends with
`--provider ollama` (local Qwen, fully free) or `--provider anthropic` (paid,
best quality on hard specs).

## LLM backends

Jarvis Jr supports three LLM backends. Pick one in `configs/default.yaml`
(`llm.provider` + `llm.model`), or override per-run with `--provider`.

### Google Gemini (default)

Free tier at aistudio.google.com. Vision + tools, generous limits.

1. Grab a free API key: https://aistudio.google.com/apikey
2. Add it to `.env`: `GOOGLE_API_KEY=...`
3. In `configs/default.yaml`:
   ```yaml
   llm:
     provider: "google"
     model: "gemini-2.5-flash"
   ```

### Ollama (local, no internet)

Runs the model on your Mac (later: Jetson). Slower than cloud, fully private,
no key required.

1. `brew install ollama` (or download from https://ollama.com/download)
2. In another terminal: `ollama serve`
3. `ollama pull qwen3.5:9b`  (~6.6GB; multimodal — vision + tool calling)
4. In `configs/default.yaml`:
   ```yaml
   llm:
     provider: "ollama"
     model: "qwen3.5:9b"
   ```

Other working options: `qwen3-vl:8b` (6.1GB, same size class) or `qwen3.5:4b`
(3.4GB, smaller/faster). Larger tags (`qwen3.5:27b` and up) **need more than
24GB of RAM in practice** — the file on disk is ~17GB but inference
context + KV cache pushes it past available memory on a 24GB Mac, causing
hard swap thrash. Don't fall back to `qwen2.5vl` — it has vision but no
tool support.

Or just run the interactive helper:
```bash
uv run python scripts/setup_ollama.py
```

### Anthropic Claude

Highest quality but paid (~$0.005/turn at current Sonnet pricing).

1. Get a key at https://console.anthropic.com/
2. Add to `.env`: `ANTHROPIC_API_KEY=sk-ant-...`
3. In `configs/default.yaml`:
   ```yaml
   llm:
     provider: "anthropic"
     model: "claude-sonnet-4-6"
   ```

## Layout

```
src/jarvis_jr/      library code
scripts/            thin entry points
configs/default.yaml
credentials/        gitignored; Google OAuth token lands here
notes/observations.md
```
