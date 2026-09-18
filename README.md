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
- [x] **M6.5** — evals: graded agent tasks in `src/jarvis_jr/evals/`
      (checks + tool-trajectory + LLM judge), runs via `scripts/run_evals.py`.
      Merged in from the standalone Harness project so M7 can be measured.
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

## Verify M6.5 (evals)

```bash
uv run pytest tests/evals                       # no model needed; runs on fakes
uv run python scripts/run_evals.py --profile evals   # live: LM Studio server running
```

Each YAML in `evals/tasks/` is one graded task: a prompt, an optional fixture
directory copied into a fresh sandbox workspace, and how to grade the run —
deterministic `checks` (file_exists / file_contains / file_matches /
answer_contains), an `expected_tools` ordered subsequence plus
`forbidden_tools`, and a `rubric` for the LLM judge. Results land in
`notes/evals/<timestamp>/` with a per-task workspace and `report.json`
(verdicts, tool sequence, full trajectory).

The agent sees the eval-side sandbox tools (file read/write/edit/list,
fetch_url, read-only Gmail when `GMAIL_*` is set) plus, with `--profile
<name>`, the assistant's own tools from that `configs/default.yaml` profile —
the `evals` profile is a read-only subset. `--graders checks` skips the model
calls for grading; `--model` / `--judge-model` A/B different models against
the same suite (defaults come from `llm.ollama` in the config).

Adding an eval-side tool: one class in `evals/tools.py` with `requires_env`,
appended to `ALL_TOOLS`. Adding an assistant tool: nothing — the registry
shim (`registry_tools`) picks it up through the profile.

## Trading (agent vs. a simulated market)

```bash
uv run pytest tests/trading                                   # contract + unit tests, no model
uv run python scripts/run_trading_episode.py evals/scenarios/dip_and_recover.yaml
```

`src/jarvis_jr/trading/` gives the agent five tools — `get_quote`,
`get_portfolio`, `place_order`, `cancel_order`, `list_orders` — over any
`Broker` (`trading/domain.py`). v1 ships `InMemoryBroker`: a daily price
replay with market orders (fill now, plus slippage/commission) and limit
orders (fill on a later day if the price crosses). The episode runner owns
the clock: one agent turn per trading day, then `advance()`. Guardrails
(max position %, max order notional, allowed symbols, long-only) run in the
tool layer before any order reaches a broker, so the model cannot argue its
way past them.

A scenario (`evals/scenarios/*.yaml`) is prices + cash + limits +
expectations (`min_end_equity`, `max_drawdown`, `max_rejections`,
`min_orders`). Reports land in `notes/evals/trading/<timestamp>/` with
per-day trajectories and P&L / drawdown metrics.

**Research feeds.** A scenario's `data:` block gives the agent research tools
alongside the trading ones: `get_price_history` (from the broker — cannot
see the future), `get_macro` (FRED, with ALFRED vintages so a figure only
appears once it was actually released; needs `FRED_API_KEY`), `get_news`
(dated JSONL fixtures per scenario), and three SecFiler-backed tools —
`get_filing_signals` (drift / risk-theme / sector features), `get_filing_summary`
(Business, Risk Factors, MD&A) and `get_risk_headings` — reading SecFiler's
data directory (`SECFILER_DATA_DIR`) and filtering on **filing date**, not
period end. Every feed obeys one rule, enforced by `tests/trading/test_feeds.py`:
nothing dated after the simulation day is ever returned. Feeds whose
env vars are missing are skipped with a warning, not a crash.

**Research playbook (`src/jarvis_jr/research/`).** "What usually happens
after this kind of release?" — an event study, not a forecast. Macro events
come from FRED/ALFRED *vintages* (first-print value + real release date, no
NLP: the label is just whether the print came in above or below a naive
expectation); reactions are measured on daily bars from Tiingo. The agent
tool `get_event_playbook(series, direction)` returns mean/median/up-rate and
worst/best over 1 and 5 days for SPY/QQQ/XLP/TLT, computed only from events
whose reaction window closed before the simulated day.

```bash
uv run python scripts/research_fetch.py             # bars (TIINGO_API_KEY) + vintages (FRED_API_KEY)
uv run python scripts/research_playbook.py CPIAUCSL --as-of 2024-06-01
```

**Event taxonomy and generated scenarios.** `evals/events.yaml` defines event
*types* — where instances come from (`fred_vintage`, `fred_daily_step`,
`threshold`, `edgar`), how they're labeled (pure rules over trailing windows:
`surprise_sign`, `surprise_tercile`, `sign_of_change`, `bps_bucket`,
`level_bucket`, `feature_sign`), and the scenario window. Nothing is labeled
from headlines. Scenarios are stamped from real events with real bars and
real first prints:

```bash
uv run python scripts/research_fetch.py --symbols --series DFEDTARU VIXCLS     # daily series for FOMC / VIX
uv run python scripts/research_scenarios.py fomc_decision --list --since 2022-01-01
uv run python scripts/research_scenarios.py cpi_release --direction above --magnitude large --since 2021-01-01
uv run python scripts/run_trading_episode.py evals/scenarios/generated/cpi_release   # run the family -> pass rate
```

Hand-written scenarios (`cpi_shock`, `dip_and_recover`) remain as plumbing
tests; generated families are the evidence.

**Reaction graph.** Events are grouped by the cross-asset *signature* they
produced, not by source: `inflation_shock` (stocks ↓ bonds ↓), `growth_scare`
(↓ ↑), `goldilocks` (↑ ↑), `reflation` (↑ ↓), `muted` — defined by sign
rules, no fitting. `scripts/research_reactions.py [type] --as-of DATE` prints
the event → archetype distribution over a trailing window; the agent gets the
same via `get_reaction_profile(event_type, label)` (scenario `data: {reactions:
taxonomy}`). Compare `--as-of 2019-12-31` with `--as-of 2023-06-01` to see the
regime shift the trailing window is there to capture. A real dated-news archive plugs in
via `data: {news: "python:your.module:YourFeed"}` — any class with
`headlines(query, as_of, n)`.

**Env and sizing policy.** `trading/env.py` is a Gym-shaped `reset()/step()`
over a scenario (action = target exposure, reward = log equity change) so a
learned policy and the LLM agent are judged on identical scenarios.
`research/sizing.py` is the one learned component: an exposure bandit whose
arm is the mean-variance identity over similar past events (no fitting,
point-in-time). It sizes risk; it does not pick names. Walk-forward it
against constant baselines — generated families are a test set, never
training data:

```bash
uv run python scripts/research_sizing.py fomc_decision --since 2019-01-01 --rows
uv run python scripts/research_sizing.py cpi_release --since 2021-01-01
```

To plug in a real exchange simulator (or later a paper/live broker):
implement the five `Broker` methods plus `advance()`, then subclass
`tests/trading/broker_contract.py::BrokerContract` with a `make_broker`
factory — green contract tests mean the agent, tools, and episode runner
work unchanged. Nothing in this package places real trades; a live broker
adapter should additionally go behind `confirmation.require_for` and a
kill-switch env check.

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

1. `brew install ollama` (or download from https://ollama.com/download).
   **Needs Ollama ≥ 0.20.2** for the gemma4 tool-call fix — `brew upgrade ollama`
   if you installed earlier.
2. In another terminal: `ollama serve`
3. `ollama pull gemma4:12b`  (~7.6GB; vision + tools + thinking, 256K context)
4. In `configs/default.yaml`:
   ```yaml
   llm:
     provider: "ollama"
     model: "gemma4:12b"
   ```

Other working options: `qwen3.5:9b` (6.6GB, vision + tools), `qwen3-vl:8b`
(6.1GB), or `qwen3.5:4b` (3.4GB, smaller/faster). Larger tags (`gemma4:26b`,
`qwen3.5:27b` and up) **need more than 24GB of RAM in practice** — the file on
disk is ~17-18GB but inference context + KV cache pushes it past available
memory on a 24GB Mac, causing hard swap thrash. Don't fall back to `qwen2.5vl`
or `gemma3` — they have vision but no tool support.

Note: gemma4 is a thinking model. The client strips its
`<|channel>thought ... <channel|>` blocks before TTS, and `max_tokens` is set
to 2048 since reasoning tokens count against the cap. Thinking adds latency
per turn; if it's too slow for voice use, qwen3.5:9b is the faster fallback.

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
src/jarvis_jr/evals/  eval harness (agent loop, graders, registry shim)
src/jarvis_jr/trading/  broker protocol, in-memory exchange, guardrails, feeds, episode runner
src/jarvis_jr/research/ bars store, macro events from vintages, event studies, playbook
evals/tasks/        graded task specs (+ evals/tasks_gmail/ needing creds)
scripts/            thin entry points
configs/default.yaml
credentials/        gitignored; Google OAuth token lands here
notes/observations.md
notes/DECISIONS.md  design bets with grade-by dates
notes/evals/        gitignored; per-run eval workspaces + report.json
```
