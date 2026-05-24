# Dev log

## 2026-05-20 — M1 + M2

- Project bootstrapped as `jarvis_jr` (package) / `jarvis-jr` (project name).
- Repo lives at `JARVIS_JR/`, replacing the original `lookout` name from the spec.
- Assistant persona name kept as "JARVIS Jr." in the (future) system prompt.

## 2026-05-21 — Heuristic: pick models by ecosystem maturity, not benchmark rank

When choosing a local model (or any model), the right pick is the newest one
whose **ecosystem support** — the inference runtime, the SDK that drives it,
the quantization tooling — is actually mature. Bleeding-edge models often
outrun their tooling by 1–2 releases.

Concrete instances from this session:

- Briefly considered Qwen 3.6 — newer benchmarks but Ollama tooling lagged.
- Initially set the Ollama default to `qwen3.5:30b-a3b` based on an
  upstream HF tag. That tag does not exist in Ollama's library — the actual
  qwen3.5 tags are `0.8b/2b/4b/9b/27b/35b/122b`. Lesson: verify the tag
  string against the runtime's own library, not the model author's release
  notes.
- I also briefly claimed Qwen 3.5 was text-only. Wrong: Ollama lists the
  full qwen3.5 family with both vision and tool capabilities. Verify by
  reading the library page, don't infer from the family name.

Settled on `qwen3.5:27b` (17GB; vision + tools) as the default, with
`qwen3.5:9b` and `qwen3-vl:8b` as smaller fallbacks. All three support
vision + tool calling, which is what the assistant needs.

**Update (same day):** 27b doesn't actually fit on a 24GB M5 in practice.
The model file is 17GB but inference needs additional headroom for context
and KV cache; user reported the machine becoming "super slow" (swap thrash)
and the smoke test timing out at 120s. Lesson: on-disk model size ≠ working
memory needed. Rule of thumb: model needs ~1.3–1.5× its file size in RAM at
modest context lengths, more for long context. So on a 24GB Mac, aim for
~12GB models or smaller. Switched default to `qwen3.5:9b` (6.6GB), with
`qwen3-vl:8b` and `qwen3.5:4b` as fallbacks.

Related: don't optimize the local-model path before the cloud-model path
works end-to-end. MLX-quantized variants on Apple Silicon may be meaningfully
faster than Ollama later — defer until the system is running.
