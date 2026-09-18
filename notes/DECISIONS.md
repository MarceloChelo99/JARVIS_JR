# Decisions

Each entry is a bet. Grade them: when the listed observation happens (or doesn't) by the review date, note the outcome here. Entries before 2026-09-18 were made in the standalone Harness project and carried over.

## 2026-09-18 — No RL trader; a Gym-shaped env + a closed-form exposure bandit instead
**Bet:** RL on daily prices fails here for three measured reasons — ~4,200 bars per ticker is far too few rewards, the reaction graph shows regimes flip, and `InMemoryBroker` has exploitable artifacts (close fills, no impact). So: `trading/env.py` is a Gym-shaped `reset/step` over the same broker, guardrails and clock the LLM uses (any learned policy is comparable to the agent on the same families), and the only learned component sizes RISK: `research/sizing.py`, state = (event type, direction), arm = exposure in {0, .3, .6, 1}, chosen by the mean-variance identity e* = mean/(λ·var) over a trailing window — no fitting, point-in-time, evaluated walk-forward with the test event never informing its own advice.
**First walk-forward (real data):** fine state (direction/magnitude × VIX regime) fell back to the prior 90% of the time → too fine. Direction-only, no regime, 6y window: CPI 2021+ (n=68) policy +16.4% cumulative vs +14.5% const-0.6 vs +24.2% const-1.0 (an up-market punishes any de-risking; same worst case −8.7%). FOMC 2019+ (n=22): policy −9.3% vs −19.0% const-0.6 vs −31.7% const-1.0, worst −6.8% vs −14.5% — it learned to step back around the 2022 hikes. Conclusion: sizing earns its keep on policy days, not print days. Exposed to the agent as `get_exposure_recommendation`; generated families remain a TEST set.
**Grade by 2027-01:** on the generated FOMC family, does the LLM agent with the sizing tool end closer to the recommended exposure and with lower drawdown than without? If the policy ever beats const-1.0 on CPI in an up-market, suspect leakage before believing it.

## 2026-09-18 — Group events by reaction signature, not by source; archetypes by identity, clustering only as a check
**Bet:** the useful grouping is the cross-asset signature the market printed, defined by the signs of stocks (SPY) and bonds (TLT): inflation_shock (↓↓), growth_scare (↓↑), goldilocks (↑↑), reflation (↑↓), muted. `research/reactions.py` maps every (event_type, label) to a distribution over these on a trailing window — the event→reaction graph — exposed as `get_reaction_profile`. First real graph (as of 2023-06): large upside CPI → 45% inflation_shock; large downside CPI → 62% growth_scare with bonds +0.9%; VIX spells → 67% growth_scare, SPY −2.5%; and the same "large upside CPI" as of 2019-12 was 67% goldilocks — the regime shift is visible, which is the point. Macro taxonomy broadened to 12 scheduled types (PCE, PPI, retail sales, GDP, claims, sentiment, housing, industrial production added). Deliberately NOT built: k-means on reaction vectors (3-year windows hold ~36 events; clusters would be unstable and unreadable to a 26B model) — if used, it validates the four archetypes on long histories. Event→next-event edges (hot CPI → hike odds) are the next layer.
**Grade by 2027-01:** if the agent with `get_reaction_profile` sizes post-event exposure better than with the per-series playbook alone on the generated CPI family, keep both; if profiles and playbook disagree systematically, the archetype thresholds (15 bp "muted") need tuning. If a fifth archetype keeps appearing in a long-history cluster check, add it by identity.

## 2026-09-18 — Event taxonomy as data; labels from data, never headlines; scenarios stamped from real events
**Bet:** `evals/events.yaml` is the single place event types are defined (source kind + label rules + scenario window + playbook minimums). Instances are extracted mechanically — FRED vintages for releases, `DFEDTARU` steps for FOMC decisions (the whole 2022–23 cycle falls out of the data with correct decision dates), VIX threshold crossings with a cooldown, SecFiler filing dates — and labeled by pure rules over TRAILING windows so a 2022 print is "large" vs 2019–22, not vs all history. Headlines are reserved for the unscheduled category (tariffs, wars) where nothing else works, and the LLM labeler there must emit this same label vocabulary. Hand-written scenarios stay as plumbing tests; generated families (11 big-upside CPI prints since 2021, real bars, real first prints, one templated headline) are the evidence, and the runner reports a pass RATE over them. Deliberately NOT built: FOMC "holds" (needs a meeting calendar), consensus-based surprises (needs paid data — trailing-3-change expectation is the stand-in), a dated news archive (Marcelo is adding one; it plugs in as `news: python:module:Class`).
**Grade by 2026-12:** the label question is now empirical — does `trailing3` separate reactions better than `prior`? Run `cpi_release` both ways through the event study and keep the one with the larger above/below spread on SPY d1. If generated families and hand-written scenarios disagree on pass/fail systematically, trust the families.

## 2026-09-18 — News→reaction is an event study (identity), not a return model; macro first
**Bet:** the learnable thing is the conditional reaction *distribution* by event type, not the next return. So `research/` computes it exactly (mean, median, up-rate, tails over 1/5 days) and exposes it as a point-in-time playbook tool. Macro releases go first because their labels are free — FRED/ALFRED vintages give the first print and the real release date; "above/below naive expectation" needs no NLP. Filings (SecFiler dates + drift features) are next; headline news with Gemma as the labeler is third. Deliberately NOT fitted: any regression/classifier — only after the tables prove insufficient. Data quirk learned: ALFRED's oldest vintage covers all back-history at once; events are kept only when release lags the period by ≤120 days.
**Graded 2026-09-18 (first real table):** pooled since 2010, "CPI above expectation" showed SPY +0.6% over 5 days, up 70% — the *opposite* of the textbook. Split by regime: 2010–2020 nothing; 2021–2023 big upside surprises → SPY −0.75% d1 (up 17%), TLT down too. Reaction is regime-dependent, so the playbook now uses a TRAILING window (default 3y) and the largest third of |surprise| (the prints markets noticed), reports n, and flags n<8 as anecdote. Second lesson: "vs prior month's change" is a weak proxy for consensus surprise — a consensus source would sharpen labels.
**Grade by 2026-12:** on `cpi_shock`, the agent with `get_event_playbook` should show measurably lower post-print exposure than without, on the same scenario and model. If it doesn't, the playbook is too coarse (add regime/surprise-size conditioning) — or the model ignores it (then it's a prompt/eval problem, not a data problem).

## 2026-09-18 — Research loops are a tool-layer problem, and "turn-limit days" are a graded metric
**Observation:** first `cpi_shock` run: every one of 8 days ended by `max_turns`, never by a final answer — the agent re-called the same research tools (3 symbols x 5 tools) until exhausted. The "held through the CPI shock" behavior was therefore *unmeasurable*: it may have been judgment or just running out of turns. `dip_and_recover` showed the same loop on 3 of 10 days.
**Bet:** fix it below the prompt. `DataTools` now dedupes identical same-day calls (returns a one-line "already retrieved, decide" reminder — no data, no context growth), and `turn_limit_days` is in the report with a `max_turn_limit_days` expectation so exhaustion can never again masquerade as a decision. Deliberately NOT done: raising `max_turns` (hides the loop), or prompt-nagging ("only call each tool once") — prompts are argued with, tools aren't.
**Grade by 2026-11:** if turn-limit days stay >20% after dedupe, the loop is in the model's planning, not tool repetition — then try a research-budget prompt line and compare on the same scenarios.
**Graded 2026-09-18 (same day):** dedupe alone did not fix it — `cpi_shock` re-run: 8/8 turn-limit days despite 6–9 dedupe hits per day (the model kept calling after being told "already retrieved"). Outcome still improved (exposure 74%→54%, return −1.7%→−0.7%). Follow-up, two changes: (1) the agent loop's LAST turn is called with no tools, so the model must answer — `hit_turn_limit` stays true, so exhaustion remains measurable but the day ends with the model's own summary, not a placeholder; (2) the daily prompt states the turn budget explicitly. Third run pending.

## 2026-09-18 — Hand-rolled agent loop; no LangChain / CrewAI
**Bet:** the loop (`evals/agent.py`, ~40 lines) plus the Tool protocol and one model adapter is *less* to know than a framework's executor, tool decorators and message handling — and three load-bearing decisions (runner owns the clock, guardrails in the tool layer, prompt-size control on an 8K local context) are things frameworks make harder. M7's planner → executor → reviewer runs *sequentially* on this hardware (model swap between phases), so CrewAI's parallel-delegation machinery would be paid for and unused.
**Grade by 2027-03:** revisit only if (a) agents must run concurrently with delegation between them, or (b) a specific integration a framework ships would cost more than a week to write. The eval suite is framework-agnostic, so a migration — if ever — is measurable.

## 2026-09-18 — Research feeds obey the simulation clock (no lookahead), enforced by contract test
**Bet:** a backtest is only meaningful if the agent sees data as it was known on the simulated day. So every feed takes `as_of` from the runner (never the wall clock, never the model): FRED via ALFRED vintages (`realtime_start=realtime_end=as_of`), SecFiler filtered on *filing_date* (not report_date — SecFiler's summaries carry the latter, so the adapter joins raw/<T>/<acc>/filing.json), news as dated fixtures. SecFiler is read as files, not imported: its data layout is the interface, keeping umap/sklearn out of JARVIS. Deliberately NOT built: live news APIs (no point-in-time guarantees), a bars feed separate from the broker (the broker *is* the price history in a sim).
**Grade by 2026-12:** any feed added later must pass `test_point_in_time_contract`; if one needs an exception, the design is wrong, not the test. If SecFiler changes its file layout twice in three months, switch the adapter to importing a small `secfiler.api` module instead.

## 2026-09-18 — Broker is a Protocol; the in-memory broker is the v1 exchange
**Bet:** broker implementations will multiply (in-memory sim → your exchange simulator → paper → live), so the agent tools and episode loop talk only to `trading.domain.Broker`. The contract lives in `tests/trading/broker_contract.py`; a new broker is certified by subclassing it. The exchange simulator you build next should be written *to* this contract rather than the contract adapted to it.
**Grade by 2026-12:** if plugging in the real simulator touched anything outside a new `broker_*.py` + one test subclass, the seam leaked.

## 2026-09-18 — Runner owns the clock; guardrails are tool-layer, not prompt-layer
**Bet:** reproducibility and safety both require the agent to be unable to bypass them. `advance()` is on the Protocol but never a tool; `guardrails.check` runs in `TradingTools.place_order` before any broker sees the order. Deliberately NOT built: shorting, margin, options, a strategy plugin system, a cost model beyond slippage_bps + flat commission.
**Grade by 2026-11:** an episode run with the same FakeModel must be bit-for-bit repeatable. Any drift = a clock leak. First guardrail bypass found in a trajectory = move that rule into the broker too.

## 2026-09-18 — Daily cadence, one Agent.run per day, fresh context + 5-day history
**Bet:** rebalance-style decisions are what a 26B local model does well in a few tool calls; giving it only a short text history (not the full prior transcript) keeps prompts small on an 8K context. The planner/executor split, if it comes, goes in `episode.run_episode` as an "analyze" call before the tool turn.
**Grade by 2026-12:** if the agent repeatedly re-buys what it sold yesterday, the history window is too thin — widen it before adding memory tools. Revisit intraday only after a daily strategy beats buy-and-hold on `dip_and_recover`.

## 2026-09-18 — Merge the Harness eval project into JARVIS_JR as `jarvis_jr.evals`
**Bet:** eval tasks change when the assistant's tools change (co-change), so they belong in the same repo. JARVIS_JR is the runner; evals is the measurement. The two connect through one shim (`registry_tools`) over the existing `schemas`/`dispatch` interface — the same seam `CoderTools` already proved. Deliberately NOT done yet: replacing the three per-provider tool loops in `jarvis_jr.llm` with the single `evals.agent` loop, or building a router into evals — JARVIS *is* the router.
**Grade by 2026-12:** if M7 (planner + coder + reviewer) ships with a before/after pass rate from this suite, the merge paid for itself. If evals rot because they live far from the tools they test, the co-change bet was wrong.

## 2026-09-18 — Planner/executor swap is a hypothesis, not an architecture
**Bet:** a smart dense planner (Qwen3.8-27B class) plus the MoE executor (Gemma 4 26B A4B) cannot coexist in 24GB; LM Studio's JIT + "keep last loaded" swaps them per request. Whether the swap latency is worth it is unknown — observations.md already records a dense 27B thrashing this machine once KV cache grew.
**Grade by 2026-11:** run the same suite one-model vs two-model and compare pass rate and wall time in report.json before committing M7 to a swap design.

## 2026-07-14 — One tools.py; visibility declared per-tool via requires_env
**Bet:** (revised from "each foreign system gets its own adapter file") all tools live in one `tools.py`, and each tool declares the env vars it needs in `requires_env`; `available_tools()` filters on that. The declarative gate scales to any credentialed tool without touching `run.py`. The IMAP vocabulary stays confined to the Gmail section of the file — the boundary survived the merge, the file split didn't.
**Grade by 2026-10:** if the file passes ~12 tools or two tools' helpers start tangling, split into a `tools/` package (the GROWTH marker) — keeping `requires_env` as the visibility mechanism.

## 2026-07-14 — Gmail via IMAP app password, read-only, env-gated
**Bet:** for a personal-account eval harness, IMAP + app password beats the Gmail API (no GCP project, no OAuth flow, stdlib only). Tools are read-only by construction (readonly mailbox + BODY.PEEK) and only appear in the agent's toolset when GMAIL_ADDRESS/GMAIL_APP_PASSWORD are set — an agent should never see a tool that can't work. Deliberately did NOT add send/delete/mark: a local model with write access to a real inbox is a bigger decision than a tool class.
**Grade by 2026-10:** if Google drops app passwords or a Workspace account is needed, swap `_connect` for an OAuth flow — the tools themselves shouldn't change. If eval tasks start needing labeled test data, build a fixture mailbox instead of hitting the live inbox.

## 2026-07-07 — Sequential task execution, results to flat JSON
**Bet:** suites stay small (<20 tasks) while the model is the bottleneck anyway (one MoE model, one Mac). No parallelism, no results database.
**Grade by 2026-10:** if a suite run exceeds ~30 min wall time, add parallel workspaces (GROWTH marker in `run.py`).
## 2026-07-07 — One model adapter, OpenAI wire format only
**Bet:** model churn happens at config level (Gemma 4 26B A4B today, whatever ships next quarter tomorrow), all served through LM Studio's OpenAI-compatible endpoint. So there is exactly one adapter (`model_lmstudio.py`) and no provider abstraction layer.
**Grade by 2026-10:** if a second *server* (not model) is needed and it took more than one new file to support, the bet lost.

## 2026-07-07 — Tool protocol with 5 members
**Bet:** the toolset is the highest-churn surface — shell exec, web search, and task-specific tools will land within weeks. Bought a `Tool` protocol + uniform `ToolResult` so adding a tool is one class and zero loop changes.
**Grade by 2026-09:** if new tools required touching `agent.py`, the seam is wrong.

## 2026-07-07 — Grader protocol with exactly 3 members
**Bet:** the three-tier evaluation (checks / trajectory / judge) is the stable shape; new evaluation ideas will be new graders, not changes to existing ones.
**Grade by 2026-10:** if grading changes keep landing inside `run.py` instead of as new graders, revisit.

## 2026-07-07 — Tasks are plain YAML files, one loader, no framework
**Bet:** task specs stay simple enough that a flat YAML schema covers them. Deliberately did NOT build task inheritance, parametrization, or a registry.
**Grade by 2026-09:** if >20% of tasks need fields the schema lacks, extend the schema once; if tasks start generating other tasks, that's the pressure to rethink.

## 2026-07-07 — No shell tool in v1
**Bet:** file + web tools cover the first round of agent evaluation; shell exec needs a sandboxing decision (subprocess limits? container?) that shouldn't be rushed.
**Grade by 2026-08:** first time a task can't be graded without running code, design the shell tool deliberately (see GROWTH marker in `tools.py`).

