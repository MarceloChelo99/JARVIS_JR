"""Extractors: an EventType's `source` spec -> dated, unlabeled EventInstances.

One function per source.kind. All of them read local stores filled by
scripts/research_fetch.py (FRED vintages, bars, SecFiler data); none read
headlines. Adding a source kind = one function here + one line in EXTRACTORS.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

from jarvis_jr.research.bars import BarStore
from jarvis_jr.research.macro_events import DEFAULT_VINTAGES_DIR, FirstPrint, first_prints
from jarvis_jr.research.taxonomy import EventInstance, EventType


class ResearchData:
    """Lazy access to the local research stores."""

    def __init__(
        self,
        vintages_dir: Path = DEFAULT_VINTAGES_DIR,
        bars: BarStore | None = None,
        secfiler_dir: str | Path = "",
    ):
        self.vintages_dir = Path(vintages_dir)
        self.bars = bars or BarStore()
        self.secfiler_dir = str(secfiler_dir or os.environ.get("SECFILER_DATA_DIR", ""))
        self._prints: dict[str, list[FirstPrint]] = {}

    def prints(self, series: str) -> list[FirstPrint]:
        series = series.upper()
        if series not in self._prints:
            path = self.vintages_dir / f"{series}.json"
            if not path.is_file():
                raise ValueError(f"no vintages for {series}; run scripts/research_fetch.py --series {series}")
            self._prints[series] = first_prints(series, json.loads(path.read_text()))
        return self._prints[series]


# ---- fred_vintage: scheduled releases with a naive expectation -----------------


def _expected(prints: list[FirstPrint], i: int, change: str, expectation: str) -> tuple[float, dict]:
    """Naive expectation for prints[i] from the prints before it, plus context numbers."""
    hist = prints[max(0, i - 4) : i]  # up to 4 prior prints -> up to 3 prior changes
    k = 3 if expectation == "trailing3" else 1
    if change == "level":
        vals = [p.value for p in hist[-k:]]
        exp = mean(vals)
        return exp, {"prior": hist[-1].value}
    if change == "pct":
        chg = [b.value / a.value - 1 for a, b in zip(hist, hist[1:]) if a.value]
        exp_pct = mean(chg[-k:])
        exp = hist[-1].value * (1 + exp_pct)
        actual_pct = prints[i].value / hist[-1].value - 1
        return exp, {"prior": hist[-1].value, "expected_pct": exp_pct, "pct_change": actual_pct}
    if change == "diff":
        chg = [b.value - a.value for a, b in zip(hist, hist[1:])]
        exp_diff = mean(chg[-k:])
        exp = hist[-1].value + exp_diff
        return exp, {"prior": hist[-1].value, "expected_diff": exp_diff, "diff": prints[i].value - hist[-1].value}
    raise ValueError(f"unknown change kind {change!r}")


def extract_fred_vintage(et: EventType, data: ResearchData) -> list[EventInstance]:
    src = et.source
    prints = data.prints(src["series"])
    change = src.get("change", "level")
    expectation = src.get("expectation", "prior")
    need = 4 if expectation == "trailing3" else 2
    out = []
    for i in range(need, len(prints)):
        exp, ctx = _expected(prints, i, change, expectation)
        ctx.update(series=src["series"], obs_day=prints[i].obs_day.isoformat(), expected=exp)
        out.append(EventInstance(et.id, prints[i].release_day, prints[i].value, exp, ctx))
    return out


# ---- fred_daily_step: a policy rate target that moves in steps --------------------


def extract_fred_daily_step(et: EventType, data: ResearchData) -> list[EventInstance]:
    prints = data.prints(et.source["series"])
    out = []
    for prev, cur in zip(prints, prints[1:]):
        if cur.value == prev.value:
            continue
        bps = (cur.value - prev.value) * 100
        # The new target is effective on cur.obs_day; the decision was announced the
        # afternoon before. The market reacted on the decision day.
        day = cur.obs_day - timedelta(days=1)
        out.append(
            EventInstance(
                et.id, day, cur.value, prev.value,
                {"bps": bps, "prior": prev.value, "effective": cur.obs_day.isoformat()},
            )
        )
    return out


# ---- threshold: first day a daily series closes at/above a level ------------------


def extract_threshold(et: EventType, data: ResearchData) -> list[EventInstance]:
    prints = data.prints(et.source["series"])
    level = float(et.source["level"])
    cooldown = int(et.source.get("cooldown_days", 15))  # one spell = one event
    out: list[EventInstance] = []
    for prev, cur in zip(prints, prints[1:]):
        if cur.value >= level and prev.value < level:
            if out and (cur.obs_day - out[-1].day).days < cooldown:
                continue
            out.append(
                EventInstance(et.id, cur.obs_day, cur.value, None, {"level": level, "prior": prev.value})
            )
    return out


# ---- edgar: filings from SecFiler's data dir, with one analysis feature -------------


def extract_edgar(et: EventType, data: ResearchData) -> list[EventInstance]:
    from jarvis_jr.trading.feed_secfiler import SecFilerFilings

    feed = SecFilerFilings(data.secfiler_dir)
    form = et.source.get("form", "10-K")
    feature = et.source.get("feature", "drift_1A")
    out = []
    for (ticker, label), (filed, f) in feed._filed.items():
        if not f.startswith(form):
            continue
        rows = [r for r in feed._features.get(ticker, []) if r["period_label"] == label and r["feature"] == feature]
        if not rows:
            continue
        out.append(
            EventInstance(
                et.id, filed, float(rows[0]["value"]), None,
                {"ticker": ticker, "period": label, "form": f, "feature": feature},
            )
        )
    return sorted(out, key=lambda e: e.day)


EXTRACTORS: dict[str, Callable[[EventType, ResearchData], list[EventInstance]]] = {
    "fred_vintage": extract_fred_vintage,
    "fred_daily_step": extract_fred_daily_step,
    "threshold": extract_threshold,
    "edgar": extract_edgar,
}


def extract(et: EventType, data: ResearchData) -> list[EventInstance]:
    kind = et.source["kind"]
    if kind not in EXTRACTORS:
        raise ValueError(f"unknown source kind {kind!r}; known: {sorted(EXTRACTORS)}")
    return sorted(EXTRACTORS[kind](et, data), key=lambda e: e.day)


def window_start(as_of: date, years: int) -> date:
    return date(as_of.year - years, as_of.month, min(as_of.day, 28))
