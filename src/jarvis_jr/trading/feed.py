"""Research data feeds: macro series, news headlines, SEC filings.

The one rule every feed obeys: nothing dated after `as_of` is ever returned.
`as_of` is the simulation day, supplied by the runner — never the wall clock,
never the model. The point-in-time contract tests in tests/trading enforce it
for every implementation.

Three small protocols rather than one big one, because the sources differ
(a scenario may provide news fixtures but no filings). Fixture feeds live
here; live adapters (FRED, SecFiler) each get their own file because they
wrap a foreign data source exactly once.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class MacroObservation:
    series: str
    day: date  # observation period
    value: float
    released: date  # when it became public; >= day


@dataclass(frozen=True)
class Headline:
    day: date
    title: str
    source: str = ""
    symbols: tuple[str, ...] = ()
    summary: str = ""


@dataclass(frozen=True)
class FilingSignal:
    feature: str  # drift_1A, risk_headings, theme:<name>, new_risk_clusters, sector
    value: float
    period_label: str
    filed: date
    note: str = ""


@dataclass(frozen=True)
class FilingSummary:
    ticker: str
    form: str
    period_label: str
    filed: date
    item: str
    title: str
    summary: str


@dataclass(frozen=True)
class RiskHeading:
    heading: str
    theme: str = ""
    cluster_label: str = ""


class MacroFeed(Protocol):
    def observations(self, series: str, as_of: date, n: int) -> list[MacroObservation]: ...


class NewsFeed(Protocol):
    def headlines(self, query: str, as_of: date, n: int) -> list[Headline]: ...


class FilingsFeed(Protocol):
    def signals(self, ticker: str, as_of: date) -> list[FilingSignal]: ...
    def summary(self, ticker: str, item: str, as_of: date) -> FilingSummary | None: ...
    def risk_headings(self, ticker: str, as_of: date, n: int) -> list[RiskHeading]: ...


class ReactionFeed(Protocol):
    """Event → cross-asset reaction archetypes, fit only on events before as_of.
    Implemented by jarvis_jr.research.reactions.ReactionGraph."""

    def available_types(self) -> list[str]: ...
    def labels_for(self, type_id: str) -> list[str]: ...
    def profile(self, type_id: str, label: str, as_of: date, lookback_years: int | None = None, horizon: int = 1): ...


class SizingFeed(Protocol):
    """Exposure advice for an event as of a date. Implemented by research.sizing.ExposurePolicy."""

    def advise(self, event_type: str, label: str, as_of: date, lookback_years: int | None = None): ...


class PlaybookFeed(Protocol):
    """Historical reaction stats for an event type, fit only on data before as_of.
    Implemented by jarvis_jr.research.playbook.MacroPlaybook."""

    def available_series(self) -> list[str]: ...
    def lookup(self, series: str, direction: str, as_of: date, lookback_years: int = 3): ...


def _to_date(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])


class FixtureMacro:
    """CSV with columns: series,date,value[,release_date]. Missing release_date = date."""

    def __init__(self, path: Path):
        self._rows: list[MacroObservation] = []
        with Path(path).open() as f:
            for r in csv.DictReader(f):
                day = _to_date(r["date"])
                released = _to_date(r["release_date"]) if r.get("release_date") else day
                self._rows.append(MacroObservation(r["series"], day, float(r["value"]), released))

    def observations(self, series: str, as_of: date, n: int) -> list[MacroObservation]:
        rows = [o for o in self._rows if o.series == series and o.released <= as_of]
        rows.sort(key=lambda o: o.day, reverse=True)
        return rows[:n]


class FixtureNews:
    """JSONL, one object per line: {date, title, source?, symbols?, summary?}."""

    def __init__(self, path: Path):
        self._items: list[Headline] = []
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            self._items.append(
                Headline(
                    day=_to_date(d["date"]),
                    title=d["title"],
                    source=d.get("source", ""),
                    symbols=tuple(s.upper() for s in d.get("symbols", [])),
                    summary=d.get("summary", ""),
                )
            )

    def headlines(self, query: str, as_of: date, n: int) -> list[Headline]:
        q = query.strip().upper()
        items = [
            h
            for h in self._items
            if h.day <= as_of and (not q or q in h.symbols or q.lower() in h.title.lower())
        ]
        items.sort(key=lambda h: h.day, reverse=True)
        return items[:n]
