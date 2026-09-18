"""SecFiler filings feed: reads SecFiler's data directory, filtered by filing date.

Deliberately reads files rather than importing the `secfiler` package — its
data layout is the interface, so SecFiler can change internals (and keep its
heavy deps) without touching JARVIS. Env: SECFILER_DATA_DIR
(e.g. ~/PycharmProjects/SecFiler/data).

Point-in-time: SecFiler's summaries carry `report_date` (period end) but the
public-availability date is `filing_date`, which lives in
raw/<TICKER>/<accession>/filing.json. Every lookup here joins on that and
filters `filing_date <= as_of`. features.csv already carries filing_date;
risk_headings.csv does not (joined via fiscal_year -> FY label).

Layout read (relative to the data dir):
  raw/<T>/<acc>/filing.json            form, filing_date, report_date
  summaries/<T>/<period>/item_<I>.json  ItemSummary (title, summary, ...)
  analysis/features.csv                 ticker, period_label, filing_date, feature, value, note
  analysis/risk_headings.csv            ticker, fiscal_year, heading, label, theme
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from datetime import date
from functools import cached_property
from pathlib import Path

from jarvis_jr.trading.feed import FilingSignal, FilingSummary, RiskHeading

ITEM_TITLES = {"1": "Business", "1A": "Risk Factors", "7": "MD&A"}


def _period_label(form: str, report_date: str) -> str:
    return f"FY{report_date[:4]}" if form.startswith("10-K") else f"Q-{report_date[:7]}"


class SecFilerFilings:
    def __init__(self, data_dir: str | Path = ""):
        raw = str(data_dir or os.environ.get("SECFILER_DATA_DIR", "")).strip()
        if not raw:
            raise ValueError("SecFiler data dir not set (SECFILER_DATA_DIR)")
        self.data_dir = Path(raw).expanduser()
        if not (self.data_dir / "raw").is_dir():
            raise ValueError(f"SecFiler data dir not found (no raw/): {self.data_dir}")

    # ---- indexes (built lazily, once) ---------------------------------------

    @cached_property
    def _filed(self) -> dict[tuple[str, str], tuple[date, str]]:
        """(ticker, period_label) -> (filing_date, form)."""
        out: dict[tuple[str, str], tuple[date, str]] = {}
        for meta in (self.data_dir / "raw").glob("*/*/filing.json"):
            try:
                d = json.loads(meta.read_text())
                label = _period_label(d["form"], d["report_date"])
                out[(d["ticker"].upper(), label)] = (date.fromisoformat(d["filing_date"]), d["form"])
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
        return out

    @cached_property
    def _features(self) -> dict[str, list[dict]]:
        by_ticker: dict[str, list[dict]] = defaultdict(list)
        path = self.data_dir / "analysis" / "features.csv"
        if path.is_file():
            with path.open() as f:
                for r in csv.DictReader(f):
                    if r.get("filing_date"):
                        by_ticker[r["ticker"].upper()].append(r)
        return by_ticker

    @cached_property
    def _headings(self) -> dict[str, list[dict]]:
        by_ticker: dict[str, list[dict]] = defaultdict(list)
        path = self.data_dir / "analysis" / "risk_headings.csv"
        if path.is_file():
            with path.open() as f:
                for r in csv.DictReader(f):
                    by_ticker[r["ticker"].upper()].append(r)
        return by_ticker

    def _filed_on(self, ticker: str, period_label: str) -> date | None:
        entry = self._filed.get((ticker, period_label))
        return entry[0] if entry else None

    # ---- FilingsFeed --------------------------------------------------------

    def signals(self, ticker: str, as_of: date) -> list[FilingSignal]:
        ticker = ticker.upper()
        latest: dict[str, FilingSignal] = {}
        for r in self._features.get(ticker, []):
            filed = date.fromisoformat(r["filing_date"])
            if filed > as_of:
                continue
            sig = FilingSignal(r["feature"], float(r["value"]), r["period_label"], filed, r.get("note", ""))
            if r["feature"] not in latest or filed > latest[r["feature"]].filed:
                latest[r["feature"]] = sig
        return sorted(latest.values(), key=lambda s: s.feature)

    def summary(self, ticker: str, item: str, as_of: date) -> FilingSummary | None:
        ticker, item = ticker.upper(), item.upper()
        best: FilingSummary | None = None
        for path in (self.data_dir / "summaries" / ticker).glob(f"*/item_{item}.json"):
            label = path.parent.name
            filed = self._filed_on(ticker, label)
            if filed is None or filed > as_of or (best and filed <= best.filed):
                continue
            d = json.loads(path.read_text())
            best = FilingSummary(
                ticker=ticker,
                form=d.get("form", self._filed[(ticker, label)][1]),
                period_label=label,
                filed=filed,
                item=item,
                title=ITEM_TITLES.get(item, d.get("title", "")),
                summary=d.get("summary", ""),
            )
        return best

    def risk_headings(self, ticker: str, as_of: date, n: int) -> list[RiskHeading]:
        ticker = ticker.upper()
        by_year: dict[int, list[dict]] = defaultdict(list)
        for r in self._headings.get(ticker, []):
            by_year[int(r["fiscal_year"])].append(r)
        # latest fiscal year whose 10-K was filed by as_of
        for fy in sorted(by_year, reverse=True):
            filed = self._filed_on(ticker, f"FY{fy}")
            if filed is not None and filed <= as_of:
                rows = by_year[fy][:n]
                return [
                    RiskHeading(r["heading"], r.get("theme", ""), r.get("label", ""))
                    for r in rows
                ]
        return []
