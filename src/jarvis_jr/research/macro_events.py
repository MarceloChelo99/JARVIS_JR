"""Macro release events from FRED/ALFRED vintage history.

One request per series with a wide realtime window returns every vintage of
every observation. The FIRST vintage of each observation is the number the
market saw on release day, and its realtime_start is the release date. From
that we build MacroEvent rows with a surprise vs. a naive expectation:

  kind "pct"   (CPIAUCSL, PAYEMS as index-like):  actual pct change vs prior pct change
  kind "diff"  (PAYEMS as level change):          actual change vs prior change
  kind "level" (UNRATE, FEDFUNDS, DGS10):         value vs prior value

direction = "above" if surprise > 0 else "below" (vs. the naive expectation).
Labels are free — no NLP — which is why macro is the first playbook.

Cached under data/macro_vintages/<SERIES>.json. Env: FRED_API_KEY.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import requests

from jarvis_jr.settings import REPO_ROOT

DEFAULT_VINTAGES_DIR = REPO_ROOT / "data" / "macro_vintages"
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

SERIES_KIND: dict[str, str] = {
    "CPIAUCSL": "pct",
    "CPILFESL": "pct",
    "PCEPI": "pct",
    "PAYEMS": "diff",
    "UNRATE": "level",
    "FEDFUNDS": "level",
    "DGS10": "level",
    "DGS2": "level",
    "RSAFS": "pct",
    "INDPRO": "pct",
}


@dataclass(frozen=True)
class FirstPrint:
    series: str
    obs_day: date
    release_day: date
    value: float


@dataclass(frozen=True)
class MacroEvent:
    series: str
    obs_day: date
    release_day: date
    value: float
    expected: float  # naive expectation (see module doc)
    surprise: float  # value - expected, in the series' change units

    @property
    def direction(self) -> str:
        return "above" if self.surprise > 0 else "below"


def _http_fetch(params: dict) -> dict:
    resp = requests.get(FRED_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


class FredVintages:
    def __init__(
        self,
        api_key: str = "",
        cache_dir: Path = DEFAULT_VINTAGES_DIR,
        fetch: Callable[[dict], dict] = _http_fetch,
    ):
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        if not self.api_key:
            raise ValueError("FredVintages needs FRED_API_KEY")
        self.cache_dir = Path(cache_dir)
        self._fetch = fetch

    def raw(self, series: str, refresh: bool = False) -> dict:
        series = series.upper()
        cache = self.cache_dir / f"{series}.json"
        if cache.is_file() and not refresh:
            return json.loads(cache.read_text())
        base = {"series_id": series, "api_key": self.api_key, "file_type": "json", "limit": 100000}
        try:
            data = self._fetch_all(
                {**base, "realtime_start": "1776-07-04", "realtime_end": "9999-12-31", "output_type": 1}
            )
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code != 400:
                raise
            # FRED refuses the full vintage window for high-frequency series (daily
            # policy rates, VIX...). Those are never revised, so the plain series is
            # its own first print: stamp each observation's release as its own date.
            data = self._fetch_all(base)
            for o in data.get("observations", []):
                o["realtime_start"] = o["date"]
            data["vintage_fallback"] = True
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))
        return data

    def _fetch_all(self, params: dict) -> dict:
        """Follow FRED's offset pagination (weekly series x vintages exceed one page)."""
        limit = int(params.get("limit", 100000))
        observations: list[dict] = []
        offset = 0
        while True:
            page = self._fetch({**params, "offset": offset})
            obs = page.get("observations", [])
            observations.extend(obs)
            if len(obs) < limit:
                break
            offset += limit
        page["observations"] = observations
        return page


MAX_RELEASE_LAG_DAYS = 120  # a monthly print lands 2-6 weeks after its period; daily within days


def first_prints(series: str, raw: dict, max_lag_days: int = MAX_RELEASE_LAG_DAYS) -> list[FirstPrint]:
    """Earliest vintage of each observation date = what was published on release day.

    ALFRED's oldest vintage covers every historical observation at once, so anything
    whose 'first' vintage is far later than its period is back-history, not a release.
    Those are dropped: an event needs a real release date.
    """
    earliest: dict[date, tuple[date, float]] = {}
    for o in raw.get("observations", []):
        if o.get("value") in (".", "", None):
            continue
        obs = date.fromisoformat(o["date"])
        released = date.fromisoformat(o["realtime_start"])
        value = float(o["value"])
        if obs not in earliest or released < earliest[obs][0]:
            earliest[obs] = (released, value)
    return [
        FirstPrint(series.upper(), obs, rel, val)
        for obs, (rel, val) in sorted(earliest.items())
        if (rel - obs).days <= max_lag_days
    ]


def events_from_prints(prints: list[FirstPrint], kind: str) -> list[MacroEvent]:
    """Turn consecutive first prints into events with a surprise vs naive expectation."""
    out: list[MacroEvent] = []
    for i in range(2, len(prints)):
        p0, p1, p2 = prints[i - 2], prints[i - 1], prints[i]
        if kind == "level":
            expected, actual = p1.value, p2.value
        elif kind == "diff":
            expected, actual = p1.value - p0.value, p2.value - p1.value
            expected, actual = p1.value + expected, p1.value + actual  # back to level units
        elif kind == "pct":
            if not p0.value or not p1.value:
                continue
            prior_pct = p1.value / p0.value - 1
            expected, actual = p1.value * (1 + prior_pct), p2.value
        else:
            raise ValueError(f"unknown series kind {kind!r}")
        out.append(
            MacroEvent(
                series=p2.series,
                obs_day=p2.obs_day,
                release_day=p2.release_day,
                value=p2.value,
                expected=expected,
                surprise=actual - expected,
            )
        )
    return out


def macro_events(series: str, vintages: FredVintages) -> list[MacroEvent]:
    kind = SERIES_KIND.get(series.upper(), "level")
    return events_from_prints(first_prints(series, vintages.raw(series)), kind)
