"""Labeling rules: (instance, prior instances) -> direction / magnitude.

Every rule is a pure function and every window is TRAILING — an instance is
ranked only against instances that happened before it, so labels are
point-in-time and a 2022 CPI print is "large" relative to 2019-2022, not to
all of history. Rules never read text.
"""

from __future__ import annotations

from collections.abc import Callable

from jarvis_jr.research.extractors import window_start
from jarvis_jr.research.taxonomy import EventInstance, EventType, LabeledEvent

MIN_WINDOW = 6  # fewer prior instances than this -> magnitude "unrated"

Rule = Callable[[EventInstance, list[EventInstance], dict], str]


# ---- direction rules ------------------------------------------------------------


def surprise_sign(inst: EventInstance, prior: list[EventInstance], spec: dict) -> str:
    s = inst.surprise
    if s is None:
        raise ValueError("surprise_sign needs an instance with an expectation")
    return "above" if s > 0 else "below"


def sign_of_change(inst: EventInstance, prior: list[EventInstance], spec: dict) -> str:
    delta = inst.context.get("bps")
    if delta is None:
        delta = inst.value - (inst.expected if inst.expected is not None else 0.0)
    return spec.get("positive", "up") if delta > 0 else spec.get("negative", "down")


def fixed(inst: EventInstance, prior: list[EventInstance], spec: dict) -> str:
    return str(spec["value"])


def feature_sign(inst: EventInstance, prior: list[EventInstance], spec: dict) -> str:
    """High if the value is above the given percentile of prior instances' values."""
    years = int(spec.get("lookback_years", 3))
    vals = sorted(p.value for p in prior if p.day >= window_start(inst.day, years))
    if len(vals) < MIN_WINDOW:
        return spec.get("low", "normal")
    cut = vals[min(len(vals) - 1, int(len(vals) * float(spec.get("percentile", 75)) / 100))]
    return spec.get("high", "high") if inst.value >= cut else spec.get("low", "normal")


# ---- magnitude rules -----------------------------------------------------------


def surprise_tercile(inst: EventInstance, prior: list[EventInstance], spec: dict) -> str:
    years = int(spec.get("lookback_years", 3))
    hist = sorted(
        abs(p.surprise) for p in prior if p.surprise is not None and p.day >= window_start(inst.day, years)
    )
    if len(hist) < MIN_WINDOW or inst.surprise is None:
        return "unrated"
    a = abs(inst.surprise)
    lo, hi = hist[len(hist) // 3], hist[len(hist) * 2 // 3]
    return "small" if a < lo else "medium" if a < hi else "large"


def bps_bucket(inst: EventInstance, prior: list[EventInstance], spec: dict) -> str:
    buckets = [int(b) for b in spec.get("buckets", [25, 50, 75])]
    a = abs(float(inst.context.get("bps", 0.0)))
    for b in buckets:
        if a <= b + 1e-9:
            return f"{b}bp"
    return f"{buckets[-1]}bp+"


def level_bucket(inst: EventInstance, prior: list[EventInstance], spec: dict) -> str:
    buckets = [float(b) for b in spec.get("buckets", [])]
    for lo, hi in zip(buckets, buckets[1:]):
        if lo <= inst.value < hi:
            return f"{lo:g}-{hi:g}"
    return f"{buckets[-1]:g}+" if buckets and inst.value >= buckets[-1] else f"<{buckets[0]:g}"


RULES: dict[str, Rule] = {
    "surprise_sign": surprise_sign,
    "sign_of_change": sign_of_change,
    "fixed": fixed,
    "feature_sign": feature_sign,
    "surprise_tercile": surprise_tercile,
    "bps_bucket": bps_bucket,
    "level_bucket": level_bucket,
}


def label_events(instances: list[EventInstance], et: EventType) -> list[LabeledEvent]:
    """Label chronologically; each instance sees only the ones before it."""
    d_spec = et.label["direction"]
    m_spec = et.label.get("magnitude")
    for spec in (d_spec, m_spec):
        if spec and spec["rule"] not in RULES:
            raise ValueError(f"unknown label rule {spec['rule']!r}; known: {sorted(RULES)}")
    out: list[LabeledEvent] = []
    ordered = sorted(instances, key=lambda e: e.day)
    for i, inst in enumerate(ordered):
        prior = ordered[:i]
        direction = RULES[d_spec["rule"]](inst, prior, d_spec)
        magnitude = RULES[m_spec["rule"]](inst, prior, m_spec) if m_spec else ""
        out.append(LabeledEvent(inst, direction, magnitude))
    return out
