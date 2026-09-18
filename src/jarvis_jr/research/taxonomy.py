"""The event taxonomy: typed view of evals/events.yaml.

An EventType says where instances come from (source), how they are labeled
(label rules), what a scenario around one looks like, and how many make a
playbook trustworthy. Instances and labels live in extractors.py / labels.py;
this module only defines the vocabulary and loads the file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from jarvis_jr.settings import REPO_ROOT

DEFAULT_TAXONOMY = REPO_ROOT / "evals" / "events.yaml"


@dataclass(frozen=True)
class EventType:
    id: str
    category: str
    source: dict  # {kind, ...}
    label: dict  # {direction: {rule, ...}, magnitude?: {rule, ...}}
    scenario: dict  # {before_days, after_days, symbols}
    playbook: dict  # {min_n, lookback_years}
    headline: str = ""


@dataclass(frozen=True)
class EventInstance:
    """A dated fact extracted from a source. Unlabeled."""

    type_id: str
    day: date  # the day the market learned it (release / decision / filing / crossing)
    value: float  # the headline number (first print, new target, VIX close, drift)
    expected: float | None = None  # naive expectation, when the source defines one
    context: dict = field(default_factory=dict)  # ticker, prior values, units...

    @property
    def surprise(self) -> float | None:
        return None if self.expected is None else self.value - self.expected


@dataclass(frozen=True)
class LabeledEvent:
    instance: EventInstance
    direction: str
    magnitude: str = ""

    @property
    def label(self) -> str:
        return f"{self.direction}/{self.magnitude}" if self.magnitude else self.direction


@dataclass(frozen=True)
class Taxonomy:
    types: dict[str, EventType]
    scenario_expectations: dict[str, dict]

    def __getitem__(self, type_id: str) -> EventType:
        if type_id not in self.types:
            raise KeyError(f"unknown event type {type_id!r}; known: {sorted(self.types)}")
        return self.types[type_id]

    def expectations_for(self, label: str) -> dict:
        out = dict(self.scenario_expectations.get("default", {}))
        for key in (label.split("/")[0], label):
            out.update(self.scenario_expectations.get(key, {}))
        return out


def load_taxonomy(path: Path = DEFAULT_TAXONOMY) -> Taxonomy:
    data = yaml.safe_load(Path(path).read_text())
    types = {}
    for raw in data.get("event_types", []):
        for key in ("id", "category", "source", "label"):
            if key not in raw:
                raise ValueError(f"{path}: event type missing `{key}`: {raw}")
        if "kind" not in raw["source"]:
            raise ValueError(f"{path}: {raw['id']}: source needs `kind`")
        if "direction" not in raw["label"]:
            raise ValueError(f"{path}: {raw['id']}: label needs `direction`")
        types[raw["id"]] = EventType(
            id=raw["id"],
            category=raw["category"],
            source=dict(raw["source"]),
            label=dict(raw["label"]),
            scenario=dict(raw.get("scenario", {"before_days": 2, "after_days": 5, "symbols": ["SPY"]})),
            playbook=dict(raw.get("playbook", {"min_n": 8, "lookback_years": 3})),
            headline=str(raw.get("headline", "")),
        )
    return Taxonomy(types=types, scenario_expectations=dict(data.get("scenario_expectations", {})))
