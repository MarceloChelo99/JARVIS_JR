"""Built-in reference knowledge: authoritative, offline, zero-hallucination.

This is the template for giving JARVIS Jr. *inherent* knowledge of finite,
structured datasets (here: every country in the world per ISO 3166-1). The data
lives in a static JSON file (data/countries.json) that ships with the repo, so
lookups are exact, instant, and work with no internet — unlike asking the LLM to
recall the list from its weights, which a small local model may get wrong.

To add another reference dataset (timezones, airport codes, a product catalog),
copy this pattern: a generated JSON file + a thin loader class + a tool entry in
registry.py.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

_DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "countries.json"


class CountryKnowledge:
    """Read-only lookups over the bundled ISO 3166-1 country dataset."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else _DEFAULT_DATA_PATH

    @cached_property
    def _countries(self) -> list[dict[str, str]]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return data["countries"]

    @cached_property
    def _by_code(self) -> dict[str, dict[str, str]]:
        index: dict[str, dict[str, str]] = {}
        for c in self._countries:
            index[c["alpha_2"].upper()] = c
            index[c["alpha_3"].upper()] = c
            index[c["numeric"]] = c
        return index

    def lookup(self, query: str = "", continent: str | None = None) -> str:
        """Resolve a country query to a JSON string.

        - Empty query: return the full list (names + alpha_2), optionally filtered
          by continent.
        - 2/3-letter or numeric code: exact match.
        - Otherwise: case-insensitive name match (exact first, then substring).
        """
        q = (query or "").strip()
        cont = (continent or "").strip().lower()

        # No specific country asked for -> list (optionally by continent).
        if not q:
            rows = self._countries
            if cont:
                rows = [c for c in rows if (c.get("continent") or "").lower() == cont]
                if not rows:
                    return f"ERROR: no countries found for continent '{continent}'."
            slim = [{"name": c["name"], "alpha_2": c["alpha_2"]} for c in rows]
            return json.dumps({"count": len(slim), "countries": slim}, ensure_ascii=False)

        # Exact code match.
        hit = self._by_code.get(q.upper())
        if hit:
            return json.dumps(hit, ensure_ascii=False)

        # Name match: exact, then substring.
        ql = q.lower()
        exact = [
            c
            for c in self._countries
            if ql in (c["name"].lower(), c["official_name"].lower())
        ]
        if exact:
            return json.dumps(exact[0], ensure_ascii=False)

        partial = [
            c
            for c in self._countries
            if ql in c["name"].lower() or ql in c["official_name"].lower()
        ]
        if not partial:
            return f"ERROR: no country matching '{query}'."
        if len(partial) == 1:
            return json.dumps(partial[0], ensure_ascii=False)
        return json.dumps(
            {
                "ambiguous": True,
                "matches": [{"name": c["name"], "alpha_2": c["alpha_2"]} for c in partial],
            },
            ensure_ascii=False,
        )
