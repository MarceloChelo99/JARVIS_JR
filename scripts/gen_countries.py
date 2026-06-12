"""Regenerate data/countries.json — JARVIS Jr.'s built-in country reference.

This bakes ISO 3166-1 into a static JSON file so the assistant has authoritative
country knowledge with zero hallucination and zero network calls at runtime. The
runtime tool (src/jarvis_jr/tools/knowledge.py) only reads the JSON, so pycountry
is a *build-time* dependency, not a runtime one.

Usage:
    uv run --with pycountry --with pycountry-convert python scripts/gen_countries.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pycountry
import pycountry_convert as pc

_CONTINENTS = {
    "AF": "Africa",
    "AS": "Asia",
    "EU": "Europe",
    "NA": "North America",
    "SA": "South America",
    "OC": "Oceania",
    "AN": "Antarctica",
}


# pycountry-convert doesn't map a handful of codes; fill them in by hand.
_CONTINENT_OVERRIDES = {
    "TL": "Asia",  # Timor-Leste
    "VA": "Europe",  # Holy See / Vatican
    "EH": "Africa",  # Western Sahara
    "SX": "North America",  # Sint Maarten
    "PN": "Oceania",  # Pitcairn
    "UM": "Oceania",  # US Minor Outlying Islands
    "TF": "Antarctica",  # French Southern Territories
    "AQ": "Antarctica",  # Antarctica
}


def _flag(alpha_2: str) -> str:
    """Regional-indicator emoji for a 2-letter country code (e.g. 'JP' -> 🇯🇵)."""
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in alpha_2.upper())


def build() -> dict:
    rows = []
    for c in pycountry.countries:
        try:
            continent = _CONTINENTS.get(pc.country_alpha2_to_continent_code(c.alpha_2))
        except Exception:
            continent = None
        continent = continent or _CONTINENT_OVERRIDES.get(c.alpha_2)
        rows.append(
            {
                "name": c.name,
                "official_name": getattr(c, "official_name", c.name),
                "alpha_2": c.alpha_2,
                "alpha_3": c.alpha_3,
                "numeric": c.numeric,
                "continent": continent,
                "flag": _flag(c.alpha_2),
            }
        )
    rows.sort(key=lambda r: r["name"])
    return {
        "_meta": {
            "source": f"ISO 3166-1 via pycountry {pycountry.__version__}",
            "count": len(rows),
            "note": "Built-in country reference for JARVIS Jr. Regenerate with scripts/gen_countries.py.",
        },
        "countries": rows,
    }


def main() -> None:
    out_path = Path(__file__).resolve().parents[1] / "data" / "countries.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = build()
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {data['_meta']['count']} countries -> {out_path}")


if __name__ == "__main__":
    main()
