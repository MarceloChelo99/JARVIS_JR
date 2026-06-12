"""Tests for the built-in country reference tool."""

from __future__ import annotations

import json

from jarvis_jr.tools.knowledge import CountryKnowledge


def test_dataset_loads_and_is_substantial():
    k = CountryKnowledge()
    data = json.loads(k.lookup())  # empty query -> full list
    assert data["count"] >= 190
    assert {"name": "Japan", "alpha_2": "JP"} in data["countries"]


def test_lookup_by_name():
    k = CountryKnowledge()
    jp = json.loads(k.lookup("Japan"))
    assert jp["alpha_3"] == "JPN"
    assert jp["continent"] == "Asia"
    assert jp["flag"] == "🇯🇵"


def test_lookup_by_code_case_insensitive():
    k = CountryKnowledge()
    assert json.loads(k.lookup("fr"))["name"] == "France"
    assert json.loads(k.lookup("DEU"))["name"] == "Germany"
    assert json.loads(k.lookup("076"))["name"] == "Brazil"


def test_partial_name_match_is_resolved_or_disambiguated():
    k = CountryKnowledge()
    # "United" matches several -> ambiguous list rather than a wrong guess.
    res = json.loads(k.lookup("United"))
    assert res.get("ambiguous") is True
    assert len(res["matches"]) >= 2


def test_filter_by_continent():
    k = CountryKnowledge()
    europe = json.loads(k.lookup("", continent="Europe"))
    names = {c["name"] for c in europe["countries"]}
    assert "France" in names
    assert "Japan" not in names


def test_unknown_country_returns_error():
    k = CountryKnowledge()
    assert k.lookup("Wakanda").startswith("ERROR")
