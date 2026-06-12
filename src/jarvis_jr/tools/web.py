"""Web search + page fetching tools.

Search uses DuckDuckGo via the `ddgs` package (no API key): `search` for
general queries with an optional recency filter, `news` for current events
(dated, sourced articles). Page fetching uses `trafilatura` to extract
readable text. Results are truncated hard — the local models are small and
slow, so we keep tool output compact.
"""

from __future__ import annotations

import json
import time

_RECENCY = {"day": "d", "week": "w", "month": "m", "year": "y"}


def _with_retry(fn, attempts: int = 2, delay_s: float = 1.5):
    """DDG's free endpoint occasionally rate-limits; one spaced retry fixes most."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # ddgs raises various exception types
            last = e
            if i + 1 < attempts:
                time.sleep(delay_s)
    raise last  # type: ignore[misc]


class WebTools:
    """web_search, web_news, and fetch_page implementations."""

    def __init__(self, max_results: int = 5, max_page_chars: int = 4000):
        self.max_results = max_results
        self.max_page_chars = max_page_chars

    def search(
        self,
        query: str,
        max_results: int | None = None,
        recency: str | None = None,
    ) -> str:
        from ddgs import DDGS

        n = min(max_results or self.max_results, 10)
        kwargs = {}
        timelimit = _RECENCY.get((recency or "").lower())
        if timelimit:
            kwargs["timelimit"] = timelimit
        results = _with_retry(lambda: list(DDGS().text(query, max_results=n, **kwargs)))
        if not results:
            return "No results."
        slim = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": (r.get("body") or "")[:300],
            }
            for r in results
        ]
        return json.dumps(slim, ensure_ascii=False)

    def news(self, query: str, max_results: int | None = None) -> str:
        from ddgs import DDGS

        n = min(max_results or self.max_results, 10)
        results = _with_retry(lambda: list(DDGS().news(query, max_results=n)))
        if not results:
            return "No news found."
        slim = [
            {
                "title": r.get("title", ""),
                "date": r.get("date", ""),
                "source": r.get("source", ""),
                "url": r.get("url", ""),
                "snippet": (r.get("body") or "")[:300],
            }
            for r in results
        ]
        return json.dumps(slim, ensure_ascii=False)

    def fetch_page(self, url: str, max_chars: int | None = None) -> str:
        import trafilatura

        html = _with_retry(lambda: trafilatura.fetch_url(url))
        if html is None:
            return f"ERROR: could not fetch {url}."
        text = trafilatura.extract(html, include_comments=False, include_tables=True)
        if not text:
            return f"ERROR: no readable text extracted from {url}."
        limit = max_chars or self.max_page_chars
        if len(text) > limit:
            text = text[:limit] + "\n[...truncated]"
        return text
