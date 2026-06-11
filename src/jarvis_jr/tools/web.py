"""Web search + page fetching tools.

Search uses DuckDuckGo via the `ddgs` package (no API key). Page fetching uses
`trafilatura` to extract readable text from HTML. Results are truncated hard —
the local models are small and slow, so we keep tool output compact.
"""

from __future__ import annotations

import json


class WebTools:
    """web_search and fetch_page implementations."""

    def __init__(self, max_results: int = 5, max_page_chars: int = 4000):
        self.max_results = max_results
        self.max_page_chars = max_page_chars

    def search(self, query: str, max_results: int | None = None) -> str:
        from ddgs import DDGS

        n = min(max_results or self.max_results, 10)
        results = list(DDGS().text(query, max_results=n))
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

    def fetch_page(self, url: str, max_chars: int | None = None) -> str:
        import trafilatura

        html = trafilatura.fetch_url(url)
        if html is None:
            return f"ERROR: could not fetch {url}."
        text = trafilatura.extract(html, include_comments=False, include_tables=True)
        if not text:
            return f"ERROR: no readable text extracted from {url}."
        limit = max_chars or self.max_page_chars
        if len(text) > limit:
            text = text[:limit] + "\n[...truncated]"
        return text
