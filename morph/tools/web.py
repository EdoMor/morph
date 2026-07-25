"""Web tools. Degrade to a clear error when offline or unconfigured (R-206)."""

from __future__ import annotations

import html
import os
import re
from typing import Any

import httpx

from . import ToolError, ToolRegistry

SEARCH_KEY_ENV = "MORPH_SEARCH_API_KEY"
SEARCH_ENDPOINT_ENV = "MORPH_SEARCH_ENDPOINT"
MAX_TEXT = 100_000

_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n{3,}")


def html_to_text(markup: str) -> str:
    """Strip markup down to readable text. Good enough for model consumption."""
    without_scripts = _SCRIPT_RE.sub(" ", markup)
    text = _TAG_RE.sub("\n", without_scripts)
    return _WS_RE.sub("\n\n", html.unescape(text)).strip()


def register_web_tools(registry: ToolRegistry, config: Any) -> None:
    del config  # web tools are workspace-independent

    async def web_fetch(url: str, max_chars: int = 20_000) -> str:
        if not url.lower().startswith(("http://", "https://")):
            raise ToolError(f"web_fetch needs an http(s) URL, got {url!r}")
        try:
            async with httpx.AsyncClient(
                timeout=30.0, follow_redirects=True, headers={"User-Agent": "morph/0.1"}
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ToolError(
                f"Cannot reach {url} — no network, or the host is down. ({exc})"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ToolError(f"{url} returned HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise ToolError(f"Fetching {url} failed: {exc}") from exc

        content_type = response.headers.get("content-type", "")
        body = response.text[:MAX_TEXT]
        text = html_to_text(body) if "html" in content_type else body
        return text[:max_chars]

    registry.register(
        "web_fetch",
        "Fetch a URL and return its text content, with HTML stripped to prose.",
        {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_chars": {"type": "integer"},
            },
            "required": ["url"],
        },
        web_fetch,
    )

    async def web_search(query: str, count: int = 5) -> str:
        api_key = os.environ.get(SEARCH_KEY_ENV)
        endpoint = os.environ.get(SEARCH_ENDPOINT_ENV, "https://api.search.brave.com/res/v1/web/search")
        if not api_key:
            raise ToolError(
                f"Web search is not configured. Set {SEARCH_KEY_ENV} (and optionally "
                f"{SEARCH_ENDPOINT_ENV}) to enable it. Use web_fetch if you already have a URL."
            )
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    endpoint,
                    params={"q": query, "count": count},
                    headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise ToolError(f"Search request failed: {exc}") from exc

        results = (data.get("web") or {}).get("results") or data.get("results") or []
        if not results:
            return f"No results for {query!r}"
        lines = []
        for item in results[:count]:
            title = item.get("title", "(untitled)")
            url = item.get("url", "")
            snippet = html_to_text(item.get("description", ""))[:300]
            lines.append(f"- {title}\n  {url}\n  {snippet}")
        return "\n".join(lines)

    registry.register(
        "web_search",
        "Search the web. Requires MORPH_SEARCH_API_KEY; errors clearly when unset.",
        {
            "type": "object",
            "properties": {"query": {"type": "string"}, "count": {"type": "integer"}},
            "required": ["query"],
        },
        web_search,
    )
