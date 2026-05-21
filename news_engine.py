"""
bot/news_engine.py — Fully async multi-source news aggregation.

DDG 403 fix:
  DuckDuckGo blocks servers that fire many simultaneous news requests.
  We now run DDG calls sequentially with a small random jitter between each
  request (1-3 seconds). RSS feeds are still fetched concurrently via aiohttp
  since they are direct HTTP to known servers that don't rate-limit this way.
"""
from __future__ import annotations
import asyncio
import hashlib
import logging
import random
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

COMPANY_TICKER_MAP: Dict[str, str] = {
    "apple":           "AAPL", "microsoft":      "MSFT",
    "google":          "GOOGL","alphabet":        "GOOGL",
    "amazon":          "AMZN", "nvidia":          "NVDA",
    "meta":            "META", "facebook":        "META",
    "tesla":           "TSLA", "amd":             "AMD",
    "netflix":         "NFLX", "oracle":          "ORCL",
    "salesforce":      "CRM",  "intel":           "INTC",
    "paypal":          "PYPL", "uber":            "UBER",
    "lyft":            "LYFT", "snap":            "SNAP",
    "alibaba":         "BABA", "jpmorgan":        "JPM",
    "bank of america": "BAC",  "goldman sachs":   "GS",
}

RSS_FEEDS: List[tuple] = [
    ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
    ("Reuters Markets",  "https://feeds.reuters.com/reuters/financialsNews"),
    ("Yahoo Finance",    "https://finance.yahoo.com/news/rssindex"),
    ("MarketWatch",      "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("CNBC",             "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    ("Benzinga",         "https://www.benzinga.com/feed"),
    ("Motley Fool",      "https://www.fool.com/feeds/index.aspx"),
]

DDG_GENERAL_QUERIES = [
    "stock market news today",
    "Federal Reserve interest rates",
    "S&P 500 nasdaq earnings today",
    "inflation CPI economic data",
]

_TIMEOUT = aiohttp.ClientTimeout(total=12, connect=5)
_HEADERS = {"User-Agent": "ApexTrader/2.0 (news aggregator)"}

# DDG rate-limit guard: max concurrent DDG calls and jitter between each
_DDG_SEMAPHORE = asyncio.Semaphore(1)   # only 1 DDG call at a time globally
_DDG_MIN_DELAY = 1.2                    # seconds between calls (minimum)
_DDG_MAX_DELAY = 3.0                    # seconds between calls (maximum)
_DDG_MAX_RETRIES = 2


class NewsItem:
    __slots__ = ["title", "summary", "source", "url", "symbols",
                 "published_at", "fingerprint"]

    def __init__(self, title: str, summary: str, source: str, url: str,
                 symbols: List[str], published_at: Optional[datetime]):
        self.title        = title.strip()
        self.summary      = summary.strip()[:500]
        self.source       = source
        self.url          = url
        self.symbols      = sorted(set(symbols))
        self.published_at = published_at or datetime.now(timezone.utc)
        self.fingerprint  = hashlib.md5(self.title.lower().encode()).hexdigest()[:12]

    def to_dict(self) -> Dict:
        return {
            "title":        self.title,
            "summary":      self.summary,
            "source":       self.source,
            "url":          self.url,
            "symbols":      self.symbols,
            "fingerprint":  self.fingerprint,
            "published_at": self.published_at.isoformat(),
        }


class NewsEngine:
    def __init__(self, watchlist: List[str], max_age_hours: int = 24):
        self.watchlist     = [s.upper() for s in watchlist]
        self.max_age_hours = max_age_hours
        self._seen: Set[str] = set()

    # ── Public API ────────────────────────────────────────────────────────────

    async def fetch_all(self) -> List[NewsItem]:
        """
        Fetch all sources. RSS feeds run concurrently.
        DDG calls run sequentially with jitter to avoid 403 rate-limiting.
        """
        # ── Step 1: RSS (concurrent, they don't rate-limit) ───────────────────
        async with aiohttp.ClientSession(timeout=_TIMEOUT, headers=_HEADERS) as session:
            rss_tasks = [
                asyncio.create_task(self._fetch_rss(session, name, url))
                for name, url in RSS_FEEDS
            ]
            rss_batches = await asyncio.gather(*rss_tasks, return_exceptions=True)

        # ── Step 2: DDG — sequential with throttling ──────────────────────────
        ddg_items: List[NewsItem] = []

        # General market queries first
        ddg_items.extend(await self._ddg_query_safe(self._ddg_general))

        # Per-ticker queries — limited to first 6 tickers to avoid long waits
        for sym in self.watchlist[:6]:
            items = await self._ddg_query_safe(self._ddg_ticker, sym)
            ddg_items.extend(items)

        # ── Combine ───────────────────────────────────────────────────────────
        all_items: List[NewsItem] = []
        for batch in rss_batches:
            if isinstance(batch, list):
                all_items.extend(batch)
        all_items.extend(ddg_items)

        return self._dedup_filter(all_items)

    async def fetch_for_symbol(self, symbol: str) -> List[NewsItem]:
        ddg_items = await self._ddg_query_safe(self._ddg_ticker, symbol)
        all_items = await self.fetch_all()
        sym_items = [i for i in all_items if symbol in i.symbols]
        merged    = {i.fingerprint: i for i in ddg_items + sym_items}
        return list(merged.values())

    # ── DDG throttled wrapper ─────────────────────────────────────────────────

    async def _ddg_query_safe(self, fn, *args) -> List[NewsItem]:
        """
        Run a DDG sync function via thread pool, but only one at a time
        (semaphore) with random jitter between calls to avoid 403s.
        Retries once on 403/Ratelimit errors with a longer wait.
        """
        async with _DDG_SEMAPHORE:
            for attempt in range(1, _DDG_MAX_RETRIES + 1):
                try:
                    items = await asyncio.to_thread(fn, *args)
                    # Small jitter AFTER a successful call too, so next call
                    # doesn't immediately fire
                    await asyncio.sleep(random.uniform(_DDG_MIN_DELAY, _DDG_MAX_DELAY))
                    return items
                except Exception as e:
                    err = str(e).lower()
                    is_rate_limit = "403" in str(e) or "ratelimit" in err or "rate limit" in err
                    if is_rate_limit and attempt < _DDG_MAX_RETRIES:
                        wait = random.uniform(8.0, 15.0)   # longer back-off on 403
                        logger.debug("DDG 403 rate-limit, backing off %.1fs (attempt %d/%d)",
                                     wait, attempt, _DDG_MAX_RETRIES)
                        await asyncio.sleep(wait)
                    else:
                        logger.debug("DDG query failed (%s): %s", fn.__name__, e)
                        return []
        return []

    # ── RSS (aiohttp fetch + feedparser in thread) ────────────────────────────

    async def _fetch_rss(
        self, session: aiohttp.ClientSession, name: str, url: str
    ) -> List[NewsItem]:
        try:
            async with session.get(url) as resp:
                content = await resp.read()
            feed  = await asyncio.to_thread(feedparser.parse, content)
            items = []
            for entry in feed.entries[:25]:
                title   = entry.get("title", "").strip()
                if not title:
                    continue
                summary = BeautifulSoup(
                    entry.get("summary", ""), "html.parser"
                ).get_text()
                link    = entry.get("link", "")
                pub     = entry.get("published_parsed") or entry.get("updated_parsed")
                pub_at  = (
                    datetime(*pub[:6], tzinfo=timezone.utc) if pub
                    else datetime.now(timezone.utc)
                )
                symbols = self._extract_symbols(title + " " + summary)
                items.append(NewsItem(title, summary, name, link, symbols, pub_at))
            return items
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("RSS %s error: %s", name, e)
            return []

    # ── DuckDuckGo (sync, called from _ddg_query_safe via to_thread) ──────────

    def _ddg_ticker(self, symbol: str) -> List[NewsItem]:
        """Fetch news for one ticker from DDG. Runs in thread pool."""
        items: List[NewsItem] = []
        try:
            with DDGS() as ddg:
                # One query per ticker (not two) to halve the request rate
                for r in ddg.news(f"{symbol} stock news", max_results=8, timelimit="d"):
                    item = self._parse_ddg(r, [symbol])
                    if item:
                        items.append(item)
        except Exception as e:
            logger.debug("DDG ticker(%s): %s", symbol, e)
        return items

    def _ddg_general(self) -> List[NewsItem]:
        """Fetch general market news from DDG. Runs in thread pool."""
        items: List[NewsItem] = []
        try:
            with DDGS() as ddg:
                # Two general queries instead of four to reduce rate-limit risk
                for q in DDG_GENERAL_QUERIES[:2]:
                    for r in ddg.news(q, max_results=8, timelimit="d"):
                        item = self._parse_ddg(r, [])
                        if item:
                            items.append(item)
                    time.sleep(random.uniform(0.8, 1.5))   # intra-session delay
        except Exception as e:
            logger.debug("DDG general: %s", e)
        return items

    def _parse_ddg(self, result: Dict, hint_symbols: List[str]) -> Optional[NewsItem]:
        try:
            title  = result.get("title", "").strip()
            if not title:
                return None
            body   = result.get("body", "")
            source = result.get("source", "DuckDuckGo")
            url    = result.get("url", "")
            date   = result.get("date")
            pub_at = None
            if date:
                try:
                    pub_at = datetime.fromisoformat(str(date).replace("Z", "+00:00"))
                except Exception:
                    pub_at = datetime.now(timezone.utc)
            symbols = list(set(hint_symbols + self._extract_symbols(title + " " + body)))
            return NewsItem(title, body, source, url, symbols, pub_at)
        except Exception:
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _dedup_filter(self, items: List[NewsItem]) -> List[NewsItem]:
        cutoff   = datetime.now(timezone.utc) - timedelta(hours=self.max_age_hours)
        new_seen: Set[str] = set()
        unique:   List[NewsItem] = []
        for item in items:
            if item.fingerprint in self._seen or item.fingerprint in new_seen:
                continue
            pub = item.published_at
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            if pub < cutoff:
                continue
            new_seen.add(item.fingerprint)
            unique.append(item)
        self._seen.update(new_seen)
        unique.sort(key=lambda x: x.published_at, reverse=True)
        logger.info("News: %d unique items (across %d symbols)", len(unique), len(self.watchlist))
        return unique

    def _extract_symbols(self, text: str) -> List[str]:
        found: Set[str] = set()
        text_lower = text.lower()

        for m in re.finditer(r'\$([A-Z]{1,5})\b', text):
            if m.group(1) in self.watchlist:
                found.add(m.group(1))

        for ticker in self.watchlist:
            if re.search(r'\b' + re.escape(ticker) + r'\b', text):
                found.add(ticker)

        for name, ticker in COMPANY_TICKER_MAP.items():
            if name in text_lower and ticker in self.watchlist:
                found.add(ticker)

        return list(found)