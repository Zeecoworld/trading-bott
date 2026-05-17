"""
bot/news_engine.py — Fully async multi-source news aggregation.
HTTP fetching via aiohttp; RSS parsing and DDG run in thread pool.
"""
from __future__ import annotations
import asyncio
import hashlib
import logging
import re
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


class NewsItem:
    __slots__ = ["title", "summary", "source", "url", "symbols",
                 "published_at", "fingerprint"]

    def __init__(self, title: str, summary: str, source: str, url: str,
                 symbols: List[str], published_at: Optional[datetime]):
        self.title       = title.strip()
        self.summary     = summary.strip()[:500]
        self.source      = source
        self.url         = url
        self.symbols     = sorted(set(symbols))
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
        """Fetch all sources concurrently, deduplicate, age-filter."""
        async with aiohttp.ClientSession(
            timeout=_TIMEOUT, headers=_HEADERS
        ) as session:
            tasks: List[asyncio.Task] = []

            # RSS feeds — fetch HTML async, parse in thread
            for name, url in RSS_FEEDS:
                tasks.append(asyncio.create_task(
                    self._fetch_rss(session, name, url)
                ))

            # DDG — entirely sync, run in thread pool
            tasks.append(asyncio.create_task(
                asyncio.to_thread(self._ddg_general)
            ))
            for sym in self.watchlist[:10]:
                tasks.append(asyncio.create_task(
                    asyncio.to_thread(self._ddg_ticker, sym)
                ))

            batches = await asyncio.gather(*tasks, return_exceptions=True)

        items: List[NewsItem] = []
        for batch in batches:
            if isinstance(batch, Exception):
                logger.debug("News source error: %s", batch)
            elif isinstance(batch, list):
                items.extend(batch)

        return self._dedup_filter(items)

    async def fetch_for_symbol(self, symbol: str) -> List[NewsItem]:
        ddg_items = await asyncio.to_thread(self._ddg_ticker, symbol)
        all_items = await self.fetch_all()
        sym_items = [i for i in all_items if symbol in i.symbols]
        merged    = {i.fingerprint: i for i in ddg_items + sym_items}
        return list(merged.values())

    # ── RSS (aiohttp fetch + feedparser in thread) ────────────────────────────

    async def _fetch_rss(
        self, session: aiohttp.ClientSession, name: str, url: str
    ) -> List[NewsItem]:
        try:
            async with session.get(url) as resp:
                content = await resp.read()
            # feedparser is CPU-bound; offload to thread
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

    # ── DuckDuckGo (sync, runs in thread) ────────────────────────────────────

    def _ddg_ticker(self, symbol: str) -> List[NewsItem]:
        items: List[NewsItem] = []
        try:
            with DDGS() as ddg:
                for q in (f"{symbol} stock news", f"${symbol} earnings analyst"):
                    for r in ddg.news(q, max_results=8, timelimit="d"):
                        item = self._parse_ddg(r, [symbol])
                        if item:
                            items.append(item)
        except Exception as e:
            logger.debug("DDG ticker(%s): %s", symbol, e)
        return items

    def _ddg_general(self) -> List[NewsItem]:
        items: List[NewsItem] = []
        try:
            with DDGS() as ddg:
                for q in DDG_GENERAL_QUERIES:
                    for r in ddg.news(q, max_results=10, timelimit="d"):
                        item = self._parse_ddg(r, [])
                        if item:
                            items.append(item)
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