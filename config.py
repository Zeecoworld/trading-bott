"""
config.py — Centralised async-ready configuration.
"""
import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default=None, required: bool = False):
    val = os.getenv(key, default)
    if required and not val:
        raise ValueError(f"Required env var '{key}' not set. Copy .env.example → .env")
    return val


@dataclass
class Config:
    # ── Alpaca ──────────────────────────────────────────────
    ALPACA_API_KEY: str       = field(default_factory=lambda: _get("ALPACA_API_KEY", required=True))
    ALPACA_SECRET_KEY: str    = field(default_factory=lambda: _get("ALPACA_SECRET_KEY", required=True))
    ALPACA_BASE_URL: str      = field(default_factory=lambda: _get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"))
    IS_PAPER_TRADING: bool    = field(init=False)

    # ── Replicate LLM ────────────────────────────────────────
    REPLICATE_API_TOKEN: str  = field(default_factory=lambda: _get("REPLICATE_API_TOKEN", required=True))
    REPLICATE_MODEL: str      = field(default_factory=lambda: _get("REPLICATE_MODEL", "meta/llama-3.3-70b-instruct"))

    # ── Redis ────────────────────────────────────────────────
    REDIS_URL: str            = field(default_factory=lambda: _get("REDIS_URL", "redis://localhost:6379/0"))
    REDIS_KEY_TTL: int        = field(default_factory=lambda: int(_get("REDIS_KEY_TTL", "604800")))  # 7 days

    # ── Risk Management ──────────────────────────────────────
    MAX_POSITION_SIZE: float  = field(default_factory=lambda: float(_get("MAX_POSITION_SIZE", "0.05")))
    MAX_OPEN_POSITIONS: int   = field(default_factory=lambda: int(_get("MAX_OPEN_POSITIONS", "10")))
    MIN_CONFIDENCE: float     = field(default_factory=lambda: float(_get("MIN_CONFIDENCE_THRESHOLD", "0.65")))
    STOP_LOSS_PCT: float      = field(default_factory=lambda: float(_get("STOP_LOSS_PCT", "0.03")))
    TAKE_PROFIT_MULT: float   = field(default_factory=lambda: float(_get("TAKE_PROFIT_MULTIPLIER", "2.5")))
    DAILY_MAX_LOSS_PCT: float = field(default_factory=lambda: float(_get("DAILY_MAX_LOSS_PCT", "0.05")))

    # ── Watchlist ────────────────────────────────────────────
    WATCHLIST: List[str]      = field(default_factory=lambda: [
        s.strip().upper() for s in
        _get("WATCHLIST", "AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,AMD,NFLX,ORCL").split(",")
    ])

    # ── Web Server ───────────────────────────────────────────
    DASHBOARD_PORT: int       = field(default_factory=lambda: int(_get("DASHBOARD_PORT", "5000")))
    DASHBOARD_HOST: str       = field(default_factory=lambda: _get("DASHBOARD_HOST", "0.0.0.0"))

    # ── Scheduling ───────────────────────────────────────────
    SCAN_INTERVAL_MINUTES: int= field(default_factory=lambda: int(_get("SCAN_INTERVAL_MINUTES", "5")))
    NEWS_MAX_AGE_HOURS: int   = field(default_factory=lambda: int(_get("NEWS_MAX_AGE_HOURS", "24")))

    # ── Logging ──────────────────────────────────────────────
    LOG_LEVEL: str            = field(default_factory=lambda: _get("LOG_LEVEL", "INFO"))
    LOG_FILE: str             = field(default_factory=lambda: _get("LOG_FILE", "logs/bot.log"))

    def __post_init__(self):
        self.IS_PAPER_TRADING = "paper" in self.ALPACA_BASE_URL.lower()
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    @property
    def take_profit_pct(self) -> float:
        return self.STOP_LOSS_PCT * self.TAKE_PROFIT_MULT


cfg = Config()

RSS_FEEDS = [
    ("Reuters Business",  "https://feeds.reuters.com/reuters/businessNews"),
    ("Reuters Markets",   "https://feeds.reuters.com/reuters/financialsNews"),
    ("Yahoo Finance",     "https://finance.yahoo.com/news/rssindex"),
    ("MarketWatch",       "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("CNBC Top",          "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    ("Benzinga",          "https://www.benzinga.com/feed"),
    ("Motley Fool",       "https://www.fool.com/feeds/index.aspx"),
]