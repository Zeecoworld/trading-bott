"""
bot/database.py — Async Redis storage layer.

Supports ALL Redis URL formats:
  redis://localhost:6379/0                     local
  redis://:password@host:6379/0               authenticated
  rediss://:password@host:6380/0              SSL/TLS  (Upstash, Redis Cloud)
  redis+sentinel://sentinel-host:26379/...    Sentinel

Key schema
──────────
trades:counter          INT   auto-increment
trades:index            ZSET  score=epoch, member=trade_id
trades:{id}             HASH  trade fields
signals:counter         INT
signals:index           ZSET
signals:{id}            HASH
news:index              ZSET  score=published_epoch, member=fingerprint
news:{fingerprint}      HASH
perf:snapshots          LIST  JSON strings, trimmed to 2880 entries
bot:state:{key}         STRING
bot:events              PUBSUB channel
"""
from __future__ import annotations
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import (
    BusyLoadingError, ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError,
)

logger = logging.getLogger(__name__)


class RedisDB:
    def __init__(self, url: str = "redis://localhost:6379/0", ttl: int = 604_800):
        self._url = url
        self._ttl = ttl
        self._r: Optional[aioredis.Redis] = None

    # ── Connection ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Create the Redis client and verify connectivity.
        Automatically handles:
          - redis://   plain TCP
          - rediss://  SSL/TLS  (Upstash, Redis Cloud, etc.)
          - passwords embedded in URL
        """
        is_ssl = self._url.startswith("rediss://")

        # Retry on transient failures (3 attempts, exponential backoff)
        retry = Retry(ExponentialBackoff(cap=4, base=0.5), retries=3)

        self._r = aioredis.from_url(
            self._url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=10,
            socket_timeout=10,
            retry=retry,
            retry_on_error=[BusyLoadingError, RedisConnectionError, RedisTimeoutError],
            # SSL settings — redis-py infers ssl=True from rediss:// automatically,
            # but we set ssl_cert_reqs explicitly for cloud providers that use
            # self-signed certs (Upstash is fine; Redis Cloud needs CERT_REQUIRED).
            **({"ssl_cert_reqs": None} if is_ssl else {}),
        )

        try:
            await self._r.ping()
        except Exception as e:
            await self._r.aclose()
            self._r = None
            raise ConnectionError(
                f"Redis ping failed at {self._safe_url()}\n"
                f"  Error: {e}\n\n"
                f"  Checklist:\n"
                f"  • Is REDIS_URL correct in your .env?\n"
                f"  • For Upstash: use  rediss://:<password>@<host>:6380\n"
                f"  • For Redis Cloud: use  redis://:<password>@<host>:<port>\n"
                f"  • For local: run  docker run -p 6379:6379 redis:7-alpine\n"
            ) from e

        logger.info("Redis connected (%s) %s",
                    "SSL/TLS" if is_ssl else "plain", self._safe_url())

    async def close(self) -> None:
        if self._r:
            await self._r.aclose()
            self._r = None

    def _safe_url(self) -> str:
        """Mask password for logging."""
        import re
        return re.sub(r":([^@/]{3,})@", ":***@", self._url)

    @property
    def r(self) -> aioredis.Redis:
        if self._r is None:
            raise RuntimeError("RedisDB not connected — call await db.connect() first")
        return self._r

    # ── Trades ────────────────────────────────────────────────────────────────

    async def save_trade(self, trade: Dict[str, Any]) -> int:
        trade_id = await self.r.incr("trades:counter")
        trade["id"] = trade_id
        trade.setdefault("opened_at", datetime.now(timezone.utc).isoformat())
        score = _epoch(trade.get("opened_at"))

        pipe = self.r.pipeline()
        pipe.hset(f"trades:{trade_id}", mapping=_flatten(trade))
        pipe.zadd("trades:index", {str(trade_id): score})
        pipe.expire(f"trades:{trade_id}", self._ttl)
        await pipe.execute()
        return trade_id

    async def update_trade(self, trade_id: int, fields: Dict[str, Any]) -> None:
        if not await self.r.exists(f"trades:{trade_id}"):
            return
        await self.r.hset(f"trades:{trade_id}", mapping=_flatten(fields))

    async def get_trade(self, trade_id: int) -> Optional[Dict]:
        data = await self.r.hgetall(f"trades:{trade_id}")
        return _parse(data) if data else None

    async def get_trades(self, limit: int = 200, status: Optional[str] = None) -> List[Dict]:
        ids = await self.r.zrevrange("trades:index", 0, limit - 1)
        if not ids:
            return []
        pipe = self.r.pipeline()
        for tid in ids:
            pipe.hgetall(f"trades:{tid}")
        results = await pipe.execute()
        trades  = [_parse(r) for r in results if r]
        if status:
            trades = [t for t in trades if t.get("status") == status]
        return trades

    async def get_open_trade(self, symbol: str) -> Optional[Dict]:
        trades = await self.get_trades(limit=500, status="open")
        return next((t for t in trades if t.get("symbol") == symbol), None)

    async def close_trade(self, trade_id: int, exit_price: float, pnl: float) -> None:
        await self.r.hset(f"trades:{trade_id}", mapping=_flatten({
            "exit_price": exit_price,
            "pnl":        round(pnl, 4),
            "status":     "closed",
            "closed_at":  datetime.now(timezone.utc).isoformat(),
        }))

    # ── Signals ───────────────────────────────────────────────────────────────

    async def save_signal(self, signal: Dict[str, Any]) -> int:
        sig_id = await self.r.incr("signals:counter")
        signal["id"] = sig_id
        signal.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        score = _epoch(signal.get("created_at"))

        pipe = self.r.pipeline()
        pipe.hset(f"signals:{sig_id}", mapping=_flatten(signal))
        pipe.zadd("signals:index", {str(sig_id): score})
        pipe.expire(f"signals:{sig_id}", self._ttl)
        pipe.zremrangebyrank("signals:index", 0, -1001)   # keep latest 1000
        await pipe.execute()
        return sig_id

    async def get_signals(self, limit: int = 100) -> List[Dict]:
        ids = await self.r.zrevrange("signals:index", 0, limit - 1)
        if not ids:
            return []
        pipe = self.r.pipeline()
        for sid in ids:
            pipe.hgetall(f"signals:{sid}")
        return [_parse(r) for r in await pipe.execute() if r]

    # ── News ──────────────────────────────────────────────────────────────────

    async def save_news_batch(self, items: List[Dict]) -> int:
        if not items:
            return 0
        pipe  = self.r.pipeline()
        saved = 0
        for item in items:
            fp = item.get("fingerprint", "")
            if not fp:
                continue
            score      = _epoch(item.get("published_at"))
            item_copy  = {k: v for k, v in item.items() if v is not None}
            if isinstance(item_copy.get("symbols"), list):
                item_copy["symbols"] = json.dumps(item_copy["symbols"])
            pipe.hset(f"news:{fp}", mapping=_flatten(item_copy))
            pipe.zadd("news:index", {fp: score}, nx=True)
            pipe.expire(f"news:{fp}", self._ttl)
            saved += 1
        pipe.zremrangebyrank("news:index", 0, -2001)
        await pipe.execute()
        return saved

    async def get_news(self, limit: int = 100) -> List[Dict]:
        fps = await self.r.zrevrange("news:index", 0, limit - 1)
        if not fps:
            return []
        pipe = self.r.pipeline()
        for fp in fps:
            pipe.hgetall(f"news:{fp}")
        items = []
        for raw in await pipe.execute():
            if not raw:
                continue
            item = _parse(raw)
            syms = item.get("symbols", "[]")
            try:
                item["symbols"] = json.loads(syms) if isinstance(syms, str) else syms
            except Exception:
                item["symbols"] = []
            items.append(item)
        return items

    # ── Performance Snapshots ─────────────────────────────────────────────────

    async def save_performance(self, snap: Dict[str, Any]) -> None:
        snap["snapshot_at"] = datetime.now(timezone.utc).isoformat()
        pipe = self.r.pipeline()
        pipe.lpush("perf:snapshots", json.dumps(snap))
        pipe.ltrim("perf:snapshots", 0, 2879)
        await pipe.execute()

    async def get_performance(self, limit: int = 288) -> List[Dict]:
        raw   = await self.r.lrange("perf:snapshots", 0, limit - 1)
        snaps = []
        for r in reversed(raw):
            try:
                snaps.append(json.loads(r))
            except Exception:
                pass
        return snaps

    # ── Bot State ─────────────────────────────────────────────────────────────

    async def set_state(self, key: str, value: Any) -> None:
        await self.r.set(f"bot:state:{key}", json.dumps(value))

    async def get_state(self, key: str, default=None) -> Any:
        val = await self.r.get(f"bot:state:{key}")
        if val is None:
            return default
        try:
            return json.loads(val)
        except Exception:
            return val

    # ── Pub/Sub ───────────────────────────────────────────────────────────────

    async def publish_event(self, event: Dict) -> None:
        try:
            await self.r.publish("bot:events", json.dumps(event, default=str))
        except Exception as e:
            logger.debug("publish_event failed: %s", e)

    def pubsub(self) -> aioredis.client.PubSub:
        return self.r.pubsub()

    # ── Stats ─────────────────────────────────────────────────────────────────

    async def trade_stats(self) -> Dict[str, Any]:
        trades    = await self.get_trades(limit=1000)
        closed    = [t for t in trades if t.get("status") == "closed"]
        wins      = [t for t in closed if float(t.get("pnl", 0)) > 0]
        total_pnl = sum(float(t.get("pnl", 0)) for t in closed)
        win_rate  = len(wins) / len(closed) if closed else 0.0
        return {
            "total_trades":  len(trades),
            "closed_trades": len(closed),
            "open_trades":   len(trades) - len(closed),
            "wins":          len(wins),
            "losses":        len(closed) - len(wins),
            "win_rate":      round(win_rate, 4),
            "total_pnl":     round(total_pnl, 2),
        }

    async def flush_all(self) -> None:
        """⚠️  Dev only — wipes all bot data from Redis."""
        keys  = await self.r.keys("trades:*")
        keys += await self.r.keys("signals:*")
        keys += await self.r.keys("news:*")
        keys += await self.r.keys("bot:*")
        if keys:
            await self.r.delete(*keys)
        await self.r.delete("perf:snapshots")
        logger.warning("Redis: all bot data flushed")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _epoch(ts: Any) -> float:
    if ts is None:
        return time.time()
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, datetime):
        return ts.timestamp()
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except Exception:
        return time.time()


def _flatten(d: Dict) -> Dict[str, str]:
    """Redis HSET requires all values to be strings."""
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        out[str(k)] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
    return out


def _parse(d: Dict[str, str]) -> Dict[str, Any]:
    """Best-effort type coercion on Redis hash data."""
    FLOATS = {"entry_price","exit_price","pnl","pnl_pct","confidence","risk_score",
               "tech_score","news_score","ai_score","stop_loss","take_profit","price",
               "sentiment","equity","cash","portfolio_value","daily_pnl","daily_pnl_pct",
               "total_pnl","win_rate"}
    INTS   = {"id","open_positions"}
    result = {}
    for k, v in d.items():
        if k in FLOATS:
            try:    result[k] = float(v)
            except: result[k] = v
        elif k in INTS:
            try:    result[k] = int(v)
            except: result[k] = v
        else:
            result[k] = v
    return result