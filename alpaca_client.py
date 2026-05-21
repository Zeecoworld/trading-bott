"""
bot/alpaca_client.py — Fully async Alpaca wrapper.
alpaca-py SDK is synchronous; all calls are offloaded via asyncio.to_thread()
so they never block the event loop.
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any, Dict, List, Optional

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, StopLossRequest,
    TakeProfitRequest, GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed          # ← required for real-time quotes

logger = logging.getLogger(__name__)


async def _t(fn, *args, **kwargs):
    """Run a synchronous function in a thread pool without blocking the loop."""
    return await asyncio.to_thread(fn, *args, **kwargs)


class AlpacaClient:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.paper   = paper
        self.trading = TradingClient(api_key, secret_key, paper=paper)
        self.data    = StockHistoricalDataClient(api_key, secret_key)
        logger.info("Alpaca client ready (%s)", "PAPER" if paper else "⚠ LIVE")

    # ── Account ──────────────────────────────────────────────────────────────

    async def get_account(self) -> Dict[str, Any]:
        acct = await _t(self.trading.get_account)
        return {
            "equity":          float(acct.equity),
            "cash":            float(acct.cash),
            "portfolio_value": float(acct.portfolio_value),
            "buying_power":    float(acct.buying_power),
            "day_trade_count": int(acct.daytrade_count),
            "trading_blocked": acct.trading_blocked,
            "account_blocked": acct.account_blocked,
        }

    async def get_clock(self) -> Dict[str, Any]:
        clock = await _t(self.trading.get_clock)
        return {
            "is_open":    clock.is_open,
            "next_open":  clock.next_open.isoformat()  if clock.next_open  else None,
            "next_close": clock.next_close.isoformat() if clock.next_close else None,
        }

    # ── Positions ────────────────────────────────────────────────────────────

    async def get_positions(self) -> List[Dict[str, Any]]:
        positions = await _t(self.trading.get_all_positions)
        return [
            {
                "symbol":          p.symbol,
                "qty":             float(p.qty),
                "side":            p.side.value,
                "entry_price":     float(p.avg_entry_price),
                "current_price":   float(p.current_price)   if p.current_price   else None,
                "market_value":    float(p.market_value)    if p.market_value    else None,
                "unrealized_pl":   float(p.unrealized_pl)   if p.unrealized_pl   else None,
                "unrealized_plpc": float(p.unrealized_plpc) if p.unrealized_plpc else None,
            }
            for p in positions
        ]

    async def get_position(self, symbol: str) -> Optional[Dict]:
        try:
            p = await _t(self.trading.get_open_position, symbol)
            return {
                "symbol":        p.symbol,
                "qty":           float(p.qty),
                "entry_price":   float(p.avg_entry_price),
                "current_price": float(p.current_price)  if p.current_price  else None,
                "unrealized_pl": float(p.unrealized_pl)  if p.unrealized_pl  else None,
            }
        except Exception:
            return None

    async def close_position(self, symbol: str) -> Optional[str]:
        try:
            order = await _t(self.trading.close_position, symbol)
            logger.info("Closed position: %s  order=%s", symbol, order.id)
            return str(order.id)
        except Exception as e:
            logger.error("close_position(%s) failed: %s", symbol, e)
            return None

    async def close_all_positions(self) -> None:
        await _t(self.trading.close_all_positions, cancel_orders=True)
        logger.warning("Closed ALL positions")

    # ── Orders ───────────────────────────────────────────────────────────────

    async def place_bracket_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_loss_price: float,
        take_profit_price: float,
        retries: int = 3,
    ) -> Optional[str]:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            order_class="bracket",
            stop_loss=StopLossRequest(stop_price=round(stop_loss_price, 2)),
            take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2)),
        )
        for attempt in range(1, retries + 1):
            try:
                order = await _t(self.trading.submit_order, req)
                logger.info(
                    "Bracket %s %dx%s  SL=%.2f TP=%.2f  id=%s",
                    side.upper(), qty, symbol, stop_loss_price, take_profit_price, order.id,
                )
                return str(order.id)
            except Exception as e:
                logger.warning("Order attempt %d/%d failed: %s", attempt, retries, e)
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)
        logger.error("Bracket order failed after %d attempts: %s", retries, symbol)
        return None

    async def place_market_order(self, symbol: str, qty: float, side: str) -> Optional[str]:
        try:
            req = MarketOrderRequest(
                symbol=symbol, qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = await _t(self.trading.submit_order, req)
            logger.info("Market order: %s %dx%s", side.upper(), qty, symbol)
            return str(order.id)
        except Exception as e:
            logger.error("Market order failed %s: %s", symbol, e)
            return None

    async def get_open_orders(self) -> List[Dict]:
        orders = await _t(self.trading.get_orders,
                          GetOrdersRequest(status=QueryOrderStatus.OPEN))
        return [
            {"id": str(o.id), "symbol": o.symbol, "side": o.side.value,
             "qty": float(o.qty), "status": o.status.value}
            for o in orders
        ]

    async def cancel_order(self, order_id: str) -> None:
        await _t(self.trading.cancel_order_by_id, order_id)

    # ── Market Data ──────────────────────────────────────────────────────────

    async def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.Hour,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        start = start or datetime.now(timezone.utc) - timedelta(days=60)
        end   = end   or datetime.now(timezone.utc)

        def _fetch():
            req  = StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=timeframe,
                start=start, end=end, limit=limit,
                feed=DataFeed.IEX,          # real-time feed; avoids 15-min delay
            )
            bars = self.data.get_stock_bars(req)
            df   = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.loc[symbol] if symbol in df.index.get_level_values(0) \
                     else df.droplevel(0)
            df.index   = pd.to_datetime(df.index)
            df.columns = [c.lower() for c in df.columns]
            return df

        for attempt in range(3):
            try:
                return await _t(_fetch)
            except Exception as e:
                if attempt == 2:
                    raise
                await asyncio.sleep(1.5 ** attempt)

    async def get_daily_bars(self, symbol: str, days: int = 252) -> pd.DataFrame:
        return await self.get_bars(
            symbol, TimeFrame.Day,
            start=datetime.now(timezone.utc) - timedelta(days=days),
        )

    async def get_latest_quote(self, symbol: str) -> Optional[float]:
        try:
            def _q():
                req   = StockLatestQuoteRequest(
                    symbol_or_symbols=symbol,
                    feed=DataFeed.IEX,      # real-time; SIP requires paid subscription
                )
                quote = self.data.get_stock_latest_quote(req)
                q     = quote[symbol]
                ask   = float(q.ask_price) if q.ask_price else 0.0
                bid   = float(q.bid_price) if q.bid_price else 0.0
                # Prefer mid-price; fall back to whichever side is non-zero
                if ask > 0 and bid > 0:
                    return (ask + bid) / 2.0
                return ask or bid or None
            price = await _t(_q)
            if price and price > 0:
                return price
            return None
        except Exception as e:
            logger.warning("Quote failed %s: %s", symbol, e)
            return None

    async def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        tasks  = [self.get_latest_quote(s) for s in symbols]
        prices = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            sym: p for sym, p in zip(symbols, prices)
            if isinstance(p, float)
        }