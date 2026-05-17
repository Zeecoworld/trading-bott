"""
bot/strategy.py — Async trading orchestrator.
All I/O (Alpaca, news, AI, Redis) runs on the asyncio event loop.
CPU-bound work (technicals, backtests) runs in thread pool via asyncio.to_thread().
"""
from __future__ import annotations
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from alpaca_client import AlpacaClient
from ai_decision   import AIDecisionEngine, TradeDecision
from backtester    import run_backtest
from database      import RedisDB
from news_engine   import NewsEngine, NewsItem
from risk_manager  import RiskManager
from technical     import analyse as tech_analyse
from config            import cfg

logger = logging.getLogger(__name__)


class TradingStrategy:
    def __init__(self, db: RedisDB):
        self.db      = db
        self.alpaca  = AlpacaClient(cfg.ALPACA_API_KEY, cfg.ALPACA_SECRET_KEY,
                                     paper=cfg.IS_PAPER_TRADING)
        self.news    = NewsEngine(cfg.WATCHLIST, cfg.NEWS_MAX_AGE_HOURS)
        self.ai      = AIDecisionEngine(cfg.REPLICATE_API_TOKEN, cfg.REPLICATE_MODEL)
        self.risk    = RiskManager(
            max_position_pct   = cfg.MAX_POSITION_SIZE,
            max_open_positions = cfg.MAX_OPEN_POSITIONS,
            stop_loss_pct      = cfg.STOP_LOSS_PCT,
            daily_max_loss_pct = cfg.DAILY_MAX_LOSS_PCT,
        )
        self._running        = False
        self._market_regime: Dict[str, Any] = {}
        self._last_scan:     Optional[str]  = None
        self._ws_callbacks:  List[Callable] = []

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        logger.info("Strategy engine started (asyncio)")
        await asyncio.gather(
            self._scan_loop(),
            self._perf_loop(),
        )

    async def stop(self) -> None:
        self._running = False

    def add_ws_callback(self, fn: Callable) -> None:
        self._ws_callbacks.append(fn)

    async def _emit(self, event: str, data: Dict) -> None:
        payload = {"event": event, "data": data,
                   "ts": datetime.now(timezone.utc).isoformat()}
        await self.db.publish_event(payload)
        for cb in self._ws_callbacks:
            try:
                await cb(payload)
            except Exception:
                pass

    # ── Loops ─────────────────────────────────────────────────────────────────

    async def _scan_loop(self) -> None:
        while self._running:
            try:
                clock = await self.alpaca.get_clock()
                if clock["is_open"]:
                    await self._scan_cycle()
                else:
                    logger.info("Market closed. Next open: %s", clock["next_open"])
            except Exception as e:
                logger.exception("Scan loop error: %s", e)
                await self._emit("error", {"message": str(e)})
            await asyncio.sleep(cfg.SCAN_INTERVAL_MINUTES * 60)

    async def _perf_loop(self) -> None:
        """Snapshot portfolio equity every 5 minutes."""
        while self._running:
            await asyncio.sleep(300)
            try:
                await self._snapshot_performance()
            except Exception as e:
                logger.error("Perf snapshot error: %s", e)

    # ── Scan Cycle ────────────────────────────────────────────────────────────

    async def _scan_cycle(self) -> None:
        logger.info("═" * 60)
        logger.info("SCAN CYCLE — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # 1 — Account & risk gate
        account = await self.alpaca.get_account()
        equity  = account["equity"]
        allowed, reason = self.risk.is_trading_allowed(equity)
        if not allowed:
            logger.warning("Trading halted: %s", reason)
            await self._emit("halt", {"reason": reason})
            return

        # 2 — News + regime (parallel)
        logger.info("Fetching news…")
        all_news, _ = await asyncio.gather(
            self.news.fetch_all(),
            asyncio.sleep(0),    # yield to event loop
        )
        await self.db.save_news_batch([n.to_dict() for n in all_news])

        self._market_regime = await self.ai.analyse_market_regime(
            [n.to_dict() for n in all_news[:20]]
        )
        await self._emit("regime", self._market_regime)
        logger.info("Regime: %s / %s  (conf=%.2f)",
                    self._market_regime.get("regime"),
                    self._market_regime.get("risk_appetite"),
                    self._market_regime.get("confidence", 0))

        # 3 — Manage open positions
        await self._manage_positions(account, all_news)

        # 4 — Scan watchlist concurrently (capped at 5 parallel AI calls)
        positions   = await self.alpaca.get_positions()
        open_syms: Set[str] = {p["symbol"] for p in positions}

        sem = asyncio.Semaphore(5)
        async def _eval_sym(sym: str):
            async with sem:
                try:
                    await self._evaluate_symbol(sym, account, all_news, open_syms)
                except Exception as e:
                    logger.error("Eval error %s: %s", sym, e)

        await asyncio.gather(*[_eval_sym(s) for s in cfg.WATCHLIST])

        await self._snapshot_performance()
        self._last_scan = datetime.now(timezone.utc).isoformat()
        await self._emit("scan_complete", {"scanned": len(cfg.WATCHLIST)})

    # ── Evaluate Single Symbol ────────────────────────────────────────────────

    async def _evaluate_symbol(
        self, symbol: str, account: Dict, all_news: List[NewsItem], open_syms: Set[str]
    ) -> None:
        if symbol in open_syms:
            return

        price, df = await asyncio.gather(
            self.alpaca.get_latest_quote(symbol),
            self.alpaca.get_bars(symbol, limit=200),
        )
        if not price or df is None or len(df) < 60:
            return

        # CPU-bound — run in thread
        tech = await asyncio.to_thread(tech_analyse, df, symbol)
        logger.info("  %-6s $%.2f  tech=%+.3f [%s]",
                    symbol, price, tech.score, tech.trend)

        sym_news = [n.to_dict() for n in all_news if symbol in n.symbols]
        decision = await self.ai.decide(
            symbol=symbol, price=price,
            tech_score=tech.score, tech_signals=tech.signals,
            news_items=sym_news, portfolio_pct=0.0,
            account_equity=account["equity"],
            stop_loss_pct=cfg.STOP_LOSS_PCT,
            take_profit_mult=cfg.TAKE_PROFIT_MULT,
        )
        logger.info("  %-6s AI: %-4s conf=%.2f — %s",
                    symbol, decision.action, decision.confidence,
                    decision.reasoning[:80])

        await self.db.save_signal({
            "symbol":     symbol,
            "action":     decision.action,
            "confidence": decision.confidence,
            "tech_score": tech.score,
            "price":      price,
            "reasoning":  decision.reasoning,
        })
        await self._emit("signal", {
            "symbol": symbol, "action": decision.action,
            "confidence": decision.confidence, "price": price,
            "tech_score": tech.score, "reasoning": decision.reasoning,
        })

        if decision.action == "BUY" and decision.confidence >= cfg.MIN_CONFIDENCE:
            await self._execute_buy(symbol, price, decision, account, open_syms)

    # ── Manage Positions ──────────────────────────────────────────────────────

    async def _manage_positions(
        self, account: Dict, all_news: List[NewsItem]
    ) -> None:
        positions = await self.alpaca.get_positions()
        if not positions:
            return
        equity = account["equity"]
        logger.info("Managing %d open positions", len(positions))

        async def _check(pos: Dict) -> None:
            symbol  = pos["symbol"]
            current = pos.get("current_price") or pos["entry_price"]
            unreal  = pos.get("unrealized_pl", 0) or 0

            price, df = await asyncio.gather(
                self.alpaca.get_latest_quote(symbol),
                self.alpaca.get_bars(symbol, limit=200),
            )
            if not price or df is None or len(df) < 60:
                return

            tech     = await asyncio.to_thread(tech_analyse, df, symbol)
            sym_news = [n.to_dict() for n in all_news if symbol in n.symbols]
            decision = await self.ai.decide(
                symbol=symbol, price=current,
                tech_score=tech.score, tech_signals=tech.signals,
                news_items=sym_news,
                portfolio_pct=abs(pos.get("market_value", 0)) / equity,
                account_equity=equity,
                stop_loss_pct=cfg.STOP_LOSS_PCT,
                take_profit_mult=cfg.TAKE_PROFIT_MULT,
            )
            if decision.action == "SELL" and decision.confidence >= 0.60:
                logger.info("  Closing %s: %s", symbol, decision.reasoning[:60])
                order_id = await self.alpaca.close_position(symbol)
                if order_id:
                    self.risk.record_pnl(unreal)
                    open_trade = await self.db.get_open_trade(symbol)
                    if open_trade:
                        await self.db.close_trade(
                            open_trade["id"], current, unreal
                        )
                    await self._emit("trade_closed", {
                        "symbol": symbol, "exit_price": current, "pnl": unreal,
                    })

        sem = asyncio.Semaphore(3)
        async def _guarded(pos):
            async with sem:
                await _check(pos)

        await asyncio.gather(*[_guarded(p) for p in positions])

    # ── Execute Buy ───────────────────────────────────────────────────────────

    async def _execute_buy(
        self, symbol: str, price: float,
        decision: TradeDecision, account: Dict, open_syms: Set[str]
    ) -> None:
        equity  = account["equity"]
        pos_map = {p["symbol"]: int(p["qty"])
                   for p in await self.alpaca.get_positions()}

        size = self.risk.size_position(
            equity=equity, price=price,
            confidence=decision.confidence,
            open_positions=len(pos_map),
            symbol_positions=pos_map.get(symbol, 0),
        )
        if not size.allowed:
            logger.info("  %s skipped: %s", symbol, size.reason)
            return

        sl, tp = self.risk.calc_stops(price, "buy")
        if decision.stop_loss and 0 < decision.stop_loss < price:
            sl = decision.stop_loss
        if decision.take_profit and decision.take_profit > price:
            tp = decision.take_profit

        order_id = await self.alpaca.place_bracket_order(
            symbol=symbol, qty=size.qty, side="buy",
            stop_loss_price=sl, take_profit_price=tp,
        )
        if order_id:
            trade_id = await self.db.save_trade({
                "symbol":          symbol,
                "side":            "buy",
                "qty":             size.qty,
                "entry_price":     price,
                "stop_loss":       sl,
                "take_profit":     tp,
                "status":          "open",
                "confidence":      decision.confidence,
                "reasoning":       decision.reasoning,
                "alpaca_order_id": order_id,
            })
            open_syms.add(symbol)
            logger.info("  ✅ BUY %dx%s @ $%.2f  SL=%.2f TP=%.2f  trade_id=%d",
                        size.qty, symbol, price, sl, tp, trade_id)
            await self._emit("trade_opened", {
                "symbol": symbol, "qty": size.qty, "price": price,
                "sl": sl, "tp": tp, "confidence": decision.confidence,
                "trade_id": trade_id,
            })

    # ── Backtesting ───────────────────────────────────────────────────────────

    async def run_backtests(self, days: int = 252) -> List[Dict]:
        logger.info("Backtesting %d symbols (%dd)…", len(cfg.WATCHLIST), days)
        sem = asyncio.Semaphore(4)

        async def _bt(symbol: str) -> Optional[Dict]:
            async with sem:
                try:
                    df = await self.alpaca.get_daily_bars(symbol, days=days)
                    result = await asyncio.to_thread(
                        run_backtest, df, symbol,
                        stop_loss_pct=cfg.STOP_LOSS_PCT,
                        take_profit_pct=cfg.take_profit_pct,
                    )
                    logger.info("  %-6s ret=%.1f%%  wr=%.0f%%  sharpe=%.2f",
                                symbol, result.total_return_pct,
                                result.win_rate * 100, result.sharpe_ratio)
                    return {
                        "symbol":          result.symbol,
                        "total_trades":    result.total_trades,
                        "win_rate":        result.win_rate,
                        "total_return_pct":result.total_return_pct,
                        "max_drawdown_pct":result.max_drawdown_pct,
                        "sharpe_ratio":    result.sharpe_ratio,
                        "profit_factor":   result.profit_factor,
                        "expectancy":      result.expectancy,
                        "equity_curve":    result.equity_curve,
                        "trades":          result.trades,
                    }
                except Exception as e:
                    logger.error("Backtest error %s: %s", symbol, e)
                    return None

        raw     = await asyncio.gather(*[_bt(s) for s in cfg.WATCHLIST])
        results = [r for r in raw if r]
        results.sort(key=lambda r: r["sharpe_ratio"], reverse=True)
        return results

    # ── Status (for dashboard API) ────────────────────────────────────────────

    async def get_status(self) -> Dict[str, Any]:
        try:
            account, clock, positions, orders, stats, signals = await asyncio.gather(
                self.alpaca.get_account(),
                self.alpaca.get_clock(),
                self.alpaca.get_positions(),
                self.alpaca.get_open_orders(),
                self.db.trade_stats(),
                self.db.get_signals(limit=100),
            )
        except Exception as e:
            return {"error": str(e)}

        return {
            "account":        account,
            "clock":          clock,
            "positions":      positions,
            "open_orders":    orders,
            "recent_signals": signals,
            "trade_stats":    stats,
            "market_regime":  self._market_regime,
            "last_scan":      self._last_scan,
            "is_running":     self._running,
            "daily_pnl":      self.risk.daily_pnl,
            "watchlist":      cfg.WATCHLIST,
            "paper_trading":  cfg.IS_PAPER_TRADING,
        }

    # ── Snapshot ──────────────────────────────────────────────────────────────

    async def _snapshot_performance(self) -> None:
        try:
            acct  = await self.alpaca.get_account()
            stats = await self.db.trade_stats()
            await self.db.save_performance({
                "equity":          acct["equity"],
                "cash":            acct["cash"],
                "portfolio_value": acct["portfolio_value"],
                "daily_pnl":       self.risk.daily_pnl,
                "daily_pnl_pct":   self.risk.daily_pnl / acct["equity"]
                                   if acct["equity"] else 0,
                "total_pnl":       stats["total_pnl"],
                "win_rate":        stats["win_rate"],
                "open_positions":  stats["open_trades"],
            })
        except Exception as e:
            logger.error("Snapshot error: %s", e)