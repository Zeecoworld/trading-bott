from __future__ import annotations
import argparse
import asyncio
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so imports work regardless of cwd
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from logger import setup_logging
from config     import cfg          # also prints which .env was loaded


async def async_main(args: argparse.Namespace) -> None:
    import logging
    logger = logging.getLogger("main")

    # ── Redis ─────────────────────────────────────────────────────────────────
    from database import RedisDB
    db = RedisDB(cfg.REDIS_URL, ttl=cfg.REDIS_KEY_TTL)

    import re as _re2
    _safe = lambda u: _re2.sub(r":([^@/]{3,})@", ":***@", u)
    logger.info("Connecting to Redis: %s", _safe(cfg.REDIS_URL))
    try:
        await db.connect()
    except ConnectionError as e:
        # Pretty-print the checklist from database.py
        print(f"\n{'─'*60}", flush=True)
        print(str(e), flush=True)
        print('─'*60, flush=True)
        sys.exit(1)

    # ── Flush Redis (dev helper) ───────────────────────────────────────────────
    if args.flush_redis:
        if input("⚠  This will DELETE all bot data in Redis. Type 'yes' to confirm: ").strip() == "yes":
            await db.flush_all()
            print("Redis bot data flushed.")
        else:
            print("Aborted.")
        await db.close()
        return

    # ── Strategy ──────────────────────────────────────────────────────────────
    from strategy import TradingStrategy
    strategy = TradingStrategy(db)

    # ── --status ──────────────────────────────────────────────────────────────
    if args.status:
        status = await strategy.get_status()
        stats  = await db.trade_stats()
        perf   = await db.get_performance(limit=1)
        print("\n── Account ─────────────────────────────────────────")
        acct = status.get("account", {})
        for k, v in acct.items():
            print(f"  {k:<22} {v}")
        print("\n── Trade Stats ─────────────────────────────────────")
        for k, v in stats.items():
            print(f"  {k:<22} {v}")
        print("\n── Market Regime ────────────────────────────────────")
        reg = status.get("market_regime", {})
        for k, v in reg.items():
            print(f"  {k:<22} {v}")
        await db.close()
        return

    # ── --backtest ────────────────────────────────────────────────────────────
    if args.backtest:
        logger.info("Backtesting %d symbols (%d days)…", len(cfg.WATCHLIST), args.backtest_days)
        results = await strategy.run_backtests(days=args.backtest_days)
        print(f"\n{'Symbol':<8} {'Return%':>8} {'WinRate':>8} {'Trades':>7} {'Sharpe':>8} {'MaxDD%':>8} {'PF':>7}")
        print("─" * 60)
        for r in results:
            print(f"{r['symbol']:<8} {r['total_return_pct']:>7.1f}% "
                  f"{r['win_rate']*100:>7.1f}% {r['total_trades']:>7} "
                  f"{r['sharpe_ratio']:>8.2f} {r['max_drawdown_pct']:>7.1f}% "
                  f"{r['profit_factor']:>6.2f}x")
        await db.close()
        return

    # ── Dashboard ─────────────────────────────────────────────────────────────
    from app import run_dashboard, set_deps
    set_deps(strategy, db)

    tasks = []
    if not args.no_dashboard:
        tasks.append(asyncio.create_task(
            run_dashboard(cfg.DASHBOARD_HOST, cfg.DASHBOARD_PORT, db=db),
            name="dashboard",
        ))
        await asyncio.sleep(0.8)
        logger.info("Dashboard → http://localhost:%d", cfg.DASHBOARD_PORT)

    # ── Trading bot ───────────────────────────────────────────────────────────
    if not args.dashboard_only:
        if not cfg.IS_PAPER_TRADING:
            logger.warning("=" * 60)
            logger.warning("⚠  LIVE TRADING MODE — REAL MONEY AT RISK")
            logger.warning("=" * 60)
            await asyncio.sleep(4)
        tasks.append(asyncio.create_task(strategy.start(), name="strategy"))
    else:
        logger.info("Dashboard-only mode — trading bot NOT started")

    if not tasks:
        logger.error("Nothing to run. Use --help.")
        await db.close()
        return

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutting down…")
        await strategy.stop()
        await db.close()
        logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="APEX TRADER v2 — asyncio + Redis + aiohttp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # run bot + dashboard
  python main.py --dashboard-only       # monitor only
  python main.py --backtest             # backtest all watchlist symbols
  python main.py --backtest-days 90     # backtest with 90 days history
  python main.py --status               # print account + trade stats
  python main.py --flush-redis          # wipe Redis bot data (dev)
        """,
    )
    parser.add_argument("--dashboard-only", action="store_true", help="Dashboard only, no trading")
    parser.add_argument("--no-dashboard",   action="store_true", help="Trading bot only, no dashboard")
    parser.add_argument("--backtest",       action="store_true", help="Run backtests and exit")
    parser.add_argument("--backtest-days",  type=int, default=252, help="Days of history for backtest")
    parser.add_argument("--status",         action="store_true", help="Print status and exit")
    parser.add_argument("--flush-redis",    action="store_true", help="⚠ Wipe all Redis bot data")
    args = parser.parse_args()

    setup_logging(cfg.LOG_LEVEL)  # stdout only — no log file on Render

    import re as _re
    def _safe_url(url: str) -> str:
        return _re.sub(r":([^@/]{3,})@", ":***@", url)

    mode      = "PAPER" if cfg.IS_PAPER_TRADING else "⚠  LIVE"
    redis_url = _safe_url(cfg.REDIS_URL)[:44]
    print(f"""
╔══════════════════════════════════════════════════════╗
║       APEX TRADER v2 — asyncio + Redis + aiohttp     ║
║  Mode   : {mode:<44}║
║  LLM    : {cfg.REPLICATE_MODEL[:44]:<44}║
║  Redis  : {redis_url:<44}║
╚══════════════════════════════════════════════════════╝
""", flush=True)

    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()