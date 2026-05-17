"""
dashboard/app.py — aiohttp async web server.
REST API + native WebSocket (no Flask, no SocketIO).
WebSocket clients subscribe to Redis pub/sub channel 'bot:events'
and also receive direct pushes from the strategy via _ws_clients set.
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Set

import aiohttp
import aiohttp_cors
from aiohttp import web

logger = logging.getLogger(__name__)

_strategy  = None
_db        = None
_ws_clients: Set[web.WebSocketResponse] = set()


def set_deps(strategy, db):
    global _strategy, _db
    _strategy = strategy
    _db       = db

    # Push live events straight to all WebSocket connections
    async def _push(event: dict):
        await _broadcast(event)

    strategy.add_ws_callback(_push)


# ── WebSocket broadcast ───────────────────────────────────────────────────────

async def _broadcast(data: dict) -> None:
    if not _ws_clients:
        return
    payload = json.dumps(data, default=str)
    dead    = set()
    for ws in list(_ws_clients):
        try:
            await ws.send_str(payload)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


# ── Redis pub/sub relay → WebSocket ─────────────────────────────────────────

async def _redis_relay(db) -> None:
    """Subscribe to Redis 'bot:events' and forward to all WS clients."""
    pubsub = db.pubsub()
    await pubsub.subscribe("bot:events")
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    await _broadcast(data)
                except Exception:
                    pass
    except asyncio.CancelledError:
        pass
    finally:
        await pubsub.unsubscribe("bot:events")
        await pubsub.aclose()


# ── Route handlers ─────────────────────────────────────────────────────────────

async def handle_index(request: web.Request) -> web.Response:
    tmpl = Path(__file__).parent / "templates" / "index.html"
    return web.Response(text=tmpl.read_text(), content_type="text/html")


async def handle_status(request: web.Request) -> web.Response:
    if not _strategy:
        return web.json_response({"error": "Strategy not initialised"}, status=503)
    data = await _strategy.get_status()
    return web.json_response(data, dumps=lambda d: json.dumps(d, default=str))


async def handle_trades(request: web.Request) -> web.Response:
    trades = await _db.get_trades(limit=200)
    return web.json_response(trades, dumps=lambda d: json.dumps(d, default=str))


async def handle_news(request: web.Request) -> web.Response:
    limit = int(request.rel_url.query.get("limit", 100))
    items = await _db.get_news(limit=limit)
    return web.json_response(items, dumps=lambda d: json.dumps(d, default=str))


async def handle_performance(request: web.Request) -> web.Response:
    limit = int(request.rel_url.query.get("limit", 288))
    snaps = await _db.get_performance(limit=limit)
    return web.json_response(snaps, dumps=lambda d: json.dumps(d, default=str))


async def handle_backtest_all(request: web.Request) -> web.Response:
    if not _strategy:
        return web.json_response({"error": "No strategy"}, status=503)
    body = await request.json()
    days = int(body.get("days", 252))
    results = await _strategy.run_backtests(days=days)
    return web.json_response(results, dumps=lambda d: json.dumps(d, default=str))


async def handle_backtest_symbol(request: web.Request) -> web.Response:
    if not _strategy:
        return web.json_response({"error": "No strategy"}, status=503)
    symbol = request.match_info["symbol"].upper()
    days   = int(request.rel_url.query.get("days", 252))
    from bot.backtester import run_backtest
    from config import cfg
    df     = await _strategy.alpaca.get_daily_bars(symbol, days=days)
    result = await asyncio.to_thread(
        run_backtest, df, symbol,
        stop_loss_pct=cfg.STOP_LOSS_PCT,
        take_profit_pct=cfg.take_profit_pct,
    )
    return web.json_response({
        "symbol":           result.symbol,
        "total_trades":     result.total_trades,
        "win_rate":         result.win_rate,
        "total_return_pct": result.total_return_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "sharpe_ratio":     result.sharpe_ratio,
        "profit_factor":    result.profit_factor,
        "equity_curve":     result.equity_curve,
        "trades":           result.trades,
    })


async def handle_force_scan(request: web.Request) -> web.Response:
    if not _strategy:
        return web.json_response({"error": "No strategy"}, status=503)
    asyncio.create_task(_strategy._scan_cycle())
    return web.json_response({"status": "scan_triggered"})


async def handle_close_position(request: web.Request) -> web.Response:
    if not _strategy:
        return web.json_response({"error": "No strategy"}, status=503)
    symbol   = request.match_info["symbol"].upper()
    order_id = await _strategy.alpaca.close_position(symbol)
    return web.json_response({"order_id": order_id})


async def handle_signals(request: web.Request) -> web.Response:
    limit   = int(request.rel_url.query.get("limit", 100))
    signals = await _db.get_signals(limit=limit)
    return web.json_response(signals, dumps=lambda d: json.dumps(d, default=str))


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def handle_websocket(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    _ws_clients.add(ws)
    logger.debug("WS client connected (%d total)", len(_ws_clients))

    # Send initial state immediately
    if _strategy:
        try:
            status = await _strategy.get_status()
            await ws.send_json({"event": "status", "data": status})
        except Exception:
            pass

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    if data.get("type") == "request_status" and _strategy:
                        status = await _strategy.get_status()
                        await ws.send_json({"event": "status", "data": status})
                except Exception:
                    pass
            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                break
    finally:
        _ws_clients.discard(ws)
        logger.debug("WS client disconnected (%d remaining)", len(_ws_clients))
    return ws


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> web.Application:
    app = web.Application()

    # Routes
    app.router.add_get("/",                           handle_index)
    app.router.add_get("/ws",                         handle_websocket)
    app.router.add_get("/api/status",                 handle_status)
    app.router.add_get("/api/trades",                 handle_trades)
    app.router.add_get("/api/news",                   handle_news)
    app.router.add_get("/api/performance",            handle_performance)
    app.router.add_get("/api/signals",                handle_signals)
    app.router.add_post("/api/backtest",              handle_backtest_all)
    app.router.add_get("/api/backtest/{symbol}",      handle_backtest_symbol)
    app.router.add_post("/api/force_scan",            handle_force_scan)
    app.router.add_post("/api/close_position/{symbol}", handle_close_position)

    # CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["GET", "POST", "OPTIONS"],
        )
    })
    for route in list(app.router.routes()):
        if route.resource.canonical != "/ws":
            try:
                cors.add(route)
            except Exception:
                pass

    return app


async def run_dashboard(host: str = "0.0.0.0", port: int = 5000, db=None) -> None:
    app = create_app()

    # Start Redis → WebSocket relay as background task
    if db:
        asyncio.create_task(_redis_relay(db))

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info("Dashboard running → http://%s:%d", host, port)