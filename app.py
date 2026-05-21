"""
dashboard/app.py — aiohttp async web server.
REST API + native WebSocket + in-memory log viewer.

Log endpoints:
  GET /api/logs               last 200 entries (JSON)
  GET /api/logs?limit=500     up to 500 entries
  GET /api/logs?level=WARNING WARNING and above only
  GET /api/logs?clear=1       wipe the buffer and return empty list
  WS  /ws                     receives {"event":"log","data":{...}} in real time
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

import logger as _logger_module   # our custom logger with the ring buffer

log = logging.getLogger(__name__)

_strategy   = None
_db         = None
_ws_clients: Set[web.WebSocketResponse] = set()


# ── WebSocket broadcast (also wired into the log buffer) ─────────────────────

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


def set_deps(strategy, db):
    global _strategy, _db
    _strategy = strategy
    _db       = db

    # Strategy events → WebSocket
    async def _push(event: dict):
        await _broadcast(event)
    strategy.add_ws_callback(_push)

    # Log records → WebSocket  (wired here so _broadcast is already defined)
    _logger_module.set_ws_broadcast(_broadcast)


# ── Redis pub/sub relay ───────────────────────────────────────────────────────

async def _redis_relay(db) -> None:
    pubsub = db.pubsub()
    await pubsub.subscribe("bot:events")
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    await _broadcast(json.loads(message["data"]))
                except Exception:
                    pass
    except asyncio.CancelledError:
        pass
    finally:
        await pubsub.unsubscribe("bot:events")
        await pubsub.aclose()


# ── Inline fallback dashboard HTML ───────────────────────────────────────────

_FALLBACK_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Apex Trader</title>
  <style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{background:#060810;color:#e8edf8;font-family:'Courier New',monospace;padding:24px}
    h1{color:#3d7aff;font-size:18px;margin-bottom:4px}
    p{color:#8892aa;font-size:12px;margin-bottom:20px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:20px}
    .card{background:#0f1422;border:1px solid #1e2640;border-radius:8px;padding:14px}
    .card h2{font-size:9px;color:#4a5470;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px}
    .card .val{font-size:18px;font-weight:700}
    .badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:9px;font-weight:700;margin-left:6px}
    .live{background:rgba(0,229,138,.1);color:#00e58a;border:1px solid rgba(0,229,138,.3)}
    .off{background:rgba(255,61,110,.1);color:#ff3d6e;border:1px solid rgba(255,61,110,.3)}
    .tabs{display:flex;gap:0;border-bottom:1px solid #1e2640;margin-bottom:0}
    .tab{padding:8px 16px;font-family:'Courier New',monospace;font-size:10px;font-weight:700;
         letter-spacing:.08em;text-transform:uppercase;color:#4a5470;background:none;
         border:none;cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px}
    .tab.active{color:#3d7aff;border-bottom-color:#3d7aff}
    .pane{display:none;height:420px;overflow-y:auto;background:#0b0f1a;
          border:1px solid #1e2640;border-top:none;border-radius:0 0 8px 8px}
    .pane.active{display:block}
    .log-line{display:flex;gap:10px;padding:4px 12px;border-bottom:1px solid rgba(30,38,64,.4);
              font-size:10px;line-height:1.5;align-items:baseline}
    .log-line:hover{background:#141928}
    .log-ts{color:#4a5470;white-space:nowrap;flex-shrink:0}
    .log-lvl{font-weight:700;width:54px;flex-shrink:0;text-align:center}
    .log-name{color:#8892aa;width:120px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .log-msg{color:#c8d0e0;word-break:break-all}
    .INFO{color:#00e58a}.WARNING{color:#ffa826}.ERROR{color:#ff3d6e}
    .CRITICAL{color:#ff3d6e;font-weight:700}.DEBUG{color:#4a5470}
    .filters{display:flex;gap:8px;align-items:center;padding:8px 12px;
             background:#0b0f1a;border:1px solid #1e2640;border-bottom:none;border-radius:8px 8px 0 0}
    .filter-btn{padding:3px 10px;border-radius:12px;border:1px solid #1e2640;
                background:none;color:#8892aa;font-family:'Courier New',monospace;
                font-size:9px;font-weight:700;cursor:pointer;letter-spacing:.06em}
    .filter-btn.active,.filter-btn:hover{border-color:#3d7aff;color:#3d7aff;background:rgba(61,122,255,.08)}
    .filter-btn.ERR.active,.filter-btn.ERR:hover{border-color:#ff3d6e;color:#ff3d6e;background:rgba(255,61,110,.08)}
    .filter-btn.WARN.active,.filter-btn.WARN:hover{border-color:#ffa826;color:#ffa826;background:rgba(255,168,38,.08)}
    .sep{color:#1e2640;margin:0 4px}
    .clear-btn{margin-left:auto;padding:3px 10px;border-radius:12px;
               border:1px solid rgba(255,61,110,.3);background:none;
               color:#ff3d6e;font-family:'Courier New',monospace;font-size:9px;cursor:pointer}
    .clear-btn:hover{background:rgba(255,61,110,.1)}
    #log-count{font-size:9px;color:#4a5470;margin-left:4px}
    .status-json{padding:12px;font-size:10px;color:#00d4ff;white-space:pre-wrap;word-break:break-all}
  </style>
</head>
<body>
  <h1>⟠ Apex Trader <span class="badge off" id="ws-badge">CONNECTING</span></h1>
  <p>Live dashboard · logs stream via WebSocket, also available at <code>/api/logs</code></p>

  <div class="grid">
    <div class="card"><h2>Equity</h2><div class="val" id="equity">—</div></div>
    <div class="card"><h2>Buying Power</h2><div class="val" id="bp">—</div></div>
    <div class="card"><h2>Positions</h2><div class="val" id="positions">—</div></div>
    <div class="card"><h2>Daily P&amp;L</h2><div class="val" id="pnl">—</div></div>
    <div class="card"><h2>Regime</h2><div class="val" id="regime">—</div></div>
    <div class="card"><h2>Circuit Breaker</h2><div class="val" id="cb">—</div></div>
  </div>

  <div class="tabs">
    <button class="tab active" onclick="switchTab('logs')">Live Logs</button>
    <button class="tab" onclick="switchTab('status')">Status JSON</button>
  </div>

  <div class="filters">
    <button class="filter-btn active" onclick="setLevel('DEBUG',this)">ALL</button>
    <span class="sep">|</span>
    <button class="filter-btn" onclick="setLevel('INFO',this)">INFO</button>
    <button class="filter-btn WARN" onclick="setLevel('WARNING',this)">WARN</button>
    <button class="filter-btn ERR" onclick="setLevel('ERROR',this)">ERROR</button>
    <span id="log-count"></span>
    <button class="clear-btn" onclick="clearLogs()">CLEAR</button>
  </div>

  <div class="pane active" id="pane-logs" style="border-radius:0 0 8px 8px">
    <div id="log-container"></div>
  </div>
  <div class="pane" id="pane-status">
    <div class="status-json" id="status-json">Waiting...</div>
  </div>

<script>
const $  = id => document.getElementById(id);
const fmt = v => v==null?'—':'$'+parseFloat(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});

// ── Logs ─────────────────────────────────────────────────────────────────────
let allLogs = [];
let minLevel = 'DEBUG';
const LEVELS = {DEBUG:10,INFO:20,WARNING:30,ERROR:40,CRITICAL:50};

function setLevel(lvl, btn) {
  minLevel = lvl;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderLogs();
}

function renderLogs() {
  const container = $('log-container');
  const minNo = LEVELS[minLevel] || 10;
  const filtered = allLogs.filter(e => (e.level_no||0) >= minNo);
  $('log-count').textContent = filtered.length + ' entries';
  container.innerHTML = filtered.slice(-500).map(e => {
    const ts = e.timestamp ? e.timestamp.slice(11,19) : '';
    const msg = (e.message||'').replace(/</g,'&lt;');
    return `<div class="log-line">
      <span class="log-ts">${ts}</span>
      <span class="log-lvl ${e.level}">${e.level}</span>
      <span class="log-name" title="${e.logger||''}">${e.logger||''}</span>
      <span class="log-msg">${msg}</span>
    </div>`;
  }).join('');
  container.scrollTop = container.scrollHeight;
}

function addLog(entry) {
  allLogs.push(entry);
  if (allLogs.length > 2000) allLogs.shift();
  renderLogs();
}

async function fetchLogs() {
  try {
    const r = await fetch('/api/logs?limit=500');
    if (!r.ok) return;
    allLogs = await r.json();
    renderLogs();
  } catch(_) {}
}

async function clearLogs() {
  allLogs = [];
  renderLogs();
  try { await fetch('/api/logs?clear=1'); } catch(_) {}
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', ['logs','status'][i]===name));
  $('pane-logs').classList.toggle('active', name==='logs');
  $('pane-status').classList.toggle('active', name==='status');
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
const proto = location.protocol==='https:'?'wss:':'ws:';
const wsUrl = `${proto}//${location.host}/ws`;
let ws;

function connect() {
  ws = new WebSocket(wsUrl);
  ws.onopen = () => {
    $('ws-badge').textContent = 'LIVE';
    $('ws-badge').className = 'badge live';
    fetchLogs();  // load history on connect
  };
  ws.onmessage = e => {
    const pkt = JSON.parse(e.data);

    // Live log entry
    if (pkt.event === 'log') { addLog(pkt.data); return; }

    // Status / full update
    const s = pkt.data || pkt;
    const a = s.account || {};
    if (a.equity !== undefined)       $('equity').textContent = fmt(a.equity);
    if (a.buying_power !== undefined) $('bp').textContent     = fmt(a.buying_power);
    if (s.positions !== undefined)    $('positions').textContent = s.positions.length;
    if (a.daily_pnl !== undefined) {
      const p = parseFloat(a.daily_pnl);
      $('pnl').textContent  = (p>=0?'+':'')+fmt(p);
      $('pnl').style.color  = p>=0?'#00e58a':'#ff3d6e';
    }
    const reg = s.market_regime || {};
    if (reg.regime) $('regime').textContent = reg.regime;
    $('cb').textContent = a.circuit_breaker ? '⚠ TRIPPED' : 'SAFE';
    $('cb').style.color = a.circuit_breaker ? '#ff3d6e' : '#00e58a';
    $('status-json').textContent = JSON.stringify(s, null, 2).slice(0, 8000);
  };
  ws.onclose = () => {
    $('ws-badge').textContent = 'OFFLINE';
    $('ws-badge').className = 'badge off';
    setTimeout(connect, 4000);
  };
  ws.onerror = () => ws.close();
}
connect();
</script>
</body>
</html>"""


# ── Route handlers ────────────────────────────────────────────────────────────

async def handle_health(request: web.Request) -> web.Response:
    return web.Response(text="OK", content_type="text/plain")


async def handle_index(request: web.Request) -> web.Response:
    tmpl = Path(__file__).parent / "templates" / "index.html"
    try:
        html = tmpl.read_text(encoding="utf-8")
    except Exception:
        html = _FALLBACK_HTML
    return web.Response(text=html, content_type="text/html")


async def handle_logs(request: web.Request) -> web.Response:
    """
    GET /api/logs
      ?limit=N      how many entries to return (default 200, max 2000)
      ?level=WARN   minimum log level: DEBUG | INFO | WARNING | ERROR | CRITICAL
      ?clear=1      wipe the in-memory buffer first
    """
    if request.rel_url.query.get("clear") == "1":
        _logger_module.clear_logs()
        return web.json_response([])

    limit     = min(int(request.rel_url.query.get("limit",  200)), 2000)
    min_level =     request.rel_url.query.get("level", "DEBUG").upper()
    entries   = _logger_module.get_logs(limit=limit, min_level=min_level)
    return web.json_response(entries)


async def handle_status(request: web.Request) -> web.Response:
    if not _strategy:
        return web.json_response({"error": "Strategy not initialised"}, status=503)
    try:
        data = await asyncio.wait_for(_strategy.get_status(), timeout=20.0)
    except asyncio.TimeoutError:
        return web.json_response({"error": "status_timeout"}, status=504)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
    return web.json_response(data, dumps=lambda d: json.dumps(d, default=str))


async def handle_trades(request: web.Request) -> web.Response:
    if not _db:
        return web.json_response([])
    trades = await _db.get_trades(limit=200)
    return web.json_response(trades, dumps=lambda d: json.dumps(d, default=str))


async def handle_news(request: web.Request) -> web.Response:
    if not _db:
        return web.json_response([])
    limit = int(request.rel_url.query.get("limit", 100))
    items = await _db.get_news(limit=limit)
    return web.json_response(items, dumps=lambda d: json.dumps(d, default=str))


async def handle_performance(request: web.Request) -> web.Response:
    if not _db:
        return web.json_response([])
    limit = int(request.rel_url.query.get("limit", 288))
    snaps = await _db.get_performance(limit=limit)
    return web.json_response(snaps, dumps=lambda d: json.dumps(d, default=str))


async def handle_signals(request: web.Request) -> web.Response:
    if not _db:
        return web.json_response([])
    limit   = int(request.rel_url.query.get("limit", 100))
    signals = await _db.get_signals(limit=limit)
    return web.json_response(signals, dumps=lambda d: json.dumps(d, default=str))


async def handle_backtest_all(request: web.Request) -> web.Response:
    if not _strategy:
        return web.json_response({"error": "No strategy"}, status=503)
    body    = await request.json()
    days    = int(body.get("days", 252))
    results = await _strategy.run_backtests(days=days)
    return web.json_response(results, dumps=lambda d: json.dumps(d, default=str))


async def handle_backtest_symbol(request: web.Request) -> web.Response:
    if not _strategy:
        return web.json_response({"error": "No strategy"}, status=503)
    symbol = request.match_info["symbol"].upper()
    days   = int(request.rel_url.query.get("days", 252))
    from backtester import run_backtest
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


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def handle_websocket(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    _ws_clients.add(ws)
    log.debug("WS client connected (%d total)", len(_ws_clients))

    # Send initial status + last 100 log lines on connect
    if _strategy:
        try:
            status = await _strategy.get_status()
            await ws.send_json({"event": "status", "data": status})
        except Exception:
            pass

    try:
        recent_logs = _logger_module.get_logs(limit=100)
        await ws.send_str(json.dumps({"event": "log_history", "data": recent_logs}))
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
                    elif data.get("type") == "request_logs":
                        logs = _logger_module.get_logs(
                            limit=int(data.get("limit", 200)),
                            min_level=data.get("level", "DEBUG"),
                        )
                        await ws.send_str(json.dumps({"event": "log_history", "data": logs}))
                except Exception:
                    pass
            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                break
    finally:
        _ws_clients.discard(ws)
        log.debug("WS client disconnected (%d remaining)", len(_ws_clients))
    return ws


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> web.Application:
    app = web.Application()

    app.router.add_get("/health",                        handle_health)
    app.router.add_get("/",                              handle_index)
    app.router.add_get("/ws",                            handle_websocket)
    app.router.add_get("/api/logs",                      handle_logs)        # ← NEW
    app.router.add_get("/api/status",                    handle_status)
    app.router.add_get("/api/trades",                    handle_trades)
    app.router.add_get("/api/news",                      handle_news)
    app.router.add_get("/api/performance",               handle_performance)
    app.router.add_get("/api/signals",                   handle_signals)
    app.router.add_post("/api/backtest",                 handle_backtest_all)
    app.router.add_get("/api/backtest/{symbol}",         handle_backtest_symbol)
    app.router.add_post("/api/force_scan",               handle_force_scan)
    app.router.add_post("/api/close_position/{symbol}",  handle_close_position)

    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["GET", "POST", "OPTIONS"],
        )
    })
    for route in list(app.router.routes()):
        if route.resource.canonical in ("/ws",):
            continue
        try:
            cors.add(route)
        except Exception:
            pass

    return app


async def run_dashboard(host: str = "0.0.0.0", port: int = 5000, db=None) -> None:
    app = create_app()
    if db:
        asyncio.create_task(_redis_relay(db))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    log.info("Dashboard  → http://%s:%d", host, port)
    log.info("Log viewer → http://%s:%d/api/logs", host, port)