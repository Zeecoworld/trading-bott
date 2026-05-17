"""
bot/backtester.py — Strategy backtesting on historical Alpaca data.
Uses a vectorised approach for speed; returns detailed metrics + equity curve.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from technical import analyse as tech_analyse

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    symbol:          str
    start_date:      str
    end_date:        str
    total_trades:    int
    wins:            int
    losses:          int
    win_rate:        float
    total_return_pct:float
    max_drawdown_pct:float
    sharpe_ratio:    float
    profit_factor:   float
    avg_win_pct:     float
    avg_loss_pct:    float
    trades:          List[dict] = field(default_factory=list)
    equity_curve:    List[float]= field(default_factory=list)

    @property
    def expectancy(self) -> float:
        return (self.win_rate * self.avg_win_pct) - ((1 - self.win_rate) * abs(self.avg_loss_pct))


def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    initial_capital: float   = 10_000.0,
    stop_loss_pct: float     = 0.03,
    take_profit_pct: float   = 0.075,
    position_size_pct: float = 0.10,
    tech_threshold: float    = 0.35,
) -> BacktestResult:
    """
    Simulate the technical strategy on historical OHLCV data.
    Generates signals every N bars using a rolling window, executes at open.
    """
    if df is None or len(df) < 60:
        logger.warning(f"Insufficient data for backtest: {symbol}")
        return _empty_result(symbol)

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df.index   = pd.to_datetime(df.index)
    df         = df.sort_index()

    capital    = initial_capital
    equity_curve: List[float] = [capital]
    trades:       List[dict]  = []

    in_trade   = False
    entry_price= 0.0
    stop_price = 0.0
    tp_price   = 0.0
    entry_idx  = 0
    shares     = 0.0

    WINDOW      = 60    # minimum bars for indicator calculation
    SIGNAL_STEP = 5     # re-evaluate every N bars

    for i in range(WINDOW, len(df)):
        bar  = df.iloc[i]
        open_= float(bar["open"])
        high_= float(bar["high"])
        low_ = float(bar["low"])
        close= float(bar["close"])

        # ── Manage open trade ─────────────────────────────────────────────
        if in_trade:
            hit_stop = low_  <= stop_price
            hit_tp   = high_ >= tp_price

            if hit_stop or hit_tp:
                exit_price = stop_price if hit_stop else tp_price
                pnl_pct    = (exit_price - entry_price) / entry_price
                pnl_dollar = shares * (exit_price - entry_price)
                capital   += shares * exit_price
                trades.append({
                    "symbol":      symbol,
                    "entry_price": round(entry_price, 2),
                    "exit_price":  round(exit_price, 2),
                    "pnl_pct":     round(pnl_pct * 100, 2),
                    "pnl_dollar":  round(pnl_dollar, 2),
                    "exit_reason": "STOP_LOSS" if hit_stop else "TAKE_PROFIT",
                    "entry_bar":   entry_idx,
                    "exit_bar":    i,
                    "bars_held":   i - entry_idx,
                    "date_entry":  df.index[entry_idx].isoformat(),
                    "date_exit":   df.index[i].isoformat(),
                })
                in_trade = False
                equity_curve.append(round(capital, 2))
            continue

        # ── Generate signal every SIGNAL_STEP bars ────────────────────────
        if (i - WINDOW) % SIGNAL_STEP != 0:
            continue

        window_df = df.iloc[max(0, i - 200): i + 1]
        score_obj = tech_analyse(window_df, symbol)
        score     = score_obj.score

        if score < tech_threshold:
            continue

        # Entry
        position_dollars = capital * position_size_pct
        shares           = int(position_dollars / open_)
        if shares < 1 or shares * open_ > capital:
            continue

        cost        = shares * open_
        capital    -= cost
        entry_price = open_
        stop_price  = round(entry_price * (1 - stop_loss_pct), 4)
        tp_price    = round(entry_price * (1 + take_profit_pct), 4)
        entry_idx   = i
        in_trade    = True

    # Close any open trade at end
    if in_trade:
        last_close = float(df.iloc[-1]["close"])
        pnl_dollar = shares * (last_close - entry_price)
        pnl_pct    = (last_close - entry_price) / entry_price
        capital   += shares * last_close
        trades.append({
            "symbol":      symbol,
            "entry_price": round(entry_price, 2),
            "exit_price":  round(last_close, 2),
            "pnl_pct":     round(pnl_pct * 100, 2),
            "pnl_dollar":  round(pnl_dollar, 2),
            "exit_reason": "END_OF_DATA",
            "entry_bar":   entry_idx,
            "exit_bar":    len(df) - 1,
            "bars_held":   len(df) - 1 - entry_idx,
            "date_entry":  df.index[entry_idx].isoformat(),
            "date_exit":   df.index[-1].isoformat(),
        })

    equity_curve.append(round(capital, 2))

    # ── Metrics ───────────────────────────────────────────────────────────────
    wins   = [t for t in trades if t["pnl_dollar"] > 0]
    losses = [t for t in trades if t["pnl_dollar"] <= 0]

    win_rate   = len(wins) / len(trades) if trades else 0.0
    total_ret  = (capital - initial_capital) / initial_capital * 100

    avg_win    = np.mean([t["pnl_pct"] for t in wins])   if wins   else 0.0
    avg_loss   = np.mean([t["pnl_pct"] for t in losses]) if losses else 0.0

    gross_profit = sum(t["pnl_dollar"] for t in wins)   if wins   else 0.0
    gross_loss   = abs(sum(t["pnl_dollar"] for t in losses)) if losses else 1e-9
    profit_factor= gross_profit / gross_loss

    # Max drawdown from equity curve
    eq = pd.Series(equity_curve)
    roll_max  = eq.cummax()
    drawdowns = (eq - roll_max) / roll_max * 100
    max_dd    = float(drawdowns.min())

    # Sharpe (daily returns of equity curve)
    eq_returns = eq.pct_change().dropna()
    sharpe     = 0.0
    if len(eq_returns) > 1 and eq_returns.std() > 0:
        sharpe = float((eq_returns.mean() / eq_returns.std()) * np.sqrt(252))

    return BacktestResult(
        symbol=symbol,
        start_date=df.index[0].isoformat(),
        end_date=df.index[-1].isoformat(),
        total_trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=round(win_rate, 4),
        total_return_pct=round(total_ret, 2),
        max_drawdown_pct=round(max_dd, 2),
        sharpe_ratio=round(sharpe, 3),
        profit_factor=round(profit_factor, 3),
        avg_win_pct=round(float(avg_win), 2),
        avg_loss_pct=round(float(avg_loss), 2),
        trades=trades,
        equity_curve=[round(e, 2) for e in equity_curve],
    )


def _empty_result(symbol: str) -> BacktestResult:
    return BacktestResult(
        symbol=symbol, start_date="", end_date="",
        total_trades=0, wins=0, losses=0,
        win_rate=0, total_return_pct=0,
        max_drawdown_pct=0, sharpe_ratio=0,
        profit_factor=0, avg_win_pct=0, avg_loss_pct=0,
    )