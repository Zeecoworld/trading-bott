"""
bot/technical.py — Technical analysis indicators & scoring.
Uses the `ta` library + manual implementations for reliability.
Returns a unified TechnicalScore object with individual signal breakdowns.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TechnicalScore:
    symbol:        str
    price:         float
    score:         float = 0.0          # –1.0 (strong sell) → +1.0 (strong buy)
    signals:       Dict[str, Any] = field(default_factory=dict)
    trend:         str = "NEUTRAL"      # BULLISH / BEARISH / NEUTRAL
    strength:      str = "WEAK"         # STRONG / MODERATE / WEAK

    @property
    def action(self) -> str:
        if self.score >= 0.4:
            return "BUY"
        elif self.score <= -0.4:
            return "SELL"
        return "HOLD"


def analyse(df: pd.DataFrame, symbol: str) -> TechnicalScore:
    """
    Run full technical analysis on OHLCV DataFrame.
    Returns TechnicalScore with composite score in [-1, +1].
    """
    if df is None or len(df) < 50:
        logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} bars")
        return TechnicalScore(symbol=symbol, price=0.0)

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Ensure required columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            return TechnicalScore(symbol=symbol, price=float(df["close"].iloc[-1]) if "close" in df.columns else 0.0)

    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)
    price  = float(close.iloc[-1])

    signals: Dict[str, Any] = {}
    votes:   list[float]    = []

    # ── 1. Trend (SMA crossover 20/50/200) ───────────────────────────────────
    sma20  = _sma(close, 20)
    sma50  = _sma(close, 50)
    sma200 = _sma(close, 200) if len(df) >= 200 else sma50

    signals["sma20"]  = round(sma20, 2)
    signals["sma50"]  = round(sma50, 2)
    signals["sma200"] = round(sma200, 2)
    signals["price_vs_sma200"] = round((price / sma200 - 1) * 100, 2)

    if price > sma200:                                    # above 200-SMA = trend up
        votes.append(0.3)
    else:
        votes.append(-0.3)

    if sma20 > sma50:                                     # golden cross
        votes.append(0.25)
    else:
        votes.append(-0.25)

    # ── 2. RSI ────────────────────────────────────────────────────────────────
    rsi = _rsi(close, 14)
    signals["rsi"] = round(rsi, 2)

    if rsi < 30:
        votes.append(0.5)     # oversold → strong buy signal
    elif rsi < 40:
        votes.append(0.25)
    elif rsi > 70:
        votes.append(-0.5)    # overbought → strong sell signal
    elif rsi > 60:
        votes.append(-0.25)
    else:
        votes.append(0.0)

    # ── 3. MACD ───────────────────────────────────────────────────────────────
    macd_line, signal_line, histogram = _macd(close)
    signals["macd"]        = round(macd_line, 4)
    signals["macd_signal"] = round(signal_line, 4)
    signals["macd_hist"]   = round(histogram, 4)

    if macd_line > signal_line and histogram > 0:
        votes.append(0.3)
    elif macd_line < signal_line and histogram < 0:
        votes.append(-0.3)
    else:
        votes.append(0.0)

    # MACD histogram momentum
    hist_series = _macd_histogram(close)
    if len(hist_series) >= 3:
        momentum = hist_series.iloc[-1] - hist_series.iloc[-3]
        votes.append(0.15 if momentum > 0 else -0.15)

    # ── 4. Bollinger Bands ────────────────────────────────────────────────────
    bb_upper, bb_mid, bb_lower = _bollinger(close, 20, 2.0)
    bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid else 0
    signals["bb_upper"]  = round(bb_upper, 2)
    signals["bb_lower"]  = round(bb_lower, 2)
    signals["bb_width"]  = round(bb_width * 100, 2)

    if price < bb_lower:
        votes.append(0.35)    # below lower band → potential reversal up
    elif price > bb_upper:
        votes.append(-0.35)
    else:
        # position within bands (0 = at lower, 1 = at upper)
        pct_b = (price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) else 0.5
        votes.append((0.5 - pct_b) * 0.2)

    # ── 5. Volume Analysis ────────────────────────────────────────────────────
    vol_avg20 = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio = float(volume.iloc[-1]) / vol_avg20 if vol_avg20 else 1.0
    signals["volume_ratio"] = round(vol_ratio, 2)

    price_change = (price - float(close.iloc[-2])) / float(close.iloc[-2])
    if vol_ratio > 1.5 and price_change > 0:
        votes.append(0.2)     # high volume breakout up
    elif vol_ratio > 1.5 and price_change < 0:
        votes.append(-0.2)    # high volume breakdown
    else:
        votes.append(0.0)

    # ── 6. ATR (volatility) ───────────────────────────────────────────────────
    atr = _atr(high, low, close, 14)
    atr_pct = atr / price * 100
    signals["atr_pct"] = round(atr_pct, 2)

    # ── 7. Stochastic Oscillator ──────────────────────────────────────────────
    stoch_k, stoch_d = _stochastic(high, low, close, 14, 3)
    signals["stoch_k"] = round(stoch_k, 2)
    signals["stoch_d"] = round(stoch_d, 2)

    if stoch_k < 20 and stoch_d < 20:
        votes.append(0.25)
    elif stoch_k > 80 and stoch_d > 80:
        votes.append(-0.25)
    else:
        votes.append(0.0)

    # ── 8. EMA momentum (13/21) ───────────────────────────────────────────────
    ema13 = _ema(close, 13)
    ema21 = _ema(close, 21)
    signals["ema13"] = round(ema13, 2)
    signals["ema21"] = round(ema21, 2)
    votes.append(0.15 if ema13 > ema21 else -0.15)

    # ── 9. Rate of Change (momentum) ─────────────────────────────────────────
    roc = (price - float(close.iloc[-10])) / float(close.iloc[-10]) * 100 if len(close) >= 10 else 0
    signals["roc_10"] = round(roc, 2)
    if roc > 5:
        votes.append(0.2)
    elif roc < -5:
        votes.append(-0.2)
    else:
        votes.append(roc / 25 * 0.2)

    # ── Composite Score ───────────────────────────────────────────────────────
    raw_score = float(np.mean(votes)) if votes else 0.0
    score     = float(np.clip(raw_score, -1.0, 1.0))

    # Trend classification
    if score >= 0.3:
        trend = "BULLISH"
    elif score <= -0.3:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    strength = "STRONG" if abs(score) >= 0.6 else ("MODERATE" if abs(score) >= 0.35 else "WEAK")

    return TechnicalScore(
        symbol=symbol, price=price, score=score,
        signals=signals, trend=trend, strength=strength,
    )


# ── Indicator implementations ─────────────────────────────────────────────────

def _sma(series: pd.Series, period: int) -> float:
    return float(series.rolling(period).mean().iloc[-1])

def _ema(series: pd.Series, period: int) -> float:
    return float(series.ewm(span=period, adjust=False).mean().iloc[-1])

def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])

def _macd_histogram(series: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    ema_fast  = series.ewm(span=fast, adjust=False).mean()
    ema_slow  = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - sig_line

def _bollinger(series: pd.Series, period=20, std=2.0):
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    return float((mid + std * sigma).iloc[-1]), float(mid.iloc[-1]), float((mid - std * sigma).iloc[-1])

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> float:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period=14, d_period=3):
    lo  = low.rolling(k_period).min()
    hi  = high.rolling(k_period).max()
    k   = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
    d   = k.rolling(d_period).mean()
    return float(k.iloc[-1]), float(d.iloc[-1])