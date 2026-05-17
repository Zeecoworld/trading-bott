"""
bot/risk_manager.py — Position sizing, daily loss limits, and risk controls.
Implements Kelly Criterion + fixed-fractional position sizing with hard guards.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import date
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SizeResult:
    allowed:       bool
    qty:           float
    dollar_value:  float
    reason:        str


class RiskManager:
    def __init__(
        self,
        max_position_pct: float = 0.05,
        max_open_positions: int  = 10,
        stop_loss_pct: float     = 0.03,
        daily_max_loss_pct: float= 0.05,
    ):
        self.max_position_pct    = max_position_pct
        self.max_open_positions  = max_open_positions
        self.stop_loss_pct       = stop_loss_pct
        self.daily_max_loss_pct  = daily_max_loss_pct
        self._daily_pnl: float   = 0.0
        self._pnl_date: date     = date.today()

    # ── Position Sizing ───────────────────────────────────────────────────────

    def size_position(
        self,
        equity:           float,
        price:            float,
        confidence:       float,
        open_positions:   int,
        symbol_positions: int,    # existing shares held
        win_rate:         float = 0.55,
    ) -> SizeResult:
        """
        Kelly-adjusted, capped position sizer.
        Returns qty to buy (whole shares only).
        """
        self._refresh_daily_pnl()

        # Hard gate: too many open positions
        if open_positions >= self.max_open_positions:
            return SizeResult(False, 0, 0, f"Max positions reached ({self.max_open_positions})")

        # Hard gate: daily loss exceeded
        if self._daily_pnl < -(equity * self.daily_max_loss_pct):
            return SizeResult(False, 0, 0,
                f"Daily loss limit hit (${self._daily_pnl:.0f}). Bot paused for today.")

        if price <= 0 or equity <= 0:
            return SizeResult(False, 0, 0, "Invalid price or equity")

        # Already holding?
        if symbol_positions > 0:
            return SizeResult(False, 0, 0, "Position already open")

        # Kelly fraction: f* = (p(b+1) - 1) / b  where b = R:R ratio
        rr_ratio     = self.stop_loss_pct * 2.5 / self.stop_loss_pct   # = 2.5
        kelly_frac   = (win_rate * (rr_ratio + 1) - 1) / rr_ratio
        kelly_frac   = max(0, min(kelly_frac, 0.25))   # cap Kelly at 25%

        # Scale by confidence and apply max cap
        frac = min(kelly_frac * confidence, self.max_position_pct)
        dollar_value = equity * frac
        qty = int(dollar_value / price)

        if qty < 1:
            return SizeResult(False, 0, 0, f"Position too small (${dollar_value:.0f})")

        return SizeResult(True, float(qty), float(qty * price),
                          f"Kelly={kelly_frac:.3f}, conf={confidence:.2f}, frac={frac:.3f}")

    # ── Stop-Loss / Take-Profit ───────────────────────────────────────────────

    def calc_stops(self, price: float, side: str = "buy",
                   stop_pct: Optional[float] = None,
                   tp_mult: float = 2.5) -> tuple[float, float]:
        sp   = stop_pct or self.stop_loss_pct
        if side == "buy":
            sl = round(price * (1 - sp), 2)
            tp = round(price * (1 + sp * tp_mult), 2)
        else:
            sl = round(price * (1 + sp), 2)
            tp = round(price * (1 - sp * tp_mult), 2)
        return sl, tp

    # ── Daily P&L Tracking ────────────────────────────────────────────────────

    def record_pnl(self, pnl: float):
        self._refresh_daily_pnl()
        self._daily_pnl += pnl

    def _refresh_daily_pnl(self):
        if date.today() != self._pnl_date:
            self._daily_pnl = 0.0
            self._pnl_date  = date.today()

    @property
    def daily_pnl(self) -> float:
        self._refresh_daily_pnl()
        return self._daily_pnl

    def is_trading_allowed(self, equity: float) -> tuple[bool, str]:
        self._refresh_daily_pnl()
        if self._daily_pnl < -(equity * self.daily_max_loss_pct):
            return False, f"Daily loss limit: ${self._daily_pnl:.0f}"
        return True, "OK"