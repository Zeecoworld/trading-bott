"""
bot/ai_decision.py — Async Replicate LLM trade decision engine.
Replicate SDK is synchronous; calls are offloaded via asyncio.to_thread().
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import replicate

logger = logging.getLogger(__name__)

MODELS: Dict[str, str] = {
    "llama3.3-70b":  "meta/llama-3.3-70b-instruct",
    "llama3.1-405b": "meta/llama-3.1-405b-instruct",
    "llama3.1-70b":  "meta/llama-3.1-70b-instruct",
    "llama3.1-8b":   "meta/llama-3.1-8b-instruct",
    "mistral-7b":    "mistralai/mistral-7b-instruct-v0.2",
    "mixtral-8x7b":  "mistralai/mixtral-8x7b-instruct-v0.1",
    "deepseek-r1":   "deepseek-ai/deepseek-r1",
}
DEFAULT_MODEL = "meta/meta-llama-3-70b-instruct"


def _collect(stream: Any) -> str:
    if isinstance(stream, str):
        return stream
    if isinstance(stream, (list, Iterator)):
        return "".join(str(t) for t in stream)
    return str(stream)


def _clean_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw   = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
    # trim anything before first '{'
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start >= 0 and end > start:
        raw = raw[start:end]
    return raw


@dataclass
class TradeDecision:
    symbol:      str
    action:      str            # BUY / SELL / HOLD
    confidence:  float          # 0.0–1.0
    reasoning:   str
    stop_loss:   Optional[float] = None
    take_profit: Optional[float] = None
    risk_score:  float = 0.5


class AIDecisionEngine:
    def __init__(self, api_token: str, model: Optional[str] = None):
        os.environ["REPLICATE_API_TOKEN"] = api_token
        self.model = MODELS.get(
            model or os.getenv("REPLICATE_MODEL", DEFAULT_MODEL),
            model or DEFAULT_MODEL,
        )
        logger.info("AI engine: Replicate / %s", self.model)

    # ── Internal sync call (run via to_thread) ────────────────────────────────

    def _run_sync(self, system: str, user: str, max_tokens: int = 512) -> str:
        output = replicate.run(
            self.model,
            input={
                "system_prompt":      system,
                "prompt":             user,
                "max_new_tokens":     max_tokens,
                "temperature":        0.1,
                "top_p":              0.9,
                "repetition_penalty": 1.1,
            },
        )
        return _collect(output)

    async def _run(self, system: str, user: str, max_tokens: int = 512) -> str:
        return await asyncio.to_thread(self._run_sync, system, user, max_tokens)

    # ── Trade Decision ────────────────────────────────────────────────────────

    async def decide(
        self,
        symbol:           str,
        price:            float,
        tech_score:       float,
        tech_signals:     dict,
        news_items:       List[dict],
        portfolio_pct:    float,
        account_equity:   float,
        stop_loss_pct:    float = 0.03,
        take_profit_mult: float = 2.5,
    ) -> TradeDecision:

        news_text = "\n".join(
            f"- [{n.get('source','')}] {n.get('title','')} "
            f"({str(n.get('published_at',''))[:10]}): {n.get('summary','')[:180]}"
            for n in news_items[:10]
        ) or "No recent news found."

        system = (
            "You are a professional quantitative trader with 20 years of experience. "
            "Respond ONLY with valid JSON — no markdown, no prose outside the JSON object."
        )
        user = f"""Analyse data for {symbol} and output a single JSON trade decision.

CURRENT PRICE: ${price:.2f}

TECHNICAL ANALYSIS
Composite score: {tech_score:.3f}  (-1.0=strong sell | 0=neutral | +1.0=strong buy)
Indicators: {json.dumps(tech_signals, indent=2)}

RECENT NEWS (last 24h)
{news_text}

PORTFOLIO CONTEXT
- Allocation {symbol}: {portfolio_pct*100:.1f}%
- Equity: ${account_equity:,.2f}
- Stop-loss: {stop_loss_pct*100:.1f}%  Take-profit: {stop_loss_pct*take_profit_mult*100:.1f}%

OUTPUT — respond with exactly this JSON:
{{
  "action": "BUY"|"SELL"|"HOLD",
  "confidence": <0.0-1.0>,
  "reasoning": "<2-3 sentences>",
  "key_factors": ["f1","f2","f3"],
  "risk_score": <0.0-1.0>,
  "suggested_stop_loss": <number|null>,
  "suggested_take_profit": <number|null>,
  "news_sentiment": "POSITIVE"|"NEGATIVE"|"NEUTRAL"|"MIXED"
}}
RULES: BUY if confidence>=0.65 AND risk_score<=0.70; SELL if confidence>=0.60; else HOLD."""

        try:
            raw  = await self._run(system, user, max_tokens=512)
            data = json.loads(_clean_json(raw))
            return TradeDecision(
                symbol=symbol,
                action=str(data.get("action", "HOLD")).upper(),
                confidence=float(data.get("confidence", 0.0)),
                reasoning=data.get("reasoning", ""),
                stop_loss=data.get("suggested_stop_loss"),
                take_profit=data.get("suggested_take_profit"),
                risk_score=float(data.get("risk_score", 0.5)),
            )
        except json.JSONDecodeError as e:
            logger.error("JSON parse error %s: %s", symbol, e)
            return TradeDecision(symbol=symbol, action="HOLD",
                                 confidence=0.0, reasoning=f"Parse error: {e}")
        except Exception as e:
            logger.error("Replicate error %s: %s", symbol, e)
            return TradeDecision(symbol=symbol, action="HOLD",
                                 confidence=0.0, reasoning=f"API error: {e}")

    # ── Market Regime ─────────────────────────────────────────────────────────

    async def analyse_market_regime(self, macro_news: List[dict]) -> dict:
        news_text = "\n".join(
            f"- {n.get('title','')} ({n.get('source','')})"
            for n in macro_news[:15]
        ) or "No macro news available."

        system = ("You are a macro market analyst. "
                  "Respond ONLY with valid JSON — no prose, no markdown.")
        user = f"""Classify the current market regime.

HEADLINES:
{news_text}

Respond with this JSON only:
{{
  "regime": "BULL"|"BEAR"|"SIDEWAYS",
  "risk_appetite": "RISK_ON"|"RISK_OFF"|"NEUTRAL",
  "key_themes": ["t1","t2","t3"],
  "confidence": <0.0-1.0>,
  "summary": "<one sentence>"
}}"""
        try:
            raw  = await self._run(system, user, max_tokens=256)
            return json.loads(_clean_json(raw))
        except Exception as e:
            logger.error("Regime analysis error: %s", e)
            return {"regime": "SIDEWAYS", "risk_appetite": "NEUTRAL",
                    "key_themes": [], "confidence": 0.0,
                    "summary": "Analysis unavailable"}