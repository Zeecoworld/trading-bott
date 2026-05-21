"""
bot/ai_decision.py — Async Replicate LLM trade decision engine.

Model ID reference (verified working on Replicate as of 2025/2026):
  meta/meta-llama-3-70b-instruct       Llama 3 70B  — default, stable
  meta/meta-llama-3-8b-instruct        Llama 3 8B   — faster / cheaper
  mistralai/mixtral-8x7b-instruct-v0.1 Mixtral 8x7B — reliable fallback
  mistralai/mistral-7b-instruct-v0.2   Mistral 7B   — lightest

NOTE on Llama 3.3:
  Replicate deprecated the short-form path "meta/llama-3.3-70b-instruct".
  The model now requires a full version hash, e.g.:
    meta/llama-3.3-70b-instruct:<sha256_version>
  Until Replicate publishes a stable alias, use meta/meta-llama-3-70b-instruct.
  You can find the current version hash at https://replicate.com/meta/llama-3.3-70b-instruct
  and set REPLICATE_MODEL=meta/llama-3.3-70b-instruct:<hash> in your env.
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import replicate

logger = logging.getLogger(__name__)

# ── Verified working model IDs on Replicate ───────────────────────────────────
MODELS: Dict[str, str] = {
    # Short aliases → full Replicate model path
    "llama3-70b":    "meta/meta-llama-3-70b-instruct",    # Llama 3 70B  ✓ stable
    "llama3-8b":     "meta/meta-llama-3-8b-instruct",     # Llama 3 8B   ✓ fast
    "mixtral-8x7b":  "mistralai/mixtral-8x7b-instruct-v0.1",
    "mistral-7b":    "mistralai/mistral-7b-instruct-v0.2",
    # Llama 3.1 series (still active but slower cold starts)
    "llama3.1-70b":  "meta/meta-llama-3.1-70b-instruct",
    "llama3.1-8b":   "meta/meta-llama-3.1-8b-instruct",
    # Keep old keys for backwards compat — mapped to working equivalents
    "llama3.3-70b":  "meta/meta-llama-3-70b-instruct",    # 3.3 path is 404; use 3
    "llama3.1-405b": "meta/meta-llama-3-70b-instruct",    # 405B too slow for trading
    "deepseek-r1":   "meta/meta-llama-3-70b-instruct",    # deepseek-r1 path inactive
}

# Default: proven-stable 70B model
DEFAULT_MODEL = "meta/meta-llama-3-70b-instruct"

# Models that use "prompt" + "system_prompt" as separate fields (Replicate-style)
# vs models that expect a single combined "prompt" field.
_SYSTEM_PROMPT_MODELS = {
    "meta/meta-llama-3-70b-instruct",
    "meta/meta-llama-3-8b-instruct",
    "meta/meta-llama-3.1-70b-instruct",
    "meta/meta-llama-3.1-8b-instruct",
    "mistralai/mixtral-8x7b-instruct-v0.1",
    "mistralai/mistral-7b-instruct-v0.2",
}


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

        # Resolve model: check short aliases first, then use as-is (allows full
        # paths like "meta/llama-3.3-70b-instruct:<version_hash>" from env var)
        raw = model or os.getenv("REPLICATE_MODEL", DEFAULT_MODEL)
        self.model = MODELS.get(raw, raw)

        logger.info("AI engine: Replicate / %s", self.model)

    # ── Internal sync call ────────────────────────────────────────────────────

    def _build_input(self, system: str, user: str, max_tokens: int) -> dict:
        """
        Build the correct input dict for this model.
        Most Replicate-hosted Meta/Mistral models accept system_prompt + prompt.
        Models specified with a version hash may use a different schema.
        """
        base_model = self.model.split(":")[0]   # strip version hash if present
        if base_model in _SYSTEM_PROMPT_MODELS:
            return {
                "system_prompt":      system,
                "prompt":             user,
                "max_new_tokens":     max_tokens,
                "temperature":        0.1,
                "top_p":              0.9,
                "repetition_penalty": 1.1,
            }
        # Fallback: combine into a single prompt (works for any model)
        return {
            "prompt":         f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]",
            "max_new_tokens": max_tokens,
            "temperature":    0.1,
            "top_p":          0.9,
        }

    def _run_sync(self, system: str, user: str, max_tokens: int = 512,
                  retries: int = 2) -> str:
        """
        Call Replicate with retry on transient errors (5xx, timeout).
        Raises immediately on 404 so the caller can log and return HOLD.
        """
        inp = self._build_input(system, user, max_tokens)
        last_exc = None
        for attempt in range(1, retries + 2):
            try:
                output = replicate.run(self.model, input=inp)
                return _collect(output)
            except Exception as e:
                err = str(e)
                # 404 = wrong model ID — retrying won't help
                if "404" in err or "not found" in err.lower():
                    raise
                last_exc = e
                if attempt <= retries:
                    wait = 2 ** attempt
                    logger.warning("Replicate attempt %d/%d failed (%s), retrying in %ds",
                                   attempt, retries + 1, err[:80], wait)
                    time.sleep(wait)
        raise last_exc

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
            err = str(e)
            if "404" in err:
                logger.error(
                    "Replicate 404 for model '%s'. The model ID is invalid or deprecated. "
                    "Set REPLICATE_MODEL to a valid model (e.g. meta/meta-llama-3-70b-instruct). "
                    "Full error: %s", self.model, err
                )
            else:
                logger.error("Replicate error %s: %s", symbol, err)
            return TradeDecision(symbol=symbol, action="HOLD",
                                 confidence=0.0, reasoning=f"API error: {err[:120]}")

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