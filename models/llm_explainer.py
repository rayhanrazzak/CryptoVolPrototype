"""
Optional trade synthesis using Google Gemini (free tier).
Generates concise, data-rich trade analysis when a GEMINI_API_KEY is configured.
Falls back gracefully — the app works fully without it.

Setup: Get a free API key from https://aistudio.google.com/apikey
       Set GEMINI_API_KEY in your .env file
"""

import os
import hashlib
import requests

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

# in-memory cache to avoid redundant calls within a session
_cache: dict[str, str] = {}


def is_available() -> bool:
    """Check if LLM synthesis is configured."""
    return bool(os.getenv("GEMINI_API_KEY"))


def synthesize_trade(
    analysis: dict,
    spot: float,
    iv_data: dict,
    rv_data: dict,
) -> str | None:
    """
    Generate a concise trade synthesis for a single market.
    Returns None if not configured or the call fails.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    context = _build_trade_context(analysis, spot, iv_data, rv_data)
    cache_key = hashlib.md5(context.encode()).hexdigest()
    if cache_key in _cache:
        return _cache[cache_key]

    prompt = (
        "You are a quantitative analyst at a top hedge fund writing a brief "
        "trade note. Be concise (3-5 sentences), cite specific numbers from "
        "the data below, and sound like a professional trading desk memo. "
        "No hedging language, no disclaimers, no bullet points. Just sharp analysis.\n\n"
        f"{context}\n\n"
        "Write a synthesis covering: (1) what the discrepancy is and why it exists, "
        "(2) the key vol/skew insight driving the pricing, "
        "(3) the main risk to the trade."
    )

    result = _call_gemini(api_key, prompt, max_tokens=200)
    if result:
        _cache[cache_key] = result
    return result


def synthesize_overview(
    analyses: list[dict],
    spot: float,
    iv_data: dict,
    rv_data: dict,
) -> str | None:
    """Generate a portfolio-level overview of all analyzed markets."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not analyses:
        return None

    lines = []
    for a in analyses[:15]:
        p = a["params"]
        sig = a["signal"]
        fv = a["fair_value"]
        label = p.get("subtitle", p.get("title", ""))[:40]
        edge = sig.get("raw_edge")
        signal = sig.get("signal", "NO TRADE")
        mkt = p.get("market_prob")
        mdl = fv.get("model_prob")
        if edge is not None and mkt is not None:
            lines.append(
                f"  {label}: mkt={mkt:.0%} model={mdl:.0%} "
                f"discrepancy={edge:+.1%} → {signal}"
            )

    dvol = iv_data.get("dvol")
    rv = rv_data.get("realized_vol")
    ratio_str = f"{dvol/rv:.2f}" if (dvol and rv and rv > 0) else "N/A"

    context = (
        f"BTC spot: ${spot:,.0f}\n"
        f"DVOL (30d IV): {dvol:.1f}%\n"
        f"Realized vol (24h): {rv:.1f}%\n"
        f"IV/RV ratio: {ratio_str}\n"
        f"Markets:\n" + "\n".join(lines)
    )

    cache_key = hashlib.md5(("overview:" + context).encode()).hexdigest()
    if cache_key in _cache:
        return _cache[cache_key]

    prompt = (
        "You are a quantitative analyst writing a morning briefing on BTC "
        "prediction market opportunities. Write 4-6 concise sentences. "
        "Reference specific numbers. Highlight the most interesting patterns. "
        "Sound like a professional desk note — no disclaimers, no bullet points.\n\n"
        f"{context}\n\n"
        "Cover: (1) overall market view, (2) where the best edges are, "
        "(3) vol regime context, (4) key risk."
    )

    result = _call_gemini(api_key, prompt, max_tokens=300)
    if result:
        _cache[cache_key] = result
    return result


def synthesize_chart_analysis(
    threshold_analyses: list[dict],
    spot: float,
    iv_data: dict,
    rv_data: dict,
    forward: float | None = None,
) -> str | None:
    """
    Generate analysis of the hero chart — options model vs prediction market
    probability curves. Receives the same data points plotted on the chart.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not threshold_analyses:
        return None

    # build a compact table of the chart data
    rows = []
    for a in threshold_analyses:
        p = a["params"]
        fv = a["fair_value"]
        sig = a["signal"]
        threshold = p.get("threshold", 0)
        dist_pct = (threshold - spot) / spot * 100 if spot else 0
        mkt = p.get("market_prob")
        mdl = fv.get("model_prob")
        edge = sig.get("raw_edge")
        strike_iv = fv.get("strike_matched_iv")
        rows.append(
            f"  ${threshold:,.0f} ({dist_pct:+.1f}%): "
            f"kalshi={mkt:.1%} model={mdl:.1%} discrepancy={edge:+.1%}"
            + (f" iv={strike_iv:.0f}%" if strike_iv else "")
        )

    dvol = iv_data.get("dvol")
    rv = rv_data.get("realized_vol")
    ratio_str = f"{dvol/rv:.2f}" if (dvol and rv and rv > 0) else "N/A"

    context = (
        f"BTC spot: ${spot:,.0f}\n"
        + (f"Kalshi forward (50% crossover): ${forward:,.0f}\n" if forward else "")
        + f"DVOL (30d IV): {dvol:.1f}%\n"
        + (f"Realized vol (24h): {rv:.1f}%\n" if rv else "")
        + f"IV/RV ratio: {ratio_str}\n"
        f"Threshold probabilities (strike, kalshi vs options model, discrepancy):\n"
        + "\n".join(rows)
    )

    cache_key = hashlib.md5(("chart:" + context).encode()).hexdigest()
    if cache_key in _cache:
        return _cache[cache_key]

    prompt = (
        "You are a quantitative analyst at a crypto trading desk. You're looking at "
        "a chart comparing Kalshi prediction market probabilities vs an options-implied "
        "model for BTC threshold contracts. Write 3-5 sharp sentences analyzing what "
        "the curves show. Reference specific strikes and numbers. Note where the "
        "biggest divergences are and what might explain them (drift, skew, vol regime). "
        "Sound like a professional desk note — no disclaimers, no bullet points, "
        "no hedging language.\n\n"
        f"{context}"
    )

    result = _call_gemini(api_key, prompt, max_tokens=500)
    if result:
        _cache[cache_key] = result
    return result


def _call_gemini(api_key: str, prompt: str, max_tokens: int = 200) -> str | None:
    """Make a Gemini API call. Returns text or None on failure."""
    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={api_key}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": max_tokens,
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            },
            timeout=15,
        )
        resp.raise_for_status()
        # extract the text part (skip any thought parts)
        parts = resp.json()["candidates"][0]["content"]["parts"]
        text_parts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
        return text_parts[-1].strip() if text_parts else None
    except Exception:
        return None


def _build_trade_context(analysis, spot, iv_data, rv_data):
    """Build structured data context for the trade synthesis prompt."""
    params = analysis.get("params", {})
    fv = analysis.get("fair_value", {})
    sig = analysis.get("signal", {})
    conf = analysis.get("confidence", {})

    threshold = params.get("threshold", 0)
    dist_pct = (threshold - spot) / spot * 100 if spot else 0
    strike_iv = fv.get("strike_matched_iv")
    dvol = iv_data.get("dvol")
    rv = rv_data.get("realized_vol")

    parts = [
        f"Contract: {params.get('subtitle', params.get('title', ''))}",
        f"BTC Spot: ${spot:,.0f}",
        f"Threshold: ${threshold:,.0f} ({dist_pct:+.1f}% from spot)",
        f"Hours to expiry: {analysis.get('hours_to_expiry', 0):.1f}",
    ]

    mkt = params.get("market_prob")
    mdl = fv.get("model_prob")
    if mkt is not None:
        parts.append(f"Market probability: {mkt:.1%}")
    if mdl is not None:
        parts.append(f"Model probability: {mdl:.1%}")

    edge = sig.get("raw_edge")
    if edge is not None:
        parts.append(f"Raw discrepancy: {edge:+.1%}")
    parts.append(f"Signal: {sig.get('signal', 'N/A')}")

    if strike_iv:
        parts.append(f"Strike-matched IV: {strike_iv:.1f}%")
    if dvol:
        parts.append(f"DVOL (30d): {dvol:.1f}%")
        if strike_iv:
            parts.append(f"Skew ratio (strike IV / DVOL): {strike_iv/dvol:.2f}x")
    if rv:
        parts.append(f"Realized vol (24h): {rv:.1f}%")

    regime = fv.get("vol_regime")
    if regime:
        parts.append(f"Vol regime: {regime}")

    tail_adj = fv.get("iv_tail_adjustment")
    if tail_adj and abs(tail_adj) > 0.005:
        parts.append(f"Fat-tail adjustment: {tail_adj:+.1%}")

    parts.append(f"Liquidity: {params.get('liquidity', 'unknown')}")

    spread = params.get("spread")
    if spread:
        parts.append(f"Spread: {spread:.1%}")

    parts.append(
        f"Confidence: {conf.get('confidence_label', 'N/A')} "
        f"({conf.get('confidence', 0):.0%})"
    )
    if conf.get("concerns"):
        parts.append(f"Concerns: {', '.join(conf['concerns'])}")

    return "\n".join(parts)
