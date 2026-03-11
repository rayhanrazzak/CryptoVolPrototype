"""
Mock paper-trade signal generator.
Produces BUY YES / BUY NO / NO TRADE signals based on edge analysis.

This is a prototype signal system for demonstration purposes only.
Not intended for live trading.
"""

import config


def compute_edge(market_prob: float, model_prob: float) -> float:
    """Raw edge = model probability - market probability."""
    if market_prob is None or model_prob is None:
        return 0.0
    return model_prob - market_prob


def assess_confidence(
    spread: float = None,
    volume: int = None,
    iv_available: bool = True,
    rv_available: bool = True,
    hours_to_expiry: float = None,
    iv_rv_divergence: float = None,
) -> dict:
    """
    Assess confidence in the signal. Returns a confidence score (0-1)
    and a list of concerns that reduce confidence.
    """
    confidence = 1.0
    concerns = []

    # spread penalty
    if spread is not None and spread > 0.10:
        penalty = min(spread * config.SPREAD_PENALTY_FACTOR, 0.3)
        confidence -= penalty
        concerns.append(f"wide spread ({spread:.0%})")

    # low volume
    if volume is not None and volume < 10:
        confidence -= 0.15
        concerns.append("low volume")
    elif volume is not None and volume < 50:
        confidence -= 0.05
        concerns.append("moderate volume")

    # missing vol data
    if not iv_available:
        confidence -= 0.10
        concerns.append("no implied vol data")
    if not rv_available:
        confidence -= 0.10
        concerns.append("no realized vol data")

    # very short expiry = noisier estimate
    if hours_to_expiry is not None and hours_to_expiry < 1:
        confidence -= 0.15
        concerns.append("very short expiry (<1hr)")
    elif hours_to_expiry is not None and hours_to_expiry < 4:
        confidence -= 0.05
        concerns.append("short expiry")

    # IV/RV disagreement suggests uncertainty
    if iv_rv_divergence is not None and abs(iv_rv_divergence) > 15:
        confidence -= 0.10
        concerns.append(f"IV-RV divergence ({iv_rv_divergence:+.1f}%)")

    confidence = max(confidence, 0.1)

    return {
        "confidence": round(confidence, 2),
        "concerns": concerns,
        "confidence_label": _confidence_label(confidence),
    }


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.8:
        return "HIGH"
    elif confidence >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"


def generate_signal(
    market_prob: float,
    model_prob: float,
    confidence: float = 1.0,
    spread: float = None,
) -> dict:
    """
    Generate a mock paper-trade signal.

    Logic:
    - Compute raw edge (model - market)
    - Apply confidence penalty
    - Apply spread cost if available
    - Compare adjusted edge to minimum threshold
    - Produce BUY YES / BUY NO / NO TRADE
    """
    if market_prob is None or model_prob is None:
        return {
            "signal": "NO TRADE",
            "raw_edge": None,
            "adjusted_edge": None,
            "reason": "insufficient pricing data",
        }

    raw_edge = model_prob - market_prob

    # adjust edge for confidence and spread
    adjusted_edge = raw_edge * confidence
    if spread is not None:
        adjusted_edge -= spread * config.SPREAD_PENALTY_FACTOR

    abs_edge = abs(adjusted_edge)
    threshold = config.MIN_EDGE_THRESHOLD

    if abs_edge < threshold:
        signal = "NO TRADE"
        reason = f"edge too small ({adjusted_edge:+.1%} vs {threshold:.1%} threshold)"
    elif adjusted_edge > 0:
        signal = "BUY YES"
        reason = f"model sees {adjusted_edge:+.1%} edge — market underpricing YES"
    else:
        signal = "BUY NO"
        reason = f"model sees {abs(adjusted_edge):.1%} edge — market overpricing YES"

    return {
        "signal": signal,
        "raw_edge": round(raw_edge, 4),
        "adjusted_edge": round(adjusted_edge, 4),
        "reason": reason,
        "edge_threshold": threshold,
    }
