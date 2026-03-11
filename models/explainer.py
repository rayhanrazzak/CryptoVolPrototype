"""
Deterministic explanation engine.
Converts quantitative model outputs into concise natural-language commentary.

No external API calls — all text is template-driven and deterministic.
Structured so an LLM-backed explainer could replace this later.
"""


def explain_market(
    market_title: str,
    market_prob: float,
    model_prob: float,
    signal: str,
    implied_vol: float = None,
    realized_vol: float = None,
    hours_to_expiry: float = None,
    confidence_label: str = None,
    concerns: list[str] = None,
    spot: float = None,
    threshold: float = None,
    direction: str = None,
    spread: float = None,
    raw_edge: float = None,
) -> str:
    """
    Generate a concise explanation of the analysis for a given market.
    Returns a few sentences of commentary.
    """
    parts = []

    # opening: what the market is
    if threshold and direction and spot:
        distance_pct = abs(spot - threshold) / spot * 100
        if direction == "above":
            rel = "above" if spot > threshold else "below"
        else:
            rel = "below" if spot < threshold else "above"
        parts.append(
            f"BTC is currently ${spot:,.0f}, sitting {distance_pct:.1f}% {rel} "
            f"the ${threshold:,.0f} threshold."
        )

    # edge assessment
    if market_prob is not None and model_prob is not None:
        edge = model_prob - market_prob
        if abs(edge) < 0.02:
            parts.append(
                f"The market ({market_prob:.0%}) and model ({model_prob:.0%}) "
                f"are closely aligned — no clear mispricing."
            )
        elif edge > 0:
            parts.append(
                f"The market prices this at {market_prob:.0%}, but the vol model "
                f"suggests {model_prob:.0%} — the market may be underpricing YES "
                f"by ~{abs(edge):.0%}."
            )
        else:
            parts.append(
                f"The market prices this at {market_prob:.0%}, but the vol model "
                f"suggests {model_prob:.0%} — the market may be overpricing YES "
                f"by ~{abs(edge):.0%}."
            )

    # vol comparison
    if implied_vol is not None and realized_vol is not None:
        iv_rv_diff = implied_vol - realized_vol
        if abs(iv_rv_diff) < 5:
            parts.append(
                f"Implied vol ({implied_vol:.1f}%) and realized vol ({realized_vol:.1f}%) "
                f"are roughly in line."
            )
        elif iv_rv_diff > 0:
            parts.append(
                f"Implied vol ({implied_vol:.1f}%) is running above realized vol "
                f"({realized_vol:.1f}%), suggesting the options market is pricing in "
                f"more uncertainty than recent price action warrants."
            )
        else:
            parts.append(
                f"Realized vol ({realized_vol:.1f}%) exceeds implied vol ({implied_vol:.1f}%), "
                f"suggesting recent BTC moves have been larger than what's priced in."
            )
    elif implied_vol is not None:
        parts.append(f"Implied vol: {implied_vol:.1f}% (no realized vol available for comparison).")
    elif realized_vol is not None:
        parts.append(f"Realized vol: {realized_vol:.1f}% (no implied vol available).")

    # confidence caveats
    if confidence_label and confidence_label != "HIGH":
        caveat_text = f"Confidence is {confidence_label}"
        if concerns:
            caveat_text += f" due to: {', '.join(concerns)}"
        caveat_text += "."
        parts.append(caveat_text)

    # spread note
    if spread is not None and spread > 0.08:
        parts.append(f"Note: bid-ask spread is wide ({spread:.0%}), which erodes tradeable edge.")

    # expiry note
    if hours_to_expiry is not None and hours_to_expiry < 2:
        parts.append(
            f"With only {hours_to_expiry:.1f}h to expiry, the model estimate is less reliable — "
            f"short-horizon vol scaling from 30-day IV is a rough approximation."
        )

    # signal summary
    if signal == "BUY YES":
        parts.append("Signal: BUY YES (prototype — not a real trade recommendation).")
    elif signal == "BUY NO":
        parts.append("Signal: BUY NO (prototype — not a real trade recommendation).")
    else:
        parts.append("Signal: NO TRADE — edge is insufficient or confidence too low.")

    return " ".join(parts)


def explain_methodology_brief() -> str:
    """Short methodology summary for the dashboard."""
    return (
        "Fair probabilities are estimated using a log-normal model with "
        "zero drift and volatility scaled from Deribit's DVOL index (implied) "
        "or recent BTC returns (realized). The model computes the probability "
        "of BTC crossing the market's threshold level before expiry. "
        "Signals are generated when the model-implied probability diverges "
        "from the market price by more than the configured edge threshold, "
        "adjusted for confidence penalties. This is a prototype — all signals "
        "are illustrative only."
    )
