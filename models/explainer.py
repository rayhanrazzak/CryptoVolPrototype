"""
Deterministic explanation engine.
Converts quantitative model outputs into concise natural-language commentary.

No external API calls — all text is template-driven and deterministic.
Structured so an API-backed explainer could replace this later.
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
    iv_source: str = None,
    vol_regime: str = None,
    tail_adjustment: float = None,
    strike_matched_iv: float = None,
    liquidity: str = None,
    forward: float = None,
) -> str:
    """
    Generate a concise explanation of the analysis for a given market.
    """
    parts = []

    # opening: forward vs threshold context (model is centered on forward)
    center = forward or spot
    if threshold and direction and center:
        distance_pct = abs(center - threshold) / center * 100
        rel = "above" if center > threshold else "below"
        if forward and spot:
            parts.append(
                f"The Kalshi forward is ${forward:,.0f} (spot ${spot:,.0f}), "
                f"{distance_pct:.1f}% {rel} the ${threshold:,.0f} threshold."
            )
        elif spot:
            parts.append(
                f"BTC is currently ${spot:,.0f}, sitting {distance_pct:.1f}% {rel} "
                f"the ${threshold:,.0f} threshold."
            )

    # discrepancy assessment
    if market_prob is not None and model_prob is not None:
        edge = model_prob - market_prob
        if abs(edge) < 0.02:
            parts.append(
                f"The market ({market_prob:.0%}) and model ({model_prob:.0%}) "
                f"are closely aligned — no clear mispricing."
            )
        elif edge > 0:
            parts.append(
                f"The market prices this at {market_prob:.0%}, but the model "
                f"suggests {model_prob:.0%} — the market may be underpricing YES "
                f"by ~{abs(edge):.0%}."
            )
        else:
            parts.append(
                f"The market prices this at {market_prob:.0%}, but the model "
                f"suggests {model_prob:.0%} — the market may be overpricing YES "
                f"by ~{abs(edge):.0%}."
            )

    # IV source — explain what vol was used and why it matters
    if iv_source == "strike_matched" and strike_matched_iv:
        parts.append(
            f"Using strike-matched IV ({strike_matched_iv:.1f}%) from Deribit options "
            f"near the ${threshold:,.0f} level"
            + (f", rather than flat DVOL ({implied_vol:.1f}%)." if implied_vol else ".")
        )
    elif iv_source == "dvol_fallback" and implied_vol:
        parts.append(
            f"Using DVOL ({implied_vol:.1f}%) as the IV input — "
            f"no close strike match available on Deribit for this threshold."
        )

    # vol comparison
    iv_display = strike_matched_iv or implied_vol
    if iv_display is not None and realized_vol is not None:
        iv_rv_diff = iv_display - realized_vol
        if abs(iv_rv_diff) < 5:
            parts.append(
                f"IV ({iv_display:.1f}%) and realized vol ({realized_vol:.1f}%) "
                f"are roughly aligned."
            )
        elif iv_rv_diff > 0:
            parts.append(
                f"IV ({iv_display:.1f}%) is above realized vol ({realized_vol:.1f}%), "
                f"suggesting the options market prices in more uncertainty than "
                f"recent price action shows."
            )
        else:
            parts.append(
                f"Realized vol ({realized_vol:.1f}%) exceeds IV ({iv_display:.1f}%), "
                f"suggesting recent BTC moves have been larger than what's priced in."
            )

    # vol regime
    if vol_regime == "vol_expansion":
        parts.append("Vol regime: expansion — IV leads RV, model tilts toward IV.")
    elif vol_regime == "vol_compression":
        parts.append("Vol regime: compression — RV exceeds IV, model tilts toward RV.")

    # tail adjustment
    if tail_adjustment is not None and abs(tail_adjustment) > 0.005:
        direction_word = "adds" if tail_adjustment > 0 else "subtracts"
        parts.append(
            f"Fat-tail adjustment {direction_word} {abs(tail_adjustment):.1%} vs. a "
            f"normal distribution — BTC's heavy tails matter here."
        )

    # liquidity warning
    if liquidity == "one_sided":
        parts.append(
            "Caution: this market has no active bid — the probability estimate "
            "is based on the ask or last trade, not a two-sided midpoint."
        )
    elif liquidity == "no_quotes":
        parts.append("Caution: no live quotes available — pricing is unreliable.")

    # confidence caveats
    if confidence_label and confidence_label != "HIGH":
        caveat_text = f"Confidence is {confidence_label}"
        if concerns:
            caveat_text += f" due to: {', '.join(concerns)}"
        caveat_text += "."
        parts.append(caveat_text)

    # spread note
    if spread is not None and spread > 0.08:
        parts.append(f"Bid-ask spread is wide ({spread:.0%}), which erodes any tradeable discrepancy.")

    # expiry caveat
    if hours_to_expiry is not None and hours_to_expiry < 2:
        parts.append(
            f"With only {hours_to_expiry:.1f}h to expiry, short-horizon vol scaling "
            f"is a rough approximation."
        )

    # signal summary
    if signal == "BUY YES":
        parts.append("Signal: BUY YES (prototype only).")
    elif signal == "BUY NO":
        parts.append("Signal: BUY NO (prototype only).")
    else:
        parts.append("Signal: NO TRADE.")

    return " ".join(parts)


def explain_methodology_brief() -> str:
    """Short methodology summary for the dashboard."""
    return (
        "Fair probabilities are estimated using a log-normal (Gaussian) framework "
        "centered on the Kalshi forward price. "
        "Volatility is sourced from Deribit's options chain — strike-matched and "
        "tenor-interpolated when possible, falling back to the DVOL index. "
        "Implied and realized vol are blended based on their ratio as a regime signal. "
        "Signals require minimum discrepancy after confidence and spread penalties. "
        "All signals are illustrative only."
    )
