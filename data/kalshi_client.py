"""
Kalshi market discovery and data client.
Read-only — no order placement or account operations.

Supports two BTC market types on Kalshi:
- KXBTCD: threshold-style ("$X or above") — ideal for modeling
- KXBTC: range-style ("$X to $Y") — also supported
"""

import re
import requests
from datetime import datetime, timezone
from typing import Optional

import config


def _get_headers() -> dict:
    headers = {"Accept": "application/json"}
    return headers


def discover_btc_events(limit: int = 10) -> list[dict]:
    """
    Discover open BTC events from known series.
    KXBTCD = threshold-style, KXBTC = range-style.
    """
    url = f"{config.KALSHI_API_BASE}/events"
    all_events = []

    for series in ["KXBTCD", "KXBTC"]:
        try:
            params = {
                "limit": limit,
                "status": "open",
                "series_ticker": series,
                "with_nested_markets": "true",
            }
            resp = requests.get(url, headers=_get_headers(), params=params, timeout=15)
            resp.raise_for_status()
            events = resp.json().get("events", [])
            for e in events:
                e["_series"] = series
            all_events.extend(events)
        except requests.RequestException:
            continue

    return all_events


def discover_btc_markets() -> list[dict]:
    """
    Pull all individual BTC markets from discovered events.
    Enriches each market with parsed metadata.
    """
    events = discover_btc_events()
    markets = []

    for event in events:
        event_title = event.get("title", "")
        series = event.get("_series", "")
        nested = event.get("markets", [])

        for m in nested:
            m["_event_title"] = event_title
            m["_series"] = series
            markets.append(m)

    return markets


def rank_btc_markets(markets: list[dict]) -> list[dict]:
    """
    Rank markets for analysis. Prefers:
    - threshold-style (KXBTCD) over range-style
    - shorter expiry
    - near-the-money (interesting probability levels)
    - active trading volume
    """
    now = datetime.now(timezone.utc)
    scored = []

    for m in markets:
        exp_str = m.get("expiration_time") or m.get("close_time")
        if not exp_str:
            continue

        try:
            exp_dt = datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue

        if exp_dt <= now:
            continue

        hours_to_expiry = (exp_dt - now).total_seconds() / 3600
        params = extract_market_params(m)

        score = 0.0

        # prefer shorter expiry
        if 1 <= hours_to_expiry <= 72:
            score += 8.0 / max(hours_to_expiry, 1.0)
        elif hours_to_expiry < 1:
            score += 3.0
        else:
            score += 0.5

        # prefer threshold-style markets — much easier to model
        if m.get("_series") == "KXBTCD":
            score += 5.0

        # prefer markets with interesting probability (not 99c or 1c)
        prob = params.get("market_prob")
        if prob is not None:
            if 0.15 <= prob <= 0.85:
                score += 4.0  # interesting range
            elif 0.05 <= prob <= 0.95:
                score += 2.0

        # volume
        vol = m.get("volume") or 0
        if vol > 0:
            score += min(vol / 200, 3.0)

        # has pricing data
        if m.get("yes_bid") and m.get("yes_ask"):
            score += 2.0
        elif m.get("last_price"):
            score += 1.0

        scored.append({
            "market": m,
            "params": params,
            "score": score,
            "hours_to_expiry": hours_to_expiry,
            "expiry": exp_dt,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def extract_market_params(market: dict) -> dict:
    """
    Parse market metadata into structured params for modeling.
    Handles both threshold ("$X or above") and range ("$X to $Y") styles.
    """
    subtitle = market.get("subtitle", "")
    ticker = market.get("ticker", "")
    series = market.get("_series", "")

    market_type = None
    threshold = None
    range_low = None
    range_high = None
    direction = None

    # threshold-style: "$X or above" / "$X or below"
    above_match = re.search(r'\$([\d,]+(?:\.\d+)?)\s+or\s+above', subtitle)
    below_match = re.search(r'\$([\d,]+(?:\.\d+)?)\s+or\s+below', subtitle)

    if above_match:
        market_type = "threshold"
        threshold = float(above_match.group(1).replace(",", ""))
        direction = "above"
    elif below_match:
        market_type = "threshold"
        threshold = float(below_match.group(1).replace(",", ""))
        direction = "below"
    else:
        # range-style: "$X to Y" or "$X to $Y"
        range_match = re.search(r'\$([\d,]+(?:\.\d+)?)\s+to\s+\$?([\d,]+(?:\.\d+)?)', subtitle)
        if range_match:
            market_type = "range"
            range_low = float(range_match.group(1).replace(",", ""))
            range_high = float(range_match.group(2).replace(",", ""))
            # for range markets, use midpoint as reference
            threshold = (range_low + range_high) / 2

    # pricing — Kalshi uses cents (0-100)
    yes_bid = market.get("yes_bid")
    yes_ask = market.get("yes_ask")
    last_price = market.get("last_price")

    if yes_bid is not None and yes_ask is not None and yes_bid > 0 and yes_ask > 0:
        market_prob = (yes_bid + yes_ask) / 200.0
        spread = (yes_ask - yes_bid) / 100.0
    elif last_price is not None and last_price > 0:
        market_prob = last_price / 100.0
        spread = None
    else:
        market_prob = None
        spread = None

    return {
        "ticker": ticker,
        "title": market.get("title", ""),
        "subtitle": subtitle,
        "event_title": market.get("_event_title", ""),
        "series": series,
        "market_type": market_type,
        "threshold": threshold,
        "direction": direction,
        "range_low": range_low,
        "range_high": range_high,
        "market_prob": market_prob,
        "spread": spread,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "last_price": last_price,
        "volume": market.get("volume"),
        "open_interest": market.get("open_interest"),
    }
