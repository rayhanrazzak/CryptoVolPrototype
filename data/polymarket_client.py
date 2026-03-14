"""
Polymarket BTC threshold market discovery and parsing.
Read-only — fetches "Bitcoin above $X on [date]" event data from Gamma API.
"""

import re
import requests
from datetime import datetime, timezone, timedelta

import config
from data.cache import get_cached, set_cached


def discover_btc_threshold_events() -> list[dict]:
    """
    Find active Polymarket BTC threshold events for upcoming dates.
    Tries slug-based lookup for the next few days, plus a broad search fallback.
    """
    cached = get_cached("polymarket_btc_events", ttl_seconds=30)
    if cached is not None:
        return cached

    events = []
    seen_ids = set()
    now = datetime.now(timezone.utc)

    # slug-based lookup for next 4 days
    months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ]
    for day_offset in range(0, 4):
        dt = now + timedelta(days=day_offset)
        month_name = months[dt.month - 1]
        slug = f"bitcoin-above-on-{month_name}-{dt.day}"
        try:
            resp = requests.get(
                f"{config.POLYMARKET_GAMMA_BASE}/events",
                params={"slug": slug},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                # API returns a list of events
                items = data if isinstance(data, list) else [data]
                for ev in items:
                    if isinstance(ev, dict) and ev.get("id") and ev["id"] not in seen_ids:
                        seen_ids.add(ev["id"])
                        events.append(ev)
        except Exception:
            continue

    # broad search fallback
    try:
        resp = requests.get(
            f"{config.POLYMARKET_GAMMA_BASE}/events",
            params={"active": "true", "closed": "false", "limit": "50"},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            items = data if isinstance(data, list) else []
            for ev in items:
                title = ev.get("title", "")
                if re.search(r"Bitcoin above .* on", title, re.IGNORECASE) and ev.get("id") not in seen_ids:
                    seen_ids.add(ev["id"])
                    events.append(ev)
    except Exception:
        pass

    set_cached("polymarket_btc_events", events)
    return events


def parse_markets(events: list[dict]) -> list[dict]:
    """
    Extract individual threshold markets from Polymarket events.
    Each event contains multiple sub-markets at different BTC thresholds.
    """
    markets = []
    now = datetime.now(timezone.utc)

    for ev in events:
        sub_markets = ev.get("markets", [])
        for m in sub_markets:
            question = m.get("question", "")
            # extract threshold from question text
            match = re.search(r'\$(\d[\d,]*)', question)
            if not match:
                continue
            threshold = float(match.group(1).replace(",", ""))

            # parse YES probability from outcomePrices
            outcome_prices = m.get("outcomePrices", "")
            if isinstance(outcome_prices, str):
                try:
                    import json
                    outcome_prices = json.loads(outcome_prices)
                except Exception:
                    continue
            if not outcome_prices or len(outcome_prices) < 1:
                continue
            try:
                yes_prob = float(outcome_prices[0])
            except (ValueError, TypeError):
                continue

            # parse expiry
            end_str = m.get("endDate", "")
            try:
                expiry = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except Exception:
                continue

            hours_to_expiry = max(0, (expiry - now).total_seconds() / 3600)
            if hours_to_expiry <= 0:
                continue

            markets.append({
                "threshold": threshold,
                "direction": "above",
                "market_prob": yes_prob,
                "expiry": expiry,
                "poly_expiry_hours": hours_to_expiry,
                "question": question,
                "slug": m.get("slug", ""),
            })

    return markets


def adjust_to_horizon(poly_markets, target_hours, vol_model_fn) -> dict:
    """
    Time-adjust Polymarket probabilities to a target expiry horizon.
    Standalone — no Kalshi date matching required.

    vol_model_fn(threshold, hours, direction) -> model_prob or None

    Returns {threshold: {"poly_prob_raw", "poly_prob_adjusted", "question", ...}}
    Uses the nearest-expiry Polymarket market per threshold.
    """
    if not poly_markets or target_hours is None:
        return {}

    # pick nearest-expiry market per threshold
    best_per_threshold = {}
    for pm in poly_markets:
        t = pm["threshold"]
        if t not in best_per_threshold or pm["poly_expiry_hours"] < best_per_threshold[t]["poly_expiry_hours"]:
            best_per_threshold[t] = pm

    result = {}
    for threshold, pm in best_per_threshold.items():
        poly_hours = pm["poly_expiry_hours"]
        poly_prob = pm["market_prob"]
        direction = pm["direction"]

        adjusted = poly_prob
        try:
            model_at_poly = vol_model_fn(threshold, poly_hours, direction)
            model_at_target = vol_model_fn(threshold, target_hours, direction)
            if model_at_poly and model_at_poly > 0.01 and model_at_target is not None:
                ratio = model_at_target / model_at_poly
                adjusted = max(0.0, min(1.0, poly_prob * ratio))
        except Exception:
            pass

        result[threshold] = {
            "poly_prob_raw": poly_prob,
            "poly_prob_adjusted": adjusted,
            "question": pm["question"],
            "time_offset_hrs": poly_hours - target_hours,
        }

    return result
