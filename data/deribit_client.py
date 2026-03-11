"""
Deribit public API client for BTC implied volatility.

Provides:
- DVOL index (30-day implied vol) as a baseline anchor
- Full vol surface from the BTC options chain for strike/tenor-matched IV
- Near-ATM option IV as a supplementary signal
"""

import time
import requests
from datetime import datetime, timezone
from collections import defaultdict

from data.cache import get_cached, set_cached

DERIBIT_BASE = "https://www.deribit.com/api/v2"


def _parse_instrument_name(name: str) -> dict | None:
    """
    Parse a Deribit instrument name like 'BTC-28MAR26-90000-C'.
    Returns expiry datetime, strike, and option type.
    """
    parts = name.split("-")
    if len(parts) != 4 or parts[0] != "BTC":
        return None
    try:
        # Deribit options settle at 08:00 UTC on expiry date
        expiry = datetime.strptime(parts[1], "%d%b%y").replace(
            tzinfo=timezone.utc, hour=8
        )
        strike = float(parts[2])
        opt_type = parts[3]  # C or P
        if opt_type not in ("C", "P"):
            return None
    except (ValueError, TypeError):
        return None

    return {
        "expiry": expiry,
        "expiry_label": parts[1],
        "strike": strike,
        "type": opt_type,
    }


def get_btc_dvol() -> dict:
    """
    Fetch the BTC DVOL index — Deribit's 30-day implied volatility index.
    Returns annualized vol as a percentage (e.g., 55.0 means 55% ann. vol).
    """
    cache_key = "deribit_btc_dvol"
    cached = get_cached(cache_key, ttl_seconds=30)
    if cached is not None:
        return cached

    now_ms = int(time.time() * 1000)
    url = f"{DERIBIT_BASE}/public/get_volatility_index_data"
    params = {
        "currency": "BTC",
        "resolution": 3600,
        "start_timestamp": now_ms - 6 * 3600 * 1000,
        "end_timestamp": now_ms,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()["result"]["data"]
    if not data:
        return {"dvol": None, "error": "no DVOL data returned"}

    latest = data[-1]
    result = {"dvol": latest[4]}  # close value
    set_cached(cache_key, result)
    return result


def get_btc_options_summary() -> list[dict]:
    """
    Fetch summary data for all active BTC options.
    Each entry includes mark_iv, underlying_price, etc.
    """
    cache_key = "deribit_btc_options_summary"
    cached = get_cached(cache_key, ttl_seconds=60)
    if cached is not None:
        return cached

    url = f"{DERIBIT_BASE}/public/get_book_summary_by_currency"
    params = {"currency": "BTC", "kind": "option"}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    result = resp.json()["result"]
    set_cached(cache_key, result)
    return result


def build_vol_surface() -> dict:
    """
    Build a structured volatility surface from the full Deribit option chain.

    Returns a dict with:
    - surface: {expiry_label: [{"strike", "iv", "instrument", "type"}, ...]}
    - expiries: sorted list of (label, expiry_dt, hours_to_expiry)
    - underlying: current underlying price from the options chain
    - raw_count: total options processed
    """
    # the underlying get_btc_options_summary() call is cached at 60s,
    # so this avoids redundant API hits even without its own cache
    summaries = get_btc_options_summary()
    now = datetime.now(timezone.utc)

    surface = defaultdict(list)
    underlying = None

    for s in summaries:
        iv = s.get("mark_iv")
        if not iv or iv <= 0:
            continue

        parsed = _parse_instrument_name(s.get("instrument_name", ""))
        if not parsed:
            continue

        if parsed["expiry"] <= now:
            continue

        if underlying is None:
            underlying = s.get("underlying_price", 0)

        hours = (parsed["expiry"] - now).total_seconds() / 3600

        surface[parsed["expiry_label"]].append({
            "strike": parsed["strike"],
            "iv": iv,
            "instrument": s["instrument_name"],
            "type": parsed["type"],
            "expiry": parsed["expiry"],
            "hours_to_expiry": hours,
        })

    # sort strikes within each expiry
    for label in surface:
        surface[label].sort(key=lambda x: x["strike"])

    # build sorted expiry list
    expiries = []
    for label, opts in surface.items():
        if opts:
            expiries.append({
                "label": label,
                "expiry": opts[0]["expiry"],
                "hours": opts[0]["hours_to_expiry"],
                "n_strikes": len(opts),
            })
    expiries.sort(key=lambda x: x["hours"])

    return {
        "surface": dict(surface),
        "expiries": expiries,
        "underlying": underlying or 0,
        "raw_count": len(summaries),
    }


def lookup_strike_tenor_iv(
    vol_surface: dict,
    target_strike: float,
    target_hours: float,
) -> dict:
    """
    Look up implied vol for a specific strike and tenor from the vol surface.

    Uses the nearest Deribit expiry and interpolates between the two
    closest strikes. If two expiries bracket the target, interpolates
    in total-variance space (sigma^2 * t) to avoid calendar arbitrage.

    Returns the matched IV, source instruments, and method description.
    """
    if not vol_surface or not vol_surface.get("expiries"):
        return {"iv": None, "method": "no_surface"}

    expiries = vol_surface["expiries"]
    surface = vol_surface["surface"]

    # find bracketing expiries
    before = None
    after = None
    for exp in expiries:
        if exp["hours"] <= target_hours:
            before = exp
        elif after is None:
            after = exp

    # determine which expiry/expiries to use
    if before and after:
        # interpolate between two expiries in total-variance space
        iv_before = _find_strike_iv(surface[before["label"]], target_strike)
        iv_after = _find_strike_iv(surface[after["label"]], target_strike)

        if iv_before and iv_after:
            # total variance interpolation: sigma^2 * t
            t1 = before["hours"] / 8760
            t2 = after["hours"] / 8760
            t_target = target_hours / 8760
            var1 = (iv_before["iv"] / 100) ** 2 * t1
            var2 = (iv_after["iv"] / 100) ** 2 * t2

            # linear interpolation in total variance
            w = (t_target - t1) / (t2 - t1) if t2 != t1 else 0.5
            var_interp = var1 + w * (var2 - var1)
            iv_interp = (var_interp / t_target) ** 0.5 * 100 if t_target > 0 else iv_before["iv"]

            return {
                "iv": round(iv_interp, 2),
                "method": "tenor_strike_interpolated",
                "near_expiry": before["label"],
                "far_expiry": after["label"],
                "near_iv": iv_before["iv"],
                "far_iv": iv_after["iv"],
                "instruments": [iv_before.get("instrument"), iv_after.get("instrument")],
            }

        # fall back to whichever side has data
        best = iv_before or iv_after
        exp_used = before if iv_before else after
    elif before:
        best = _find_strike_iv(surface[before["label"]], target_strike)
        exp_used = before
    elif after:
        best = _find_strike_iv(surface[after["label"]], target_strike)
        exp_used = after
    else:
        return {"iv": None, "method": "no_matching_expiry"}

    if not best:
        return {"iv": None, "method": "no_matching_strike"}

    return {
        "iv": best["iv"],
        "method": "nearest_strike_tenor",
        "expiry_used": exp_used["label"],
        "strike_used": best["strike"],
        "instrument": best.get("instrument"),
        "interpolated_strike": best.get("interpolated", False),
    }


def _find_strike_iv(options: list[dict], target_strike: float) -> dict | None:
    """
    Find or interpolate the IV at a target strike from a sorted list of options.
    Linear interpolation between the two bracketing strikes.
    """
    if not options:
        return None

    strikes = [o["strike"] for o in options]

    # exact match
    for o in options:
        if o["strike"] == target_strike:
            return o

    # find bracketing strikes
    below = None
    above = None
    for o in options:
        if o["strike"] <= target_strike:
            below = o
        elif above is None:
            above = o

    if below and above:
        # linear interpolation
        w = (target_strike - below["strike"]) / (above["strike"] - below["strike"])
        iv_interp = below["iv"] + w * (above["iv"] - below["iv"])
        return {
            "strike": target_strike,
            "iv": round(iv_interp, 2),
            "instrument": f"{below['instrument']}↔{above['instrument']}",
            "interpolated": True,
        }

    # only one side available — use nearest
    nearest = below or above
    return nearest


def get_iv_summary() -> dict:
    """
    Combined IV summary: DVOL index + full vol surface.
    The vol surface enables strike/tenor-matched IV lookup.
    """
    dvol_data = {}
    vol_surface = {}

    try:
        dvol_data = get_btc_dvol()
    except Exception as e:
        dvol_data = {"dvol": None, "error": str(e)}

    try:
        vol_surface = build_vol_surface()
    except Exception as e:
        vol_surface = {"surface": {}, "expiries": [], "error": str(e)}

    return {
        "dvol": dvol_data.get("dvol"),
        "dvol_raw": dvol_data,
        "vol_surface": vol_surface,
    }
