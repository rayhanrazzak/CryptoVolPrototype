"""
Deribit public API client for BTC implied volatility.
Uses the DVOL index (Deribit's 30-day implied vol index) as the primary
IV anchor, with option-level IV as a fallback / supplementary signal.
"""

import requests
from datetime import datetime, timezone


DERIBIT_BASE = "https://www.deribit.com/api/v2"


def get_btc_dvol() -> dict:
    """
    Fetch the BTC DVOL index — Deribit's 30-day implied volatility index.
    This is the cleanest single-number IV anchor available.
    Returns annualized vol as a percentage (e.g., 55.0 means 55% ann. vol).
    """
    import time
    now_ms = int(time.time() * 1000)
    # fetch the last few hours of DVOL data and take the most recent value
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

    # each row: [timestamp, open, high, low, close]
    latest = data[-1]
    return {
        "dvol": latest[4],  # close value
    }


def get_btc_options_summary() -> list[dict]:
    """
    Fetch summary data for all active BTC options.
    Each entry includes mark_iv, underlying_price, expiration, etc.
    Useful for finding near-term option IV if we want more granularity.
    """
    url = f"{DERIBIT_BASE}/public/get_book_summary_by_currency"
    params = {"currency": "BTC", "kind": "option"}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()["result"]


def get_nearest_expiry_atm_iv() -> dict:
    """
    Find the nearest-expiry ATM option and return its implied vol.
    More targeted than DVOL but noisier for very short expirations.
    """
    summaries = get_btc_options_summary()
    if not summaries:
        return {"iv": None, "instrument": None, "expiry": None}

    now = datetime.now(timezone.utc)

    # filter to options with reasonable IV and near-the-money
    candidates = []
    for s in summaries:
        iv = s.get("mark_iv")
        underlying = s.get("underlying_price", 0)
        mid = s.get("mid_price")
        expiry_ts = s.get("expiration_timestamp", 0)
        instrument = s.get("instrument_name", "")

        if iv is None or iv <= 0 or underlying <= 0:
            continue
        if expiry_ts <= now.timestamp() * 1000:
            continue

        # parse strike from instrument name (e.g., BTC-28MAR25-90000-C)
        parts = instrument.split("-")
        if len(parts) < 3:
            continue
        try:
            strike = float(parts[2])
        except ValueError:
            continue

        # "near ATM" = within 10% of underlying
        moneyness = abs(strike - underlying) / underlying
        if moneyness > 0.10:
            continue

        time_to_expiry = (expiry_ts / 1000) - now.timestamp()
        candidates.append({
            "instrument": instrument,
            "iv": iv,
            "strike": strike,
            "underlying": underlying,
            "moneyness": moneyness,
            "time_to_expiry_hours": time_to_expiry / 3600,
            "expiry_ts": expiry_ts,
        })

    if not candidates:
        return {"iv": None, "instrument": None, "expiry": None}

    # sort by expiry (nearest first), then by moneyness (closest to ATM)
    candidates.sort(key=lambda c: (c["expiry_ts"], c["moneyness"]))
    best = candidates[0]

    return {
        "iv": best["iv"],
        "instrument": best["instrument"],
        "expiry_hours": best["time_to_expiry_hours"],
        "strike": best["strike"],
        "underlying": best["underlying"],
    }


def get_iv_summary() -> dict:
    """
    Combined IV summary: DVOL index + nearest ATM option IV.
    Returns both for display and model use.
    """
    dvol_data = {}
    atm_data = {}

    try:
        dvol_data = get_btc_dvol()
    except Exception as e:
        dvol_data = {"dvol": None, "error": str(e)}

    try:
        atm_data = get_nearest_expiry_atm_iv()
    except Exception as e:
        atm_data = {"iv": None, "error": str(e)}

    return {
        "dvol": dvol_data.get("dvol"),
        "dvol_raw": dvol_data,
        "nearest_atm_iv": atm_data.get("iv"),
        "nearest_atm_detail": atm_data,
    }
