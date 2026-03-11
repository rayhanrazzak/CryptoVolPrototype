"""
BTC spot price client using CoinGecko public API.
Also provides recent price history for realized vol computation.
"""

import time
import requests
import pandas as pd

from data.cache import get_cached, get_stale, set_cached


COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def get_btc_spot() -> dict:
    """Fetch current BTC/USD spot price and 24h change."""
    cache_key = "spot_btc_spot"
    cached = get_cached(cache_key, ttl_seconds=15)
    if cached is not None:
        return cached

    try:
        url = f"{COINGECKO_BASE}/simple/price"
        params = {
            "ids": "bitcoin",
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_last_updated_at": "true",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()["bitcoin"]
        result = {
            "price": data["usd"],
            "change_24h_pct": data.get("usd_24h_change"),
            "last_updated": data.get("last_updated_at"),
        }
        set_cached(cache_key, result)
        return result
    except Exception:
        stale = get_stale(cache_key)
        if stale is not None:
            stale["_from_cache"] = True
            return stale
        raise


def get_btc_price_history(hours: int = 24) -> pd.DataFrame:
    """
    Fetch recent BTC price data for realized vol calculation.
    Uses CoinGecko market_chart endpoint which gives ~5-min granularity
    for ranges <= 1 day, hourly for ranges <= 90 days.
    """
    cache_key = f"spot_btc_history_{hours}h"
    cached = get_cached(cache_key, ttl_seconds=60)
    if cached is not None:
        df = pd.DataFrame(cached, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        return df

    try:
        url = f"{COINGECKO_BASE}/coins/bitcoin/market_chart"
        # for sub-day granularity we need hours <= 24
        # CoinGecko returns ~5-min intervals for 1-day range
        params = {
            "vs_currency": "usd",
            "days": max(hours / 24, 1),
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        prices = resp.json()["prices"]
        set_cached(cache_key, prices)

        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        return df
    except Exception:
        stale = get_stale(cache_key)
        if stale is not None:
            df = pd.DataFrame(stale, columns=["timestamp", "price"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp").sort_index()
            df.attrs["_from_cache"] = True
            return df
        raise


def get_btc_ohlc_history(days: int = 1) -> pd.DataFrame:
    """
    Fetch OHLC candle data. CoinGecko gives 30-min candles for 1-2 day range.
    Useful as a secondary data source.
    """
    url = f"{COINGECKO_BASE}/coins/bitcoin/ohlc"
    params = {
        "vs_currency": "usd",
        "days": days,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    return df
