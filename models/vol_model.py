"""
Volatility model for BTC threshold-crossing probability estimation.

Uses a simple log-normal framework:
  ln(S_T / S_0) ~ N(mu * t, sigma^2 * t)

where sigma is the annualized volatility input (implied or realized).
For short horizons, the drift term is negligible relative to vol,
so we approximate with mu=0.

This gives a clean, defensible probability estimate that's easy to explain.
"""

import math
import numpy as np
from scipy import stats
import pandas as pd


def compute_realized_vol(prices: pd.DataFrame, window_hours: int = 24) -> dict:
    """
    Compute annualized realized volatility from recent price data.

    Uses log returns and scales to annualized terms.
    Returns vol as a percentage (e.g., 55.0 = 55% annualized).
    """
    if prices.empty or len(prices) < 5:
        return {"realized_vol": None, "n_observations": 0, "error": "insufficient data"}

    df = prices.copy()
    if "price" in df.columns:
        series = df["price"]
    elif "close" in df.columns:
        series = df["close"]
    else:
        series = df.iloc[:, 0]

    log_returns = np.log(series / series.shift(1)).dropna()

    if len(log_returns) < 3:
        return {"realized_vol": None, "n_observations": len(log_returns), "error": "too few returns"}

    # infer sampling frequency from median time delta
    if hasattr(df.index, 'freq') and df.index.freq:
        samples_per_year = pd.Timedelta("365.25D") / df.index.freq
    else:
        median_delta = df.index.to_series().diff().median()
        samples_per_year = pd.Timedelta("365.25D") / median_delta

    sample_std = log_returns.std()
    annualized_vol = sample_std * math.sqrt(float(samples_per_year)) * 100

    return {
        "realized_vol": round(annualized_vol, 2),
        "n_observations": len(log_returns),
        "sample_std": float(sample_std),
        "samples_per_year": float(samples_per_year),
        "mean_return": float(log_returns.mean()),
        "latest_price": float(series.iloc[-1]),
    }


def threshold_probability(
    spot: float,
    threshold: float,
    vol_annual_pct: float,
    hours_to_expiry: float,
    direction: str = "above",
) -> float:
    """
    Estimate probability that BTC crosses a threshold before expiry.

    Uses log-normal assumption with zero drift (reasonable for short horizons).

    P(S_T > K) = N(-d2) where d2 = [ln(K/S) - 0.5*sigma^2*t] / (sigma*sqrt(t))

    For 'below': P(S_T < K) = 1 - P(S_T > K)
    """
    if vol_annual_pct <= 0 or hours_to_expiry <= 0 or spot <= 0 or threshold <= 0:
        return 0.5  # can't estimate, return agnostic

    sigma = vol_annual_pct / 100.0
    t = hours_to_expiry / 8760.0  # convert hours to years

    d2 = (math.log(threshold / spot) - 0.5 * sigma**2 * t) / (sigma * math.sqrt(t))

    # P(S_T > threshold) = N(-d2)
    prob_above = stats.norm.cdf(-d2)

    if direction == "above":
        return float(prob_above)
    else:
        return float(1.0 - prob_above)


def range_probability(
    spot: float,
    range_low: float,
    range_high: float,
    vol_annual_pct: float,
    hours_to_expiry: float,
) -> float:
    """
    Probability that BTC lands in a price range [low, high) at expiry.
    P(low <= S_T < high) = P(S_T > low) - P(S_T > high)
    """
    prob_above_low = threshold_probability(spot, range_low, vol_annual_pct, hours_to_expiry, "above")
    prob_above_high = threshold_probability(spot, range_high, vol_annual_pct, hours_to_expiry, "above")
    return max(prob_above_low - prob_above_high, 0.0)


def compute_fair_value(
    spot: float,
    threshold: float,
    hours_to_expiry: float,
    direction: str,
    implied_vol: float = None,
    realized_vol: float = None,
    market_type: str = "threshold",
    range_low: float = None,
    range_high: float = None,
) -> dict:
    """
    Compute fair probability estimates using available vol inputs.
    Supports both threshold-style and range-style markets.
    Returns separate estimates for IV-based and RV-based scenarios.
    """
    result = {
        "spot": spot,
        "threshold": threshold,
        "hours_to_expiry": hours_to_expiry,
        "direction": direction,
        "market_type": market_type,
    }

    def _calc_prob(vol_pct):
        if market_type == "range" and range_low and range_high:
            return range_probability(spot, range_low, range_high, vol_pct, hours_to_expiry)
        else:
            return threshold_probability(spot, threshold, vol_pct, hours_to_expiry, direction or "above")

    if implied_vol and implied_vol > 0:
        result["iv_fair_prob"] = round(_calc_prob(implied_vol), 4)
        result["implied_vol_used"] = implied_vol
    else:
        result["iv_fair_prob"] = None

    if realized_vol and realized_vol > 0:
        result["rv_fair_prob"] = round(_calc_prob(realized_vol), 4)
        result["realized_vol_used"] = realized_vol
    else:
        result["rv_fair_prob"] = None

    # primary model estimate: prefer IV when available, fall back to RV
    if result["iv_fair_prob"] is not None:
        result["model_prob"] = result["iv_fair_prob"]
        result["vol_source"] = "implied"
    elif result["rv_fair_prob"] is not None:
        result["model_prob"] = result["rv_fair_prob"]
        result["vol_source"] = "realized"
    else:
        result["model_prob"] = None
        result["vol_source"] = None

    return result
