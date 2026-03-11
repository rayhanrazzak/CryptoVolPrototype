"""
Volatility model for BTC threshold/range probability estimation.

Core framework:
  ln(S_T / S_0) ~ Distribution(0, sigma^2 * t)

Supports:
- Log-normal (Gaussian) baseline
- Student-t fat-tail adjustment for more realistic BTC tail probabilities
- Strike-matched IV from Deribit options surface (vs flat DVOL)
- IV-RV blended vol as a regime-aware input
"""

import math
import numpy as np
from scipy import stats
import pandas as pd

import config


def compute_realized_vol(prices: pd.DataFrame, window_hours: int = 24) -> dict:
    """
    Compute annualized realized volatility from recent price data.
    Uses log returns scaled by sqrt(samples_per_year).
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


# ─── Probability estimation ──────────────────────────────────────────────

def threshold_probability(
    spot: float,
    threshold: float,
    vol_annual_pct: float,
    hours_to_expiry: float,
    direction: str = "above",
    use_t: bool = True,
    df: int = None,
) -> dict:
    """
    Estimate probability of BTC crossing a threshold.

    Returns both Gaussian and Student-t estimates for comparison.
    The t-distribution captures fat tails — important for BTC where
    large moves happen more often than a normal distribution predicts.
    """
    if df is None:
        df = config.T_DIST_DF

    if vol_annual_pct <= 0 or hours_to_expiry <= 0 or spot <= 0 or threshold <= 0:
        return {"prob": 0.5, "prob_normal": 0.5, "prob_t": 0.5, "d2": 0}

    sigma = vol_annual_pct / 100.0
    t = hours_to_expiry / 8760.0

    d2 = (math.log(threshold / spot) - 0.5 * sigma**2 * t) / (sigma * math.sqrt(t))

    # Gaussian estimate
    prob_above_normal = float(stats.norm.cdf(-d2))

    # Student-t estimate: same standardized stat, fatter-tailed CDF
    # scale the t-distribution to have unit variance: t_df has variance df/(df-2)
    # so we scale d2 by sqrt((df-2)/df) to get comparable standardized units
    scale_factor = math.sqrt((df - 2) / df) if df > 2 else 1.0
    d2_scaled = d2 * scale_factor
    prob_above_t = float(stats.t.sf(d2_scaled, df=df))  # sf = 1 - cdf = P(X > x)

    if direction == "above":
        p_normal = prob_above_normal
        p_t = prob_above_t
    else:
        p_normal = 1.0 - prob_above_normal
        p_t = 1.0 - prob_above_t

    # primary estimate uses t-distribution (more realistic for crypto)
    primary = p_t if use_t else p_normal

    return {
        "prob": primary,
        "prob_normal": p_normal,
        "prob_t": p_t,
        "tail_adjustment": round(p_t - p_normal, 4),
        "d2": round(d2, 4),
    }


def range_probability(
    spot: float,
    range_low: float,
    range_high: float,
    vol_annual_pct: float,
    hours_to_expiry: float,
    use_t: bool = True,
    df: int = None,
) -> dict:
    """
    Probability that BTC lands in [range_low, range_high) at expiry.
    P(low <= S_T < high) = P(S_T > low) - P(S_T > high)
    """
    above_low = threshold_probability(spot, range_low, vol_annual_pct, hours_to_expiry, "above", use_t, df)
    above_high = threshold_probability(spot, range_high, vol_annual_pct, hours_to_expiry, "above", use_t, df)

    prob = max(above_low["prob"] - above_high["prob"], 0.0)
    prob_normal = max(above_low["prob_normal"] - above_high["prob_normal"], 0.0)
    prob_t = max(above_low["prob_t"] - above_high["prob_t"], 0.0)

    return {
        "prob": prob,
        "prob_normal": prob_normal,
        "prob_t": prob_t,
        "tail_adjustment": round(prob_t - prob_normal, 4),
    }


# ─── IV-RV blending ──────────────────────────────────────────────────────

def compute_blended_vol(implied_vol: float, realized_vol: float) -> dict:
    """
    Blend implied and realized vol based on their ratio as a regime signal.

    Rationale:
    - IV >> RV: options market expects upcoming volatility expansion
      (event risk, macro uncertainty). Trust IV more — it's forward-looking.
    - RV >> IV: recent price action is wilder than what options price.
      The market may be slow to update. Tilt toward RV.
    - Aligned: comfortable blend, slight IV preference (forward-looking).
    """
    ratio = implied_vol / realized_vol if realized_vol > 0 else 1.0

    if ratio > config.BLEND_RATIO_HIGH:
        regime = "vol_expansion"
        iv_weight = config.BLEND_IV_WEIGHT_EXPANSION
    elif ratio < config.BLEND_RATIO_LOW:
        regime = "vol_compression"
        iv_weight = config.BLEND_IV_WEIGHT_COMPRESSION
    else:
        regime = "neutral"
        iv_weight = config.BLEND_IV_WEIGHT_NEUTRAL

    rv_weight = 1.0 - iv_weight
    blended = implied_vol * iv_weight + realized_vol * rv_weight

    return {
        "blended_vol": round(blended, 2),
        "regime": regime,
        "iv_weight": iv_weight,
        "rv_weight": rv_weight,
        "iv_rv_ratio": round(ratio, 3),
    }


# ─── Fair value computation ──────────────────────────────────────────────

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
    vol_surface: dict = None,
) -> dict:
    """
    Compute fair probability estimates using the best available vol inputs.

    Priority:
    1. Strike-matched IV from Deribit options surface (skew + term structure)
    2. DVOL as flat IV fallback
    3. Realized vol as last resort

    When both IV and RV are available, also computes a blended estimate.
    All probabilities use Student-t CDF for fat-tail realism.
    """
    result = {
        "spot": spot,
        "threshold": threshold,
        "hours_to_expiry": hours_to_expiry,
        "direction": direction,
        "market_type": market_type,
    }

    # ─── Step 1: Resolve the best IV input ────────────────────
    strike_iv = None
    iv_source = None
    iv_detail = {}

    # try strike-matched IV from vol surface
    if vol_surface and vol_surface.get("surface"):
        from data.deribit_client import lookup_strike_tenor_iv
        match = lookup_strike_tenor_iv(vol_surface, threshold, hours_to_expiry)
        if match.get("iv"):
            strike_iv = match["iv"]
            iv_source = "strike_matched"
            iv_detail = match

    # determine which IV to use for the primary estimate
    if strike_iv:
        primary_iv = strike_iv
        result["iv_source"] = "strike_matched"
        result["iv_detail"] = iv_detail
    elif implied_vol and implied_vol > 0:
        primary_iv = implied_vol
        result["iv_source"] = "dvol_fallback"
    else:
        primary_iv = None
        result["iv_source"] = None

    result["strike_matched_iv"] = strike_iv
    result["dvol"] = implied_vol

    # ─── Step 2: Compute probabilities under each vol scenario ─

    def _calc(vol_pct):
        if market_type == "range" and range_low and range_high:
            return range_probability(spot, range_low, range_high, vol_pct, hours_to_expiry)
        else:
            return threshold_probability(spot, threshold, vol_pct, hours_to_expiry, direction or "above")

    # IV-based estimate (using best available IV)
    if primary_iv and primary_iv > 0:
        iv_result = _calc(primary_iv)
        result["iv_fair_prob"] = round(iv_result["prob"], 4)
        result["iv_fair_prob_normal"] = round(iv_result["prob_normal"], 4)
        result["iv_tail_adjustment"] = iv_result["tail_adjustment"]
        result["implied_vol_used"] = primary_iv
    else:
        result["iv_fair_prob"] = None
        result["iv_fair_prob_normal"] = None

    # DVOL-based estimate (for comparison when strike-matched is primary)
    if strike_iv and implied_vol and implied_vol > 0:
        dvol_result = _calc(implied_vol)
        result["dvol_fair_prob"] = round(dvol_result["prob"], 4)
    else:
        result["dvol_fair_prob"] = None

    # RV-based estimate
    if realized_vol and realized_vol > 0:
        rv_result = _calc(realized_vol)
        result["rv_fair_prob"] = round(rv_result["prob"], 4)
        result["rv_fair_prob_normal"] = round(rv_result["prob_normal"], 4)
        result["rv_tail_adjustment"] = rv_result["tail_adjustment"]
        result["realized_vol_used"] = realized_vol
    else:
        result["rv_fair_prob"] = None

    # ─── Step 3: Blended estimate ─────────────────────────────
    blend_iv = primary_iv or implied_vol
    if blend_iv and blend_iv > 0 and realized_vol and realized_vol > 0:
        blend = compute_blended_vol(blend_iv, realized_vol)
        blended_result = _calc(blend["blended_vol"])
        result["blended_fair_prob"] = round(blended_result["prob"], 4)
        result["blended_vol"] = blend["blended_vol"]
        result["vol_regime"] = blend["regime"]
        result["iv_rv_ratio"] = blend["iv_rv_ratio"]
        result["blend_weights"] = {"iv": blend["iv_weight"], "rv": blend["rv_weight"]}
    else:
        result["blended_fair_prob"] = None
        result["vol_regime"] = None

    # ─── Step 4: Select primary model probability ─────────────
    # priority: blended (best of both worlds) > IV-only > RV-only
    if result.get("blended_fair_prob") is not None:
        result["model_prob"] = result["blended_fair_prob"]
        result["vol_source"] = "blended"
    elif result.get("iv_fair_prob") is not None:
        result["model_prob"] = result["iv_fair_prob"]
        result["vol_source"] = "implied"
    elif result.get("rv_fair_prob") is not None:
        result["model_prob"] = result["rv_fair_prob"]
        result["vol_source"] = "realized"
    else:
        result["model_prob"] = None
        result["vol_source"] = None

    return result
