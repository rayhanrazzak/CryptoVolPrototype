"""
BTC Volatility-Based Market Analyzer
Streamlit dashboard for comparing Kalshi market probabilities
against a volatility-based fair value model.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone

from data import kalshi_client, deribit_client, spot_client
from models import vol_model, signal_engine, explainer
import config


st.set_page_config(
    page_title="BTC Market Analyzer",
    page_icon="₿",
    layout="wide",
)


@st.cache_data(ttl=config.REFRESH_INTERVAL)
def fetch_all_data():
    """Fetch all live data sources. Cached briefly to avoid hammering APIs."""
    errors = []

    # BTC spot
    try:
        spot = spot_client.get_btc_spot()
    except Exception as e:
        spot = {"price": None, "error": str(e)}
        errors.append(f"Spot: {e}")

    # price history for realized vol
    try:
        price_history = spot_client.get_btc_price_history(hours=24)
    except Exception as e:
        price_history = pd.DataFrame()
        errors.append(f"Price history: {e}")

    # implied vol
    try:
        iv_data = deribit_client.get_iv_summary()
    except Exception as e:
        iv_data = {"dvol": None, "nearest_atm_iv": None, "error": str(e)}
        errors.append(f"IV: {e}")

    # realized vol
    rv_data = vol_model.compute_realized_vol(price_history)

    # Kalshi markets
    try:
        raw_markets = kalshi_client.discover_btc_markets()
        ranked = kalshi_client.rank_btc_markets(raw_markets)
    except Exception as e:
        raw_markets = []
        ranked = []
        errors.append(f"Kalshi: {e}")

    return {
        "spot": spot,
        "price_history": price_history,
        "iv_data": iv_data,
        "rv_data": rv_data,
        "ranked_markets": ranked,
        "raw_markets": raw_markets,
        "errors": errors,
        "timestamp": datetime.now(timezone.utc),
    }


def analyze_market(market_entry: dict, spot_price: float, iv: float, rv: float) -> dict:
    """Run the full analysis pipeline for a single market."""
    market = market_entry["market"]
    # use pre-computed params if available from ranking
    params = market_entry.get("params") or kalshi_client.extract_market_params(market)

    hours = market_entry["hours_to_expiry"]
    threshold = params["threshold"]
    direction = params["direction"] or "above"
    market_prob = params["market_prob"]
    market_type = params.get("market_type", "threshold")

    # compute fair value
    if threshold and spot_price:
        fv = vol_model.compute_fair_value(
            spot=spot_price,
            threshold=threshold,
            hours_to_expiry=hours,
            direction=direction,
            implied_vol=iv,
            realized_vol=rv,
            market_type=market_type,
            range_low=params.get("range_low"),
            range_high=params.get("range_high"),
        )
    else:
        fv = {"model_prob": None, "iv_fair_prob": None, "rv_fair_prob": None}

    model_prob = fv.get("model_prob")

    # confidence assessment
    iv_rv_div = (iv - rv) if (iv and rv) else None
    conf = signal_engine.assess_confidence(
        spread=params["spread"],
        volume=params["volume"],
        iv_available=iv is not None,
        rv_available=rv is not None,
        hours_to_expiry=hours,
        iv_rv_divergence=iv_rv_div,
    )

    # generate signal
    sig = signal_engine.generate_signal(
        market_prob=market_prob,
        model_prob=model_prob,
        confidence=conf["confidence"],
        spread=params["spread"],
    )

    # generate explanation
    explanation = explainer.explain_market(
        market_title=params["title"],
        market_prob=market_prob,
        model_prob=model_prob,
        signal=sig["signal"],
        implied_vol=iv,
        realized_vol=rv,
        hours_to_expiry=hours,
        confidence_label=conf["confidence_label"],
        concerns=conf["concerns"],
        spot=spot_price,
        threshold=threshold,
        direction=direction,
        spread=params["spread"],
        raw_edge=sig["raw_edge"],
    )

    return {
        "params": params,
        "fair_value": fv,
        "confidence": conf,
        "signal": sig,
        "explanation": explanation,
        "hours_to_expiry": hours,
    }


# ─── UI ────────────────────────────────────────────────────────────────────

def render_header(data: dict):
    """Top-level status bar."""
    spot = data["spot"]
    iv = data["iv_data"]
    rv = data["rv_data"]

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if spot.get("price"):
            st.metric("BTC Spot", f"${spot['price']:,.0f}",
                      f"{spot.get('change_24h_pct', 0):.1f}% (24h)")
        else:
            st.metric("BTC Spot", "N/A")

    with col2:
        dvol = iv.get("dvol")
        st.metric("DVOL (30d IV)", f"{dvol:.1f}%" if dvol else "N/A")

    with col3:
        atm_iv = iv.get("nearest_atm_iv")
        st.metric("Near ATM IV", f"{atm_iv:.1f}%" if atm_iv else "N/A")

    with col4:
        rv_val = rv.get("realized_vol")
        st.metric("Realized Vol (24h)", f"{rv_val:.1f}%" if rv_val else "N/A")

    with col5:
        n_markets = len(data["ranked_markets"])
        st.metric("BTC Markets Found", n_markets)


def render_dashboard(data: dict):
    """Main dashboard tab."""
    st.subheader("Market Opportunities")

    spot_price = data["spot"].get("price")
    iv = data["iv_data"].get("dvol")
    rv = data["rv_data"].get("realized_vol")
    ranked = data["ranked_markets"]

    if not ranked:
        st.warning(
            "No active BTC markets found on Kalshi. "
            "This can happen outside of active trading hours or if market listings have changed."
        )
        if data["errors"]:
            with st.expander("API Errors"):
                for e in data["errors"]:
                    st.text(e)
        return

    # analyze top markets
    analyses = []
    for entry in ranked[:10]:
        try:
            analysis = analyze_market(entry, spot_price, iv, rv)
            analyses.append(analysis)
        except Exception as e:
            continue

    if not analyses:
        st.warning("Could not analyze any markets. Check data availability.")
        return

    # summary table
    table_data = []
    for a in analyses:
        p = a["params"]
        fv = a["fair_value"]
        sig = a["signal"]
        conf = a["confidence"]

        table_data.append({
            "Market": p["title"][:60],
            "Expiry (hrs)": f"{a['hours_to_expiry']:.1f}",
            "Market Prob": f"{p['market_prob']:.0%}" if p["market_prob"] else "—",
            "Model Prob": f"{fv.get('model_prob'):.0%}" if fv.get("model_prob") else "—",
            "Edge": f"{sig['raw_edge']:+.1%}" if sig["raw_edge"] else "—",
            "Signal": sig["signal"],
            "Confidence": conf["confidence_label"],
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # detailed cards for top opportunities
    st.subheader("Analysis Detail")
    for i, a in enumerate(analyses[:5]):
        p = a["params"]
        sig = a["signal"]

        signal_color = {"BUY YES": "🟢", "BUY NO": "🔴", "NO TRADE": "⚪"}.get(sig["signal"], "⚪")

        with st.expander(f"{signal_color} {p['title'][:60]} — {sig['signal']}", expanded=(i == 0)):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Market Probability",
                          f"{p['market_prob']:.1%}" if p["market_prob"] else "N/A")
            with col2:
                mp = a["fair_value"].get("model_prob")
                st.metric("Model Probability", f"{mp:.1%}" if mp else "N/A")
            with col3:
                st.metric("Signal", sig["signal"])

            st.markdown(f"**Explanation:** {a['explanation']}")

            if sig["raw_edge"] is not None:
                sub1, sub2, sub3, sub4 = st.columns(4)
                with sub1:
                    st.metric("Raw Edge", f"{sig['raw_edge']:+.1%}")
                with sub2:
                    st.metric("Adj. Edge", f"{sig['adjusted_edge']:+.1%}")
                with sub3:
                    st.metric("Confidence", a["confidence"]["confidence_label"])
                with sub4:
                    st.metric("Spread", f"{p['spread']:.1%}" if p["spread"] else "N/A")

    # vol chart
    if not data["price_history"].empty:
        render_price_chart(data["price_history"])


def render_price_chart(price_history: pd.DataFrame):
    """Simple BTC price chart for context."""
    st.subheader("BTC Price — Last 24h")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_history.index,
        y=price_history["price"],
        mode="lines",
        name="BTC/USD",
        line=dict(color="#F7931A", width=2),
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="",
        yaxis_title="USD",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_market_details(data: dict):
    """Detailed view for a selected market."""
    ranked = data["ranked_markets"]
    if not ranked:
        st.info("No markets available for detailed view.")
        return

    market_titles = [
        f"{r['market'].get('title', 'Unknown')[:50]} ({r['hours_to_expiry']:.1f}h)"
        for r in ranked[:10]
    ]

    selected_idx = st.selectbox("Select a market:", range(len(market_titles)),
                                format_func=lambda i: market_titles[i])

    entry = ranked[selected_idx]
    market = entry["market"]
    spot_price = data["spot"].get("price")
    iv = data["iv_data"].get("dvol")
    rv = data["rv_data"].get("realized_vol")

    try:
        analysis = analyze_market(entry, spot_price, iv, rv)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return

    params = analysis["params"]
    fv = analysis["fair_value"]
    sig = analysis["signal"]
    conf = analysis["confidence"]

    # market metadata
    st.subheader("Market Metadata")
    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        st.markdown(f"**Title:** {params['title']}")
        st.markdown(f"**Ticker:** {params['ticker']}")
        st.markdown(f"**Direction:** {params.get('direction', 'N/A')}")
        st.markdown(f"**Threshold:** ${params['threshold']:,.0f}" if params["threshold"] else "**Threshold:** N/A")
    with meta_col2:
        st.markdown(f"**Hours to Expiry:** {analysis['hours_to_expiry']:.2f}")
        st.markdown(f"**Volume:** {params.get('volume', 'N/A')}")
        st.markdown(f"**Open Interest:** {params.get('open_interest', 'N/A')}")
        st.markdown(f"**Spread:** {params['spread']:.1%}" if params["spread"] else "**Spread:** N/A")

    # pricing
    st.subheader("Pricing & Model")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        st.metric("Yes Bid", f"{params['yes_bid']}" if params["yes_bid"] else "N/A")
    with p_col2:
        st.metric("Yes Ask", f"{params['yes_ask']}" if params["yes_ask"] else "N/A")
    with p_col3:
        st.metric("Market Prob", f"{params['market_prob']:.1%}" if params["market_prob"] else "N/A")
    with p_col4:
        mp = fv.get("model_prob")
        st.metric("Model Prob", f"{mp:.1%}" if mp else "N/A")

    # vol inputs
    st.subheader("Volatility Inputs")
    v_col1, v_col2, v_col3, v_col4 = st.columns(4)
    with v_col1:
        st.metric("DVOL (IV)", f"{iv:.1f}%" if iv else "N/A")
    with v_col2:
        st.metric("Realized Vol", f"{rv:.1f}%" if rv else "N/A")
    with v_col3:
        iv_prob = fv.get("iv_fair_prob")
        st.metric("IV-Based Prob", f"{iv_prob:.1%}" if iv_prob else "N/A")
    with v_col4:
        rv_prob = fv.get("rv_fair_prob")
        st.metric("RV-Based Prob", f"{rv_prob:.1%}" if rv_prob else "N/A")

    # signal
    st.subheader("Signal")
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        st.metric("Signal", sig["signal"])
    with s_col2:
        st.metric("Adjusted Edge", f"{sig['adjusted_edge']:+.1%}" if sig["adjusted_edge"] else "N/A")
    with s_col3:
        st.metric("Confidence", conf["confidence_label"])

    if conf["concerns"]:
        st.markdown(f"**Confidence concerns:** {', '.join(conf['concerns'])}")

    # explanation
    st.subheader("Explanation")
    st.markdown(analysis["explanation"])

    # raw market data
    with st.expander("Raw Market Data"):
        st.json(market)


def render_methodology():
    """Methodology explanation tab."""
    st.subheader("Methodology")

    st.markdown("""
### Market Selection

Markets are discovered dynamically from Kalshi's API by searching for BTC/Bitcoin-related
event contracts. Markets are ranked by a scoring function that prefers:
- Shorter time to expiry (more relevant for short-horizon analysis)
- Active markets with volume and pricing
- Threshold-style binary contracts (e.g., "BTC above $X by time Y")

### Data Sources

| Source | Data | Auth Required |
|--------|------|---------------|
| **CoinGecko** | BTC spot price, 24h price history | No |
| **Deribit** | DVOL index (30-day implied vol), near-ATM option IV | No |
| **Kalshi** | Market listings, pricing, order book | No (read-only) |

### Implied Volatility

The primary IV input is Deribit's **DVOL index**, a 30-day implied volatility measure
for BTC options. This is a clean, well-known benchmark.

**Caveat:** Using 30-day IV as a proxy for very short-horizon (minutes to hours) event
probabilities involves a horizon mismatch. Short-horizon realized vol can differ
meaningfully from the 30-day implied level. The model accounts for this by scaling
vol to the appropriate time horizon using √t scaling, but the underlying assumption
of constant vol across horizons is a simplification.

### Realized Volatility

Realized vol is computed from the last 24 hours of BTC price data (approximately
5-minute intervals from CoinGecko). The standard deviation of log returns is
annualized using √(samples_per_year) scaling.

This provides a backward-looking vol estimate that captures recent market conditions
but may miss sudden regime changes.

### Fair Value Model

For a threshold-style market ("BTC above $K by time T"), the model estimates:

**P(S_T > K)** under a log-normal assumption:

```
ln(S_T / S_0) ~ N(0, σ²t)
```

where:
- S_0 = current BTC spot price
- K = threshold level
- σ = annualized volatility (from IV or RV)
- t = time to expiry in years
- Drift is set to zero (negligible for short horizons)

The probability is computed using the standard normal CDF:
```
P(S_T > K) = Φ(-d₂)
d₂ = [ln(K/S₀) - 0.5σ²t] / (σ√t)
```

The model produces separate estimates using implied vol and realized vol
when both are available, defaulting to IV as the primary input.

### Signal Generation

Signals are generated using a simple edge framework:

1. **Raw edge** = model probability − market probability
2. **Adjusted edge** = raw edge × confidence − spread cost
3. If adjusted edge > threshold (default 5%): **BUY YES**
4. If adjusted edge < −threshold: **BUY NO**
5. Otherwise: **NO TRADE**

Confidence is reduced by:
- Wide bid-ask spreads
- Low volume
- Missing data (no IV or RV)
- Very short expiry (< 1 hour)
- Large IV-RV disagreement

### Assumptions & Limitations

- **Log-normal returns:** BTC returns have fat tails; the model underestimates
  extreme moves. A more realistic model would use a fat-tailed distribution.
- **Constant volatility:** Vol is assumed constant over the horizon, which is
  unrealistic for very short periods.
- **Zero drift:** Reasonable for short horizons but ignores any intraday momentum.
- **Vol scaling:** √t scaling from 30-day IV to sub-day horizons is a rough
  approximation. Actual short-dated vol can be significantly different.
- **No microstructure:** The model doesn't account for order flow, market maker
  behavior, or event-driven pricing (e.g., around scheduled macro releases).
- **Prototype only:** Signals are illustrative and not intended for actual trading.
""")


# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():
    st.title("₿ BTC Market Analyzer")
    st.caption("Comparing Kalshi market probabilities against a volatility-based fair value model")

    # fetch data
    with st.spinner("Fetching live data..."):
        data = fetch_all_data()

    # show errors if any
    if data["errors"]:
        with st.sidebar.expander("⚠ Data Warnings", expanded=False):
            for e in data["errors"]:
                st.warning(e)

    # header metrics
    render_header(data)
    st.divider()

    # tabs
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Market Details", "Methodology"])

    with tab1:
        render_dashboard(data)

    with tab2:
        render_market_details(data)

    with tab3:
        render_methodology()

    # footer
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last updated: {data['timestamp'].strftime('%H:%M:%S UTC')}")
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
