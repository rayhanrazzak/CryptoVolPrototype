"""
BTC Volatility-Based Market Analyzer
Streamlit dashboard for comparing Kalshi market probabilities
against a volatility-based fair value model.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone

from data import kalshi_client, deribit_client, spot_client
from models import vol_model, signal_engine, explainer
import config


st.set_page_config(
    page_title="BTC Market Analyzer",
    page_icon="₿",
    layout="wide",
)

# ─── Custom styling ───────────────────────────────────────────────────────

st.markdown("""
<style>
    .signal-buy-yes {
        background-color: #0e4429;
        color: #3fb950;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85em;
        display: inline-block;
    }
    .signal-buy-no {
        background-color: #4a1419;
        color: #f85149;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85em;
        display: inline-block;
    }
    .signal-no-trade {
        background-color: #2d2d2d;
        color: #8b949e;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85em;
        display: inline-block;
    }
    .confidence-high { color: #3fb950; }
    .confidence-medium { color: #d29922; }
    .confidence-low { color: #f85149; }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 8px;
    }
    .explanation-box {
        background-color: #161b22;
        border-left: 3px solid #58a6ff;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin: 8px 0;
        font-size: 0.92em;
        line-height: 1.5;
    }
    div[data-testid="stMetric"] {
        background-color: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


def signal_badge(signal: str) -> str:
    css_class = {
        "BUY YES": "signal-buy-yes",
        "BUY NO": "signal-buy-no",
        "NO TRADE": "signal-no-trade",
    }.get(signal, "signal-no-trade")
    return f'<span class="{css_class}">{signal}</span>'


def confidence_colored(label: str) -> str:
    css_class = {
        "HIGH": "confidence-high",
        "MEDIUM": "confidence-medium",
        "LOW": "confidence-low",
    }.get(label, "")
    return f'<span class="{css_class}">{label}</span>'


def format_market_label(params: dict) -> str:
    """Human-readable label for a market contract."""
    sub = params.get("subtitle", "")
    if sub:
        return sub
    if params.get("threshold"):
        direction = params.get("direction", "above")
        return f"${params['threshold']:,.0f} or {direction}"
    return params.get("title", "Unknown")[:50]


def format_expiry(hours: float) -> str:
    if hours < 1:
        return f"{hours * 60:.0f}m"
    elif hours < 48:
        return f"{hours:.1f}h"
    else:
        return f"{hours / 24:.1f}d"


# ─── Data fetching ────────────────────────────────────────────────────────

@st.cache_data(ttl=config.REFRESH_INTERVAL)
def fetch_all_data():
    """Fetch all live data sources. Cached briefly to avoid hammering APIs."""
    errors = []

    try:
        spot = spot_client.get_btc_spot()
    except Exception as e:
        spot = {"price": None, "error": str(e)}
        errors.append(f"Spot: {e}")

    try:
        price_history = spot_client.get_btc_price_history(hours=24)
    except Exception as e:
        price_history = pd.DataFrame()
        errors.append(f"Price history: {e}")

    try:
        iv_data = deribit_client.get_iv_summary()
    except Exception as e:
        iv_data = {"dvol": None, "nearest_atm_iv": None, "error": str(e)}
        errors.append(f"IV: {e}")

    rv_data = vol_model.compute_realized_vol(price_history)

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
    params = market_entry.get("params") or kalshi_client.extract_market_params(market)

    hours = market_entry["hours_to_expiry"]
    threshold = params["threshold"]
    direction = params["direction"] or "above"
    market_prob = params["market_prob"]
    market_type = params.get("market_type", "threshold")

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

    iv_rv_div = (iv - rv) if (iv and rv) else None
    conf = signal_engine.assess_confidence(
        spread=params["spread"],
        volume=params["volume"],
        iv_available=iv is not None,
        rv_available=rv is not None,
        hours_to_expiry=hours,
        iv_rv_divergence=iv_rv_div,
    )

    sig = signal_engine.generate_signal(
        market_prob=market_prob,
        model_prob=model_prob,
        confidence=conf["confidence"],
        spread=params["spread"],
    )

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


# ─── UI Components ────────────────────────────────────────────────────────

def render_header(data: dict):
    """Top-level status bar with key metrics."""
    spot = data["spot"]
    iv = data["iv_data"]
    rv = data["rv_data"]

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if spot.get("price"):
            change = spot.get("change_24h_pct", 0)
            st.metric("BTC Spot", f"${spot['price']:,.0f}",
                      f"{change:+.1f}% 24h" if change else None)
        else:
            st.metric("BTC Spot", "Unavailable")

    with col2:
        dvol = iv.get("dvol")
        st.metric("Implied Vol (DVOL)", f"{dvol:.1f}%" if dvol else "N/A")

    with col3:
        rv_val = rv.get("realized_vol")
        st.metric("Realized Vol (24h)", f"{rv_val:.1f}%" if rv_val else "N/A")

    with col4:
        # vol regime indicator
        dvol_val = iv.get("dvol")
        rv_val = rv.get("realized_vol")
        if dvol_val and rv_val:
            spread = dvol_val - rv_val
            label = "IV > RV" if spread > 2 else ("RV > IV" if spread < -2 else "Aligned")
            st.metric("Vol Regime", label, f"{spread:+.1f}%")
        else:
            st.metric("Vol Regime", "N/A")

    with col5:
        n_markets = len(data["ranked_markets"])
        st.metric("BTC Markets", n_markets)


def render_dashboard(data: dict):
    """Main dashboard tab with market opportunities table and charts."""
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

    # ─── Sidebar filters ──────────────────────────────────────
    with st.sidebar:
        st.subheader("Filters")
        show_type = st.radio("Market Type", ["All", "Threshold", "Range"], horizontal=True)
        min_volume = st.slider("Min Volume", 0, 500, 0, step=10)
        n_display = st.slider("Markets to Show", 5, 30, 15)

    # filter
    filtered = ranked
    if show_type == "Threshold":
        filtered = [r for r in filtered if r["params"].get("market_type") == "threshold"]
    elif show_type == "Range":
        filtered = [r for r in filtered if r["params"].get("market_type") == "range"]
    if min_volume > 0:
        filtered = [r for r in filtered if (r["market"].get("volume") or 0) >= min_volume]

    # analyze
    analyses = []
    for entry in filtered[:n_display]:
        try:
            analyses.append(analyze_market(entry, spot_price, iv, rv))
        except Exception:
            continue

    if not analyses:
        st.warning("No markets match the current filters.")
        return

    # ─── Summary table ────────────────────────────────────────
    st.subheader("Market Opportunities")

    table_data = []
    for a in analyses:
        p = a["params"]
        fv = a["fair_value"]
        sig = a["signal"]
        conf = a["confidence"]
        label = format_market_label(p)

        table_data.append({
            "Contract": label,
            "Type": (p.get("market_type") or "?").title(),
            "Expiry": format_expiry(a["hours_to_expiry"]),
            "Mkt Prob": f"{p['market_prob']:.0%}" if p["market_prob"] is not None else "—",
            "Model Prob": f"{fv.get('model_prob'):.0%}" if fv.get("model_prob") is not None else "—",
            "Edge": f"{sig['raw_edge']:+.1%}" if sig["raw_edge"] is not None else "—",
            "Signal": sig["signal"],
            "Confidence": conf["confidence_label"],
            "Vol": (p.get("volume") or 0),
        })

    df = pd.DataFrame(table_data)

    # style the dataframe
    def style_signal(val):
        colors = {"BUY YES": "#0e4429", "BUY NO": "#4a1419", "NO TRADE": "#2d2d2d"}
        text = {"BUY YES": "#3fb950", "BUY NO": "#f85149", "NO TRADE": "#8b949e"}
        return f"background-color: {colors.get(val, '')}; color: {text.get(val, '')}"

    def style_confidence(val):
        colors = {"HIGH": "#3fb950", "MEDIUM": "#d29922", "LOW": "#f85149"}
        return f"color: {colors.get(val, '')}"

    styled = df.style.map(style_signal, subset=["Signal"]).map(
        style_confidence, subset=["Confidence"]
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=min(len(df) * 38 + 40, 600))

    # ─── Charts row ──────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2)

    # edge bar chart — show markets with actual edge values
    with chart_col1:
        edge_data = [
            (format_market_label(a["params"])[:30], a["signal"]["raw_edge"], a["signal"]["signal"])
            for a in analyses
            if a["signal"]["raw_edge"] is not None and a["params"]["market_prob"] is not None
        ]
        if edge_data:
            render_edge_chart(edge_data)

    # price chart
    with chart_col2:
        if not data["price_history"].empty:
            render_price_chart(data["price_history"])

    # ─── Detailed analysis cards ──────────────────────────────
    # only show actionable markets (those with signals)
    actionable = [a for a in analyses if a["signal"]["signal"] != "NO TRADE"]
    display_list = actionable if actionable else analyses[:5]

    if display_list:
        st.subheader("Analysis Detail")
        for i, a in enumerate(display_list[:8]):
            render_analysis_card(a, expanded=(i == 0))


def render_edge_chart(edge_data: list[tuple]):
    """Bar chart showing edge across markets."""
    labels, edges, signals = zip(*edge_data)

    colors = []
    for sig in signals:
        if sig == "BUY YES":
            colors.append("#3fb950")
        elif sig == "BUY NO":
            colors.append("#f85149")
        else:
            colors.append("#484f58")

    fig = go.Figure(go.Bar(
        x=list(edges),
        y=list(labels),
        orientation="h",
        marker_color=colors,
        text=[f"{e:+.1%}" for e in edges],
        textposition="outside",
        textfont=dict(size=11),
    ))

    # threshold lines
    threshold = config.MIN_EDGE_THRESHOLD
    fig.add_vline(x=threshold, line_dash="dot", line_color="#3fb950", opacity=0.4)
    fig.add_vline(x=-threshold, line_dash="dot", line_color="#f85149", opacity=0.4)
    fig.add_vline(x=0, line_color="#484f58", opacity=0.5)

    fig.update_layout(
        title="Edge by Market",
        height=max(len(edge_data) * 32 + 80, 200),
        margin=dict(l=10, r=60, t=40, b=20),
        xaxis_title="Edge (Model - Market)",
        xaxis_tickformat=".0%",
        template="plotly_dark",
        showlegend=False,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_price_chart(price_history: pd.DataFrame):
    """BTC price chart with area fill."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_history.index,
        y=price_history["price"],
        mode="lines",
        name="BTC/USD",
        line=dict(color="#F7931A", width=2),
        fill="tozeroy",
        fillcolor="rgba(247, 147, 26, 0.08)",
    ))
    fig.update_layout(
        title="BTC Price — Last 24h",
        height=max(200, 300),
        margin=dict(l=10, r=10, t=40, b=20),
        xaxis_title="",
        yaxis_title="USD",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_analysis_card(analysis: dict, expanded: bool = False):
    """Detailed analysis card for a single market."""
    p = analysis["params"]
    sig = analysis["signal"]
    fv = analysis["fair_value"]
    conf = analysis["confidence"]
    label = format_market_label(p)

    icon = {"BUY YES": "🟢", "BUY NO": "🔴", "NO TRADE": "⚪"}.get(sig["signal"], "⚪")
    header = f"{icon} {label} — {sig['signal']} ({format_expiry(analysis['hours_to_expiry'])})"

    with st.expander(header, expanded=expanded):
        # top metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Market Prob",
                      f"{p['market_prob']:.1%}" if p["market_prob"] is not None else "N/A")
        with m2:
            mp = fv.get("model_prob")
            st.metric("Model Prob", f"{mp:.1%}" if mp is not None else "N/A")
        with m3:
            if sig["raw_edge"] is not None:
                st.metric("Raw Edge", f"{sig['raw_edge']:+.1%}")
            else:
                st.metric("Raw Edge", "N/A")
        with m4:
            if sig["adjusted_edge"] is not None:
                st.metric("Adj. Edge", f"{sig['adjusted_edge']:+.1%}")
            else:
                st.metric("Adj. Edge", "N/A")

        # vol breakdown
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            iv_prob = fv.get("iv_fair_prob")
            st.metric("IV Model Prob", f"{iv_prob:.1%}" if iv_prob is not None else "N/A")
        with v2:
            rv_prob = fv.get("rv_fair_prob")
            st.metric("RV Model Prob", f"{rv_prob:.1%}" if rv_prob is not None else "N/A")
        with v3:
            st.metric("Confidence", conf["confidence_label"])
        with v4:
            st.metric("Spread", f"{p['spread']:.1%}" if p["spread"] else "N/A")

        # explanation
        st.markdown(
            f'<div class="explanation-box">{analysis["explanation"]}</div>',
            unsafe_allow_html=True,
        )

        # confidence concerns
        if conf["concerns"]:
            st.caption(f"Concerns: {', '.join(conf['concerns'])}")


def render_market_details(data: dict):
    """Detailed view for a selected market with probability comparison chart."""
    ranked = data["ranked_markets"]
    if not ranked:
        st.info("No markets available for detailed view.")
        return

    spot_price = data["spot"].get("price")
    iv = data["iv_data"].get("dvol")
    rv = data["rv_data"].get("realized_vol")

    # build meaningful selector labels
    selector_labels = []
    for r in ranked[:30]:
        p = r["params"]
        label = format_market_label(p)
        expiry = format_expiry(r["hours_to_expiry"])
        prob = f" | Mkt: {p['market_prob']:.0%}" if p.get("market_prob") is not None else ""
        selector_labels.append(f"{label} ({expiry}{prob})")

    selected_idx = st.selectbox(
        "Select a market to analyze:",
        range(len(selector_labels)),
        format_func=lambda i: selector_labels[i],
    )

    entry = ranked[selected_idx]
    market = entry["market"]

    try:
        analysis = analyze_market(entry, spot_price, iv, rv)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return

    params = analysis["params"]
    fv = analysis["fair_value"]
    sig = analysis["signal"]
    conf = analysis["confidence"]

    # ─── Signal header ────────────────────────────────────────
    sig_col, info_col = st.columns([1, 3])
    with sig_col:
        st.markdown(signal_badge(sig["signal"]), unsafe_allow_html=True)
        st.caption(sig.get("reason", ""))
    with info_col:
        st.markdown(f"**{params.get('event_title', params['title'])}**")
        st.markdown(f"Contract: **{format_market_label(params)}** | "
                    f"Ticker: `{params['ticker']}` | "
                    f"Expires in **{format_expiry(analysis['hours_to_expiry'])}**")

    st.divider()

    # ─── Main content: two columns ────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        # probability comparison chart
        render_probability_chart(params, fv, sig)

        # explanation
        st.markdown("#### Analysis")
        st.markdown(
            f'<div class="explanation-box">{analysis["explanation"]}</div>',
            unsafe_allow_html=True,
        )

    with right:
        # market metadata
        st.markdown("#### Market Data")
        meta = {
            "Type": (params.get("market_type") or "unknown").title(),
            "Direction": params.get("direction", "N/A"),
            "Threshold": f"${params['threshold']:,.0f}" if params["threshold"] else "N/A",
        }
        if params.get("range_low"):
            meta["Range"] = f"${params['range_low']:,.0f} – ${params['range_high']:,.0f}"
        meta.update({
            "Yes Bid": f"{params['yes_bid']}¢" if params["yes_bid"] else "—",
            "Yes Ask": f"{params['yes_ask']}¢" if params["yes_ask"] else "—",
            "Spread": f"{params['spread']:.1%}" if params["spread"] else "—",
            "Volume": f"{params.get('volume', 0):,}",
            "Open Interest": f"{params.get('open_interest', 0):,}" if params.get("open_interest") else "—",
        })

        for k, v in meta.items():
            st.markdown(f"**{k}:** {v}")

        st.markdown("---")

        # vol inputs
        st.markdown("#### Volatility")
        v1, v2 = st.columns(2)
        with v1:
            st.metric("DVOL (IV)", f"{iv:.1f}%" if iv else "N/A")
        with v2:
            st.metric("Realized Vol", f"{rv:.1f}%" if rv else "N/A")

        iv_prob = fv.get("iv_fair_prob")
        rv_prob = fv.get("rv_fair_prob")
        p1, p2 = st.columns(2)
        with p1:
            st.metric("IV Fair Prob", f"{iv_prob:.1%}" if iv_prob is not None else "N/A")
        with p2:
            st.metric("RV Fair Prob", f"{rv_prob:.1%}" if rv_prob is not None else "N/A")

        st.markdown("---")

        # confidence
        st.markdown("#### Confidence")
        st.metric("Score", f"{conf['confidence']:.0%}")
        if conf["concerns"]:
            for c in conf["concerns"]:
                st.caption(f"• {c}")

    # raw data
    with st.expander("Raw Market JSON"):
        st.json(market)


def render_probability_chart(params: dict, fv: dict, sig: dict):
    """Side-by-side comparison of market vs model probability."""
    market_prob = params.get("market_prob")
    model_prob = fv.get("model_prob")
    iv_prob = fv.get("iv_fair_prob")
    rv_prob = fv.get("rv_fair_prob")

    categories = []
    values = []
    colors = []

    if market_prob is not None:
        categories.append("Market")
        values.append(market_prob)
        colors.append("#58a6ff")

    if model_prob is not None:
        categories.append("Model (primary)")
        values.append(model_prob)
        colors.append("#f0883e")

    if iv_prob is not None and iv_prob != model_prob:
        categories.append("IV-based")
        values.append(iv_prob)
        colors.append("#a371f7")

    if rv_prob is not None and rv_prob != model_prob:
        categories.append("RV-based")
        values.append(rv_prob)
        colors.append("#7ee787")

    if not categories:
        return

    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        textfont=dict(size=13),
    ))

    # edge annotation
    if market_prob is not None and model_prob is not None:
        edge = model_prob - market_prob
        edge_color = "#3fb950" if edge > 0 else "#f85149"
        fig.add_annotation(
            text=f"Edge: {edge:+.1%}",
            xref="paper", yref="paper",
            x=0.95, y=0.95,
            showarrow=False,
            font=dict(size=14, color=edge_color),
            bgcolor="#0d1117",
            bordercolor=edge_color,
            borderwidth=1,
            borderpad=6,
        )

    fig.update_layout(
        title="Probability Comparison",
        height=300,
        margin=dict(l=10, r=10, t=40, b=20),
        yaxis_tickformat=".0%",
        yaxis_range=[0, max(values) * 1.25] if values else [0, 1],
        template="plotly_dark",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_methodology():
    """Methodology explanation tab."""
    st.subheader("Methodology")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
### Market Selection

Markets are discovered dynamically from Kalshi's API using the **KXBTCD** (threshold)
and **KXBTC** (range) series. Markets are ranked by a scoring function that prefers:
- Shorter time to expiry
- Threshold-style contracts (easier to model and explain)
- Active markets with volume and pricing
- "Interesting" probability levels (not deep ITM/OTM at 99¢/1¢)

### Data Sources

| Source | Data | Auth |
|--------|------|------|
| **CoinGecko** | BTC spot, 24h price history | Public |
| **Deribit** | DVOL index (30-day IV) | Public |
| **Kalshi** | Market listings, pricing | Public |

### Implied Volatility

The primary IV input is Deribit's **DVOL index**, a 30-day implied volatility measure
for BTC options.

**Caveat:** Using 30-day IV for shorter-horizon events involves a horizon mismatch.
The model scales vol using √t, but this assumes constant vol across horizons —
a known simplification.

### Realized Volatility

Computed from 24h of BTC price data (~5-minute intervals from CoinGecko).
Annualized standard deviation of log returns.
""")

    with col2:
        st.markdown("""
### Fair Value Model

**Threshold markets** ("BTC above $K"):

P(S_T > K) = Φ(−d₂) where d₂ = [ln(K/S₀) − 0.5σ²t] / (σ√t)

**Range markets** ("BTC between $A and $B"):

P(A ≤ S_T < B) = P(S_T > A) − P(S_T > B)

Both use a log-normal assumption with **zero drift** (reasonable for short horizons)
and volatility from either IV or RV.

### Signal Generation

| Step | Logic |
|------|-------|
| Raw edge | model prob − market prob |
| Adjusted edge | raw × confidence − spread cost |
| BUY YES | adj. edge > +5% |
| BUY NO | adj. edge < −5% |
| NO TRADE | otherwise |

### Assumptions & Limitations

- **Log-normal tails**: underestimates extreme BTC moves
- **Constant vol**: breaks down during volatile regimes
- **Zero drift**: ignores momentum, reasonable for short horizons
- **√t scaling**: rough approximation from 30-day IV to shorter periods
- **No microstructure**: ignores order flow and market impact
- **Prototype only**: signals are illustrative, not trading advice
""")


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    st.title("₿ BTC Market Analyzer")
    st.caption("Live comparison of Kalshi BTC market pricing vs. volatility-based fair value estimates")

    with st.spinner("Fetching live data..."):
        data = fetch_all_data()

    if data["errors"]:
        with st.sidebar.expander("Data Warnings", expanded=False):
            for e in data["errors"]:
                st.warning(e)

    render_header(data)
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Dashboard", "Market Details", "Methodology"])

    with tab1:
        render_dashboard(data)

    with tab2:
        render_market_details(data)

    with tab3:
        render_methodology()

    # sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last refresh: {data['timestamp'].strftime('%H:%M:%S UTC')}")
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
