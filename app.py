"""
Volatility Arbitrage Terminal
Dashboard for comparing Kalshi prediction market pricing against
an options-implied fair value model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
from collections import defaultdict

from data import kalshi_client, deribit_client, spot_client
from models import vol_model, signal_engine, explainer, llm_explainer
import config


st.set_page_config(
    page_title="Crypto Vol Arb",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Theme & CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Red+Hat+Display:wght@300;400;500;600;700&family=Azeret+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-deep: #F8F7F5;
        --bg-card: #FFFFFF;
        --bg-elevated: #F0EFEC;
        --border: #E2E0DB;
        --border-subtle: #ECEAE6;
        --text-primary: #1A1814;
        --text-secondary: #5C5850;
        --text-muted: #9C9890;
        --accent-copper: #9A7B4F;
        --accent-teal: #3A8BA8;
        --signal-long: #2D8F4E;
        --signal-short: #B83B36;
        --synthesis: #7E5EA8;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        font-family: 'Red Hat Display', sans-serif !important;
        color: var(--text-primary);
    }
    [data-testid="stAppViewContainer"] {
        background: var(--bg-deep);
    }
    /* Hide sidebar entirely */
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
    button[kind="header"] {
        display: none !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }
    code, pre, .stCode, [data-testid="stMetricValue"] {
        font-family: 'Azeret Mono', monospace !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] {
        font-family: 'Red Hat Display', sans-serif !important;
        font-size: 0.68rem !important;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted) !important;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Azeret Mono', monospace !important;
        font-size: 1.35rem !important;
        font-weight: 600;
        color: var(--text-primary) !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'Azeret Mono', monospace !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        border-bottom: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Red Hat Display', sans-serif !important;
        font-weight: 600;
        font-size: 0.82rem;
        letter-spacing: 0.05em;
        padding: 10px 24px;
        color: var(--text-muted);
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-copper) !important;
        border-bottom: 2px solid var(--accent-copper) !important;
        background: transparent !important;
    }

    /* Signal badges */
    .sig-long {
        background: rgba(45, 143, 78, 0.08);
        color: #2D8F4E;
        padding: 5px 14px;
        border-radius: 3px;
        font-family: 'Azeret Mono', monospace;
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        display: inline-block;
        border: 1px solid rgba(45, 143, 78, 0.18);
    }
    .sig-short {
        background: rgba(184, 59, 54, 0.08);
        color: #B83B36;
        padding: 5px 14px;
        border-radius: 3px;
        font-family: 'Azeret Mono', monospace;
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        display: inline-block;
        border: 1px solid rgba(184, 59, 54, 0.18);
    }
    .sig-flat {
        background: rgba(156, 152, 144, 0.1);
        color: #8C8880;
        padding: 5px 14px;
        border-radius: 3px;
        font-family: 'Azeret Mono', monospace;
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        display: inline-block;
        border: 1px solid rgba(156, 152, 144, 0.15);
    }

    /* Confidence colors */
    .conf-high { color: #2D8F4E; }
    .conf-med { color: #9A7B4F; }
    .conf-low { color: #B83B36; }

    /* Synthesis card */
    .synthesis-card {
        background: var(--bg-card);
        border-left: 3px solid var(--synthesis);
        padding: 16px 20px;
        border-radius: 0 6px 6px 0;
        margin: 12px 0;
        font-size: 0.88rem;
        line-height: 1.7;
        color: var(--text-secondary);
    }
    .synthesis-card strong {
        color: var(--text-primary);
    }

    /* Analysis card */
    .analysis-box {
        background: var(--bg-card);
        border-left: 3px solid var(--accent-teal);
        padding: 14px 18px;
        border-radius: 0 6px 6px 0;
        margin: 8px 0;
        font-size: 0.86rem;
        line-height: 1.65;
        color: var(--text-secondary);
    }

    /* Overview card */
    .overview-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 18px 22px;
        margin: 8px 0;
    }

    /* Liquidity badge */
    .liq-good { color: #2D8F4E; font-weight: 500; }
    .liq-warn { color: #9A7B4F; font-weight: 500; }
    .liq-bad { color: #B83B36; font-weight: 500; }

    /* Muted label */
    .muted-label {
        font-family: 'Instrument Sans', 'Red Hat Display', sans-serif;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 4px;
    }

    /* Accent text */
    .gold { color: var(--accent-copper); }
    .cyan { color: var(--accent-teal); }

    /* Divider override */
    hr {
        border-color: var(--border) !important;
        margin: 16px 0 !important;
        opacity: 0.6;
    }

    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    /* Expander — Apple-like rounded card */
    details {
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        background: var(--bg-card) !important;
    }
    details summary {
        font-family: 'Red Hat Display', sans-serif !important;
        font-weight: 500;
    }

    /* Section title */
    .section-title {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: var(--text-primary);
        margin: 24px 0 12px 0;
        letter-spacing: -0.01em;
    }

    /* Smooth scrolling */
    * { scroll-behavior: smooth; }

    /* ── Buttons ─────────────────────────────────────── */
    .stButton > button {
        font-family: 'Red Hat Display', sans-serif !important;
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 0.03em;
        background: var(--bg-elevated);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 8px 20px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: var(--accent-copper);
        color: #FFFFFF;
        border-color: var(--accent-copper);
    }
    .stButton > button:active {
        transform: scale(0.97);
    }

    /* ── Radio buttons (pill-style segmented control) ── */
    div[role="radiogroup"] {
        gap: 0 !important;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 3px;
        display: inline-flex !important;
    }
    div[role="radiogroup"] label {
        font-family: 'Red Hat Display', sans-serif !important;
        font-size: 0.78rem !important;
        font-weight: 500;
        color: var(--text-muted) !important;
        border-radius: 8px;
        padding: 6px 16px !important;
        margin: 0 !important;
        transition: all 0.2s ease;
        border: none !important;
        background: transparent !important;
    }
    div[role="radiogroup"] label[data-checked="true"],
    div[role="radiogroup"] label:has(input:checked) {
        background: var(--accent-copper) !important;
        color: #FFFFFF !important;
        font-weight: 600;
    }
    /* Hide the actual radio circle */
    div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    /* ── Slider ──────────────────────────────────────── */
    [data-testid="stSlider"] label {
        font-family: 'Red Hat Display', sans-serif !important;
        font-size: 0.78rem !important;
        font-weight: 500;
        color: var(--text-muted) !important;
    }
    [data-testid="stSlider"] [role="slider"] {
        background: var(--accent-copper) !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.12);
        width: 18px !important;
        height: 18px !important;
    }
    [data-testid="stSlider"] [data-testid="stTickBar"] {
        display: none;
    }
    /* Slider track */
    [data-testid="stSlider"] div[role="slider"] ~ div {
        background: var(--border) !important;
    }

    /* ── Selectbox ───────────────────────────────────── */
    [data-testid="stSelectbox"] label {
        font-family: 'Red Hat Display', sans-serif !important;
        font-size: 0.78rem !important;
        font-weight: 500;
        color: var(--text-muted) !important;
    }
    [data-testid="stSelectbox"] > div > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        font-family: 'Red Hat Display', sans-serif !important;
        font-size: 0.85rem !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stSelectbox"] svg {
        color: var(--text-muted) !important;
    }

    /* ── Spinner ─────────────────────────────────────── */
    .stSpinner > div > div {
        border-top-color: var(--accent-copper) !important;
    }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Plotly chart defaults
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHART_COLORS = {
    "gold": "#9A7B4F",
    "cyan": "#2E7D96",
    "emerald": "#2D8F4E",
    "rose": "#B83B36",
    "purple": "#7E5EA8",
    "orange": "#C07D3A",
    "slate": "#8C8880",
    "btc": "#E8860F",
}

EXPIRY_COLORS = ["#2E7D96", "#2D8F4E", "#9A7B4F", "#7E5EA8", "#C07D3A", "#B83B36"]


def chart_layout(**overrides) -> dict:
    """Base Plotly layout — light, Apple-esque."""
    base = dict(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAF8",
        font=dict(family="Azeret Mono, monospace", color="#5C5850", size=11),
        margin=dict(l=12, r=12, t=44, b=12),
        xaxis=dict(
            gridcolor="#ECEAE6", zerolinecolor="#E2E0DB",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            gridcolor="#ECEAE6", zerolinecolor="#E2E0DB",
            tickfont=dict(size=10),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0)", bordercolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF", bordercolor="#E2E0DB",
            font=dict(family="Azeret Mono", size=11, color="#1A1814"),
        ),
    )
    base.update(overrides)
    return base


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def sig_badge(signal: str) -> str:
    cls = {"BUY YES": "sig-long", "BUY NO": "sig-short"}.get(signal, "sig-flat")
    return f'<span class="{cls}">{signal}</span>'


def conf_badge(label: str) -> str:
    cls = {"HIGH": "conf-high", "MEDIUM": "conf-med", "LOW": "conf-low"}.get(label, "")
    return f'<span class="{cls}">{label}</span>'


def liq_badge(liq: str) -> str:
    if liq == "two_sided":
        return '<span class="liq-good">Two-Sided</span>'
    elif liq == "one_sided":
        return '<span class="liq-warn">One-Sided</span>'
    return f'<span class="liq-bad">{liq or "Unknown"}</span>'


def fmt_label(params: dict) -> str:
    sub = params.get("subtitle", "")
    if sub:
        return sub
    if params.get("threshold"):
        d = params.get("direction", "above")
        return f"${params['threshold']:,.0f} or {d}"
    return params.get("title", "Unknown")[:50]


def fmt_expiry(hours: float) -> str:
    if hours < 1:
        return f"{hours * 60:.0f}m"
    elif hours < 48:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def fmt_pct(val, decimals=1) -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}%}"


def fmt_vol(val) -> str:
    if val is None:
        return "—"
    return f"{val:.1f}%"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data fetching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(ttl=config.REFRESH_INTERVAL, show_spinner=False)
def fetch_all_data():
    errors = []

    try:
        iv_data = deribit_client.get_iv_summary()
    except Exception as e:
        iv_data = {"dvol": None, "vol_surface": {}, "error": str(e)}
        errors.append(f"IV: {e}")

    # Use Deribit underlying as primary spot — real-time index price
    # CoinGecko free tier can be delayed or rate limited
    deribit_px = (iv_data.get("vol_surface") or {}).get("underlying")
    try:
        cg_spot = spot_client.get_btc_spot()
    except Exception:
        cg_spot = {"price": None}

    if deribit_px and deribit_px > 0:
        spot = {"price": deribit_px, "source": "deribit"}
        # include CoinGecko 24h change if available
        if cg_spot.get("change_24h_pct"):
            spot["change_24h_pct"] = cg_spot["change_24h_pct"]
    elif cg_spot.get("price"):
        spot = cg_spot
    else:
        spot = {"price": None}
        errors.append("No spot price available from Deribit or CoinGecko")

    try:
        price_history = spot_client.get_btc_price_history(hours=24)
    except Exception as e:
        price_history = pd.DataFrame()
        errors.append(f"Price history: {e}")

    rv_data = vol_model.compute_realized_vol(price_history)

    try:
        raw_markets = kalshi_client.discover_btc_markets()
        ranked = kalshi_client.rank_btc_markets(raw_markets)
    except Exception as e:
        raw_markets, ranked = [], []
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


def analyze_market(entry, spot_price, iv, rv, vol_surface=None, forward=None):
    market = entry["market"]
    params = entry.get("params") or kalshi_client.extract_market_params(market)
    hours = entry["hours_to_expiry"]
    threshold = params["threshold"]
    direction = params["direction"] or "above"
    market_prob = params["market_prob"]
    market_type = params.get("market_type", "threshold")

    if threshold and spot_price:
        fv = vol_model.compute_fair_value(
            spot=spot_price, threshold=threshold,
            hours_to_expiry=hours, direction=direction,
            implied_vol=iv, realized_vol=rv,
            market_type=market_type,
            range_low=params.get("range_low"),
            range_high=params.get("range_high"),
            vol_surface=vol_surface,
            forward=forward,
        )
    else:
        fv = {"model_prob": None, "iv_fair_prob": None, "rv_fair_prob": None}

    model_prob = fv.get("model_prob")
    best_iv = fv.get("strike_matched_iv") or fv.get("dvol") or iv
    iv_rv_div = (best_iv - rv) if (best_iv and rv) else None

    conf = signal_engine.assess_confidence(
        spread=params["spread"], volume=params["volume"],
        iv_available=best_iv is not None, rv_available=rv is not None,
        hours_to_expiry=hours, iv_rv_divergence=iv_rv_div,
        liquidity=params.get("liquidity"),
    )

    sig = signal_engine.generate_signal(
        market_prob=market_prob, model_prob=model_prob,
        confidence=conf["confidence"], spread=params["spread"],
    )

    explanation = explainer.explain_market(
        market_title=params["title"], market_prob=market_prob,
        model_prob=model_prob, signal=sig["signal"],
        implied_vol=iv, realized_vol=rv, hours_to_expiry=hours,
        confidence_label=conf["confidence_label"], concerns=conf["concerns"],
        spot=spot_price, threshold=threshold, direction=direction,
        spread=params["spread"], raw_edge=sig["raw_edge"],
        iv_source=fv.get("iv_source"), vol_regime=fv.get("vol_regime"),
        tail_adjustment=fv.get("iv_tail_adjustment"),
        strike_matched_iv=fv.get("strike_matched_iv"),
        liquidity=params.get("liquidity"),
        forward=forward,
    )

    return {
        "params": params,
        "fair_value": fv,
        "confidence": conf,
        "signal": sig,
        "explanation": explanation,
        "hours_to_expiry": hours,
        "expiry": entry.get("expiry"),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart renderers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_vol_smile(vol_surface, spot):
    """IV vs Strike for nearest expiries — reveals the put skew."""
    if not vol_surface or not vol_surface.get("surface"):
        st.caption("No vol surface data available.")
        return

    expiries = vol_surface["expiries"][:5]
    surface = vol_surface["surface"]
    fig = go.Figure()

    # filter strikes to ±35% of spot
    lo, hi = spot * 0.65, spot * 1.35

    for i, exp in enumerate(expiries):
        opts = surface.get(exp["label"], [])
        # deduplicate by strike (average put/call IVs)
        strike_ivs = defaultdict(list)
        for o in opts:
            if lo <= o["strike"] <= hi:
                strike_ivs[o["strike"]].append(o["iv"])
        if not strike_ivs:
            continue

        strikes = sorted(strike_ivs.keys())
        ivs = [np.mean(strike_ivs[s]) for s in strikes]
        hours = exp["hours"]
        label = f"{exp['label']} ({fmt_expiry(hours)})"

        fig.add_trace(go.Scatter(
            x=strikes, y=ivs,
            mode="lines+markers",
            name=label,
            line=dict(color=EXPIRY_COLORS[i % len(EXPIRY_COLORS)], width=2),
            marker=dict(size=3),
            hovertemplate="Strike: $%{x:,.0f}<br>IV: %{y:.1f}%<extra>" + label + "</extra>",
        ))

    # spot line
    fig.add_vline(
        x=spot, line_dash="dash", line_color=CHART_COLORS["btc"], line_width=1.5,
        annotation_text=f"Spot ${spot:,.0f}", annotation_position="top right",
        annotation_font=dict(color=CHART_COLORS["btc"], size=10),
    )

    fig.update_layout(**chart_layout(
        title=dict(text="Volatility Smile", font=dict(size=14)),
        xaxis_title="Strike ($)", yaxis_title="Implied Volatility (%)",
        height=380,
        xaxis=dict(tickformat="$,.0f", gridcolor="#ECEAE6"),
    ))
    st.plotly_chart(fig, use_container_width=True)


def render_term_structure(vol_surface, spot):
    """ATM implied volatility across expiry tenors."""
    if not vol_surface or not vol_surface.get("expiries"):
        return

    expiries = vol_surface["expiries"]
    surface = vol_surface["surface"]

    hours_list, ivs = [], []
    for exp in expiries:
        opts = surface.get(exp["label"], [])
        # find nearest strike to spot
        if not opts:
            continue
        nearest = min(opts, key=lambda o: abs(o["strike"] - spot))
        hours_list.append(exp["hours"])
        ivs.append(nearest["iv"])

    if len(hours_list) < 2:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours_list, y=ivs,
        mode="lines+markers",
        line=dict(color=CHART_COLORS["gold"], width=2.5),
        marker=dict(size=7, color=CHART_COLORS["gold"],
                    line=dict(width=1.5, color="#FFFFFF")),
        hovertemplate="Tenor: %{x:.0f}h<br>ATM IV: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(**chart_layout(
        title=dict(text="ATM Term Structure", font=dict(size=14)),
        xaxis_title="Hours to Expiry", yaxis_title="ATM IV (%)",
        height=340,
        xaxis=dict(type="log"),
    ))
    st.plotly_chart(fig, use_container_width=True)


def render_vol_heatmap(vol_surface, spot):
    """2D heatmap: Strike x Tenor colored by IV."""
    if not vol_surface or not vol_surface.get("surface"):
        return

    expiries = vol_surface["expiries"][:8]
    surface = vol_surface["surface"]
    lo, hi = spot * 0.75, spot * 1.25

    # collect all unique strikes in range
    all_strikes = set()
    for exp in expiries:
        for o in surface.get(exp["label"], []):
            if lo <= o["strike"] <= hi:
                all_strikes.add(o["strike"])

    if not all_strikes:
        return

    strikes = sorted(all_strikes)
    tenor_labels = [f"{exp['label']}\n({fmt_expiry(exp['hours'])})" for exp in expiries]

    # build IV matrix
    z = []
    for s in strikes:
        row = []
        for exp in expiries:
            opts = surface.get(exp["label"], [])
            # find closest strike
            closest = min(opts, key=lambda o: abs(o["strike"] - s)) if opts else None
            if closest and abs(closest["strike"] - s) < spot * 0.02:
                row.append(closest["iv"])
            else:
                row.append(None)
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=tenor_labels,
        y=[f"${s/1000:.0f}k" for s in strikes],
        colorscale=[
            [0, "#F0EFEC"], [0.25, "#C8DDE6"], [0.5, "#6BA3BE"],
            [0.75, "#C9A87C"], [1, "#B83B36"]
        ],
        colorbar=dict(title=dict(text="IV %", font=dict(size=10))),
        hovertemplate="Strike: %{y}<br>Expiry: %{x}<br>IV: %{z:.1f}%<extra></extra>",
    ))

    fig.update_layout(**chart_layout(
        title=dict(text="Volatility Surface", font=dict(size=14)),
        height=420,
        xaxis_title="Expiry", yaxis_title="Strike",
    ))
    st.plotly_chart(fig, use_container_width=True)


def render_edge_scatter(analyses, spot):
    """Discrepancy vs distance from spot — shows favorite-longshot bias."""
    pts = []
    for a in analyses:
        threshold = a["params"].get("threshold")
        edge = a["signal"].get("raw_edge")
        if threshold and edge is not None and spot and a["params"].get("market_type") == "threshold":
            dist = (threshold - spot) / spot * 100
            pts.append({
                "dist": dist, "edge": edge * 100,
                "signal": a["signal"]["signal"],
                "label": fmt_label(a["params"]),
                "mkt": a["params"].get("market_prob", 0),
            })

    if len(pts) < 3:
        return

    fig = go.Figure()

    for sig_type, color, name in [
        ("BUY YES", CHART_COLORS["emerald"], "BUY YES"),
        ("BUY NO", CHART_COLORS["rose"], "BUY NO"),
        ("NO TRADE", CHART_COLORS["slate"], "NO TRADE"),
    ]:
        subset = [p for p in pts if p["signal"] == sig_type]
        if not subset:
            continue
        fig.add_trace(go.Scatter(
            x=[p["dist"] for p in subset],
            y=[p["edge"] for p in subset],
            mode="markers",
            name=name,
            marker=dict(color=color, size=9, line=dict(width=1, color="#FFFFFF")),
            text=[p["label"] for p in subset],
            hovertemplate="%{text}<br>Distance: %{x:+.1f}%<br>Discrepancy: %{y:+.1f}%<extra></extra>",
        ))

    # reference lines
    fig.add_hline(y=0, line_color="#E2E0DB", line_width=1)
    fig.add_vline(x=0, line_color="#E2E0DB", line_width=1)
    fig.add_hline(y=5, line_dash="dot", line_color=CHART_COLORS["emerald"], opacity=0.3)
    fig.add_hline(y=-5, line_dash="dot", line_color=CHART_COLORS["rose"], opacity=0.3)

    # annotation for the pattern
    fig.add_annotation(
        text="Below spot: model sees<br>lower prob than market",
        x=-5, y=-10, showarrow=False,
        font=dict(size=9, color=CHART_COLORS["slate"]),
        xanchor="center",
    )
    fig.add_annotation(
        text="Above spot: model sees<br>higher prob than market",
        x=5, y=15, showarrow=False,
        font=dict(size=9, color=CHART_COLORS["slate"]),
        xanchor="center",
    )

    fig.update_layout(**chart_layout(
        title=dict(text="Discrepancy vs Distance from Spot", font=dict(size=14)),
        xaxis_title="Distance from Spot (%)",
        yaxis_title="Discrepancy: Model − Market (%)",
        height=380,
    ))
    st.plotly_chart(fig, use_container_width=True)





def render_price_chart(price_history, spot=None, threshold=None):
    """BTC price chart auto-scaled to actual price range."""
    if price_history.empty:
        return

    prices = price_history["price"]
    y_min = prices.min()
    y_max = prices.max()
    y_pad = (y_max - y_min) * 0.15 or 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_history.index, y=prices,
        mode="lines", name="BTC/USD",
        line=dict(color=CHART_COLORS["btc"], width=2),
    ))

    # show a single threshold if provided (the contract being analyzed)
    if threshold and y_min - y_pad < threshold < y_max + y_pad:
        fig.add_hline(
            y=threshold, line_dash="dot", line_color=CHART_COLORS["cyan"], opacity=0.5,
            annotation_text=f"${threshold:,.0f}",
            annotation_font=dict(size=10, color=CHART_COLORS["cyan"]),
            annotation_position="right",
        )

    fig.update_layout(**chart_layout(
        title=dict(text="BTC — 24h", font=dict(size=14)),
        height=260,
        yaxis=dict(tickformat="$,.0f", gridcolor="#ECEAE6",
                   range=[y_min - y_pad, y_max + y_pad]),
    ))
    st.plotly_chart(fig, use_container_width=True)


def render_prob_comparison_detail(params, fv, sig):
    """Probability comparison bar chart for single market detail."""
    entries = []
    colors = []

    mkt = params.get("market_prob")
    mdl = fv.get("model_prob")
    iv_p = fv.get("iv_fair_prob")
    rv_p = fv.get("rv_fair_prob")

    if mkt is not None:
        entries.append(("Market", mkt))
        colors.append(CHART_COLORS["cyan"])
    if mdl is not None:
        entries.append(("Model", mdl))
        colors.append(CHART_COLORS["gold"])
    if iv_p is not None and iv_p != mdl:
        entries.append(("IV-Only", iv_p))
        colors.append(CHART_COLORS["purple"])
    if rv_p is not None and rv_p != mdl:
        entries.append(("RV-Only", rv_p))
        colors.append(CHART_COLORS["emerald"])

    if not entries:
        return

    names, vals = zip(*entries)
    fig = go.Figure(go.Bar(
        x=list(names), y=list(vals),
        marker_color=colors,
        text=[f"{v:.1%}" for v in vals],
        textposition="outside",
        textfont=dict(size=12, family="Azeret Mono"),
    ))

    if mkt is not None and mdl is not None:
        edge = mdl - mkt
        edge_color = CHART_COLORS["emerald"] if edge > 0 else CHART_COLORS["rose"]
        fig.add_annotation(
            text=f"Discrepancy: {edge:+.1%}",
            xref="paper", yref="paper", x=0.95, y=0.95,
            showarrow=False,
            font=dict(size=13, color=edge_color, family="Azeret Mono"),
            bgcolor="#FFFFFF", bordercolor=edge_color,
            borderwidth=1, borderpad=6,
        )

    fig.update_layout(**chart_layout(
        title=dict(text="Probability Comparison", font=dict(size=14)),
        height=300,
        yaxis_tickformat=".0%",
        yaxis_range=[0, max(vals) * 1.3] if vals else [0, 1],
    ))
    st.plotly_chart(fig, use_container_width=True)


def render_skew_metrics(vol_surface, spot):
    """Table showing skew metrics by tenor."""
    if not vol_surface or not vol_surface.get("expiries"):
        return

    rows = []
    for exp in vol_surface["expiries"][:8]:
        opts = vol_surface["surface"].get(exp["label"], [])
        if not opts:
            continue

        # find ATM and 25-delta-ish strikes
        atm = min(opts, key=lambda o: abs(o["strike"] - spot))
        put_25d = min(opts, key=lambda o: abs(o["strike"] - spot * 0.93))
        call_25d = min(opts, key=lambda o: abs(o["strike"] - spot * 1.07))

        atm_iv = atm["iv"]
        put_iv = put_25d["iv"]
        call_iv = call_25d["iv"]

        rows.append({
            "Expiry": exp["label"],
            "Tenor": fmt_expiry(exp["hours"]),
            "ATM IV": f"{atm_iv:.1f}%",
            "25d Put IV": f"{put_iv:.1f}%",
            "25d Call IV": f"{call_iv:.1f}%",
            "Put/ATM": f"{put_iv/atm_iv:.2f}x" if atm_iv > 0 else "—",
            "RR": f"{call_iv - put_iv:+.1f}%",
        })

    if rows:
        st.markdown('<p class="section-title">Skew Metrics by Tenor</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab: Trading Desk
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_trading_desk(data):
    spot_price = data["spot"].get("price")
    iv = data["iv_data"].get("dvol")
    rv = data["rv_data"].get("realized_vol")
    vol_surface = data["iv_data"].get("vol_surface")
    ranked = data["ranked_markets"]

    if not ranked:
        st.warning("No active BTC markets found on Kalshi.")
        return

    # only threshold markets — range contracts aren't reliably available
    filtered = [r for r in ranked if r["params"].get("market_type") == "threshold"]

    # group by expiry to compute per-expiry forwards
    _expiry_groups = {}
    for entry in filtered:
        h = entry["hours_to_expiry"]
        matched = False
        for bucket_h in _expiry_groups:
            if abs(h - bucket_h) < 1.0:
                _expiry_groups[bucket_h].append(entry)
                matched = True
                break
        if not matched:
            _expiry_groups[h] = [entry]

    # compute forward per expiry bucket
    _forwards = {}
    for bucket_h, entries in _expiry_groups.items():
        fwd = _estimate_market_forward(entries)
        _forwards[bucket_h] = fwd

    def _get_forward(hours):
        for bucket_h, fwd in _forwards.items():
            if abs(hours - bucket_h) < 1.0:
                return fwd
        return None

    # analyze all available
    analyses = []
    for entry in filtered:
        try:
            fwd = _get_forward(entry["hours_to_expiry"])
            analyses.append(analyze_market(entry, spot_price, iv, rv, vol_surface, forward=fwd))
        except Exception:
            continue

    if not analyses:
        st.warning("No markets match the current filters.")
        return

    # ── Hero chart: Options Model vs Prediction Market ────
    all_threshold = [
        a for a in analyses
        if a["params"].get("market_type") == "threshold"
        and a["params"].get("direction") == "above"
        and a["params"].get("market_prob") is not None
        and a["fair_value"].get("model_prob") is not None
    ]

    # group by expiry bucket (contracts within 1hr of each other)
    expiry_buckets = {}
    for a in all_threshold:
        h = a["hours_to_expiry"]
        matched = False
        for bucket_h in expiry_buckets:
            if abs(h - bucket_h) < 1.0:
                expiry_buckets[bucket_h].append(a)
                matched = True
                break
        if not matched:
            expiry_buckets[h] = [a]

    if expiry_buckets:
        bucket_keys = sorted(expiry_buckets.keys())
        # show expiry in Eastern time (Kalshi's reference timezone)
        from zoneinfo import ZoneInfo
        _et = ZoneInfo("America/New_York")
        def _expiry_label(bucket_analyses):
            exp = bucket_analyses[0].get("expiry")
            if exp:
                et = exp.astimezone(_et)
                return et.strftime("%b %-d %-I:%M%p ET").replace("AM", "am").replace("PM", "pm")
            return fmt_expiry(bucket_analyses[0]["hours_to_expiry"])
        expiry_labels = [_expiry_label(expiry_buckets[k]) for k in bucket_keys]
        sel_exp = st.radio("Expiry", expiry_labels, horizontal=True)
        sel_bucket = bucket_keys[expiry_labels.index(sel_exp)]
        # deduplicate by threshold within the bucket
        _seen = {}
        for a in expiry_buckets[sel_bucket]:
            t = a["params"]["threshold"]
            if t not in _seen:
                _seen[t] = a
        threshold_analyses = sorted(_seen.values(), key=lambda a: a["params"]["threshold"])
    else:
        threshold_analyses = []

    if len(threshold_analyses) >= 2 and spot_price:
        chart_forward = _estimate_market_forward(threshold_analyses)
        _render_hero_chart(threshold_analyses, spot_price, forward=chart_forward)

        # LLM analysis of the chart data (cached in session to avoid repeat API calls)
        if llm_explainer.is_available():
            cache_key = f"chart_llm_{sel_exp}" if 'sel_exp' in dir() else "chart_llm"
            if cache_key not in st.session_state:
                st.session_state[cache_key] = llm_explainer.synthesize_chart_analysis(
                    threshold_analyses, spot_price,
                    data["iv_data"], data["rv_data"],
                    forward=chart_forward,
                )
            chart_note = st.session_state[cache_key]
            if chart_note:
                st.markdown(f'<div class="synthesis-card">{chart_note}</div>', unsafe_allow_html=True)

    # ── Contract selector ──────────────────────────────────
    contract_labels = [
        f"{fmt_label(a['params'])} · {fmt_expiry(a['hours_to_expiry'])} · "
        f"Δ {a['signal']['raw_edge']:+.1%}" if a["signal"]["raw_edge"] is not None
        else f"{fmt_label(a['params'])} · {fmt_expiry(a['hours_to_expiry'])}"
        for a in analyses
    ]

    st.markdown('<p class="section-title">Select Contract</p>', unsafe_allow_html=True)
    sel_idx = st.selectbox(
        "Contract", range(len(contract_labels)),
        format_func=lambda i: contract_labels[i],
        label_visibility="collapsed",
    )

    selected = analyses[sel_idx]
    _render_contract_detail(selected)

    # ── Discrepancy scatter in expander ─────────────────────
    if spot_price and len(threshold_analyses) >= 3:
        with st.expander("Discrepancy vs Distance from Spot"):
            render_edge_scatter(analyses, spot_price)

    # ── Full table in expander ─────────────────────────────
    with st.expander("All Contracts"):
        _render_contract_table(analyses)


def _render_hero_chart(threshold_analyses, spot, forward=None):
    """Full-width probability curve — the core visualization."""
    pts = threshold_analyses

    fig = go.Figure()

    # Kalshi (prediction market)
    fig.add_trace(go.Scatter(
        x=[a["params"]["threshold"] for a in pts],
        y=[a["params"]["market_prob"] for a in pts],
        mode="lines+markers",
        name="Kalshi (Prediction Mkt)",
        line=dict(color=CHART_COLORS["cyan"], width=2.5),
        marker=dict(size=7),
        hovertemplate="$%{x:,.0f}<br>Kalshi: %{y:.1%}<extra></extra>",
    ))

    # Options model
    fig.add_trace(go.Scatter(
        x=[a["params"]["threshold"] for a in pts],
        y=[a["fair_value"]["model_prob"] for a in pts],
        mode="lines+markers",
        name="Options Model",
        line=dict(color=CHART_COLORS["gold"], width=2.5),
        marker=dict(size=7),
        hovertemplate="$%{x:,.0f}<br>Model: %{y:.1%}<extra></extra>",
    ))

    # shade the edge between curves
    fig.add_trace(go.Scatter(
        x=[a["params"]["threshold"] for a in pts] + [a["params"]["threshold"] for a in pts][::-1],
        y=[a["params"]["market_prob"] for a in pts] + [a["fair_value"]["model_prob"] for a in pts][::-1],
        fill="toself",
        fillcolor="rgba(154, 123, 79, 0.07)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Kalshi forward reference (what the model is centered on)
    ref_price = forward if forward else spot
    ref_label = f"Kalshi Fwd ${ref_price:,.0f}" if forward else f"BTC Spot ${spot:,.0f}"
    fig.add_vline(
        x=ref_price, line_dash="dash", line_color=CHART_COLORS["btc"], line_width=1.5,
        annotation_text=ref_label, annotation_position="top",
        annotation_font=dict(color=CHART_COLORS["btc"], size=10),
    )

    fig.update_layout(**chart_layout(
        title=dict(
            text="Options Market vs Prediction Market",
            font=dict(size=16, family="Playfair Display, serif"),
        ),
        xaxis_title="BTC Threshold ($)",
        yaxis_title="P(BTC Above Threshold)",
        yaxis_tickformat=".0%",
        xaxis=dict(tickformat="$,.0f", gridcolor="#ECEAE6"),
        hovermode="x unified",
        height=440,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0)", bordercolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
    ))
    st.plotly_chart(fig, use_container_width=True)



def _render_contract_detail(analysis):
    """Compact detail card for the selected contract."""
    p = analysis["params"]
    fv = analysis["fair_value"]
    sig = analysis["signal"]
    conf = analysis["confidence"]

    # metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Market", fmt_pct(p["market_prob"]))
    with m2:
        st.metric("Model", fmt_pct(fv.get("model_prob")))
    with m3:
        st.metric("Discrepancy", f"{sig['raw_edge']:+.1%}" if sig["raw_edge"] is not None else "—")
    with m4:
        matched_iv = fv.get("strike_matched_iv")
        st.metric("Strike IV", f"{matched_iv:.1f}%" if matched_iv else "—")
    with m5:
        st.metric("Signal", sig["signal"])

    # explanation
    st.markdown(f'<div class="analysis-box">{analysis["explanation"]}</div>', unsafe_allow_html=True)

    if conf["concerns"]:
        st.caption(f"Concerns: {', '.join(conf['concerns'])}")


def _render_contract_table(analyses):
    """Compact table of all analyzed contracts."""
    table_rows = []
    for a in analyses:
        p = a["params"]
        fv = a["fair_value"]
        sig = a["signal"]
        conf = a["confidence"]
        table_rows.append({
            "Contract": fmt_label(p),
            "Expiry": fmt_expiry(a["hours_to_expiry"]),
            "Mkt": fmt_pct(p["market_prob"], 0),
            "Model": fmt_pct(fv.get("model_prob"), 0),
            "Discrepancy": f"{sig['raw_edge']:+.1%}" if sig["raw_edge"] is not None else "—",
            "Signal": sig["signal"],
        })

    df = pd.DataFrame(table_rows)

    def _style_sig(val):
        if val == "BUY YES":
            return "background-color: rgba(45,143,78,0.08); color: #2D8F4E; font-weight: 600"
        elif val == "BUY NO":
            return "background-color: rgba(184,59,54,0.08); color: #B83B36; font-weight: 600"
        return "color: #8C8880"

    styled = df.style.map(_style_sig, subset=["Signal"])
    st.dataframe(styled, use_container_width=True, hide_index=True,
                 height=min(len(df) * 36 + 42, 520))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab: Vol Analytics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_vol_analytics(data):
    spot_price = data["spot"].get("price")
    vol_surface = data["iv_data"].get("vol_surface")

    if not vol_surface or not vol_surface.get("surface"):
        st.warning("No Deribit vol surface data available.")
        return

    if not spot_price:
        st.warning("No spot price available.")
        return

    # ── Smile: the hero chart (full width) ────────────────
    render_vol_smile(vol_surface, spot_price)

    # ── Term Structure + Skew Metrics ─────────────────────
    c1, c2 = st.columns([3, 2])
    with c1:
        render_term_structure(vol_surface, spot_price)
    with c2:
        render_skew_metrics(vol_surface, spot_price)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab: Deep Dive
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_deep_dive(data):
    ranked = data["ranked_markets"]
    if not ranked:
        st.info("No markets available.")
        return

    spot_price = data["spot"].get("price")
    iv = data["iv_data"].get("dvol")
    rv = data["rv_data"].get("realized_vol")
    vol_surface = data["iv_data"].get("vol_surface")

    # pre-analyze to get discrepancies, then sort by largest
    # group by expiry for per-expiry forwards
    _dd_expiry_groups = {}
    for entry in ranked:
        h = entry["hours_to_expiry"]
        matched = False
        for bh in _dd_expiry_groups:
            if abs(h - bh) < 1.0:
                _dd_expiry_groups[bh].append(entry)
                matched = True
                break
        if not matched:
            _dd_expiry_groups[h] = [entry]

    _dd_forwards = {bh: _estimate_market_forward(entries) for bh, entries in _dd_expiry_groups.items()}

    def _dd_fwd(hours):
        for bh, fwd in _dd_forwards.items():
            if abs(hours - bh) < 1.0:
                return fwd
        return None

    pre_analyzed = []
    for r in ranked:
        try:
            fwd = _dd_fwd(r["hours_to_expiry"])
            a = analyze_market(r, spot_price, iv, rv, vol_surface, forward=fwd)
            edge = a["signal"].get("raw_edge")
            pre_analyzed.append((r, a, abs(edge) if edge is not None else 0))
        except Exception:
            continue

    # sort by absolute discrepancy descending
    pre_analyzed.sort(key=lambda x: x[2], reverse=True)

    labels = []
    for r, a, abs_edge in pre_analyzed:
        p = r["params"]
        label = fmt_label(p)
        exp = fmt_expiry(r["hours_to_expiry"])
        edge = a["signal"].get("raw_edge")
        edge_str = f" | Δ {edge:+.1%}" if edge is not None else ""
        liq_warn = " ⚠" if p.get("liquidity") != "two_sided" else ""
        labels.append(f"{label} ({exp}{edge_str}){liq_warn}")

    idx = st.selectbox("Select market:", range(len(labels)), format_func=lambda i: labels[i])
    entry, analysis, _ = pre_analyzed[idx]

    try:
        pass  # analysis already computed above
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return

    params = analysis["params"]
    fv = analysis["fair_value"]
    sig = analysis["signal"]
    conf = analysis["confidence"]

    # ── Header ────────────────────────────────────────────
    h1, h2 = st.columns([1, 4])
    with h1:
        st.markdown(sig_badge(sig["signal"]), unsafe_allow_html=True)
    with h2:
        st.markdown(f"**{params.get('event_title', params['title'])}**")
        liq_note = f" | {liq_badge(params.get('liquidity'))}" if params.get("liquidity") != "two_sided" else ""
        st.markdown(
            f"`{params['ticker']}` · {fmt_label(params)} · "
            f"Expires in **{fmt_expiry(analysis['hours_to_expiry'])}**{liq_note}",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Main layout ───────────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        render_prob_comparison_detail(params, fv, sig)

        # LLM synthesis
        if llm_explainer.is_available():
            synthesis = llm_explainer.synthesize_trade(
                analysis, spot_price, data["iv_data"], data["rv_data"]
            )
            if synthesis:
                st.markdown('<p class="muted-label">Trade Synthesis</p>', unsafe_allow_html=True)
                st.markdown(f'<div class="synthesis-card">{synthesis}</div>', unsafe_allow_html=True)

        # deterministic analysis
        st.markdown('<p class="muted-label">Analysis</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="analysis-box">{analysis["explanation"]}</div>', unsafe_allow_html=True)

    with right:
        # ── Pricing ──────────────────────
        st.markdown('<p class="muted-label">Pricing</p>', unsafe_allow_html=True)
        st.markdown(
            f"**Bid / Ask:** {params['yes_bid'] or '—'}¢ / {params['yes_ask'] or '—'}¢ · "
            f"**Spread:** {params['spread']:.1%}" if params["spread"] else ""
        )
        st.markdown(
            f"**Liquidity:** {(params.get('liquidity') or '?').replace('_', ' ').title()} · "
            f"**Volume:** {params.get('volume', 0):,}"
        )

        st.divider()

        # ── Volatility ──────────────────
        st.markdown('<p class="muted-label">Volatility</p>', unsafe_allow_html=True)
        matched_iv = fv.get("strike_matched_iv")
        v1, v2, v3 = st.columns(3)
        with v1:
            st.metric("Strike IV", f"{matched_iv:.1f}%" if matched_iv else "—")
        with v2:
            st.metric("DVOL", fmt_vol(iv))
        with v3:
            st.metric("RV (24h)", fmt_vol(rv))

        regime = fv.get("vol_regime")
        if regime:
            regime_name = {"vol_expansion": "Expansion", "vol_compression": "Compression", "neutral": "Neutral"}.get(regime, regime)
            bw = fv.get("blend_weights", {})
            blend_str = f" ({bw.get('iv',0):.0%} IV / {bw.get('rv',0):.0%} RV)" if bw else ""
            st.caption(f"Regime: {regime_name}{blend_str}")

        st.divider()

        # ── Model Breakdown ──────────────
        st.markdown('<p class="muted-label">Model</p>', unsafe_allow_html=True)
        prob_items = []
        if fv.get("iv_fair_prob") is not None:
            prob_items.append(("IV-Based", fmt_pct(fv["iv_fair_prob"])))
        if fv.get("rv_fair_prob") is not None:
            prob_items.append(("RV-Based", fmt_pct(fv["rv_fair_prob"])))
        if fv.get("blended_fair_prob") is not None:
            prob_items.append(("**Blended**", f"**{fmt_pct(fv['blended_fair_prob'])}**"))
        tail = fv.get("iv_tail_adjustment")
        if tail is not None and abs(tail) > 0.001:
            prob_items.append(("Tail Adj.", f"{tail:+.1%}"))
        for label, val in prob_items:
            st.markdown(f"{label}: {val}")

        if conf["concerns"]:
            st.caption("⚠ " + ", ".join(conf["concerns"]))

        st.divider()

        # ── Price context ────────────────
        render_price_chart(data["price_history"], spot_price, params.get("threshold"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab: Methodology
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_methodology():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
### Market Selection

Markets are discovered dynamically from Kalshi's API using the **KXBTCD** (threshold)
and **KXBTC** (range) series. Ranked by a scoring function that prefers:
- Shorter time to expiry
- Threshold-style contracts (cleaner to model)
- Two-sided quotes (bid and ask both present)
- Active volume and meaningful probability levels

Markets with only one-sided quotes (zero bid) are flagged with reduced confidence —
the midpoint of 0/ask is not a meaningful probability estimate.

### Data Sources

| Source | Data | Auth |
|--------|------|------|
| **CoinGecko** | BTC spot, 24h price history | Public |
| **Deribit** | ~900 BTC options, DVOL index | Public |
| **Kalshi** | Market listings, pricing | Public |

### Implied Volatility

The model uses **strike-matched, tenor-interpolated IV** from Deribit's full options
chain. For each Kalshi threshold, the system finds the Deribit option with the
closest strike and expiry, capturing both **skew** (OTM puts trade at higher IV)
and **term structure** (short-dated vol differs from 30-day vol).

Interpolation between expiry tenors uses **total variance space** (σ²t)
to avoid calendar spread arbitrage artifacts.

Falls back to the **DVOL index** (30-day IV) when no close match exists.

### Realized Volatility

24h of BTC price data at ~5-minute intervals (CoinGecko).
Annualized standard deviation of log returns.

### IV-RV Blending

| Regime | Condition | IV Weight | Interpretation |
|--------|-----------|-----------|----------------|
| Expansion | IV/RV > 1.3 | 70% | Market expects more vol |
| Neutral | 0.7–1.3 | 55% | Aligned, slight IV tilt |
| Compression | IV/RV < 0.7 | 30% | Recent moves exceed priced vol |
""")

    with col2:
        st.markdown("""
### Fair Value Model

**Threshold markets** ("BTC above $K"):

`P(S_T > K)` using a **Student-t CDF** (df=5):

`d₂ = [ln(K/S₀) − 0.5σ²t] / (σ√t)`

The t-distribution captures BTC's fat tails — large moves happen more often
than a Gaussian predicts. df=5 is a standard pragmatic choice for crypto.

**Range markets** ("BTC between $A and $B"):

`P(A ≤ S_T < B) = P(S_T > A) − P(S_T > B)`

Zero drift assumed (reasonable for short horizons). Best available vol input
used: strike-matched IV > DVOL > RV.

### Signal Generation

| Step | Logic |
|------|-------|
| Raw discrepancy | model prob − market prob |
| Adjusted discrepancy | raw × confidence − spread cost |
| BUY YES | adj. discrepancy > +5% |
| BUY NO | adj. discrepancy < −5% |
| NO TRADE | otherwise |

Confidence reduced by: illiquid quotes, wide spreads, low volume,
missing data, short expiry, large IV-RV divergence.

### Trade Synthesis (Optional)

When a `GEMINI_API_KEY` is configured, the app generates concise, data-rich
trade notes using Google Gemini. The LLM receives structured quantitative
context (discrepancy, vol inputs, skew ratios, tail adjustments) and produces
institutional-style analysis. Falls back to the deterministic engine
when unconfigured.

### Assumptions & Limitations

- **Student-t df=5** approximates BTC's tail behavior but varies over time
- **Constant vol per horizon**: surface lookup helps but within-horizon
  clustering not modeled
- **Zero drift**: ignores momentum, reasonable for short horizons
- **No jump diffusion**: macro events create discrete gaps
- **No microstructure**: order flow, slippage, market impact not modeled
- **Prototype only**: signals are illustrative, not trading advice
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Header
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _estimate_market_forward(ranked_markets) -> float | None:
    """Find the Kalshi market's implied BTC forward price (where prob ≈ 50%)."""
    pts = []
    for entry in ranked_markets:
        p = entry.get("params", {})
        if (p.get("market_type") == "threshold" and p.get("direction") == "above"
                and p.get("market_prob") is not None and p.get("threshold")):
            pts.append((p["threshold"], p["market_prob"]))
    if len(pts) < 3:
        return None
    pts.sort(key=lambda x: x[0])
    # find where probability crosses 50% (linear interpolation)
    for i in range(len(pts) - 1):
        t1, p1 = pts[i]
        t2, p2 = pts[i + 1]
        if (p1 >= 0.5 and p2 <= 0.5) or (p1 <= 0.5 and p2 >= 0.5):
            if abs(p1 - p2) < 0.001:
                return (t1 + t2) / 2
            w = (0.5 - p2) / (p1 - p2)
            return t1 * w + t2 * (1 - w)
    return None


def render_header(data):
    spot = data["spot"]
    iv_data = data["iv_data"]
    rv_data = data["rv_data"]

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        if spot.get("price"):
            change = spot.get("change_24h_pct", 0)
            st.metric("BTC Spot", f"${spot['price']:,.0f}",
                      f"{change:+.1f}%" if change else None)
        else:
            st.metric("BTC Spot", "—")

    with c2:
        # use nearest expiry for the header forward
        ranked = data["ranked_markets"]
        threshold_markets = [r for r in ranked if r["params"].get("market_type") == "threshold"]
        if threshold_markets:
            nearest_h = min(r["hours_to_expiry"] for r in threshold_markets)
            nearest_bucket = [r for r in threshold_markets if abs(r["hours_to_expiry"] - nearest_h) < 1.0]
            fwd = _estimate_market_forward(nearest_bucket)
            # get expiry label
            from zoneinfo import ZoneInfo
            _et = ZoneInfo("America/New_York")
            exp_dt = nearest_bucket[0].get("expiry")
            exp_label = exp_dt.astimezone(_et).strftime("%-I%p ET").replace("AM", "am").replace("PM", "pm") if exp_dt else ""
        else:
            fwd = None
            exp_label = ""
        spot_px = spot.get("price")
        if fwd and spot_px:
            drift_pct = (fwd - spot_px) / spot_px * 100
            st.metric(f"Kalshi Fwd ({exp_label})", f"${fwd:,.0f}", f"{drift_pct:+.1f}%")
        else:
            st.metric("Kalshi Fwd", "—")

    with c3:
        dvol = iv_data.get("dvol")
        st.metric("DVOL (30d)", fmt_vol(dvol))

    with c4:
        rv = rv_data.get("realized_vol")
        st.metric("Realized Vol", fmt_vol(rv))

    with c5:
        dvol = iv_data.get("dvol")
        rv = rv_data.get("realized_vol")
        if dvol and rv and rv > 0:
            ratio = dvol / rv
            label = "Expansion" if ratio > 1.3 else ("Compression" if ratio < 0.7 else "Neutral")
            st.metric("Regime", label, f"{ratio:.2f}")
        else:
            st.metric("Regime", "—")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    # title
    st.markdown(
        '<h1 style="font-family: \'Playfair Display\', serif; font-weight: 700; '
        'letter-spacing: -0.02em; margin-bottom: 0; font-size: 2rem;">'
        'Crypto Vol Arb</h1>',
        unsafe_allow_html=True,
    )
    st.caption("Live Kalshi market analysis · Deribit vol surface · Strike-matched IV")

    with st.spinner("Loading live data..."):
        data = fetch_all_data()

    if data["errors"]:
        with st.expander("Data Warnings", expanded=False):
            for e in data["errors"]:
                st.warning(e)

    render_header(data)
    st.divider()

    t1, t2, t3, t4 = st.tabs(["Trading Desk", "Vol Analytics", "Deep Dive", "Methodology"])

    with t1:
        render_trading_desk(data)
    with t2:
        render_vol_analytics(data)
    with t3:
        render_deep_dive(data)
    with t4:
        render_methodology()

    # footer — refresh + status
    st.divider()
    fc1, fc2, fc3 = st.columns([1, 1, 1])
    with fc1:
        st.caption(f"Last refresh: {data['timestamp'].strftime('%H:%M:%S UTC')}")
    with fc2:
        if llm_explainer.is_available():
            st.caption("Gemini connected")
    with fc3:
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    main()
