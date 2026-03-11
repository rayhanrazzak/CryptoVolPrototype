"""
BTC Volatility Desk
Institutional-grade dashboard for comparing Kalshi prediction market
pricing against a volatility-based fair value model.
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
    page_title="BTC Vol Desk",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Theme & CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Red+Hat+Display:wght@300;400;500;600;700&family=Azeret+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-deep: #0C0B09;
        --bg-card: #141310;
        --bg-elevated: #1A1916;
        --border: #252320;
        --border-subtle: #1C1B18;
        --text-primary: #E8E4DE;
        --text-secondary: #A09A90;
        --text-muted: #706A60;
        --accent-copper: #C9A87C;
        --accent-teal: #6BA3BE;
        --signal-long: #7AB87A;
        --signal-short: #C75B5B;
        --synthesis: #B08FC7;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        font-family: 'Red Hat Display', sans-serif !important;
        color: var(--text-primary);
    }
    [data-testid="stAppViewContainer"] {
        background: var(--bg-deep);
    }
    [data-testid="stSidebar"] {
        background: var(--bg-card);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * {
        font-family: 'Red Hat Display', sans-serif !important;
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
        background: rgba(122, 184, 122, 0.1);
        color: #7AB87A;
        padding: 5px 14px;
        border-radius: 3px;
        font-family: 'Azeret Mono', monospace;
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        display: inline-block;
        border: 1px solid rgba(122, 184, 122, 0.2);
    }
    .sig-short {
        background: rgba(199, 91, 91, 0.1);
        color: #C75B5B;
        padding: 5px 14px;
        border-radius: 3px;
        font-family: 'Azeret Mono', monospace;
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        display: inline-block;
        border: 1px solid rgba(199, 91, 91, 0.2);
    }
    .sig-flat {
        background: rgba(112, 106, 96, 0.1);
        color: #A09A90;
        padding: 5px 14px;
        border-radius: 3px;
        font-family: 'Azeret Mono', monospace;
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        display: inline-block;
        border: 1px solid rgba(112, 106, 96, 0.15);
    }

    /* Confidence colors */
    .conf-high { color: #7AB87A; }
    .conf-med { color: #C9A87C; }
    .conf-low { color: #C75B5B; }

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
    .liq-good { color: #7AB87A; font-weight: 500; }
    .liq-warn { color: #C9A87C; font-weight: 500; }
    .liq-bad { color: #C75B5B; font-weight: 500; }

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

    /* Apple-style smooth scrolling */
    * { scroll-behavior: smooth; }

    /* Selectbox / input styling */
    [data-testid="stSelectbox"] label {
        font-family: 'Red Hat Display', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 500;
        color: var(--text-secondary) !important;
    }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Plotly chart defaults
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHART_COLORS = {
    "gold": "#D4A76A",
    "cyan": "#6BA3BE",
    "emerald": "#7AB87A",
    "rose": "#C75B5B",
    "purple": "#B08FC7",
    "orange": "#D4956A",
    "slate": "#636366",
    "btc": "#F7931A",
}

EXPIRY_COLORS = ["#6BA3BE", "#7AB87A", "#D4A76A", "#B08FC7", "#D4956A", "#C75B5B"]


def chart_layout(**overrides) -> dict:
    """Base Plotly layout — warm dark, Apple-esque."""
    base = dict(
        paper_bgcolor="#0C0B09",
        plot_bgcolor="#141310",
        font=dict(family="Azeret Mono, monospace", color="#A09A90", size=11),
        margin=dict(l=12, r=12, t=44, b=12),
        xaxis=dict(
            gridcolor="#252320", zerolinecolor="#252320",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            gridcolor="#252320", zerolinecolor="#252320",
            tickfont=dict(size=10),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
        hoverlabel=dict(
            bgcolor="#1A1916", bordercolor="#252320",
            font=dict(family="Azeret Mono", size=11),
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

@st.cache_data(ttl=config.REFRESH_INTERVAL)
def fetch_all_data():
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
        iv_data = {"dvol": None, "vol_surface": {}, "error": str(e)}
        errors.append(f"IV: {e}")

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


def analyze_market(entry, spot_price, iv, rv, vol_surface=None):
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
    )

    return {
        "params": params,
        "fair_value": fv,
        "confidence": conf,
        "signal": sig,
        "explanation": explanation,
        "hours_to_expiry": hours,
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
        xaxis=dict(tickformat="$,.0f", gridcolor="#1E2330"),
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
                    line=dict(width=1.5, color="#0B0D11")),
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
            [0, "#0C0B09"], [0.25, "#2A3540"], [0.5, "#6BA3BE"],
            [0.75, "#D4A76A"], [1, "#C75B5B"]
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
    """Edge vs distance from spot — shows favorite-longshot bias."""
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
            marker=dict(color=color, size=9, line=dict(width=1, color="#0B0D11")),
            text=[p["label"] for p in subset],
            hovertemplate="%{text}<br>Distance: %{x:+.1f}%<br>Edge: %{y:+.1f}%<extra></extra>",
        ))

    # reference lines
    fig.add_hline(y=0, line_color="#1E2330", line_width=1)
    fig.add_vline(x=0, line_color="#1E2330", line_width=1)
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
        title=dict(text="Edge vs Distance from Spot", font=dict(size=14)),
        xaxis_title="Distance from Spot (%)",
        yaxis_title="Edge: Model − Market (%)",
        height=380,
    ))
    st.plotly_chart(fig, use_container_width=True)


def render_prob_curves(analyses, spot):
    """Market probability vs Model probability across all thresholds."""
    pts = []
    for a in analyses:
        p = a["params"]
        fv = a["fair_value"]
        if (p.get("market_type") == "threshold" and p.get("market_prob") is not None
                and fv.get("model_prob") is not None):
            pts.append({
                "threshold": p["threshold"],
                "market": p["market_prob"],
                "model": fv["model_prob"],
                "label": fmt_label(p),
            })

    if len(pts) < 3:
        return

    pts.sort(key=lambda x: x["threshold"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[p["threshold"] for p in pts],
        y=[p["market"] for p in pts],
        mode="lines+markers",
        name="Market",
        line=dict(color=CHART_COLORS["cyan"], width=2),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=[p["threshold"] for p in pts],
        y=[p["model"] for p in pts],
        mode="lines+markers",
        name="Model",
        line=dict(color=CHART_COLORS["gold"], width=2),
        marker=dict(size=5),
    ))

    # shade the edge area
    fig.add_trace(go.Scatter(
        x=[p["threshold"] for p in pts] + [p["threshold"] for p in pts][::-1],
        y=[p["market"] for p in pts] + [p["model"] for p in pts][::-1],
        fill="toself",
        fillcolor="rgba(232, 185, 49, 0.06)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.add_vline(
        x=spot, line_dash="dash", line_color=CHART_COLORS["btc"], line_width=1.5,
        annotation_text="Spot", annotation_position="top",
        annotation_font=dict(color=CHART_COLORS["btc"], size=10),
    )

    fig.update_layout(**chart_layout(
        title=dict(text="Market vs Model Probability", font=dict(size=14)),
        xaxis_title="Threshold ($)", yaxis_title="Probability",
        yaxis_tickformat=".0%",
        xaxis=dict(tickformat="$,.0f", gridcolor="#1E2330"),
        height=380,
    ))
    st.plotly_chart(fig, use_container_width=True)


def render_edge_bars(analyses):
    """Horizontal bar chart of edge by market."""
    edge_data = []
    for a in analyses:
        edge = a["signal"]["raw_edge"]
        if edge is not None and a["params"]["market_prob"] is not None:
            edge_data.append({
                "label": fmt_label(a["params"])[:28],
                "edge": edge,
                "signal": a["signal"]["signal"],
            })

    if not edge_data:
        return

    edge_data.sort(key=lambda x: x["edge"])
    colors = [
        CHART_COLORS["emerald"] if d["signal"] == "BUY YES"
        else CHART_COLORS["rose"] if d["signal"] == "BUY NO"
        else CHART_COLORS["slate"]
        for d in edge_data
    ]

    fig = go.Figure(go.Bar(
        x=[d["edge"] for d in edge_data],
        y=[d["label"] for d in edge_data],
        orientation="h",
        marker_color=colors,
        text=[f"{d['edge']:+.1%}" for d in edge_data],
        textposition="outside",
        textfont=dict(size=10, family="IBM Plex Mono"),
    ))

    thr = config.MIN_EDGE_THRESHOLD
    fig.add_vline(x=thr, line_dash="dot", line_color=CHART_COLORS["emerald"], opacity=0.3)
    fig.add_vline(x=-thr, line_dash="dot", line_color=CHART_COLORS["rose"], opacity=0.3)
    fig.add_vline(x=0, line_color="#1E2330")

    fig.update_layout(**chart_layout(
        title=dict(text="Edge Distribution", font=dict(size=14)),
        height=max(len(edge_data) * 30 + 80, 220),
        xaxis_title="Edge (Model − Market)",
        xaxis_tickformat=".0%",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=12, r=60, t=44, b=12),
    ))
    st.plotly_chart(fig, use_container_width=True)


def render_price_chart(price_history, spot=None, thresholds=None):
    """BTC price chart with optional threshold lines."""
    if price_history.empty:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_history.index, y=price_history["price"],
        mode="lines", name="BTC/USD",
        line=dict(color=CHART_COLORS["btc"], width=2),
        fill="tozeroy",
        fillcolor="rgba(247, 147, 26, 0.04)",
    ))

    if thresholds:
        for thr in thresholds[:8]:
            fig.add_hline(
                y=thr, line_dash="dot", line_color="#38BDF8", opacity=0.2,
                annotation_text=f"${thr/1000:.1f}k",
                annotation_font=dict(size=9, color="#38BDF8"),
                annotation_position="right",
            )

    fig.update_layout(**chart_layout(
        title=dict(text="BTC Price — 24h", font=dict(size=14)),
        height=300,
        yaxis_title="USD",
        yaxis=dict(tickformat="$,.0f", gridcolor="#1E2330"),
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
        textfont=dict(size=12, family="IBM Plex Mono"),
    ))

    if mkt is not None and mdl is not None:
        edge = mdl - mkt
        edge_color = CHART_COLORS["emerald"] if edge > 0 else CHART_COLORS["rose"]
        fig.add_annotation(
            text=f"Edge: {edge:+.1%}",
            xref="paper", yref="paper", x=0.95, y=0.95,
            showarrow=False,
            font=dict(size=13, color=edge_color, family="IBM Plex Mono"),
            bgcolor="#0B0D11", bordercolor=edge_color,
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
        if data["errors"]:
            with st.expander("API Errors"):
                for e in data["errors"]:
                    st.text(e)
        return

    # ── Sidebar filters ────────────────────────────────────
    with st.sidebar:
        st.markdown('<p class="muted-label">Filters</p>', unsafe_allow_html=True)
        show_type = st.radio("Market Type", ["All", "Threshold", "Range"], horizontal=True)
        show_liq = st.radio("Liquidity", ["All", "Two-Sided"], horizontal=True)
        min_vol = st.slider("Min Volume", 0, 500, 0, step=10)
        n_show = st.slider("Display Count", 5, 40, 20)

    # filter
    filtered = ranked
    if show_type == "Threshold":
        filtered = [r for r in filtered if r["params"].get("market_type") == "threshold"]
    elif show_type == "Range":
        filtered = [r for r in filtered if r["params"].get("market_type") == "range"]
    if show_liq == "Two-Sided":
        filtered = [r for r in filtered if r["params"].get("liquidity") == "two_sided"]
    if min_vol > 0:
        filtered = [r for r in filtered if (r["market"].get("volume") or 0) >= min_vol]

    # analyze
    analyses = []
    for entry in filtered[:n_show]:
        try:
            analyses.append(analyze_market(entry, spot_price, iv, rv, vol_surface))
        except Exception:
            continue

    if not analyses:
        st.warning("No markets match the current filters.")
        return

    # ── LLM overview ─────────────────────────────────────
    if llm_explainer.is_available():
        overview = llm_explainer.synthesize_overview(analyses, spot_price, data["iv_data"], data["rv_data"])
        if overview:
            st.markdown(f'<div class="synthesis-card">{overview}</div>', unsafe_allow_html=True)

    # ── Market table ────────────────────────────────────────
    st.markdown('<p class="section-title">Market Opportunities</p>', unsafe_allow_html=True)

    table_rows = []
    for a in analyses:
        p = a["params"]
        fv = a["fair_value"]
        sig = a["signal"]
        conf = a["confidence"]
        iv_src = fv.get("iv_source", "—")
        iv_short = {"strike_matched": "Matched", "dvol_fallback": "DVOL"}.get(iv_src, "—")

        table_rows.append({
            "Contract": fmt_label(p),
            "Expiry": fmt_expiry(a["hours_to_expiry"]),
            "Mkt": fmt_pct(p["market_prob"], 0),
            "Model": fmt_pct(fv.get("model_prob"), 0),
            "Edge": f"{sig['raw_edge']:+.1%}" if sig["raw_edge"] is not None else "—",
            "Signal": sig["signal"],
            "IV Src": iv_short,
            "Conf": conf["confidence_label"],
            "Liq": (p.get("liquidity") or "?").replace("_", " ").title(),
            "Vol": p.get("volume") or 0,
        })

    df = pd.DataFrame(table_rows)

    def _style_sig(val):
        if val == "BUY YES":
            return "background-color: rgba(122,184,122,0.1); color: #7AB87A; font-weight: 600"
        elif val == "BUY NO":
            return "background-color: rgba(199,91,91,0.1); color: #C75B5B; font-weight: 600"
        return "color: #636366"

    def _style_conf(val):
        return {"HIGH": "color:#7AB87A", "MEDIUM": "color:#C9A87C", "LOW": "color:#C75B5B"}.get(val, "")

    styled = df.style.map(_style_sig, subset=["Signal"]).map(_style_conf, subset=["Conf"])
    st.dataframe(styled, use_container_width=True, hide_index=True,
                 height=min(len(df) * 36 + 42, 620))

    # ── Charts row 1: Edge + Prob curves ─────────────────
    c1, c2 = st.columns(2)
    with c1:
        render_edge_bars(analyses)
    with c2:
        if spot_price:
            render_prob_curves(analyses, spot_price)

    # ── Charts row 2: Price + Edge scatter ───────────────
    c3, c4 = st.columns(2)
    with c3:
        thresholds = [a["params"]["threshold"] for a in analyses
                      if a["params"].get("threshold") and a["params"].get("market_type") == "threshold"]
        render_price_chart(data["price_history"], spot_price, thresholds[:6])
    with c4:
        if spot_price:
            render_edge_scatter(analyses, spot_price)

    # ── Actionable trades ─────────────────────────────────
    actionable = [a for a in analyses if a["signal"]["signal"] != "NO TRADE"]
    show_list = actionable if actionable else analyses[:4]

    if show_list:
        st.markdown('<p class="section-title">Trade Detail</p>', unsafe_allow_html=True)
        for i, a in enumerate(show_list[:6]):
            _render_trade_card(a, expanded=(i == 0))


def _render_trade_card(analysis, expanded=False):
    p = analysis["params"]
    sig = analysis["signal"]
    fv = analysis["fair_value"]
    conf = analysis["confidence"]
    label = fmt_label(p)

    icon = {"BUY YES": "▲", "BUY NO": "▼", "NO TRADE": "—"}.get(sig["signal"], "—")
    header = f"{icon} {label} — {sig['signal']} ({fmt_expiry(analysis['hours_to_expiry'])})"

    with st.expander(header, expanded=expanded):
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Market", fmt_pct(p["market_prob"]))
        with m2:
            st.metric("Model", fmt_pct(fv.get("model_prob")))
        with m3:
            st.metric("Edge", f"{sig['raw_edge']:+.1%}" if sig["raw_edge"] is not None else "—")
        with m4:
            st.metric("Adj Edge", f"{sig['adjusted_edge']:+.1%}" if sig["adjusted_edge"] is not None else "—")
        with m5:
            st.metric("Conf", conf["confidence_label"])

        st.markdown(f'<div class="analysis-box">{analysis["explanation"]}</div>', unsafe_allow_html=True)

        if conf["concerns"]:
            st.caption(f"Concerns: {', '.join(conf['concerns'])}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab: Vol Analytics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_vol_analytics(data):
    spot_price = data["spot"].get("price")
    vol_surface = data["iv_data"].get("vol_surface")
    iv = data["iv_data"].get("dvol")
    rv = data["rv_data"].get("realized_vol")

    if not vol_surface or not vol_surface.get("surface"):
        st.warning("No Deribit vol surface data available.")
        return

    n_opts = vol_surface.get("raw_count", 0)
    n_expiries = len(vol_surface.get("expiries", []))
    underlying = vol_surface.get("underlying", 0)

    # ── Surface summary ─────────────────────────────────
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.metric("Options Loaded", f"{n_opts:,}")
    with s2:
        st.metric("Expiries", n_expiries)
    with s3:
        st.metric("DVOL (30d)", fmt_vol(iv))
    with s4:
        st.metric("Realized Vol", fmt_vol(rv))
    with s5:
        if iv and rv and rv > 0:
            ratio = iv / rv
            st.metric("IV/RV Ratio", f"{ratio:.2f}")
        else:
            st.metric("IV/RV Ratio", "—")

    st.divider()

    # ── Row 1: Smile + Term Structure ─────────────────────
    c1, c2 = st.columns([3, 2])
    with c1:
        if spot_price:
            render_vol_smile(vol_surface, spot_price)
    with c2:
        if spot_price:
            render_term_structure(vol_surface, spot_price)

    # ── Row 2: Heatmap + Skew Metrics ────────────────────
    c3, c4 = st.columns([3, 2])
    with c3:
        if spot_price:
            render_vol_heatmap(vol_surface, spot_price)
    with c4:
        if spot_price:
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

    # selector
    labels = []
    for r in ranked[:30]:
        p = r["params"]
        label = fmt_label(p)
        exp = fmt_expiry(r["hours_to_expiry"])
        prob = f" | {p['market_prob']:.0%}" if p.get("market_prob") is not None else ""
        liq_warn = " ⚠" if p.get("liquidity") != "two_sided" else ""
        labels.append(f"{label} ({exp}{prob}){liq_warn}")

    idx = st.selectbox("Select market:", range(len(labels)), format_func=lambda i: labels[i])
    entry = ranked[idx]

    try:
        analysis = analyze_market(entry, spot_price, iv, rv, vol_surface)
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
        # ── Market Data ───────────────────
        st.markdown('<p class="muted-label">Market Data</p>', unsafe_allow_html=True)
        meta_items = [
            ("Type", (params.get("market_type") or "?").title()),
            ("Direction", params.get("direction", "—")),
            ("Threshold", f"${params['threshold']:,.0f}" if params["threshold"] else "—"),
            ("Liquidity", (params.get("liquidity") or "unknown").replace("_", " ").title()),
        ]
        if params.get("range_low"):
            meta_items.append(("Range", f"${params['range_low']:,.0f} – ${params['range_high']:,.0f}"))
        meta_items.extend([
            ("Bid / Ask", f"{params['yes_bid'] or '—'}¢ / {params['yes_ask'] or '—'}¢"),
            ("Spread", f"{params['spread']:.1%}" if params["spread"] else "—"),
            ("Volume", f"{params.get('volume', 0):,}"),
            ("Open Interest", f"{params.get('open_interest', 0):,}" if params.get("open_interest") else "—"),
        ])
        for k, v in meta_items:
            st.markdown(f"**{k}:** {v}")

        st.divider()

        # ── Volatility ───────────────────
        st.markdown('<p class="muted-label">Volatility</p>', unsafe_allow_html=True)
        matched_iv = fv.get("strike_matched_iv")
        if matched_iv:
            st.metric("Strike-Matched IV", f"{matched_iv:.1f}%")
            method = fv.get("iv_detail", {}).get("method", "")
            if method:
                st.caption(f"Method: {method}")

        v1, v2 = st.columns(2)
        with v1:
            st.metric("DVOL", fmt_vol(iv))
        with v2:
            st.metric("RV (24h)", fmt_vol(rv))

        regime = fv.get("vol_regime")
        if regime:
            regime_name = {"vol_expansion": "Expansion", "vol_compression": "Compression", "neutral": "Neutral"}.get(regime, regime)
            bw = fv.get("blend_weights", {})
            st.metric("Regime", regime_name)
            if bw:
                st.caption(f"Blend: {bw.get('iv', 0):.0%} IV / {bw.get('rv', 0):.0%} RV")

        st.divider()

        # ── Probability Breakdown ─────────
        st.markdown('<p class="muted-label">Probability Breakdown</p>', unsafe_allow_html=True)
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

        st.divider()

        # ── Confidence ────────────────────
        st.markdown('<p class="muted-label">Confidence</p>', unsafe_allow_html=True)
        st.metric("Score", f"{conf['confidence']:.0%}")
        if conf["concerns"]:
            for c in conf["concerns"]:
                st.caption(f"• {c}")

    with st.expander("Raw Market JSON"):
        st.json(entry["market"])


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
| Raw edge | model prob − market prob |
| Adjusted edge | raw × confidence − spread cost |
| BUY YES | adj. edge > +5% |
| BUY NO | adj. edge < −5% |
| NO TRADE | otherwise |

Confidence reduced by: illiquid quotes, wide spreads, low volume,
missing data, short expiry, large IV-RV divergence.

### Trade Synthesis (Optional)

When a `GEMINI_API_KEY` is configured, the app generates concise, data-rich
trade notes using Google Gemini. The LLM receives structured quantitative
context (edge, vol inputs, skew ratios, tail adjustments) and produces
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

def render_header(data):
    spot = data["spot"]
    iv_data = data["iv_data"]
    rv_data = data["rv_data"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        if spot.get("price"):
            change = spot.get("change_24h_pct", 0)
            st.metric("BTC Spot", f"${spot['price']:,.0f}",
                      f"{change:+.1f}%" if change else None)
        else:
            st.metric("BTC Spot", "—")

    with c2:
        dvol = iv_data.get("dvol")
        st.metric("DVOL (30d)", fmt_vol(dvol))

    with c3:
        rv = rv_data.get("realized_vol")
        st.metric("Realized Vol", fmt_vol(rv))

    with c4:
        dvol = iv_data.get("dvol")
        rv = rv_data.get("realized_vol")
        if dvol and rv and rv > 0:
            ratio = dvol / rv
            label = "Expansion" if ratio > 1.3 else ("Compression" if ratio < 0.7 else "Neutral")
            st.metric("Regime", label, f"{ratio:.2f}")
        else:
            st.metric("Regime", "—")

    with c5:
        n_mkts = len(data["ranked_markets"])
        st.metric("Markets", n_mkts)

    with c6:
        vs = data["iv_data"].get("vol_surface", {})
        n_opts = vs.get("raw_count", 0)
        st.metric("Options", f"{n_opts:,}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    # title
    st.markdown(
        '<h1 style="font-family: \'Playfair Display\', serif; font-weight: 700; '
        'letter-spacing: -0.02em; margin-bottom: 0; font-size: 2rem;">'
        '<span class="gold">₿</span> BTC Vol Desk</h1>',
        unsafe_allow_html=True,
    )
    st.caption("Live Kalshi market analysis · Deribit vol surface · Strike-matched IV · Fat-tail model")

    with st.spinner("Loading live data..."):
        data = fetch_all_data()

    if data["errors"]:
        with st.sidebar.expander("Data Warnings", expanded=False):
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

    # sidebar footer
    st.sidebar.markdown("---")
    if llm_explainer.is_available():
        st.sidebar.markdown('<span class="liq-good">● Gemini Connected</span>', unsafe_allow_html=True)
    else:
        st.sidebar.caption("LLM synthesis: not configured")
        st.sidebar.caption("Set GEMINI_API_KEY for trade notes")

    st.sidebar.caption(f"Last refresh: {data['timestamp'].strftime('%H:%M:%S UTC')}")
    if st.sidebar.button("Refresh"):
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
