# BTC Volatility Market Analyzer

A live prototype that compares Kalshi BTC prediction market probabilities against a volatility-based fair value model. Uses strike-matched implied volatility from Deribit's options chain and realized volatility from recent BTC price action to identify potential mispricings in short-dated BTC event markets.

## Key Features

- **Live Kalshi market discovery** -- dynamically finds KXBTCD (threshold) and KXBTC (range) BTC markets, ranked by expiry, liquidity, and interpretability
- **Strike-matched implied volatility** -- pulls ~900 BTC options from Deribit and matches IV by strike and tenor, capturing skew and term structure
- **Realized volatility** -- computes 24-hour annualized vol from recent BTC spot returns via CoinGecko
- **IV-RV regime blending** -- blends implied and realized vol based on their ratio as a vol regime signal
- **Fat-tail adjustment** -- Student-t probability (df=5) instead of normal CDF for more realistic BTC tail risk
- **Liquidity-aware pricing** -- flags one-sided quotes and applies confidence penalties for illiquid markets
- **Vol surface analytics** -- interactive smile, term structure, heatmap, and skew metrics
- **Mock paper-trade signals** -- BUY YES / BUY NO / NO TRADE based on edge, spread cost, and confidence
- **Optional trade synthesis** -- concise trade notes via Google Gemini (free tier) when configured, deterministic fallback otherwise

## Architecture

```
app.py                  # Streamlit dashboard (4 tabs)
config.py               # Model parameters, blending weights, tail distribution settings
data/
  kalshi_client.py      # Kalshi market discovery, pricing, and liquidity classification
  deribit_client.py     # Vol surface builder from full options chain + DVOL index
  spot_client.py        # BTC spot price and recent price history via CoinGecko
models/
  vol_model.py          # Strike-matched IV lookup, Student-t probability, IV-RV blending
  signal_engine.py      # Edge-based signal generation with liquidity-aware confidence
  explainer.py          # Deterministic natural-language explanation engine
  llm_explainer.py      # Optional Gemini-powered trade synthesis
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy the example environment file and fill in values if needed:

```bash
cp .env.example .env
```

All core functionality works with public endpoints and requires no API keys.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `KALSHI_API_KEY` | No | Kalshi API key (public endpoints used by default) |
| `KALSHI_PRIVATE_KEY_PATH` | No | Path to Kalshi private key file |
| `GEMINI_API_KEY` | No | Google Gemini API key for trade synthesis (free tier) |

## How to Run

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

## Dashboard Tabs

1. **Trading Desk** -- ranked market opportunities, edge distribution, probability curves, BTC price with threshold overlays, actionable trade detail
2. **Vol Analytics** -- interactive volatility smile showing put skew, ATM term structure, vol surface heatmap, skew metrics by tenor
3. **Deep Dive** -- single-market analysis with probability comparison, vol breakdown, regime classification, and optional trade synthesis
4. **Methodology** -- full model documentation

## Methodology

### Volatility Surface

The model builds a full implied volatility surface from Deribit's BTC options chain (~900 active options across ~11 expiries). For each Kalshi contract, it looks up the IV at the closest strike and tenor, using **total variance interpolation** (sigma^2 * t) between expiries to avoid calendar arbitrage. This captures both **skew** (OTM puts trade at higher IV than ATM due to crash risk) and **term structure** (short-dated vol differs from 30-day vol).

Falls back to the DVOL index (30-day IV) when no close match exists.

### Fair Value Model

For threshold markets ("BTC above $K"), the model estimates:

- **P(S_T > K)** using a Student-t CDF (df=5) applied to the standard d2 statistic
- Zero drift assumption (reasonable for short horizons)
- The t-distribution adds probability mass to the tails vs. normal, which matters for BTC

For range markets, the probability of landing in [A, B) is computed as the difference of two threshold probabilities.

### IV-RV Blending

When both IV and RV are available, they are blended based on their ratio:
- **IV/RV > 1.3** (expansion): weight IV 70% -- options market expects higher vol
- **IV/RV < 0.7** (compression): weight IV 30% -- recent moves exceed priced vol
- **Otherwise** (neutral): weight IV 55%

### Signal Generation

- Edge = model probability - market probability
- Adjusted for confidence penalties (liquidity, spread, data quality) and spread cost
- Minimum 5% adjusted edge to trigger BUY YES or BUY NO
- Signals are mock/paper-trade only

## Assumptions and Limitations

- **Student-t tails** improve on log-normal but df=5 is still an approximation of BTC's true tail behavior
- **Constant vol per horizon**: the vol surface helps, but within-horizon vol clustering is not modeled
- **Zero drift**: ignores momentum, reasonable for short horizons
- **No jump diffusion**: macro events, ETF flows, and regulatory news can cause discrete gaps
- **No microstructure**: order flow, slippage, and market impact are not modeled
- **Prototype only**: signals are illustrative and not trading advice

## Demo Flow

1. Open the dashboard (`streamlit run app.py`)
2. **Trading Desk** shows ranked BTC markets with edge, signal, and confidence
3. Use sidebar filters to focus on threshold vs range markets, two-sided liquidity
4. Charts show edge distribution, market vs model probability curves, and BTC price with threshold overlays
5. **Vol Analytics** reveals the options skew, term structure, and full surface heatmap
6. **Deep Dive** into a specific contract for vol breakdown, regime analysis, and optional trade synthesis
7. **Methodology** tab documents the full approach
