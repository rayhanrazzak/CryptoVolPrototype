# BTC Volatility Market Analyzer

**[Live Demo](https://cryptovolprototype.streamlit.app/)**

A live prototype that compares Kalshi BTC prediction market probabilities against a volatility-based fair value model. Uses strike-matched implied volatility from Deribit's options chain and realized volatility from recent BTC price action to surface discrepancies in short-dated BTC event markets.

## Key Features

- **Live Kalshi market discovery** -- dynamically finds KXBTCD threshold BTC markets, grouped by expiry with Eastern time labels
- **Forward-centered model** -- centers the probability distribution on the Kalshi forward price (market-implied expected BTC price) rather than spot, isolating vol-driven discrepancies from directional disagreement
- **Strike-matched implied volatility** -- pulls ~900 BTC options from Deribit and matches IV by strike and tenor, capturing skew and term structure
- **Realized volatility** -- computes 24-hour annualized vol from recent BTC spot returns
- **IV-RV regime blending** -- blends implied and realized vol based on their ratio as a vol regime signal
- **Liquidity-aware pricing** -- flags one-sided quotes and applies confidence penalties for illiquid markets
- **Vol surface analytics** -- interactive smile, term structure, and skew metrics (including 25-delta risk reversals)
- **Mock paper-trade signals** -- BUY YES / BUY NO / NO TRADE based on discrepancy, spread cost, and confidence
- **LLM chart analysis** -- optional Gemini-powered analysis of the probability curves, cached per expiry to minimize API calls
- **API response caching** -- disk-based cache with TTL to handle rate limits and support offline demos

## Architecture

```
app.py                  # Streamlit dashboard (4 tabs)
config.py               # Model parameters, blending weights, distribution settings
data/
  cache.py              # Disk-based API response caching with TTL
  kalshi_client.py      # Kalshi market discovery, pricing, and liquidity classification
  deribit_client.py     # Vol surface builder from full options chain + DVOL index
  spot_client.py        # BTC spot price and recent price history
models/
  vol_model.py          # Strike-matched IV lookup, forward-centered probability estimation
  signal_engine.py      # Discrepancy-based signal generation with confidence adjustments
  explainer.py          # Deterministic natural-language explanation engine
  llm_explainer.py      # Optional Gemini-powered chart and trade analysis
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
| `GEMINI_API_KEY` | No | Google Gemini API key for chart analysis and trade synthesis (free tier) |

## How to Run

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

Or visit the [live deployment](https://cryptovolprototype.streamlit.app/).

## Dashboard Tabs

1. **Trading Desk** -- expiry selector, hero chart comparing options-implied vs prediction market probabilities, optional LLM analysis, contract selector with discrepancy detail
2. **Vol Analytics** -- interactive volatility smile showing put skew, ATM term structure, skew metrics by tenor
3. **Deep Dive** -- single-market analysis with probability comparison, vol breakdown, regime classification, and optional trade synthesis
4. **Methodology** -- full model documentation

## Methodology

### Volatility Surface

The model builds a full implied volatility surface from Deribit's BTC options chain (~900 active options across ~11 expiries). For each Kalshi contract, it looks up the IV at the closest strike and tenor, using **total variance interpolation** (sigma^2 * t) between expiries to avoid calendar arbitrage. This captures both **skew** (OTM puts trade at higher IV than ATM due to crash risk) and **term structure** (short-dated vol differs from 30-day vol).

Falls back to the DVOL index (30-day IV) when no close match exists.

### Fair Value Model

For threshold markets ("BTC above $K"), the model estimates:

- **P(S_T > K)** using the standard d2 statistic under a log-normal framework
- Centered on the **Kalshi forward** (market-implied expected price), not spot -- this removes directional disagreement and isolates vol/skew-driven discrepancies
- Gaussian CDF chosen over Student-t after calibration against live Kalshi data showed better fit

### IV-RV Blending

When both IV and RV are available, they are blended based on their ratio:
- **IV/RV > 1.3** (expansion): weight IV 70% -- options market expects higher vol
- **IV/RV < 0.7** (compression): weight IV 30% -- recent moves exceed priced vol
- **Otherwise** (neutral): weight IV 55%

### Signal Generation

- Discrepancy = model probability - market probability
- Adjusted for confidence penalties (liquidity, spread, data quality) and spread cost
- Minimum 5% adjusted discrepancy to trigger BUY YES or BUY NO
- Signals are mock/paper-trade only

## Assumptions and Limitations

- **Forward-centered**: the model uses the Kalshi forward as the distribution center, which assumes the prediction market's directional view is correct
- **Constant vol per horizon**: the vol surface helps, but within-horizon vol clustering is not modeled
- **Risk-neutral vs physical**: options IV contains a risk premium, so model probabilities may overstate tail event likelihood
- **Settlement basis**: Kalshi settles on CF Benchmarks BRTI; the model uses Deribit's index for spot and vol surface -- these can diverge slightly
- **No jump diffusion**: macro events, ETF flows, and regulatory news can cause discrete gaps
- **No microstructure**: order flow, slippage, and market impact are not modeled
- **Prototype only**: signals are illustrative and not trading advice

## Demo Flow

1. Open the [live dashboard](https://cryptovolprototype.streamlit.app/) or run locally
2. **Header** shows BTC spot, Kalshi forward price, DVOL, realized vol, and regime
3. **Trading Desk** -- select an expiry, hero chart compares options model vs prediction market
4. Select a contract to see discrepancy breakdown, strike IV, signal, and explanation
5. **Vol Analytics** reveals the options skew, term structure, and skew metrics
6. **Deep Dive** into a specific contract for full vol breakdown and optional trade synthesis
7. **Methodology** tab documents the full approach
