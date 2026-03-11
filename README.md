# BTC Volatility Market Analyzer

A live prototype that compares Kalshi BTC prediction market probabilities against a log-normal fair value model, using implied volatility from Deribit and realized volatility from recent BTC price action. The goal is to identify and explain potential mispricings in short-dated BTC event markets.

## Key Features

- **Live Kalshi market discovery** -- dynamically finds KXBTCD (threshold) and KXBTC (range) BTC markets, ranked by expiry and interpretability
- **Deribit DVOL implied volatility** -- pulls the BTC Volatility Index as a forward-looking vol anchor
- **CoinGecko realized volatility** -- computes 24-hour realized vol from recent BTC spot returns
- **Fair value model** -- log-normal probability estimation for threshold-crossing events
- **Mock paper-trade signals** -- BUY YES / BUY NO / NO TRADE based on edge, spread cost, and confidence
- **Deterministic explanation engine** -- concise natural-language commentary on each market opportunity, no external API dependencies

## Architecture

```
app.py                  # Streamlit dashboard (3 tabs: Dashboard, Market Details, Methodology)
config.py               # Constants, thresholds, environment variable loading
data/
  kalshi_client.py      # Kalshi market discovery and pricing (read-only, public endpoints)
  deribit_client.py     # Deribit DVOL implied volatility
  spot_client.py        # BTC spot price and recent price history via CoinGecko
models/
  vol_model.py          # Log-normal fair value probability estimation
  signal_engine.py      # Edge-based signal generation with confidence adjustments
  explainer.py          # Deterministic natural-language explanation engine
utils/
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

Most functionality works with public endpoints and requires no API keys. Kalshi credentials are optional and only needed if the public endpoints require authentication in the future.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `KALSHI_API_KEY` | No | Kalshi API key (public endpoints used by default) |
| `KALSHI_PRIVATE_KEY_PATH` | No | Path to Kalshi private key file |
| `COINGECKO_API_KEY` | No | CoinGecko API key (free tier works without one) |

## How to Run

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

## Methodology

### Fair Value Model

The model estimates the probability that BTC spot will be above (or below) a given threshold at expiry using a log-normal assumption:

- **P(S_T > K) = Phi(-d2)** where d2 = (ln(S/K)) / (sigma * sqrt(T))
- Zero drift assumption (reasonable for short horizons)
- Volatility scaled by sqrt(T) from annualized input
- DVOL used as the primary implied vol anchor; realized vol computed separately for comparison

### Volatility

- **Implied vol**: Deribit DVOL, a 30-day forward-looking BTC options volatility index
- **Realized vol**: Annualized standard deviation of log returns from the most recent 24 hours of BTC spot prices

Both are displayed separately in the UI. The model computes fair values under each scenario when both are available.

### Signal Generation

- Edge = model probability - market probability (adjusted for direction)
- Minimum edge threshold required to generate a signal
- Spread cost penalty applied when bid-ask data is available
- Confidence downgraded when data quality is weak (missing vol, low liquidity, wide spreads)
- Signals are mock/paper-trade only

## Assumptions and Limitations

- **Log-normal tails**: real BTC returns have fatter tails than log-normal, meaning the model underestimates extreme move probabilities
- **Constant volatility**: vol is treated as fixed over the contract horizon, which breaks down during volatile periods
- **Horizon mismatch**: DVOL is a 30-day implied vol measure applied to contracts expiring in hours or days; this is a known approximation
- **No microstructure modeling**: order flow, slippage, and market impact are not modeled
- **Prototype signals only**: this is not a production trading system; signals are illustrative and should not be used for real trading decisions

## Demo Flow

1. Open the dashboard (`streamlit run app.py`)
2. The **Dashboard** tab shows ranked BTC markets with market-implied probability, model-implied probability, edge, signal, and a short explanation
3. Click into **Market Details** for deeper analysis of a specific contract: vol inputs, model breakdown, threshold interpretation, and assumptions
4. Read the **Methodology** tab for a full explanation of the approach, data sources, and caveats
