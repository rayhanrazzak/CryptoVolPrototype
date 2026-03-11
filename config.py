"""
Central configuration for the BTC vol prototype.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Kalshi
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_API_KEY = os.getenv("KALSHI_API_KEY")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")

# Deribit (public endpoints, no auth needed)
DERIBIT_API_BASE = "https://www.deribit.com/api/v2"

# BTC spot
BTC_SPOT_SOURCE = os.getenv("BTC_SPOT_SOURCE", "coingecko")

# Model parameters
MIN_EDGE_THRESHOLD = 0.05       # 5% minimum edge for signal
CONFIDENCE_PENALTY = 0.02       # penalize edge if data quality is weak
SPREAD_PENALTY_FACTOR = 0.5     # fraction of spread to deduct from edge

# Realized vol
REALIZED_VOL_WINDOW_HOURS = 24  # lookback for realized vol
REALIZED_VOL_SAMPLE_MINUTES = 5 # sampling frequency for returns

# Refresh
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL_SECONDS", "30"))
