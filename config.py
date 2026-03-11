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

# Fat-tail adjustment: Student-t degrees of freedom
# df=5 is a standard pragmatic choice for crypto (fatter than normal, not extreme)
# higher df → closer to normal; lower df → fatter tails
T_DIST_DF = 5

# IV-RV blending regime thresholds
# when IV/RV ratio exceeds these bounds, tilt the blend toward the dominant signal
BLEND_RATIO_HIGH = 1.3    # IV/RV > 1.3 → vol expansion regime, favor IV
BLEND_RATIO_LOW = 0.7     # IV/RV < 0.7 → vol compression regime, favor RV
BLEND_IV_WEIGHT_EXPANSION = 0.70   # weight on IV during vol expansion
BLEND_IV_WEIGHT_COMPRESSION = 0.30 # weight on IV during vol compression
BLEND_IV_WEIGHT_NEUTRAL = 0.55     # slight IV tilt when aligned (forward-looking)

# Liquidity filtering
LIQUIDITY_PENALTY_ONE_SIDED = 0.20  # confidence penalty for zero-bid markets
LIQUIDITY_PENALTY_NO_QUOTES = 0.35  # confidence penalty for no-quote markets

# Realized vol
REALIZED_VOL_WINDOW_HOURS = 24  # lookback for realized vol
REALIZED_VOL_SAMPLE_MINUTES = 5 # sampling frequency for returns

# Gemini (optional, free tier for trade synthesis)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Refresh
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL_SECONDS", "30"))
