"""
Configuration for Stock Analyzer
"""

# Nifty 50 Large Cap Stocks (NSE tickers for yfinance)
NIFTY_50_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "WIPRO.NS",
    "NESTLEIND.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "M&M.NS",
    "TATAMOTORS.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "COALINDIA.NS", "BAJAJFINSV.NS", "TECHM.NS", "GRASIM.NS", "INDUSINDBK.NS",
    "HINDALCO.NS", "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS",
    "EICHERMOT.NS", "HEROMOTOCO.NS", "BPCL.NS", "TATACONSUM.NS", "BRITANNIA.NS",
    "SBILIFE.NS", "HDFCLIFE.NS", "BAJAJ-AUTO.NS", "LTIM.NS", "SHRIRAMFIN.NS"
]

# Analysis Settings
HISTORICAL_PERIOD = "2y"  # 2 years of historical data
NEWS_COUNT = 10           # Number of news articles per stock

# Scoring Weights (must sum to 1.0)
WEIGHTS = {
    "sentiment": 0.25,      # News sentiment score
    "time_series": 0.25,    # Price trend prediction
    "fundamentals": 0.50    # Financial health (most important for long-term)
}

# Fundamental Analysis Thresholds (for scoring)
FUNDAMENTAL_CRITERIA = {
    "pe_ratio": {"ideal": 25, "max": 40},           # Lower is better
    "pb_ratio": {"ideal": 3, "max": 6},             # Lower is better
    "debt_to_equity": {"ideal": 0.5, "max": 1.5},   # Lower is better
    "roe": {"min": 15, "ideal": 20},                # Higher is better (%)
    "revenue_growth": {"min": 5, "ideal": 15},      # Higher is better (%)
    "profit_margin": {"min": 10, "ideal": 20},      # Higher is better (%)
}

# Output Settings
TOP_N_STOCKS = 10
OUTPUT_FILE = "output/stock_recommendations.csv"

# Qwen Model Settings (local)
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Can change based on your setup
