# 📈 Stock Analyzer - NSE Large Cap Investment Tool

A comprehensive Python-based stock analysis tool for **long-term investment decisions** in NSE (National Stock Exchange) large-cap stocks. The tool combines **Fundamental Analysis**, **Sentiment Analysis**, and **Technical Analysis** to generate actionable buy/hold/avoid recommendations.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

---

## 🎯 Features

| Component | Description | Technology |
|-----------|-------------|------------|
| **Fundamental Analysis** | P/E, P/B, ROE, Debt/Equity, Profit Margins | yfinance |
| **Sentiment Analysis** | News sentiment scoring with interpretations | Qwen (local) / Keywords |
| **Technical Analysis** | RSI, MACD, Moving Averages, Bollinger Bands | PyTorch / NumPy |
| **Dashboard** | Interactive web dashboard for visualization | HTML/CSS/JS + Chart.js |

### What You Get

- ✅ **Top 10 Stock Recommendations** ranked by composite score
- ✅ **Detailed CSV Report** with 40+ data points per stock
- ✅ **Human-readable interpretations** (Good/Bad/Neutral ratings)
- ✅ **Interactive Dashboard** to explore and filter results
- ✅ **All 50 Nifty stocks** analyzed in one run

---

## 📁 Project Structure

```
stock_analyzer/
├── main.py                     # Entry point - runs full analysis
├── config.py                   # Settings, stock list, scoring weights
├── requirements.txt            # Python dependencies
│
├── data/
│   ├── __init__.py
│   ├── stock_fetcher.py        # Fetches data from yfinance + Google News
│   └── mock_data.py            # Mock data generator for testing
│
├── analysis/
│   ├── __init__.py
│   ├── fundamentals.py         # Fundamental analysis (P/E, ROE, etc.)
│   ├── sentiment.py            # News sentiment analysis (Qwen/Keywords)
│   └── time_series.py          # Technical indicators (RSI, MACD, etc.)
│
├── scoring/
│   ├── __init__.py
│   └── ranker.py               # Combines scores & generates rankings
│
├── output/
│   ├── stock_recommendations.csv    # Simple ranking output
│   └── detailed_analysis.csv        # Full analysis with interpretations
│
└── stock_dashboard.html        # Interactive web dashboard
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd stock_analyzer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Analysis

```bash
# Full analysis of all 50 Nifty stocks
python main.py

# Test mode with mock data (no internet required)
python main.py --test

# Analyze specific number of stocks
python main.py --stocks 10
```

### 3. View Results

```bash
# Results saved to:
# - output/stock_recommendations.csv (simple ranking)
# - output/detailed_analysis.csv (full analysis)

# Open dashboard in browser
open stock_dashboard.html  # or double-click the file
# Then upload detailed_analysis.csv to the dashboard
```

---

## 📊 Analysis Components

### 1. Fundamental Analysis

Evaluates financial health using key metrics:

| Metric | Excellent | Good | Neutral | Bad | Very Bad |
|--------|-----------|------|---------|-----|----------|
| **P/E Ratio** | < 15 | 15-25 | 25-35 | 35-50 | > 50 |
| **P/B Ratio** | < 1.5 | 1.5-3 | 3-5 | 5-8 | > 8 |
| **Debt/Equity** | < 30% | 30-70% | 70-100% | 100-150% | > 150% |
| **ROE** | > 25% | 18-25% | 12-18% | 5-12% | < 5% |
| **Profit Margin** | > 20% | 12-20% | 7-12% | 3-7% | < 3% |
| **Revenue Growth** | > 20% | 10-20% | 5-10% | 0-5% | < 0% |

**Output Example:**
```
✓ ROE: 27.5% | Rating: Excellent
  → ROE of 27.5% is outstanding. Excellent capital efficiency.

✗ Debt/Equity: 185% | Rating: Very Bad
  → D/E of 185% is very high. High financial risk, vulnerable to rate hikes.
```

### 2. Sentiment Analysis

Analyzes recent news for market sentiment:

**Two Modes:**
- **Qwen Model** (Local): Uses Qwen2.5 LLM for intelligent sentiment analysis
- **Keyword-based** (Default): Uses curated positive/negative keyword dictionaries

**Positive Keywords:** beats, exceeds, surges, upgraded, expansion, profit rises, buy rating

**Negative Keywords:** misses, plunges, downgraded, fraud, layoffs, guidance cut, investigation

**Output Example:**
```
Overall Sentiment: Positive
Score: 0.67 (-1 to +1 scale)

Article Breakdown:
  ✓ Positive: 5
  ✗ Negative: 2
  ○ Neutral: 3

Summary: Recent news suggests favorable outlook for the stock.
```

### 3. Technical Analysis

Evaluates price trends using technical indicators:

| Indicator | Bullish Signal | Bearish Signal |
|-----------|---------------|----------------|
| **Moving Averages** | Price > SMA50 > SMA200 (Golden Cross) | Price < SMA50 < SMA200 (Death Cross) |
| **RSI (14)** | < 30 (Oversold - potential bounce) | > 70 (Overbought - potential pullback) |
| **MACD** | MACD > Signal Line | MACD < Signal Line |
| **Bollinger Bands** | Price at lower band | Price at upper band |
| **Momentum** | > +5% (10-day) | < -5% (10-day) |

**Output Example:**
```
Trend: Bullish
Score: 0.82 (0=Bearish, 1=Bullish)

Signals:
  ✓ MA Trend: Bullish → Price above SMA20 > SMA50
  ✓ Long-term: Bullish → Golden Cross (SMA50 > SMA200)
  ○ RSI: Overbought → RSI at 74.8, potential pullback
  ✓ MACD: Bullish → MACD above signal line
```

---

## ⚖️ Scoring System

### Weights (Configurable in `config.py`)

```python
WEIGHTS = {
    "sentiment": 0.25,      # 25% - News sentiment
    "time_series": 0.25,    # 25% - Technical trend
    "fundamentals": 0.50    # 50% - Financial health (most important)
}
```

### Final Recommendation

| Total Score | Recommendation |
|-------------|----------------|
| ≥ 0.75 | **Strong Buy** 🟢 |
| 0.60 - 0.75 | **Buy** 🔵 |
| 0.45 - 0.60 | **Hold** 🟡 |
| 0.30 - 0.45 | **Weak** 🟠 |
| < 0.30 | **Avoid** 🔴 |

---

## 📈 Dashboard

An interactive HTML dashboard to visualize your analysis:

### Features
- 📊 **Stats Overview**: Total stocks, Buy/Hold/Avoid counts
- 🔍 **Filters**: By sector, recommendation, trend, search
- 📋 **Sortable Table**: Click headers to sort by any column
- 🏆 **Top 5 Picks**: Quick view of best recommendations
- 📈 **Charts**: Score distribution & sector breakdown
- 🔎 **Stock Details**: Click any stock for full analysis modal
- ⬇️ **Export**: Download filtered data as CSV

### How to Use
1. Open `stock_dashboard.html` in your browser
2. Click "Load CSV" or drag & drop `detailed_analysis.csv`
3. Explore, filter, and analyze your stocks!

---

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Stocks to analyze (default: Nifty 50)
NIFTY_50_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", ...
]

# Analysis settings
HISTORICAL_PERIOD = "2y"    # Historical data period
NEWS_COUNT = 10             # News articles per stock

# Scoring weights
WEIGHTS = {
    "sentiment": 0.25,
    "time_series": 0.25,
    "fundamentals": 0.50
}

# Output settings
TOP_N_STOCKS = 10
OUTPUT_FILE = "output/stock_recommendations.csv"
```

---

## 🔌 Data Sources

| Data | Source | Cost |
|------|--------|------|
| Stock Prices | Yahoo Finance (yfinance) | Free |
| Fundamentals | Yahoo Finance (yfinance) | Free |
| News | Google News RSS | Free |
| Sentiment (optional) | Qwen LLM (local) | Free |

### News Fetching

The tool uses a **fallback mechanism** for news:
1. First tries **yfinance** news API
2. If unavailable, uses **Google News RSS** feed

Google News RSS URL format:
```
https://news.google.com/rss/search?q={company}+stock&hl=en-IN&gl=IN&ceid=IN:en
```

---

## 🧠 Optional: Enable Qwen for Better Sentiment Analysis

For more accurate sentiment analysis, enable local Qwen model:

### 1. Install additional dependencies
```bash
pip install transformers accelerate torch
```

### 2. Update `main.py`
```python
# Change this line:
sentiment_analyzer = SentimentAnalyzer(use_qwen=False)

# To:
sentiment_analyzer = SentimentAnalyzer(use_qwen=True)
```

### 3. First run will download the model (~1GB)

**Note:** Qwen requires 4GB+ RAM. If you have limited resources, the keyword-based analysis works well too!

---

## 📋 Output CSV Columns

The `detailed_analysis.csv` contains:

### Basic Info
- Rank, Symbol, Company Name, Sector, Current Price

### Scores
- Total Score, Recommendation
- Sentiment Score, Trend Score, Fundamental Score

### Sentiment Analysis
- News Sentiment, Sentiment Summary
- Positive/Negative/Neutral News Count

### Technical Analysis
- Technical Trend, Trend Confidence, Technical Summary
- RSI, MACD, 1M/3M/1Y Change %, Volatility

### Fundamental Analysis
- P/E Ratio (Value, Rating, Analysis)
- P/B Ratio (Value, Rating, Analysis)
- Debt/Equity (Value, Rating, Analysis)
- ROE (Value, Rating, Analysis)
- Profit Margin (Value, Rating, Analysis)
- Revenue Growth (Value, Rating, Analysis)
- Dividend Yield (Value, Rating, Analysis)
- Overall Summary

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. 

- ❌ NOT financial advice
- ❌ NOT a recommendation to buy or sell
- ❌ Do NOT rely solely on this for investment decisions
- ✅ Always do your own research
- ✅ Consult a financial advisor before investing

Stock markets are subject to risks. Past performance does not guarantee future results.

---

## 🛠️ Troubleshooting

### "No news articles found"
- This is normal for some stocks
- The tool uses Google News RSS as fallback
- Check your internet connection

### "yfinance error" or "Failed to fetch"
- Yahoo Finance may be temporarily blocking requests
- Wait a few minutes and try again
- Use `--test` mode to verify the tool works

### "Module not found"
```bash
pip install -r requirements.txt
```

### Dashboard not loading CSV
- Make sure the CSV has the correct columns
- Try opening the HTML file directly (not via file server)
- Check browser console for errors (F12)

---

## 📝 License

MIT License - Feel free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance API wrapper
- [Qwen](https://github.com/QwenLM/Qwen) - Large Language Model for sentiment analysis
- [Chart.js](https://www.chartjs.org/) - Dashboard charts
- [PapaParse](https://www.papaparse.com/) - CSV parsing in browser

---

## 📞 Support

If you find this useful, ⭐ star the repo!

For issues or feature requests, open a GitHub issue.

---

**Happy Investing! 📈💰**