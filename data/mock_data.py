"""
Sample Data Generator - For testing when yfinance API is unavailable
Generates realistic mock data for development/testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import random


def generate_mock_historical(symbol: str, days: int = 500) -> pd.DataFrame:
    """Generate realistic mock historical price data"""
    np.random.seed(hash(symbol) % 2**32)
    
    # Base price varies by stock
    base_prices = {
        "RELIANCE.NS": 2500, "TCS.NS": 3800, "HDFCBANK.NS": 1600,
        "INFY.NS": 1500, "ICICIBANK.NS": 1100, "HINDUNILVR.NS": 2400,
        "SBIN.NS": 600, "BHARTIARTL.NS": 1500, "ITC.NS": 450,
    }
    base_price = base_prices.get(symbol, random.randint(500, 3000))
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    
    # Generate price with trend and volatility
    returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
    prices = base_price * np.cumprod(1 + returns)
    
    # Generate OHLCV
    data = {
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, days)),
        'High': prices * (1 + np.random.uniform(0, 0.02, days)),
        'Low': prices * (1 - np.random.uniform(0, 0.02, days)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 50000000, days),
    }
    
    df = pd.DataFrame(data, index=dates)
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    return df


def generate_mock_fundamentals(symbol: str) -> Dict:
    """Generate realistic mock fundamental data"""
    np.random.seed(hash(symbol) % 2**32)
    
    sectors = ["Technology", "Financial Services", "Consumer Defensive", 
               "Energy", "Healthcare", "Industrials", "Basic Materials"]
    
    names = {
        "RELIANCE.NS": "Reliance Industries Ltd",
        "TCS.NS": "Tata Consultancy Services Ltd",
        "HDFCBANK.NS": "HDFC Bank Ltd",
        "INFY.NS": "Infosys Ltd",
        "ICICIBANK.NS": "ICICI Bank Ltd",
        "HINDUNILVR.NS": "Hindustan Unilever Ltd",
        "SBIN.NS": "State Bank of India",
        "BHARTIARTL.NS": "Bharti Airtel Ltd",
        "ITC.NS": "ITC Ltd",
        "KOTAKBANK.NS": "Kotak Mahindra Bank Ltd",
        "LT.NS": "Larsen & Toubro Ltd",
        "AXISBANK.NS": "Axis Bank Ltd",
        "ASIANPAINT.NS": "Asian Paints Ltd",
        "MARUTI.NS": "Maruti Suzuki India Ltd",
        "HCLTECH.NS": "HCL Technologies Ltd",
    }
    
    return {
        "symbol": symbol,
        "name": names.get(symbol, symbol.replace(".NS", " Ltd")),
        "sector": random.choice(sectors),
        "industry": "Large Cap",
        "market_cap": random.randint(100000, 2000000) * 1000000,
        "pe_ratio": round(random.uniform(10, 50), 2),
        "forward_pe": round(random.uniform(8, 40), 2),
        "pb_ratio": round(random.uniform(1, 8), 2),
        "debt_to_equity": round(random.uniform(0, 150), 2),
        "roe": round(random.uniform(5, 35), 2),
        "roa": round(random.uniform(2, 20), 2),
        "profit_margin": round(random.uniform(5, 30), 2),
        "revenue_growth": round(random.uniform(-5, 30), 2),
        "earnings_growth": round(random.uniform(-10, 40), 2),
        "dividend_yield": round(random.uniform(0, 4), 2),
        "current_price": round(random.uniform(500, 4000), 2),
        "52_week_high": round(random.uniform(600, 4500), 2),
        "52_week_low": round(random.uniform(300, 3000), 2),
        "avg_volume": random.randint(1000000, 20000000),
        "beta": round(random.uniform(0.5, 1.5), 2),
    }


def generate_mock_news(symbol: str, count: int = 10) -> List[Dict]:
    """Generate mock news headlines"""
    
    company_name = symbol.replace(".NS", "")
    
    positive_templates = [
        f"{company_name} reports strong quarterly earnings, beats estimates",
        f"{company_name} announces expansion plans, stock rallies",
        f"Analysts upgrade {company_name} to 'Buy' rating",
        f"{company_name} wins major contract worth Rs 5000 crore",
        f"{company_name} launches innovative product line",
        f"Foreign investors increase stake in {company_name}",
        f"{company_name} declares special dividend for shareholders",
    ]
    
    negative_templates = [
        f"{company_name} faces regulatory scrutiny over operations",
        f"{company_name} misses revenue expectations in Q3",
        f"Analysts downgrade {company_name} citing growth concerns",
        f"{company_name} CFO resigns citing personal reasons",
        f"Rising costs impact {company_name} profit margins",
    ]
    
    neutral_templates = [
        f"{company_name} board meeting scheduled for next week",
        f"{company_name} to announce quarterly results on Monday",
        f"Market awaits {company_name} management commentary",
        f"{company_name} shares trade flat amid market volatility",
    ]
    
    all_templates = positive_templates + negative_templates + neutral_templates
    random.shuffle(all_templates)
    
    news = []
    publishers = ["Economic Times", "Moneycontrol", "NDTV Profit", "Business Standard", "LiveMint"]
    
    for i in range(min(count, len(all_templates))):
        news.append({
            "title": all_templates[i],
            "publisher": random.choice(publishers),
            "link": f"https://example.com/news/{symbol.lower()}/{i}",
            "timestamp": int((datetime.now() - timedelta(days=i)).timestamp()),
            "type": "STORY",
        })
    
    return news


class MockStockDataFetcher:
    """Mock data fetcher for testing"""
    
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
    
    def fetch_historical_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        days = {"1mo": 22, "3mo": 66, "6mo": 132, "1y": 252, "2y": 504}.get(period, 504)
        return generate_mock_historical(symbol, days)
    
    def fetch_fundamentals(self, symbol: str) -> Dict:
        return generate_mock_fundamentals(symbol)
    
    def fetch_news(self, symbol: str, count: int = 10) -> List[Dict]:
        return generate_mock_news(symbol, count)
    
    def fetch_all_data(self, period: str = "2y", news_count: int = 10) -> Dict:
        all_data = {}
        for i, symbol in enumerate(self.tickers, 1):
            print(f"[{i}/{len(self.tickers)}] Generating mock data for {symbol}...")
            all_data[symbol] = {
                "historical": self.fetch_historical_data(symbol, period),
                "fundamentals": self.fetch_fundamentals(symbol),
                "news": self.fetch_news(symbol, news_count),
            }
        return all_data


if __name__ == "__main__":
    # Test mock data generator
    print("Testing Mock Data Generator")
    print("=" * 50)
    
    fetcher = MockStockDataFetcher(["RELIANCE.NS", "TCS.NS"])
    
    hist = fetcher.fetch_historical_data("RELIANCE.NS", "1mo")
    print(f"\nHistorical Data:\n{hist.tail()}")
    
    fund = fetcher.fetch_fundamentals("RELIANCE.NS")
    print(f"\nFundamentals:\n{fund}")
    
    news = fetcher.fetch_news("RELIANCE.NS", 3)
    print(f"\nNews:\n{news}")
