"""
Stock Data Fetcher - Uses yfinance for price/fundamentals and Google News RSS for news
"""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import re
import warnings
warnings.filterwarnings('ignore')


class StockDataFetcher:
    """Fetches stock data from Yahoo Finance and news from Google News RSS"""
    
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.data_cache = {}
        
        # Map NSE tickers to company names for news search
        self.company_names = {
            "RELIANCE.NS": "Reliance Industries",
            "TCS.NS": "TCS Tata Consultancy",
            "HDFCBANK.NS": "HDFC Bank",
            "INFY.NS": "Infosys",
            "ICICIBANK.NS": "ICICI Bank",
            "HINDUNILVR.NS": "Hindustan Unilever",
            "SBIN.NS": "State Bank of India SBI",
            "BHARTIARTL.NS": "Bharti Airtel",
            "KOTAKBANK.NS": "Kotak Mahindra Bank",
            "ITC.NS": "ITC Limited",
            "LT.NS": "Larsen Toubro",
            "AXISBANK.NS": "Axis Bank",
            "ASIANPAINT.NS": "Asian Paints",
            "MARUTI.NS": "Maruti Suzuki",
            "HCLTECH.NS": "HCL Technologies",
            "SUNPHARMA.NS": "Sun Pharma",
            "TITAN.NS": "Titan Company",
            "ULTRACEMCO.NS": "UltraTech Cement",
            "BAJFINANCE.NS": "Bajaj Finance",
            "WIPRO.NS": "Wipro",
            "NESTLEIND.NS": "Nestle India",
            "ONGC.NS": "ONGC",
            "NTPC.NS": "NTPC Limited",
            "POWERGRID.NS": "Power Grid Corporation",
            "M&M.NS": "Mahindra Mahindra",
            "TATAMOTORS.NS": "Tata Motors",
            "JSWSTEEL.NS": "JSW Steel",
            "TATASTEEL.NS": "Tata Steel",
            "ADANIENT.NS": "Adani Enterprises",
            "ADANIPORTS.NS": "Adani Ports",
            "COALINDIA.NS": "Coal India",
            "BAJAJFINSV.NS": "Bajaj Finserv",
            "TECHM.NS": "Tech Mahindra",
            "GRASIM.NS": "Grasim Industries",
            "INDUSINDBK.NS": "IndusInd Bank",
            "HINDALCO.NS": "Hindalco",
            "DIVISLAB.NS": "Divis Laboratories",
            "DRREDDY.NS": "Dr Reddys",
            "CIPLA.NS": "Cipla",
            "APOLLOHOSP.NS": "Apollo Hospitals",
            "EICHERMOT.NS": "Eicher Motors Royal Enfield",
            "HEROMOTOCO.NS": "Hero MotoCorp",
            "BPCL.NS": "BPCL Bharat Petroleum",
            "TATACONSUM.NS": "Tata Consumer",
            "BRITANNIA.NS": "Britannia",
            "SBILIFE.NS": "SBI Life Insurance",
            "HDFCLIFE.NS": "HDFC Life",
            "BAJAJ-AUTO.NS": "Bajaj Auto",
            "LTIM.NS": "LTIMindtree",
            "SHRIRAMFIN.NS": "Shriram Finance",
        }
    
    def get_ticker_object(self, symbol: str) -> yf.Ticker:
        """Get or create ticker object"""
        if symbol not in self.data_cache:
            self.data_cache[symbol] = yf.Ticker(symbol)
        return self.data_cache[symbol]
    
    def fetch_historical_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        try:
            ticker = self.get_ticker_object(symbol)
            hist = ticker.history(period=period)
            return hist
        except Exception as e:
            print(f"  ⚠ Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_fundamentals(self, symbol: str) -> Dict:
        """Fetch fundamental data for a stock"""
        try:
            ticker = self.get_ticker_object(symbol)
            info = ticker.info
            
            if not info:
                return {"symbol": symbol, "error": "No data available"}
            
            fundamentals = {
                "symbol": symbol,
                "name": info.get("shortName", info.get("longName", "N/A")),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", None),
                "forward_pe": info.get("forwardPE", None),
                "pb_ratio": info.get("priceToBook", None),
                "debt_to_equity": info.get("debtToEquity", None),
                "roe": info.get("returnOnEquity", None),
                "roa": info.get("returnOnAssets", None),
                "profit_margin": info.get("profitMargins", None),
                "revenue_growth": info.get("revenueGrowth", None),
                "earnings_growth": info.get("earningsGrowth", None),
                "dividend_yield": info.get("dividendYield", None),
                "current_price": info.get("currentPrice", info.get("regularMarketPrice", None)),
                "52_week_high": info.get("fiftyTwoWeekHigh", None),
                "52_week_low": info.get("fiftyTwoWeekLow", None),
                "avg_volume": info.get("averageVolume", None),
                "beta": info.get("beta", None),
            }
            
            # Convert percentages (yfinance returns as decimals)
            if fundamentals["roe"]:
                fundamentals["roe"] = fundamentals["roe"] * 100
            if fundamentals["roa"]:
                fundamentals["roa"] = fundamentals["roa"] * 100
            if fundamentals["profit_margin"]:
                fundamentals["profit_margin"] = fundamentals["profit_margin"] * 100
            if fundamentals["revenue_growth"]:
                fundamentals["revenue_growth"] = fundamentals["revenue_growth"] * 100
            if fundamentals["dividend_yield"]:
                fundamentals["dividend_yield"] = fundamentals["dividend_yield"] * 100
                
            return fundamentals
            
        except Exception as e:
            print(f"  ⚠ Error fetching fundamentals for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
    
    def _get_company_search_term(self, symbol: str) -> str:
        """Get the company name for news search"""
        # Check if we have a mapped name
        if symbol in self.company_names:
            return self.company_names[symbol]
        
        # Otherwise, try to extract from symbol
        base = symbol.replace(".NS", "").replace(".BO", "")
        return base + " stock NSE"
    
    def fetch_news_google(self, symbol: str, count: int = 10) -> List[Dict]:
        """Fetch news from Google News RSS feed"""
        try:
            search_term = self._get_company_search_term(symbol)
            query = urllib.parse.quote(f"{search_term} stock")
            
            # Google News RSS URL
            url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
            
            # Create request with headers to avoid blocking
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read().decode('utf-8')
            
            # Parse XML
            root = ET.fromstring(xml_data)
            
            news_list = []
            items = root.findall('.//item')
            
            for item in items[:count]:
                title = item.find('title')
                link = item.find('link')
                pub_date = item.find('pubDate')
                source = item.find('source')
                
                if title is not None:
                    # Clean the title (remove source suffix like " - Economic Times")
                    title_text = title.text or ""
                    title_text = re.sub(r'\s*-\s*[^-]+$', '', title_text).strip()
                    
                    news_list.append({
                        "title": title_text,
                        "publisher": source.text if source is not None else "Unknown",
                        "link": link.text if link is not None else "",
                        "timestamp": pub_date.text if pub_date is not None else "",
                        "type": "NEWS",
                    })
            
            return news_list
            
        except Exception as e:
            # Silently fail and return empty list
            return []
    
    def fetch_news(self, symbol: str, count: int = 10) -> List[Dict]:
        """Fetch news - tries yfinance first, then Google News RSS"""
        
        # Try yfinance first
        try:
            ticker = self.get_ticker_object(symbol)
            news = ticker.news
            
            if news and len(news) > 0:
                news_list = []
                for article in news[:count]:
                    title = article.get("title", "")
                    if title:  # Only add if title exists
                        news_list.append({
                            "title": title,
                            "publisher": article.get("publisher", "Unknown"),
                            "link": article.get("link", ""),
                            "timestamp": article.get("providerPublishTime", 0),
                            "type": article.get("type", "NEWS"),
                        })
                
                if len(news_list) >= 3:  # If we got enough news from yfinance
                    return news_list
        except:
            pass
        
        # Fallback to Google News RSS
        return self.fetch_news_google(symbol, count)
    
    def fetch_all_data(self, period: str = "2y", news_count: int = 10) -> Dict:
        """Fetch all data for all tickers"""
        all_data = {}
        total = len(self.tickers)
        
        for i, symbol in enumerate(self.tickers, 1):
            print(f"[{i}/{total}] Fetching data for {symbol}...")
            
            all_data[symbol] = {
                "historical": self.fetch_historical_data(symbol, period),
                "fundamentals": self.fetch_fundamentals(symbol),
                "news": self.fetch_news(symbol, news_count),
            }
            
            # Show news count
            news_count_fetched = len(all_data[symbol]["news"])
            if news_count_fetched > 0:
                print(f"        → {news_count_fetched} news articles found")
        
        return all_data
    
    def get_fundamentals_df(self) -> pd.DataFrame:
        """Get fundamentals for all stocks as DataFrame"""
        fundamentals_list = []
        
        for symbol in self.tickers:
            fund = self.fetch_fundamentals(symbol)
            fundamentals_list.append(fund)
        
        return pd.DataFrame(fundamentals_list)


# Quick test function
def test_fetcher():
    """Test the data fetcher with a single stock"""
    fetcher = StockDataFetcher(["RELIANCE.NS"])
    
    print("=" * 50)
    print("Testing Stock Data Fetcher")
    print("=" * 50)
    
    # Test historical data
    print("\n1. Historical Data (last 5 rows):")
    hist = fetcher.fetch_historical_data("RELIANCE.NS", period="1mo")
    print(hist.tail())
    
    # Test fundamentals
    print("\n2. Fundamentals:")
    fund = fetcher.fetch_fundamentals("RELIANCE.NS")
    for key, value in list(fund.items())[:10]:
        print(f"   {key}: {value}")
    
    # Test news (Google RSS)
    print("\n3. Recent News (Google News RSS):")
    news = fetcher.fetch_news("RELIANCE.NS", count=5)
    print(f"   Found {len(news)} articles")
    for article in news[:3]:
        print(f"   - [{article['publisher']}] {article['title'][:60]}...")
    
    print("\n" + "=" * 50)
    print("Test Complete!")


if __name__ == "__main__":
    test_fetcher()