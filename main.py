"""
Stock Analyzer - Main Entry Point
Analyzes NSE large cap stocks for long-term investment recommendations

Usage:
    python main.py                    # Run full analysis
    python main.py --test             # Run with mock data (for testing)
    python main.py --stocks 5         # Analyze only first 5 stocks
"""

import argparse
import pandas as pd
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    NIFTY_50_STOCKS, HISTORICAL_PERIOD, NEWS_COUNT, 
    WEIGHTS, TOP_N_STOCKS, OUTPUT_FILE
)


def run_analysis(use_mock: bool = False, num_stocks: int = None):
    """Run the complete stock analysis pipeline"""
    
    print("=" * 60)
    print("  STOCK ANALYZER - Long Term Investment Recommendations")
    print("=" * 60)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Mode: {'Mock Data (Testing)' if use_mock else 'Live Data (yfinance)'}")
    print("=" * 60)
    
    # Select stocks
    stocks = NIFTY_50_STOCKS[:num_stocks] if num_stocks else NIFTY_50_STOCKS
    print(f"\nAnalyzing {len(stocks)} stocks...")
    
    # Step 1: Fetch Data
    print("\n[STEP 1/4] Fetching Stock Data...")
    print("-" * 40)
    
    if use_mock:
        from data.mock_data import MockStockDataFetcher
        fetcher = MockStockDataFetcher(stocks)
    else:
        from data.stock_fetcher import StockDataFetcher
        fetcher = StockDataFetcher(stocks)
    
    all_data = fetcher.fetch_all_data(period=HISTORICAL_PERIOD, news_count=NEWS_COUNT)
    print(f"✓ Data fetched for {len(all_data)} stocks")
    
    # Step 2: Sentiment Analysis
    print("\n[STEP 2/4] Analyzing News Sentiment...")
    print("-" * 40)
    
    from analysis.sentiment import SentimentAnalyzer
    # Set use_qwen=True when you have Qwen installed locally
    sentiment_analyzer = SentimentAnalyzer(use_qwen=False)
    sentiment_scores = sentiment_analyzer.analyze_all(all_data)
    
    # Get detailed sentiment reports
    sentiment_reports = sentiment_analyzer.get_detailed_report(all_data)
    print(f"✓ Sentiment analysis complete ({len(sentiment_scores)} stocks)")
    
    # Step 3: Time Series Analysis
    print("\n[STEP 3/4] Running Time Series Analysis (Technical Indicators)...")
    print("-" * 40)
    
    from analysis.time_series import TimeSeriesAnalyzer
    # Set use_lstm=True if you have PyTorch and want LSTM predictions
    ts_analyzer = TimeSeriesAnalyzer(use_lstm=False)
    ts_scores = ts_analyzer.analyze_all(all_data)
    
    # Get detailed time series reports
    ts_reports = ts_analyzer.get_detailed_report(all_data)
    print(f"✓ Time series analysis complete ({len(ts_scores)} stocks)")
    
    # Step 4: Fundamental Analysis
    print("\n[STEP 4/4] Analyzing Fundamentals...")
    print("-" * 40)
    
    try:
        from analysis.fundamentals import FundamentalAnalyzer
        fund_analyzer = FundamentalAnalyzer()
        fund_scores = fund_analyzer.analyze_all(all_data)
        print(f"✓ Fundamental analysis complete")
    except ImportError:
        print("⚠ Fundamental analyzer not implemented yet, using placeholder scores")
        fund_scores = {s: 0.5 for s in stocks}
    
    # Combine Scores
    print("\n[SCORING] Combining all scores...")
    print("-" * 40)
    
    from scoring.ranker import StockRanker, create_summary_report
    ranker = StockRanker(WEIGHTS)
    
    # Generate detailed CSV with interpretations
    detailed_df = ranker.generate_detailed_csv(
        sentiment_scores, ts_scores, fund_scores, all_data,
        output_path="output/detailed_analysis.csv",
        sentiment_reports=sentiment_reports,
        ts_reports=ts_reports
    )
    print(f"✓ Detailed analysis saved to: output/detailed_analysis.csv")
    
    # Also generate simple ranking
    final_ranking = ranker.rank_stocks(sentiment_scores, ts_scores, fund_scores, all_data)
    
    # Create DataFrame
    df = pd.DataFrame(final_ranking)
    
    # Save to CSV
    os.makedirs("output", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Full results saved to: {OUTPUT_FILE}")
    
    # Display Top 10
    print("\n" + "=" * 60)
    print(f"  TOP {TOP_N_STOCKS} STOCK RECOMMENDATIONS")
    print("=" * 60)
    
    top_stocks = df.head(TOP_N_STOCKS)
    for i, row in top_stocks.iterrows():
        rank = i + 1
        print(f"\n#{rank} {row['symbol']} - {row['recommendation']}")
        print(f"   Name: {row['name']}")
        print(f"   Sector: {row['sector']}")
        print(f"   Total Score: {row['total_score']:.3f}")
        print(f"   └─ Sentiment: {row['sentiment_score']:.3f} | "
              f"Trend: {row['time_series_score']:.3f} | "
              f"Fundamentals: {row['fundamental_score']:.3f}")
    
    print("\n" + "=" * 60)
    print("  Analysis Complete!")
    print("=" * 60)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Stock Analyzer for Long-Term Investment")
    parser.add_argument("--test", action="store_true", help="Use mock data for testing")
    parser.add_argument("--stocks", type=int, default=None, help="Number of stocks to analyze")
    
    args = parser.parse_args()
    
    run_analysis(use_mock=args.test, num_stocks=args.stocks)


if __name__ == "__main__":
    main()