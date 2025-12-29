"""
Stock Ranker Module
Combines all analysis scores and generates final rankings with detailed reports
"""

import pandas as pd
from typing import Dict, List
from datetime import datetime


class StockRanker:
    """
    Combines sentiment, time series, and fundamental scores
    to generate final stock rankings
    """
    
    def __init__(self, weights: Dict[str, float]):
        """
        Args:
            weights: Dict with keys 'sentiment', 'time_series', 'fundamentals'
                     Values should sum to 1.0
        """
        self.weights = weights
    
    def rank_stocks(
        self,
        sentiment_scores: Dict[str, float],
        ts_scores: Dict[str, float],
        fund_scores: Dict[str, float],
        all_data: Dict
    ) -> List[Dict]:
        """
        Combine all scores and rank stocks
        
        Returns list of dicts sorted by total score (descending)
        """
        rankings = []
        
        for symbol in all_data.keys():
            fundamentals = all_data[symbol].get("fundamentals", {})
            
            # Get individual scores (default to 0.5 if missing)
            sent_score = sentiment_scores.get(symbol, 0.5)
            ts_score = ts_scores.get(symbol, 0.5)
            fund_score = fund_scores.get(symbol, 0.5)
            
            # Calculate weighted total
            total_score = (
                self.weights["sentiment"] * sent_score +
                self.weights["time_series"] * ts_score +
                self.weights["fundamentals"] * fund_score
            )
            
            # Determine recommendation
            if total_score >= 0.75:
                recommendation = "Strong Buy"
            elif total_score >= 0.60:
                recommendation = "Buy"
            elif total_score >= 0.45:
                recommendation = "Hold"
            elif total_score >= 0.30:
                recommendation = "Weak"
            else:
                recommendation = "Avoid"
            
            rankings.append({
                "symbol": symbol,
                "name": fundamentals.get("name", "N/A"),
                "sector": fundamentals.get("sector", "N/A"),
                "current_price": fundamentals.get("current_price"),
                "market_cap": fundamentals.get("market_cap"),
                "sentiment_score": round(sent_score, 3),
                "time_series_score": round(ts_score, 3),
                "fundamental_score": round(fund_score, 3),
                "total_score": round(total_score, 3),
                "recommendation": recommendation,
                # Fundamental metrics
                "pe_ratio": fundamentals.get("pe_ratio"),
                "pb_ratio": fundamentals.get("pb_ratio"),
                "debt_to_equity": fundamentals.get("debt_to_equity"),
                "roe": fundamentals.get("roe"),
                "profit_margin": fundamentals.get("profit_margin"),
                "revenue_growth": fundamentals.get("revenue_growth"),
                "dividend_yield": fundamentals.get("dividend_yield"),
            })
        
        # Sort by total score
        rankings.sort(key=lambda x: x["total_score"], reverse=True)
        
        return rankings
    
    def generate_detailed_csv(
        self,
        sentiment_scores: Dict[str, float],
        ts_scores: Dict[str, float],
        fund_scores: Dict[str, float],
        all_data: Dict,
        output_path: str = "output/detailed_analysis.csv",
        sentiment_reports: Dict = None,
        ts_reports: Dict = None
    ) -> pd.DataFrame:
        """
        Generate detailed CSV with fundamental, sentiment, and time series interpretations
        """
        from analysis.fundamentals import FundamentalAnalyzer
        
        fund_analyzer = FundamentalAnalyzer()
        rows = []
        
        for symbol in all_data.keys():
            fundamentals = all_data[symbol].get("fundamentals", {})
            analysis = fund_analyzer.analyze_stock(fundamentals)
            
            # Get scores
            sent_score = sentiment_scores.get(symbol, 0.5)
            ts_score = ts_scores.get(symbol, 0.5)
            fund_score = fund_scores.get(symbol, 0.5)
            
            total_score = (
                self.weights["sentiment"] * sent_score +
                self.weights["time_series"] * ts_score +
                self.weights["fundamentals"] * fund_score
            )
            
            # Determine recommendation
            if total_score >= 0.75:
                recommendation = "Strong Buy"
            elif total_score >= 0.60:
                recommendation = "Buy"
            elif total_score >= 0.45:
                recommendation = "Hold"
            elif total_score >= 0.30:
                recommendation = "Weak"
            else:
                recommendation = "Avoid"
            
            row = {
                "Rank": 0,  # Will be set after sorting
                "Symbol": symbol,
                "Company Name": fundamentals.get("name", "N/A"),
                "Sector": fundamentals.get("sector", "N/A"),
                "Current Price": fundamentals.get("current_price"),
                "Total Score": round(total_score, 3),
                "Recommendation": recommendation,
                "Sentiment Score": round(sent_score, 3),
                "Trend Score": round(ts_score, 3),
                "Fundamental Score": round(fund_score, 3),
            }
            
            # Add sentiment analysis details if available
            if sentiment_reports and symbol in sentiment_reports:
                sent_report = sentiment_reports[symbol]
                row["News Sentiment"] = sent_report.get("overall_sentiment", "N/A")
                row["Sentiment Summary"] = sent_report.get("summary", "")
                row["Positive News"] = sent_report.get("positive_count", 0)
                row["Negative News"] = sent_report.get("negative_count", 0)
                row["Neutral News"] = sent_report.get("neutral_count", 0)
            
            # Add time series analysis details if available
            if ts_reports and symbol in ts_reports:
                ts_report = ts_reports[symbol]
                row["Technical Trend"] = ts_report.get("trend", "N/A")
                row["Trend Confidence"] = ts_report.get("confidence", 0)
                row["Technical Summary"] = ts_report.get("prediction", "")
                
                # Add key indicators
                indicators = ts_report.get("indicators", {})
                if indicators:
                    row["RSI"] = indicators.get("rsi")
                    row["MACD"] = indicators.get("macd")
                    row["1M Change %"] = indicators.get("change_1m")
                    row["3M Change %"] = indicators.get("change_3m")
                    row["1Y Change %"] = indicators.get("change_1y")
                    row["Volatility"] = indicators.get("volatility")
            
            # Add each fundamental metric with value and interpretation
            metrics_order = [
                "pe_ratio", "pb_ratio", "debt_to_equity", 
                "roe", "profit_margin", "revenue_growth", "dividend_yield"
            ]
            
            for metric in metrics_order:
                metric_data = analysis["metrics"].get(metric, {})
                metric_name = metric.replace("_", " ").title()
                
                value = metric_data.get("value")
                rating = metric_data.get("rating", "N/A")
                interpretation = metric_data.get("interpretation", "")
                
                row[f"{metric_name}"] = value
                row[f"{metric_name} Rating"] = rating
                row[f"{metric_name} Analysis"] = interpretation
            
            row["Overall Summary"] = analysis.get("summary", "")
            rows.append(row)
        
        # Create DataFrame and sort
        df = pd.DataFrame(rows)
        df = df.sort_values("Total Score", ascending=False).reset_index(drop=True)
        df["Rank"] = df.index + 1
        
        # Reorder columns
        cols = ["Rank"] + [c for c in df.columns if c != "Rank"]
        df = df[cols]
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        return df


def create_summary_report(df: pd.DataFrame, top_n: int = 10) -> str:
    """Create a text summary report"""
    
    report = []
    report.append("=" * 70)
    report.append("  STOCK ANALYSIS REPORT - TOP RECOMMENDATIONS")
    report.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("=" * 70)
    
    for _, row in df.head(top_n).iterrows():
        report.append(f"\n#{int(row['Rank'])} {row['Symbol']} - {row['Recommendation']}")
        report.append(f"   Company: {row['Company Name']}")
        report.append(f"   Sector: {row['Sector']}")
        report.append(f"   Total Score: {row['Total Score']:.3f}")
        report.append(f"   Price: ₹{row['Current Price']:,.2f}" if row['Current Price'] else "   Price: N/A")
        report.append("")
        report.append("   Key Metrics:")
        
        # Show key metrics with ratings
        metrics = [
            ("P/E Ratio", "Pe Ratio", "Pe Ratio Rating"),
            ("ROE", "Roe", "Roe Rating"),
            ("Debt/Equity", "Debt To Equity", "Debt To Equity Rating"),
            ("Profit Margin", "Profit Margin", "Profit Margin Rating"),
        ]
        
        for label, val_col, rating_col in metrics:
            if val_col in row and row[val_col] is not None:
                report.append(f"   • {label}: {row[val_col]:.1f} ({row[rating_col]})")
        
        report.append(f"\n   Summary: {row['Overall Summary']}")
        report.append("-" * 70)
    
    return "\n".join(report)