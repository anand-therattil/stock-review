"""
Fundamental Analysis Module
Analyzes financial health metrics and provides interpretations (Good/Bad/Neutral)
"""

import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Result for a single metric analysis"""
    value: float
    score: float          # 0-1 normalized score
    rating: str           # Good, Bad, Neutral
    interpretation: str   # Human-readable explanation


class FundamentalAnalyzer:
    """
    Analyzes stock fundamentals for long-term investment suitability
    
    Metrics analyzed:
    - P/E Ratio (Price to Earnings)
    - P/B Ratio (Price to Book)
    - Debt to Equity
    - ROE (Return on Equity)
    - Profit Margin
    - Revenue Growth
    - Dividend Yield
    """
    
    def __init__(self):
        # Thresholds for each metric (Indian market adjusted)
        self.thresholds = {
            "pe_ratio": {
                "excellent": (0, 15),
                "good": (15, 25),
                "neutral": (25, 35),
                "bad": (35, 50),
                "very_bad": (50, float('inf')),
                "lower_is_better": True,
            },
            "pb_ratio": {
                "excellent": (0, 1.5),
                "good": (1.5, 3),
                "neutral": (3, 5),
                "bad": (5, 8),
                "very_bad": (8, float('inf')),
                "lower_is_better": True,
            },
            "debt_to_equity": {
                "excellent": (0, 30),
                "good": (30, 70),
                "neutral": (70, 100),
                "bad": (100, 150),
                "very_bad": (150, float('inf')),
                "lower_is_better": True,
            },
            "roe": {
                "excellent": (25, float('inf')),
                "good": (18, 25),
                "neutral": (12, 18),
                "bad": (5, 12),
                "very_bad": (float('-inf'), 5),
                "lower_is_better": False,
            },
            "profit_margin": {
                "excellent": (20, float('inf')),
                "good": (12, 20),
                "neutral": (7, 12),
                "bad": (3, 7),
                "very_bad": (float('-inf'), 3),
                "lower_is_better": False,
            },
            "revenue_growth": {
                "excellent": (20, float('inf')),
                "good": (10, 20),
                "neutral": (5, 10),
                "bad": (0, 5),
                "very_bad": (float('-inf'), 0),
                "lower_is_better": False,
            },
            "dividend_yield": {
                "excellent": (3, 6),
                "good": (1.5, 3),
                "neutral": (0.5, 1.5),
                "bad": (0, 0.5),
                "very_bad": (6, float('inf')),  # Too high can be unsustainable
                "lower_is_better": False,
            },
        }
        
        # Weights for final score calculation
        self.weights = {
            "pe_ratio": 0.15,
            "pb_ratio": 0.10,
            "debt_to_equity": 0.20,
            "roe": 0.20,
            "profit_margin": 0.15,
            "revenue_growth": 0.15,
            "dividend_yield": 0.05,
        }
    
    def _get_rating_and_score(self, metric: str, value: float) -> Tuple[str, float, str]:
        """
        Get rating, score, and interpretation for a metric value
        
        Returns: (rating, score, interpretation)
        """
        if value is None:
            return "N/A", 0.5, "Data not available"
        
        thresholds = self.thresholds.get(metric)
        if not thresholds:
            return "N/A", 0.5, "Unknown metric"
        
        lower_is_better = thresholds["lower_is_better"]
        
        # Determine rating based on value range
        if self._in_range(value, thresholds["excellent"]):
            rating = "Excellent"
            score = 1.0
        elif self._in_range(value, thresholds["good"]):
            rating = "Good"
            score = 0.75
        elif self._in_range(value, thresholds["neutral"]):
            rating = "Neutral"
            score = 0.5
        elif self._in_range(value, thresholds["bad"]):
            rating = "Bad"
            score = 0.25
        else:
            rating = "Very Bad"
            score = 0.0
        
        # Get interpretation
        interpretation = self._get_interpretation(metric, value, rating)
        
        return rating, score, interpretation
    
    def _in_range(self, value: float, range_tuple: Tuple[float, float]) -> bool:
        """Check if value is in range (inclusive)"""
        return range_tuple[0] <= value < range_tuple[1]
    
    def _get_interpretation(self, metric: str, value: float, rating: str) -> str:
        """Get human-readable interpretation for each metric"""
        
        interpretations = {
            "pe_ratio": {
                "Excellent": f"P/E of {value:.1f} is very attractive. Stock may be undervalued relative to earnings.",
                "Good": f"P/E of {value:.1f} is reasonable. Fair valuation for a quality company.",
                "Neutral": f"P/E of {value:.1f} is moderate. Market expects decent growth.",
                "Bad": f"P/E of {value:.1f} is high. Stock might be overvalued or has high growth expectations.",
                "Very Bad": f"P/E of {value:.1f} is very high. Risky unless exceptional growth is guaranteed.",
            },
            "pb_ratio": {
                "Excellent": f"P/B of {value:.2f} suggests stock trades near book value. Potential value buy.",
                "Good": f"P/B of {value:.2f} is reasonable for a profitable company.",
                "Neutral": f"P/B of {value:.2f} indicates market premium for brand/intangibles.",
                "Bad": f"P/B of {value:.2f} is elevated. Heavy premium over book value.",
                "Very Bad": f"P/B of {value:.2f} is very high. Significant overvaluation risk.",
            },
            "debt_to_equity": {
                "Excellent": f"D/E of {value:.1f}% is very low. Strong balance sheet, minimal financial risk.",
                "Good": f"D/E of {value:.1f}% is manageable. Company has healthy leverage.",
                "Neutral": f"D/E of {value:.1f}% is moderate. Monitor debt levels during downturns.",
                "Bad": f"D/E of {value:.1f}% is high. Significant debt burden may impact profitability.",
                "Very Bad": f"D/E of {value:.1f}% is very high. High financial risk, vulnerable to rate hikes.",
            },
            "roe": {
                "Excellent": f"ROE of {value:.1f}% is outstanding. Excellent capital efficiency.",
                "Good": f"ROE of {value:.1f}% is solid. Good returns on shareholder equity.",
                "Neutral": f"ROE of {value:.1f}% is average. Acceptable but room for improvement.",
                "Bad": f"ROE of {value:.1f}% is below average. Inefficient use of capital.",
                "Very Bad": f"ROE of {value:.1f}% is poor. Company struggles to generate returns.",
            },
            "profit_margin": {
                "Excellent": f"Margin of {value:.1f}% is excellent. Strong pricing power and efficiency.",
                "Good": f"Margin of {value:.1f}% is healthy. Good profitability.",
                "Neutral": f"Margin of {value:.1f}% is moderate. Competitive but not exceptional.",
                "Bad": f"Margin of {value:.1f}% is thin. Limited buffer for cost increases.",
                "Very Bad": f"Margin of {value:.1f}% is very low. Profitability concerns.",
            },
            "revenue_growth": {
                "Excellent": f"Growth of {value:.1f}% is strong. Company expanding rapidly.",
                "Good": f"Growth of {value:.1f}% is healthy. Solid business momentum.",
                "Neutral": f"Growth of {value:.1f}% is modest. Stable but not exciting.",
                "Bad": f"Growth of {value:.1f}% is slow. Limited expansion.",
                "Very Bad": f"Growth of {value:.1f}% indicates decline. Business may be shrinking.",
            },
            "dividend_yield": {
                "Excellent": f"Yield of {value:.2f}% is attractive for income investors.",
                "Good": f"Yield of {value:.2f}% provides decent passive income.",
                "Neutral": f"Yield of {value:.2f}% is modest. Growth focus over dividends.",
                "Bad": f"Yield of {value:.2f}% is minimal. Not suitable for income strategy.",
                "Very Bad": f"Yield of {value:.2f}% is unusually high. May indicate distress or unsustainability.",
            },
        }
        
        return interpretations.get(metric, {}).get(rating, "No interpretation available")
    
    def analyze_stock(self, fundamentals: Dict) -> Dict:
        """
        Analyze a single stock's fundamentals
        
        Returns dict with scores, ratings, and interpretations for each metric
        """
        results = {
            "symbol": fundamentals.get("symbol", "N/A"),
            "name": fundamentals.get("name", "N/A"),
            "sector": fundamentals.get("sector", "N/A"),
            "metrics": {},
            "overall_score": 0.0,
            "overall_rating": "",
            "summary": "",
        }
        
        total_score = 0.0
        total_weight = 0.0
        metric_summaries = []
        
        for metric, weight in self.weights.items():
            value = fundamentals.get(metric)
            rating, score, interpretation = self._get_rating_and_score(metric, value)
            
            results["metrics"][metric] = {
                "value": value,
                "score": score,
                "rating": rating,
                "interpretation": interpretation,
            }
            
            if value is not None:
                total_score += score * weight
                total_weight += weight
                
                # Collect key insights
                if rating in ["Excellent", "Very Bad"]:
                    metric_summaries.append(f"{metric.replace('_', ' ').title()}: {rating}")
        
        # Calculate overall score
        if total_weight > 0:
            results["overall_score"] = total_score / total_weight
        
        # Determine overall rating
        if results["overall_score"] >= 0.75:
            results["overall_rating"] = "Strong Buy"
        elif results["overall_score"] >= 0.6:
            results["overall_rating"] = "Buy"
        elif results["overall_score"] >= 0.45:
            results["overall_rating"] = "Hold"
        elif results["overall_score"] >= 0.3:
            results["overall_rating"] = "Weak"
        else:
            results["overall_rating"] = "Avoid"
        
        # Generate summary
        results["summary"] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: Dict) -> str:
        """Generate a human-readable summary of the analysis"""
        
        symbol = results["symbol"]
        rating = results["overall_rating"]
        score = results["overall_score"]
        
        # Count good and bad metrics
        good_metrics = []
        bad_metrics = []
        
        for metric, data in results["metrics"].items():
            if data["rating"] in ["Excellent", "Good"]:
                good_metrics.append(metric.replace("_", " ").title())
            elif data["rating"] in ["Bad", "Very Bad"]:
                bad_metrics.append(metric.replace("_", " ").title())
        
        summary_parts = [f"{symbol} is rated '{rating}' (Score: {score:.2f}/1.00)."]
        
        if good_metrics:
            summary_parts.append(f"Strengths: {', '.join(good_metrics[:3])}.")
        
        if bad_metrics:
            summary_parts.append(f"Concerns: {', '.join(bad_metrics[:3])}.")
        
        return " ".join(summary_parts)
    
    def analyze_all(self, all_data: Dict) -> Dict[str, float]:
        """
        Analyze all stocks and return scores dict
        
        Args:
            all_data: Dict with stock symbols as keys, containing 'fundamentals' data
            
        Returns:
            Dict mapping symbol -> overall_score (0-1)
        """
        scores = {}
        
        for symbol, data in all_data.items():
            fundamentals = data.get("fundamentals", {})
            analysis = self.analyze_stock(fundamentals)
            scores[symbol] = analysis["overall_score"]
        
        return scores
    
    def get_detailed_report(self, all_data: Dict) -> pd.DataFrame:
        """
        Generate detailed report for all stocks
        
        Returns DataFrame with all metrics and interpretations
        """
        reports = []
        
        for symbol, data in all_data.items():
            fundamentals = data.get("fundamentals", {})
            analysis = self.analyze_stock(fundamentals)
            
            row = {
                "Symbol": symbol,
                "Name": analysis["name"],
                "Sector": analysis["sector"],
                "Overall Score": analysis["overall_score"],
                "Rating": analysis["overall_rating"],
            }
            
            # Add each metric
            for metric, metric_data in analysis["metrics"].items():
                col_name = metric.replace("_", " ").title()
                row[f"{col_name}"] = metric_data["value"]
                row[f"{col_name} Rating"] = metric_data["rating"]
            
            row["Summary"] = analysis["summary"]
            reports.append(row)
        
        df = pd.DataFrame(reports)
        df = df.sort_values("Overall Score", ascending=False)
        
        return df
    
    def print_stock_analysis(self, fundamentals: Dict):
        """Pretty print analysis for a single stock"""
        
        analysis = self.analyze_stock(fundamentals)
        
        print("\n" + "=" * 70)
        print(f"  FUNDAMENTAL ANALYSIS: {analysis['symbol']}")
        print(f"  {analysis['name']} | {analysis['sector']}")
        print("=" * 70)
        
        print(f"\n  Overall Rating: {analysis['overall_rating']} "
              f"(Score: {analysis['overall_score']:.2f}/1.00)")
        
        print("\n  " + "-" * 66)
        print("  METRIC BREAKDOWN:")
        print("  " + "-" * 66)
        
        for metric, data in analysis["metrics"].items():
            metric_name = metric.replace("_", " ").title()
            value_str = f"{data['value']:.2f}" if data['value'] is not None else "N/A"
            
            # Color coding for terminal (optional)
            rating = data['rating']
            if rating in ["Excellent", "Good"]:
                rating_symbol = "✓"
            elif rating in ["Bad", "Very Bad"]:
                rating_symbol = "✗"
            else:
                rating_symbol = "○"
            
            print(f"\n  {rating_symbol} {metric_name}")
            print(f"    Value: {value_str} | Rating: {rating}")
            print(f"    → {data['interpretation']}")
        
        print("\n  " + "-" * 66)
        print(f"  SUMMARY: {analysis['summary']}")
        print("=" * 70)


# Test function
def test_analyzer():
    """Test the fundamental analyzer"""
    
    # Sample data
    test_fundamentals = {
        "symbol": "RELIANCE.NS",
        "name": "Reliance Industries Ltd",
        "sector": "Energy",
        "pe_ratio": 28.5,
        "pb_ratio": 2.1,
        "debt_to_equity": 45.0,
        "roe": 22.5,
        "profit_margin": 15.8,
        "revenue_growth": 12.5,
        "dividend_yield": 0.8,
    }
    
    analyzer = FundamentalAnalyzer()
    analyzer.print_stock_analysis(test_fundamentals)
    
    # Test with bad stock
    bad_stock = {
        "symbol": "BAD.NS",
        "name": "Bad Company Ltd",
        "sector": "Unknown",
        "pe_ratio": 85.0,
        "pb_ratio": 9.5,
        "debt_to_equity": 180.0,
        "roe": 3.0,
        "profit_margin": 2.5,
        "revenue_growth": -5.0,
        "dividend_yield": 0.1,
    }
    
    analyzer.print_stock_analysis(bad_stock)


if __name__ == "__main__":
    test_analyzer()
