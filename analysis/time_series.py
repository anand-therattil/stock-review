"""
Time Series Analysis Module
Uses PyTorch LSTM for price prediction and technical indicators for trend analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TrendResult:
    """Result for time series analysis of a stock"""
    symbol: str
    trend: str              # Bullish, Bearish, Neutral
    score: float            # 0 to 1 (bearish to bullish)
    confidence: float       # 0 to 1
    prediction: str         # Short description of prediction
    indicators: Dict        # Technical indicator values


class TimeSeriesAnalyzer:
    """
    Analyzes stock price trends using:
    1. Technical Indicators (MA, RSI, MACD, Bollinger Bands)
    2. PyTorch LSTM for price direction prediction (optional)
    
    Provides trend signals for long-term investment decisions
    """
    
    def __init__(self, use_lstm: bool = False):
        """
        Args:
            use_lstm: Whether to use LSTM model (requires more compute)
                      If False, uses technical indicators only
        """
        self.use_lstm = use_lstm
        self.model = None
        
        if use_lstm:
            self._build_lstm_model()
    
    def _build_lstm_model(self):
        """Build PyTorch LSTM model for price prediction"""
        try:
            import torch
            import torch.nn as nn
            
            class LSTMPredictor(nn.Module):
                def __init__(self, input_size=5, hidden_size=64, num_layers=2):
                    super(LSTMPredictor, self).__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    
                    self.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True,
                        dropout=0.2
                    )
                    
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size, 32),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(32, 1),
                        nn.Sigmoid()  # Output between 0 and 1
                    )
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    last_output = lstm_out[:, -1, :]
                    prediction = self.fc(last_output)
                    return prediction
            
            self.model = LSTMPredictor()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            print(f"✓ LSTM model initialized on {self.device}")
            
        except ImportError:
            print("⚠ PyTorch not available, using technical indicators only")
            self.model = None
            self.use_lstm = False
    
    def calculate_sma(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=window).mean()
    
    def calculate_ema(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=window, adjust=False).mean()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema12 = self.calculate_ema(prices, 12)
        ema26 = self.calculate_ema(prices, 26)
        macd_line = ema12 - ema26
        signal_line = self.calculate_ema(macd_line, 9)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(prices, window)
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, sma, lower_band
    
    def calculate_momentum(self, prices: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Price Momentum"""
        return prices.pct_change(periods=period) * 100
    
    def calculate_volatility(self, prices: pd.Series, window: int = 20) -> float:
        """Calculate annualized volatility"""
        returns = prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        return volatility
    
    def analyze_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Analyze all technical indicators for a stock
        
        Args:
            df: DataFrame with OHLCV data (must have 'Close' column)
            
        Returns:
            Dict with indicator values and signals
        """
        if df.empty or len(df) < 50:
            return {
                "error": "Insufficient data",
                "score": 0.5,
                "trend": "Neutral"
            }
        
        close = df['Close']
        current_price = close.iloc[-1]
        
        # Calculate all indicators
        sma_20 = self.calculate_sma(close, 20)
        sma_50 = self.calculate_sma(close, 50)
        sma_200 = self.calculate_sma(close, 200)
        ema_12 = self.calculate_ema(close, 12)
        ema_26 = self.calculate_ema(close, 26)
        rsi = self.calculate_rsi(close)
        macd_line, signal_line, macd_hist = self.calculate_macd(close)
        upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(close)
        momentum = self.calculate_momentum(close)
        volatility = self.calculate_volatility(close)
        
        # Get latest values
        latest = {
            "current_price": current_price,
            "sma_20": sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else None,
            "sma_50": sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else None,
            "sma_200": sma_200.iloc[-1] if len(df) >= 200 and not pd.isna(sma_200.iloc[-1]) else None,
            "ema_12": ema_12.iloc[-1],
            "ema_26": ema_26.iloc[-1],
            "rsi": rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
            "macd": macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0,
            "macd_signal": signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0,
            "macd_histogram": macd_hist.iloc[-1] if not pd.isna(macd_hist.iloc[-1]) else 0,
            "upper_bb": upper_bb.iloc[-1] if not pd.isna(upper_bb.iloc[-1]) else None,
            "lower_bb": lower_bb.iloc[-1] if not pd.isna(lower_bb.iloc[-1]) else None,
            "momentum_10d": momentum.iloc[-1] if not pd.isna(momentum.iloc[-1]) else 0,
            "volatility": volatility,
        }
        
        # Calculate price changes
        latest["change_1m"] = ((current_price / close.iloc[-22]) - 1) * 100 if len(df) >= 22 else 0
        latest["change_3m"] = ((current_price / close.iloc[-66]) - 1) * 100 if len(df) >= 66 else 0
        latest["change_6m"] = ((current_price / close.iloc[-132]) - 1) * 100 if len(df) >= 132 else 0
        latest["change_1y"] = ((current_price / close.iloc[-252]) - 1) * 100 if len(df) >= 252 else 0
        
        # Generate signals
        signals = self._generate_signals(latest)
        
        return {
            "indicators": latest,
            "signals": signals,
            "score": signals["overall_score"],
            "trend": signals["trend"],
            "summary": signals["summary"]
        }
    
    def _generate_signals(self, indicators: Dict) -> Dict:
        """Generate trading signals from indicators"""
        
        signals = []
        bullish_count = 0
        bearish_count = 0
        
        price = indicators["current_price"]
        
        # 1. Moving Average Signals
        if indicators["sma_20"] and indicators["sma_50"]:
            if price > indicators["sma_20"] > indicators["sma_50"]:
                signals.append(("MA Trend", "Bullish", "Price above SMA20 > SMA50"))
                bullish_count += 2
            elif price < indicators["sma_20"] < indicators["sma_50"]:
                signals.append(("MA Trend", "Bearish", "Price below SMA20 < SMA50"))
                bearish_count += 2
            elif price > indicators["sma_50"]:
                signals.append(("MA Trend", "Slightly Bullish", "Price above SMA50"))
                bullish_count += 1
            else:
                signals.append(("MA Trend", "Slightly Bearish", "Price below SMA50"))
                bearish_count += 1
        
        # 2. Golden/Death Cross (SMA50 vs SMA200)
        if indicators["sma_50"] and indicators["sma_200"]:
            if indicators["sma_50"] > indicators["sma_200"]:
                signals.append(("Long-term Trend", "Bullish", "Golden Cross (SMA50 > SMA200)"))
                bullish_count += 2
            else:
                signals.append(("Long-term Trend", "Bearish", "Death Cross (SMA50 < SMA200)"))
                bearish_count += 2
        
        # 3. RSI Signal
        rsi = indicators["rsi"]
        if rsi > 70:
            signals.append(("RSI", "Overbought", f"RSI at {rsi:.1f} - potential pullback"))
            bearish_count += 1
        elif rsi < 30:
            signals.append(("RSI", "Oversold", f"RSI at {rsi:.1f} - potential bounce"))
            bullish_count += 1
        elif rsi > 50:
            signals.append(("RSI", "Bullish", f"RSI at {rsi:.1f} - positive momentum"))
            bullish_count += 1
        else:
            signals.append(("RSI", "Bearish", f"RSI at {rsi:.1f} - negative momentum"))
            bearish_count += 1
        
        # 4. MACD Signal
        if indicators["macd"] > indicators["macd_signal"]:
            signals.append(("MACD", "Bullish", "MACD above signal line"))
            bullish_count += 1
        else:
            signals.append(("MACD", "Bearish", "MACD below signal line"))
            bearish_count += 1
        
        # 5. MACD Histogram trend
        if indicators["macd_histogram"] > 0:
            bullish_count += 1
        else:
            bearish_count += 1
        
        # 6. Bollinger Bands
        if indicators["upper_bb"] and indicators["lower_bb"]:
            if price > indicators["upper_bb"]:
                signals.append(("Bollinger", "Overbought", "Price above upper band"))
                bearish_count += 1
            elif price < indicators["lower_bb"]:
                signals.append(("Bollinger", "Oversold", "Price below lower band"))
                bullish_count += 1
            else:
                signals.append(("Bollinger", "Neutral", "Price within bands"))
        
        # 7. Momentum
        mom = indicators["momentum_10d"]
        if mom > 5:
            signals.append(("Momentum", "Strong Bullish", f"{mom:.1f}% gain in 10 days"))
            bullish_count += 2
        elif mom > 0:
            signals.append(("Momentum", "Bullish", f"{mom:.1f}% gain in 10 days"))
            bullish_count += 1
        elif mom > -5:
            signals.append(("Momentum", "Bearish", f"{mom:.1f}% loss in 10 days"))
            bearish_count += 1
        else:
            signals.append(("Momentum", "Strong Bearish", f"{mom:.1f}% loss in 10 days"))
            bearish_count += 2
        
        # 8. Long-term performance (for long-term investing)
        change_1y = indicators.get("change_1y", 0)
        if change_1y > 20:
            signals.append(("1Y Performance", "Strong", f"+{change_1y:.1f}% in 1 year"))
            bullish_count += 2
        elif change_1y > 0:
            signals.append(("1Y Performance", "Positive", f"+{change_1y:.1f}% in 1 year"))
            bullish_count += 1
        elif change_1y > -20:
            signals.append(("1Y Performance", "Negative", f"{change_1y:.1f}% in 1 year"))
            bearish_count += 1
        else:
            signals.append(("1Y Performance", "Weak", f"{change_1y:.1f}% in 1 year"))
            bearish_count += 2
        
        # Calculate overall score (0 to 1)
        total = bullish_count + bearish_count
        if total > 0:
            score = bullish_count / total
        else:
            score = 0.5
        
        # Determine trend
        if score >= 0.65:
            trend = "Bullish"
        elif score <= 0.35:
            trend = "Bearish"
        else:
            trend = "Neutral"
        
        # Generate summary
        summary = self._generate_summary(trend, score, indicators, signals)
        
        return {
            "signals": signals,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "overall_score": round(score, 3),
            "trend": trend,
            "summary": summary
        }
    
    def _generate_summary(self, trend: str, score: float, indicators: Dict, signals: List) -> str:
        """Generate human-readable summary"""
        
        rsi = indicators["rsi"]
        mom = indicators["momentum_10d"]
        change_1y = indicators.get("change_1y", 0)
        
        summaries = {
            "Bullish": f"Technical outlook is bullish (score: {score:.2f}). "
                      f"RSI at {rsi:.0f}, momentum +{mom:.1f}%, "
                      f"1Y return {change_1y:+.1f}%. Favorable for long-term entry.",
            "Bearish": f"Technical outlook is bearish (score: {score:.2f}). "
                      f"RSI at {rsi:.0f}, momentum {mom:.1f}%, "
                      f"1Y return {change_1y:+.1f}%. Consider waiting for better entry.",
            "Neutral": f"Technical outlook is neutral (score: {score:.2f}). "
                      f"RSI at {rsi:.0f}, momentum {mom:+.1f}%, "
                      f"1Y return {change_1y:+.1f}%. Mixed signals, monitor closely."
        }
        
        return summaries.get(trend, "Unable to determine trend.")
    
    def analyze_with_lstm(self, df: pd.DataFrame) -> float:
        """
        Use LSTM to predict price direction
        
        Returns:
            float: Probability of price going up (0 to 1)
        """
        if not self.model or df.empty or len(df) < 60:
            return 0.5
        
        try:
            import torch
            
            # Prepare features
            close = df['Close'].values
            high = df['High'].values
            low = df['Low'].values
            volume = df['Volume'].values
            
            # Normalize
            def normalize(arr):
                return (arr - arr.mean()) / (arr.std() + 1e-8)
            
            features = np.column_stack([
                normalize(close),
                normalize(high),
                normalize(low),
                normalize(volume),
                normalize(np.diff(close, prepend=close[0]))  # Price changes
            ])
            
            # Use last 60 days
            sequence = features[-60:]
            
            # Convert to tensor
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(x)
            
            return prediction.item()
            
        except Exception as e:
            print(f"LSTM prediction error: {e}")
            return 0.5
    
    def analyze_stock(self, symbol: str, historical_data: pd.DataFrame) -> TrendResult:
        """
        Complete analysis for a single stock
        
        Args:
            symbol: Stock symbol
            historical_data: DataFrame with OHLCV data
            
        Returns:
            TrendResult with trend, score, and indicators
        """
        # Technical analysis
        tech_analysis = self.analyze_technical_indicators(historical_data)
        
        if "error" in tech_analysis:
            return TrendResult(
                symbol=symbol,
                trend="Neutral",
                score=0.5,
                confidence=0.0,
                prediction="Insufficient data for analysis",
                indicators={}
            )
        
        # LSTM prediction (if enabled)
        if self.use_lstm and self.model:
            lstm_score = self.analyze_with_lstm(historical_data)
            # Blend technical and LSTM scores
            final_score = 0.6 * tech_analysis["score"] + 0.4 * lstm_score
        else:
            final_score = tech_analysis["score"]
        
        # Determine final trend
        if final_score >= 0.65:
            trend = "Bullish"
        elif final_score <= 0.35:
            trend = "Bearish"
        else:
            trend = "Neutral"
        
        # Confidence based on signal agreement
        signals = tech_analysis.get("signals", {})
        bull = signals.get("bullish_count", 0)
        bear = signals.get("bearish_count", 0)
        total = bull + bear
        confidence = abs(bull - bear) / total if total > 0 else 0
        
        return TrendResult(
            symbol=symbol,
            trend=trend,
            score=round(final_score, 3),
            confidence=round(confidence, 3),
            prediction=tech_analysis.get("summary", ""),
            indicators=tech_analysis.get("indicators", {})
        )
    
    def analyze_all(self, all_data: Dict) -> Dict[str, float]:
        """
        Analyze all stocks and return scores
        
        Args:
            all_data: Dict with stock symbols as keys, containing 'historical' data
            
        Returns:
            Dict mapping symbol -> score (0-1)
        """
        scores = {}
        
        for symbol, data in all_data.items():
            historical = data.get("historical", pd.DataFrame())
            result = self.analyze_stock(symbol, historical)
            scores[symbol] = result.score
        
        return scores
    
    def get_detailed_report(self, all_data: Dict) -> Dict[str, Dict]:
        """
        Get detailed technical analysis for all stocks
        
        Returns:
            Dict mapping symbol -> full analysis
        """
        reports = {}
        
        for symbol, data in all_data.items():
            historical = data.get("historical", pd.DataFrame())
            result = self.analyze_stock(symbol, historical)
            
            reports[symbol] = {
                "trend": result.trend,
                "score": result.score,
                "confidence": result.confidence,
                "prediction": result.prediction,
                "indicators": result.indicators
            }
        
        return reports
    
    def print_analysis(self, symbol: str, historical_data: pd.DataFrame):
        """Pretty print analysis for a stock"""
        
        result = self.analyze_stock(symbol, historical_data)
        tech = self.analyze_technical_indicators(historical_data)
        
        print("\n" + "=" * 70)
        print(f"  TIME SERIES ANALYSIS: {symbol}")
        print("=" * 70)
        
        print(f"\n  Trend: {result.trend}")
        print(f"  Score: {result.score:.3f} (0=Bearish, 1=Bullish)")
        print(f"  Confidence: {result.confidence:.1%}")
        
        indicators = result.indicators
        if indicators:
            print("\n  " + "-" * 66)
            print("  PRICE & MOVING AVERAGES:")
            print("  " + "-" * 66)
            print(f"  Current Price: ₹{indicators.get('current_price', 0):,.2f}")
            if indicators.get('sma_20'):
                print(f"  SMA 20: ₹{indicators['sma_20']:,.2f}")
            if indicators.get('sma_50'):
                print(f"  SMA 50: ₹{indicators['sma_50']:,.2f}")
            if indicators.get('sma_200'):
                print(f"  SMA 200: ₹{indicators['sma_200']:,.2f}")
            
            print("\n  " + "-" * 66)
            print("  MOMENTUM INDICATORS:")
            print("  " + "-" * 66)
            print(f"  RSI (14): {indicators.get('rsi', 0):.1f}")
            print(f"  MACD: {indicators.get('macd', 0):.2f}")
            print(f"  10-Day Momentum: {indicators.get('momentum_10d', 0):+.1f}%")
            print(f"  Volatility (Annual): {indicators.get('volatility', 0):.1%}")
            
            print("\n  " + "-" * 66)
            print("  PERFORMANCE:")
            print("  " + "-" * 66)
            print(f"  1 Month: {indicators.get('change_1m', 0):+.1f}%")
            print(f"  3 Months: {indicators.get('change_3m', 0):+.1f}%")
            print(f"  6 Months: {indicators.get('change_6m', 0):+.1f}%")
            print(f"  1 Year: {indicators.get('change_1y', 0):+.1f}%")
        
        if tech.get("signals", {}).get("signals"):
            print("\n  " + "-" * 66)
            print("  SIGNALS:")
            print("  " + "-" * 66)
            for name, signal, desc in tech["signals"]["signals"][:6]:
                icon = "✓" if "Bullish" in signal else "✗" if "Bearish" in signal else "○"
                print(f"  {icon} {name}: {signal}")
                print(f"    → {desc}")
        
        print("\n  " + "-" * 66)
        print(f"  SUMMARY: {result.prediction}")
        print("=" * 70)


# Test function
def test_analyzer():
    """Test the time series analyzer with sample data"""
    
    import numpy as np
    
    # Generate sample price data (uptrend)
    np.random.seed(42)
    dates = pd.date_range(end='2024-01-01', periods=300, freq='B')
    base_price = 1000
    returns = np.random.normal(0.001, 0.02, 300)  # Slight upward bias
    prices = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, 300)),
        'High': prices * (1 + np.random.uniform(0, 0.02, 300)),
        'Low': prices * (1 - np.random.uniform(0, 0.02, 300)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 300)
    }, index=dates)
    
    # Fix High/Low
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    # Analyze
    analyzer = TimeSeriesAnalyzer(use_lstm=False)
    analyzer.print_analysis("TEST.NS", df)


if __name__ == "__main__":
    test_analyzer()