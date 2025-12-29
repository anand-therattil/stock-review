"""
Sentiment Analysis Module
Uses Qwen model for analyzing news sentiment (local inference)
Falls back to keyword-based analysis if Qwen is not available
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class SentimentResult:
    """Result for sentiment analysis of a single news article"""
    title: str
    sentiment: str       # Positive, Negative, Neutral
    score: float         # -1 to 1 (negative to positive)
    confidence: float    # 0 to 1
    explanation: str     # Why this sentiment


class SentimentAnalyzer:
    """
    Analyzes news sentiment using Qwen model (local) or keyword fallback
    
    Sentiment categories:
    - Positive: Good news for stock (earnings beat, expansion, upgrades)
    - Negative: Bad news (misses, downgrades, scandals, layoffs)
    - Neutral: Informational without clear sentiment
    """
    
    def __init__(self, use_qwen: bool = True):
        """
        Args:
            use_qwen: Whether to try loading Qwen model (set False for keyword-only)
        """
        self.use_qwen = use_qwen
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        
        if use_qwen:
            self._load_qwen_model()
        
        # Keyword dictionaries for fallback
        self._init_keyword_dicts()
    
    def _load_qwen_model(self):
        """Try to load Qwen model for local inference"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            print("Loading Qwen model for sentiment analysis...")
            
            # Use smaller Qwen model for efficiency
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Smaller model, faster inference
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print(f"✓ Qwen model loaded on {self.device}")
            
        except Exception as e:
            print(f"⚠ Could not load Qwen model: {e}")
            print("  Using keyword-based sentiment analysis instead")
            self.model = None
            self.tokenizer = None
    
    def _init_keyword_dicts(self):
        """Initialize keyword dictionaries for fallback analysis"""
        
        self.positive_keywords = {
            # Strong positive (weight 2)
            "strong": ["beats", "exceeds", "surges", "soars", "jumps", "rallies",
                      "record high", "all-time high", "breakthrough", "blockbuster",
                      "outperforms", "upgraded", "buy rating", "strong buy",
                      "expansion", "acquisition", "partnership", "deal signed",
                      "profit rises", "revenue grows", "dividend increase",
                      "market leader", "innovation", "patent", "approval"],
            # Moderate positive (weight 1)
            "moderate": ["rises", "gains", "up", "growth", "positive", "optimistic",
                        "confident", "bullish", "recovery", "improvement", "better",
                        "success", "wins", "launches", "announces", "new product",
                        "stable", "steady", "maintains", "resilient"]
        }
        
        self.negative_keywords = {
            # Strong negative (weight 2)
            "strong": ["plunges", "crashes", "tanks", "collapses", "plummets",
                      "misses", "disappoints", "downgraded", "sell rating",
                      "fraud", "scandal", "investigation", "lawsuit", "fine",
                      "bankruptcy", "default", "layoffs", "cuts jobs",
                      "profit falls", "revenue drops", "dividend cut",
                      "warning", "guidance cut", "recall", "breach"],
            # Moderate negative (weight 1)
            "moderate": ["falls", "drops", "down", "decline", "slips", "dips",
                        "concerns", "worries", "fears", "risks", "challenges",
                        "weak", "bearish", "caution", "uncertainty", "volatile",
                        "slowdown", "pressure", "headwinds", "losses"]
        }
        
        self.neutral_keywords = [
            "announces", "reports", "scheduled", "meeting", "conference",
            "quarterly", "fiscal", "year-end", "board", "shareholders",
            "trading", "volume", "market", "sector", "industry"
        ]
    
    def _analyze_with_qwen(self, title: str) -> Tuple[str, float, str]:
        """
        Analyze sentiment using Qwen model
        
        Returns: (sentiment, score, explanation)
        """
        if not self.model or not self.tokenizer:
            return self._analyze_with_keywords(title)
        
        try:
            prompt = f"""Analyze the sentiment of this stock market news headline.
Respond with exactly one line in this format:
SENTIMENT|SCORE|REASON

Where:
- SENTIMENT is one of: POSITIVE, NEGATIVE, NEUTRAL
- SCORE is a number from -1.0 (very negative) to 1.0 (very positive)
- REASON is a brief explanation (10 words max)

Headline: "{title}"

Response:"""

            messages = [
                {"role": "system", "content": "You are a financial sentiment analyzer. Be concise and precise."},
                {"role": "user", "content": prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse response
            response_line = response.split("Response:")[-1].strip().split("\n")[0]
            parts = response_line.split("|")
            
            if len(parts) >= 3:
                sentiment = parts[0].strip().upper()
                if sentiment not in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
                    sentiment = "NEUTRAL"
                
                try:
                    score = float(parts[1].strip())
                    score = max(-1.0, min(1.0, score))
                except:
                    score = 0.0
                
                explanation = parts[2].strip()[:100]
            else:
                # Fallback to keywords if parsing fails
                return self._analyze_with_keywords(title)
            
            return sentiment.capitalize(), score, explanation
            
        except Exception as e:
            print(f"Qwen inference error: {e}, falling back to keywords")
            return self._analyze_with_keywords(title)
    
    def _analyze_with_keywords(self, title: str) -> Tuple[str, float, str]:
        """
        Analyze sentiment using keyword matching (fallback method)
        
        Returns: (sentiment, score, explanation)
        """
        title_lower = title.lower()
        
        pos_score = 0
        neg_score = 0
        matched_pos = []
        matched_neg = []
        
        # Check strong positive keywords (weight 2)
        for keyword in self.positive_keywords["strong"]:
            if keyword in title_lower:
                pos_score += 2
                matched_pos.append(keyword)
        
        # Check moderate positive keywords (weight 1)
        for keyword in self.positive_keywords["moderate"]:
            if keyword in title_lower:
                pos_score += 1
                matched_pos.append(keyword)
        
        # Check strong negative keywords (weight 2)
        for keyword in self.negative_keywords["strong"]:
            if keyword in title_lower:
                neg_score += 2
                matched_neg.append(keyword)
        
        # Check moderate negative keywords (weight 1)
        for keyword in self.negative_keywords["moderate"]:
            if keyword in title_lower:
                neg_score += 1
                matched_neg.append(keyword)
        
        # Calculate final score
        total = pos_score + neg_score
        if total == 0:
            sentiment = "Neutral"
            score = 0.0
            explanation = "No strong sentiment indicators found"
        else:
            net_score = pos_score - neg_score
            score = net_score / max(total, 1)  # Normalize to -1 to 1
            score = max(-1.0, min(1.0, score))
            
            if score > 0.2:
                sentiment = "Positive"
                explanation = f"Positive indicators: {', '.join(matched_pos[:3])}"
            elif score < -0.2:
                sentiment = "Negative"
                explanation = f"Negative indicators: {', '.join(matched_neg[:3])}"
            else:
                sentiment = "Neutral"
                explanation = "Mixed or weak sentiment signals"
        
        return sentiment, score, explanation
    
    def analyze_headline(self, title: str) -> SentimentResult:
        """
        Analyze a single news headline
        
        Args:
            title: News headline text
            
        Returns:
            SentimentResult with sentiment, score, and explanation
        """
        if self.model and self.tokenizer:
            sentiment, score, explanation = self._analyze_with_qwen(title)
        else:
            sentiment, score, explanation = self._analyze_with_keywords(title)
        
        # Calculate confidence based on score magnitude
        confidence = min(abs(score) + 0.3, 1.0)
        
        return SentimentResult(
            title=title,
            sentiment=sentiment,
            score=score,
            confidence=confidence,
            explanation=explanation
        )
    
    def analyze_news_list(self, news_list: List[Dict]) -> Dict:
        """
        Analyze a list of news articles for a stock
        
        Args:
            news_list: List of news dicts with 'title' key
            
        Returns:
            Dict with overall sentiment analysis
        """
        if not news_list:
            return {
                "overall_sentiment": "Neutral",
                "overall_score": 0.0,
                "confidence": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "articles_analyzed": 0,
                "summary": "No news available for analysis",
                "details": []
            }
        
        results = []
        total_score = 0.0
        pos_count = 0
        neg_count = 0
        neu_count = 0
        
        for news in news_list:
            title = news.get("title", "")
            if not title:
                continue
            
            result = self.analyze_headline(title)
            results.append({
                "title": result.title,
                "sentiment": result.sentiment,
                "score": result.score,
                "explanation": result.explanation
            })
            
            total_score += result.score
            
            if result.sentiment == "Positive":
                pos_count += 1
            elif result.sentiment == "Negative":
                neg_count += 1
            else:
                neu_count += 1
        
        # Calculate overall metrics
        num_articles = len(results)
        if num_articles > 0:
            avg_score = total_score / num_articles
        else:
            avg_score = 0.0
        
        # Determine overall sentiment
        if avg_score > 0.15:
            overall_sentiment = "Positive"
        elif avg_score < -0.15:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"
        
        # Generate summary
        summary = self._generate_summary(overall_sentiment, avg_score, pos_count, neg_count, neu_count)
        
        return {
            "overall_sentiment": overall_sentiment,
            "overall_score": round(avg_score, 3),
            "confidence": round(min(abs(avg_score) + 0.3, 1.0), 3),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": neu_count,
            "articles_analyzed": num_articles,
            "summary": summary,
            "details": results
        }
    
    def _generate_summary(self, sentiment: str, score: float, pos: int, neg: int, neu: int) -> str:
        """Generate human-readable summary of sentiment analysis"""
        
        total = pos + neg + neu
        
        if total == 0:
            return "No news articles available for sentiment analysis."
        
        summaries = {
            "Positive": f"Overall positive sentiment (score: {score:.2f}). "
                       f"{pos} positive, {neg} negative, {neu} neutral articles. "
                       f"Recent news suggests favorable outlook.",
            "Negative": f"Overall negative sentiment (score: {score:.2f}). "
                       f"{pos} positive, {neg} negative, {neu} neutral articles. "
                       f"Recent news raises concerns.",
            "Neutral": f"Mixed/neutral sentiment (score: {score:.2f}). "
                      f"{pos} positive, {neg} negative, {neu} neutral articles. "
                      f"No clear directional bias in recent news."
        }
        
        return summaries.get(sentiment, "Unable to determine sentiment.")
    
    def analyze_all(self, all_data: Dict) -> Dict[str, float]:
        """
        Analyze sentiment for all stocks
        
        Args:
            all_data: Dict with stock symbols as keys, containing 'news' data
            
        Returns:
            Dict mapping symbol -> normalized score (0-1)
        """
        scores = {}
        
        for symbol, data in all_data.items():
            news_list = data.get("news", [])
            analysis = self.analyze_news_list(news_list)
            
            # Convert -1 to 1 score to 0 to 1 scale
            raw_score = analysis["overall_score"]
            normalized_score = (raw_score + 1) / 2  # Maps -1,1 to 0,1
            
            scores[symbol] = round(normalized_score, 3)
        
        return scores
    
    def get_detailed_report(self, all_data: Dict) -> Dict[str, Dict]:
        """
        Get detailed sentiment report for all stocks
        
        Returns:
            Dict mapping symbol -> full sentiment analysis
        """
        reports = {}
        
        for symbol, data in all_data.items():
            news_list = data.get("news", [])
            reports[symbol] = self.analyze_news_list(news_list)
        
        return reports
    
    def print_analysis(self, symbol: str, news_list: List[Dict]):
        """Pretty print sentiment analysis for a stock"""
        
        analysis = self.analyze_news_list(news_list)
        
        print("\n" + "=" * 70)
        print(f"  SENTIMENT ANALYSIS: {symbol}")
        print("=" * 70)
        
        print(f"\n  Overall Sentiment: {analysis['overall_sentiment']}")
        print(f"  Score: {analysis['overall_score']:.3f} (-1 to +1 scale)")
        print(f"  Confidence: {analysis['confidence']:.1%}")
        
        print(f"\n  Article Breakdown:")
        print(f"    ✓ Positive: {analysis['positive_count']}")
        print(f"    ✗ Negative: {analysis['negative_count']}")
        print(f"    ○ Neutral: {analysis['neutral_count']}")
        
        print(f"\n  Summary: {analysis['summary']}")
        
        if analysis['details']:
            print("\n  " + "-" * 66)
            print("  NEWS ARTICLES:")
            print("  " + "-" * 66)
            
            for i, article in enumerate(analysis['details'][:5], 1):
                sentiment_icon = {"Positive": "✓", "Negative": "✗", "Neutral": "○"}.get(article['sentiment'], "?")
                print(f"\n  {i}. {sentiment_icon} [{article['sentiment']}] (Score: {article['score']:.2f})")
                print(f"     {article['title'][:65]}...")
                print(f"     → {article['explanation']}")
        
        print("\n" + "=" * 70)


# Test function
def test_analyzer():
    """Test the sentiment analyzer"""
    
    # Test news articles
    test_news = [
        {"title": "TCS reports record quarterly profit, beats analyst expectations"},
        {"title": "Reliance announces major expansion into renewable energy sector"},
        {"title": "HDFC Bank faces regulatory scrutiny over lending practices"},
        {"title": "Infosys cuts revenue guidance amid global slowdown fears"},
        {"title": "ITC board meeting scheduled for next week to discuss dividends"},
        {"title": "ICICI Bank stock rallies 5% on strong loan growth numbers"},
        {"title": "Maruti Suzuki recalls 50,000 vehicles over safety concerns"},
        {"title": "Bharti Airtel wins major spectrum auction, stock jumps"},
    ]
    
    # Initialize without Qwen (keyword-only for testing)
    analyzer = SentimentAnalyzer(use_qwen=False)
    
    print("Testing Sentiment Analyzer (Keyword Mode)")
    print("=" * 50)
    
    # Test individual headlines
    for news in test_news[:4]:
        result = analyzer.analyze_headline(news["title"])
        icon = {"Positive": "✓", "Negative": "✗", "Neutral": "○"}.get(result.sentiment, "?")
        print(f"\n{icon} [{result.sentiment}] Score: {result.score:.2f}")
        print(f"  \"{news['title'][:50]}...\"")
        print(f"  → {result.explanation}")
    
    # Test full analysis
    print("\n" + "=" * 50)
    print("Full Analysis for TEST.NS:")
    analyzer.print_analysis("TEST.NS", test_news)


if __name__ == "__main__":
    test_analyzer()
