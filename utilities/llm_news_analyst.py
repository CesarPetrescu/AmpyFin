"""News-driven sentiment analysis utilities for trading decisions."""
from __future__ import annotations

import json
import logging
import os
from typing import List, Tuple

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)


class NewsAnalyst:
    """Aggregates financial news and produces LLM-based sentiment scores."""

    def __init__(self) -> None:
        self.newsdata_key = os.getenv("NEWSDATA_API_KEY")
        self.marketaux_key = os.getenv("MARKETAUX_API_KEY")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

        self.client: OpenAI | None = None
        if self.deepseek_api_key:
            try:
                self.client = OpenAI(
                    api_key=self.deepseek_api_key,
                    base_url="https://api.deepseek.com",
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to initialize DeepSeek client: %s", exc)
        else:
            logger.warning("DEEPSEEK_API_KEY not set. News sentiment will be neutral.")

    def fetch_marketaux(self, ticker: str) -> List[str]:
        """Fetches finance-specific news from MarketAux."""
        if not self.marketaux_key:
            logger.warning("MARKETAUX_API_KEY not set. Skipping MarketAux fetch.")
            return []

        url = "https://api.marketaux.com/v1/news/all"
        params = {
            "api_token": self.marketaux_key,
            "symbols": ticker,
            "filter_entities": "true",
            "language": "en",
        }
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            headlines = []
            for art in data.get("data", [])[:3]:
                entity = art.get("entities", [{}])
                sentiment = entity[0].get("sentiment_score", "N/A") if entity else "N/A"
                headlines.append(f"{art.get('title', 'Untitled')} (Sentiment: {sentiment})")
            return headlines
        except Exception as exc:  # pragma: no cover - external API guard
            logger.warning("MarketAux Error for %s: %s", ticker, exc)
            return []

    def fetch_newsdata(self, ticker: str) -> List[str]:
        """Fetches broader market news from NewsData.io."""
        if not self.newsdata_key:
            logger.warning("NEWSDATA_API_KEY not set. Skipping NewsData fetch.")
            return []

        url = "https://newsdata.io/api/1/news"
        params = {
            "apikey": self.newsdata_key,
            "q": ticker,
            "language": "en",
            "category": "business",
        }
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            headlines = []
            for article in data.get("results", [])[:3]:
                title = article.get("title") or "Untitled"
                source = article.get("source_id") or "unknown source"
                headlines.append(f"{title} (Source: {source})")
            return headlines
        except Exception as exc:  # pragma: no cover - external API guard
            logger.warning("NewsData Error for %s: %s", ticker, exc)
            return []

    def get_aggregated_sentiment(self, ticker: str) -> Tuple[float, str]:
        """Fetches recent news and returns an LLM-evaluated sentiment score."""
        headlines = self.fetch_marketaux(ticker) + self.fetch_newsdata(ticker)

        if not headlines:
            return 0.0, "No news found."

        if not self.client:
            return 0.0, "LLM client unavailable."

        news_text = "\n".join(f"- {headline}" for headline in headlines)

        system_prompt = (
            "You are a Hedge Fund Risk Manager. Analyze these headlines for a specific asset.\n"
            "Output strictly valid JSON: {\"sentiment_score\": float (-1.0 to 1.0), \"reason\": \"brief string\"}"
        )

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Asset: {ticker}\nNews:\n{news_text}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            res = json.loads(response.choices[0].message.content)
            return float(res.get("sentiment_score", 0.0)), res.get("reason", "No reason provided")
        except Exception as exc:  # pragma: no cover - external API guard
            logger.warning("LLM Error for %s: %s", ticker, exc)
            return 0.0, "Error"
