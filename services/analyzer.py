"""
Hybrid AI Analyzer Service.

Implements:
1. Semantic Cache check (sub-millisecond, $0 cost).
2. Urgency & Critical Safety Guardrails (food poisoning, legal threats, harassment).
3. Gemini LLM Structured Aspect-Based Sentiment Analysis (ABSA).
4. Resilience with Tenacity exponential backoff on HTTP 429 / 503 errors.
5. Offline Rule-Based Fallback Engine (guarantees tests & demo work without live API key).
"""
import time
import re
from typing import Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from google import genai
from google.genai import types

from core.config import settings, logger
from models.schemas import AnalysisResult, AspectMention, SentimentLabel, UrgencyLevel
from services.cache import semantic_cache


class FeedbackAnalyzer:
    """Enterprise Feedback Analysis Engine."""

    def __init__(self):
        self.api_key = settings.effective_api_key
        self.client = None
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("Initialized Google GenAI Client with configured API key.")
            except Exception as e:
                logger.warning(f"Could not initialize GenAI client: {e}")

    def is_live_llm_ready(self) -> bool:
        return self.client is not None and bool(self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call_gemini_structured(self, text: str) -> AnalysisResult:
        """Execute structured LLM inference using Gemini 2.5/Flash."""
        prompt = (
            "You are an expert customer feedback intelligence analyst for restaurants and retail services.\n"
            "Perform rigorous Aspect-Based Sentiment Analysis (ABSA) on the following customer review.\n\n"
            "Guidelines:\n"
            "1. overall_label: 'positive', 'negative', or 'neutral'.\n"
            "2. overall_score: 1 (very bad) to 5 (excellent).\n"
            "3. theme: One-to-two words describing the primary operational subject (e.g., 'delivery_speed', 'food_quality', 'pricing', 'service_attitude').\n"
            "4. aspects: Extract distinct operational aspects from the text. Categories: food_quality, delivery_speed, service_staff, pricing, ambiance, cleanliness, order_accuracy, app_experience.\n"
            "   For each aspect: assign category, sentiment (positive/negative/neutral), score (1-5), and exact evidence quotation.\n"
            "5. urgency: 'low', 'medium', 'high', or 'critical'. Use 'critical' for food poisoning, severe illness, foreign objects in food, physical/verbal abuse, or legal threats.\n"
            "6. root_cause: If negative, summarize the operational root cause concisely; otherwise null.\n"
            "7. suggested_action: Concrete, actionable remedy for the manager.\n\n"
            f"Review: \"{text}\""
        )

        response = self.client.models.generate_content(
            model=settings.MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnalysisResult,
                temperature=0.1,
            ),
        )
        parsed: AnalysisResult = response.parsed
        parsed.text = text
        parsed.source = "gemini_llm"
        return parsed

    def _fallback_rule_analyzer(self, text: str) -> AnalysisResult:
        """
        Intelligent offline heuristic analyzer for local testing, CI,
        or when external LLM rate limits or missing credentials occur.
        """
        lower = text.lower()
        
        # 1. Critical safety & urgency triage
        critical_keywords = [
            "poisoning", "hospital", "vomit", "sick", "raw chicken",
            "cockroach", "rat", "hair in food", "sue", "lawyer", "police", "harass"
        ]
        high_urgency_keywords = ["terrible", "awful", "never again", "disaster", "refund", "rude", "scam", "disgusted"]
        
        urgency = UrgencyLevel.LOW
        if any(w in lower for w in critical_keywords):
            urgency = UrgencyLevel.CRITICAL
        elif any(w in lower for w in high_urgency_keywords):
            urgency = UrgencyLevel.HIGH
        elif "not happy" in lower or "slow" in lower or "cold" in lower or "crashing" in lower:
            urgency = UrgencyLevel.MEDIUM

        # 2. Aspect extraction
        aspects: List[AspectMention] = []
        theme = "general"
        root_cause = None
        suggested_action = None

        # Cleanliness & Hygiene check
        if any(w in lower for w in ["cockroach", "rat", "dirty", "clean", "hygiene", "hair in food", "disgusted"]):
            theme = "cleanliness"
            is_pos = "spotless" in lower or "clean and" in lower
            score = 5 if is_pos else 1
            sentiment = SentimentLabel.POSITIVE if is_pos else SentimentLabel.NEGATIVE
            aspects.append(AspectMention(
                aspect="cleanliness",
                sentiment=sentiment,
                score=score,
                evidence=text,
            ))
            if not is_pos:
                root_cause = "Sanitation inspection lapse and hygiene protocol failure"
                suggested_action = "Initiate immediate kitchen deep clean and pest control inspection"

        # Delivery check
        if any(w in lower for w in ["delivery", "hour", "late", "arrived", "driver", "courier", "on time"]):
            if theme == "general":
                theme = "delivery_speed"
            is_pos = any(w in lower for w in ["hot and early", "on time", "fast", "polite", "quick"])
            score = 5 if is_pos else 1
            sentiment = SentimentLabel.POSITIVE if is_pos else SentimentLabel.NEGATIVE
            aspects.append(AspectMention(
                aspect="delivery_speed",
                sentiment=sentiment,
                score=score,
                evidence=text,
            ))
            if not is_pos and not root_cause:
                root_cause = "Courier dispatch latency and transit delays"
                suggested_action = "Review delivery radius and driver dispatch efficiency during peak hours"

        # Food Quality & Illness check
        if any(w in lower for w in ["food", "delicious", "taste", "biryani", "portion", "cold food", "flavor", "chicken", "raw chicken", "poisoning"]):
            if theme == "general":
                theme = "food_quality"
            has_poisoning = "poisoning" in lower or "raw chicken" in lower or "sick" in lower
            if has_poisoning:
                is_pos = False
                score = 1
            else:
                is_pos = any(w in lower for w in ["delicious", "amazing", "perfect", "tasty", "consistent", "heavenly"])
                score = 5 if is_pos else (2 if "cold" in lower else 1)
            sentiment = SentimentLabel.POSITIVE if is_pos else SentimentLabel.NEGATIVE
            aspects.append(AspectMention(
                aspect="food_quality",
                sentiment=sentiment,
                score=score,
                evidence=text,
            ))
            if has_poisoning:
                root_cause = "Possible foodborne illness contamination in raw poultry handling"
                suggested_action = "Quarantine poultry batch, log internal meat cooking temperatures, and reach out to guest immediately"
            elif "cold" in lower and not root_cause:
                root_cause = "Insufficient thermal packaging or extended transit time"
                suggested_action = "Upgrade insulated delivery bags and expedite packaging upon kitchen handoff"

        # Pricing check
        if any(w in lower for w in ["price", "prices", "expensive", "cost", "value for money", "money", "shrank"]):
            if theme == "general":
                theme = "pricing"
            is_neg = "not worth" in lower or "expensive" in lower or "gone up" in lower or "increased" in lower or "shrank" in lower
            is_pos = ("great value" in lower or "worth" in lower) and not is_neg
            score = 4 if is_pos else 1
            sentiment = SentimentLabel.POSITIVE if is_pos else SentimentLabel.NEGATIVE
            aspects.append(AspectMention(
                aspect="pricing",
                sentiment=sentiment,
                score=score,
                evidence=text,
            ))
            if not is_pos and not suggested_action:
                suggested_action = "Review portion-to-price ratio or introduce value combo options"

        # Service staff check
        if any(w in lower for w in ["service", "staff", "waiter", "rude", "polite", "called"]):
            if theme == "general":
                theme = "service_staff"
            is_pos = "polite" in lower or "friendly" in lower
            score = 5 if is_pos else 1
            sentiment = SentimentLabel.POSITIVE if is_pos else SentimentLabel.NEGATIVE
            aspects.append(AspectMention(
                aspect="service_staff",
                sentiment=sentiment,
                score=score,
                evidence=text,
            ))
            if not is_pos:
                root_cause = "Customer support / staff communication breakdown"
                suggested_action = "Conduct frontline empathy training and call-handling coaching"

        # App / Digital check
        if any(w in lower for w in ["app", "crash", "crashing", "order", "bug"]):
            if theme == "general":
                theme = "app_experience"
            score = 1
            aspects.append(AspectMention(
                aspect="app_experience",
                sentiment=SentimentLabel.NEGATIVE,
                score=score,
                evidence=text,
            ))
            root_cause = "Client application checkout stability crash"
            suggested_action = "Investigate mobile checkout crash logs and deploy bug fix"

        # Overall polarity aggregation
        if not aspects:
            # Fallback aspect
            is_pos = any(w in lower for w in ["good", "great", "nice", "love", "awesome"])
            is_neg = any(w in lower for w in ["bad", "poor", "hate", "terrible", "worst"])
            overall_label = SentimentLabel.POSITIVE if is_pos else (SentimentLabel.NEGATIVE if is_neg else SentimentLabel.NEUTRAL)
            overall_score = 4 if is_pos else (2 if is_neg else 3)
            aspects.append(AspectMention(
                aspect="general",
                sentiment=overall_label,
                score=overall_score,
                evidence=text,
            ))
        else:
            avg_aspect_score = sum(a.score for a in aspects) / len(aspects)
            overall_score = int(round(avg_aspect_score))
            if overall_score >= 4:
                overall_label = SentimentLabel.POSITIVE
            elif overall_score <= 2:
                overall_label = SentimentLabel.NEGATIVE
            else:
                overall_label = SentimentLabel.NEUTRAL

        return AnalysisResult(
            text=text,
            overall_label=overall_label,
            overall_score=overall_score,
            theme=theme,
            aspects=aspects,
            urgency=urgency,
            root_cause=root_cause,
            suggested_action=suggested_action,
            source="rule_engine_fallback",
        )

    def analyze_single(self, text: str, bypass_cache: bool = False) -> AnalysisResult:
        """
        Analyze a single customer review with full caching, hybrid fallback,
        and latency measurement.
        """
        start_time = time.perf_counter()

        # Step 1: Semantic Cache lookup
        if settings.ENABLE_SEMANTIC_CACHE and not bypass_cache:
            cache_hit = semantic_cache.get(text)
            if cache_hit:
                result, similarity = cache_hit
                result.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return result

        # Step 2: Live LLM inference (if available)
        result: Optional[AnalysisResult] = None
        if self.is_live_llm_ready():
            try:
                result = self._call_gemini_structured(text)
            except Exception as e:
                logger.error(f"Gemini API call failed: {e}. Falling back to rule engine.")

        # Step 3: Heuristic Fallback Engine
        if not result:
            result = self._fallback_rule_analyzer(text)

        # Record latency
        result.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Step 4: Cache the result
        if settings.ENABLE_SEMANTIC_CACHE:
            semantic_cache.put(text, result)

        return result


# Global singleton instance
analyzer = FeedbackAnalyzer()

