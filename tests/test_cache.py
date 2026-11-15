"""
Unit tests for the Semantic Caching layer.
"""
from services.cache import SemanticCache
from models.schemas import AnalysisResult, SentimentLabel, UrgencyLevel, AspectMention


def test_exact_cache_hit():
    cache = SemanticCache(similarity_threshold=0.90)
    
    sample_result = AnalysisResult(
        text="Great pizza and quick service",
        overall_label=SentimentLabel.POSITIVE,
        overall_score=5,
        theme="food_quality",
        aspects=[AspectMention(aspect="food_quality", sentiment=SentimentLabel.POSITIVE, score=5, evidence="Great pizza")],
        urgency=UrgencyLevel.LOW,
    )
    
    # Put in cache
    cache.put("Great pizza and quick service", sample_result)
    
    # Query with exact string
    hit = cache.get("Great pizza and quick service")
    assert hit is not None
    result, sim = hit
    assert sim == 1.0
    assert result.cached is True
    assert result.overall_label == SentimentLabel.POSITIVE
    assert cache.hits == 1


def test_semantic_cache_hit_on_paraphrase():
    cache = SemanticCache(similarity_threshold=0.85)
    
    sample_result = AnalysisResult(
        text="The delivery took over an hour and food was cold.",
        overall_label=SentimentLabel.NEGATIVE,
        overall_score=1,
        theme="delivery_speed",
        aspects=[],
        urgency=UrgencyLevel.MEDIUM,
    )
    
    cache.put("The delivery took over an hour and food was cold.", sample_result)
    
    # Query with slight variation / paraphrase
    hit = cache.get("The delivery was over an hour and food was cold.")
    assert hit is not None
    result, sim = hit
    assert sim >= 0.85
    assert result.cached is True
    assert result.overall_label == SentimentLabel.NEGATIVE


def test_semantic_cache_miss_on_different_text():
    cache = SemanticCache(similarity_threshold=0.85)
    
    sample_result = AnalysisResult(
        text="Delicious burger and crunchy fries.",
        overall_label=SentimentLabel.POSITIVE,
        overall_score=5,
        theme="food_quality",
        aspects=[],
        urgency=UrgencyLevel.LOW,
    )
    cache.put("Delicious burger and crunchy fries.", sample_result)
    
    # Different text should miss
    hit = cache.get("The mobile app crashed during checkout.")
    assert hit is None
    assert cache.misses == 1

