"""
Unit tests for the Topic Clustering Service.
"""
from services.clustering import clusterer


def test_topic_clustering_basic():
    reviews = [
        {"review": "The food was delicious and hot.", "theme": "food_quality", "overall_label": "positive", "score": 5},
        {"review": "Amazing taste, perfectly spiced biryani.", "theme": "food_quality", "overall_label": "positive", "score": 5},
        {"review": "Delivery took 2 hours and food was stone cold.", "theme": "delivery_speed", "overall_label": "negative", "score": 1},
        {"review": "Driver was rude and courier got lost on delivery.", "theme": "delivery_speed", "overall_label": "negative", "score": 1},
        {"review": "The iOS mobile checkout app crashed during payment.", "theme": "app_experience", "overall_label": "negative", "score": 1},
        {"review": "App keeps crashing on the login screen.", "theme": "app_experience", "overall_label": "negative", "score": 1},
    ]

    result = clusterer.cluster_reviews(reviews, max_clusters=3)
    assert "clusters" in result
    assert "points" in result
    assert len(result["points"]) == len(reviews)
    assert result["num_clusters"] >= 2
    
    # Check that each point has 2D coordinates and cluster metadata
    for p in result["points"]:
        assert "x" in p
        assert "y" in p
        assert "cluster_name" in p
        assert "sentiment" in p


def test_topic_clustering_empty():
    result = clusterer.cluster_reviews([])
    assert result["clusters"] == []
    assert result["points"] == []

