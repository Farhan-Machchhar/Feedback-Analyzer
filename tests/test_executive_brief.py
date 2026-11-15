"""
Unit tests for the Executive Operational Brief Synthesis Engine.
"""
from services.executive_brief import brief_generator


def test_executive_brief_generation():
    reviews = [
        {
            "text": "The delivery took 90 minutes and arrived cold.",
            "overall_label": "negative",
            "overall_score": 1,
            "theme": "delivery_speed",
            "urgency": "medium",
            "root_cause": "Driver transit delay",
            "suggested_action": "Reduce delivery dispatch radius during peak hours",
        },
        {
            "text": "Food poisoning from raw chicken, went to emergency room!",
            "overall_label": "negative",
            "overall_score": 1,
            "theme": "food_quality",
            "urgency": "critical",
            "root_cause": "Raw poultry contamination",
            "suggested_action": "Audit internal meat cooking logs immediately",
        },
        {
            "text": "The biryani was sensational, warm and perfectly seasoned.",
            "overall_label": "positive",
            "overall_score": 5,
            "theme": "food_quality",
            "urgency": "low",
        },
    ]

    brief = brief_generator.generate_brief(reviews)
    assert "markdown" in brief
    assert "data" in brief
    assert "Executive" in brief["markdown"]
    assert "Operational" in brief["markdown"]
    assert len(brief["data"]["critical_risk_alerts"]) > 0


def test_executive_brief_empty():
    brief = brief_generator.generate_brief([])
    assert "No reviews provided" in brief["markdown"]

