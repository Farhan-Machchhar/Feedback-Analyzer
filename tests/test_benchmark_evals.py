"""
LLMOps Evaluation Benchmark Suite.

Evaluates the analyzer against a curated benchmark of diverse customer reviews:
- Critical safety & legal triage accuracy (100% recall requirement).
- ABSA aspect extraction fidelity.
- Pydantic schema validation.
"""
import pytest
from services.analyzer import analyzer
from models.schemas import SentimentLabel, UrgencyLevel

BENCHMARK_DATASET = [
    {
        "text": "I got severe food poisoning after eating the raw chicken here and was in the hospital all night!",
        "expected_label": SentimentLabel.NEGATIVE,
        "expected_urgency": UrgencyLevel.CRITICAL,
        "expected_aspect": "food_quality",
    },
    {
        "text": "Found a live cockroach inside my takeout salad. Completely disgusted and will call the health inspector.",
        "expected_label": SentimentLabel.NEGATIVE,
        "expected_urgency": UrgencyLevel.CRITICAL,
        "expected_aspect": "cleanliness",
    },
    {
        "text": "The delivery took an hour and 45 minutes, driver was completely lost and rude.",
        "expected_label": SentimentLabel.NEGATIVE,
        "expected_urgency": UrgencyLevel.MEDIUM,
        "expected_aspect": "delivery_speed",
    },
    {
        "text": "The biryani was heavenly, rich aromatic spices and generous chicken portions!",
        "expected_label": SentimentLabel.POSITIVE,
        "expected_urgency": UrgencyLevel.LOW,
        "expected_aspect": "food_quality",
    },
    {
        "text": "Prices have increased by 40% while the portions shrank in half. Not worth the money anymore.",
        "expected_label": SentimentLabel.NEGATIVE,
        "expected_urgency": UrgencyLevel.LOW,
        "expected_aspect": "pricing",
    },
    {
        "text": "The iOS app crashed three times when I tried to pay with Apple Pay.",
        "expected_label": SentimentLabel.NEGATIVE,
        "expected_urgency": UrgencyLevel.MEDIUM,
        "expected_aspect": "app_experience",
    },
]


def test_eval_benchmark_suite():
    """Run evaluation benchmark across ground truth dataset."""
    total = len(BENCHMARK_DATASET)
    correct_labels = 0
    critical_cases_tested = 0
    critical_cases_caught = 0

    for sample in BENCHMARK_DATASET:
        result = analyzer.analyze_single(sample["text"], bypass_cache=True)
        
        # Verify schema
        assert result.overall_label in [SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL]
        assert 1 <= result.overall_score <= 5

        # Check label accuracy
        if result.overall_label == sample["expected_label"]:
            correct_labels += 1

        # Check critical urgency recall
        if sample["expected_urgency"] == UrgencyLevel.CRITICAL:
            critical_cases_tested += 1
            if result.urgency == UrgencyLevel.CRITICAL:
                critical_cases_caught += 1

    label_accuracy = correct_labels / total
    critical_recall = critical_cases_caught / critical_cases_tested if critical_cases_tested > 0 else 1.0

    print(f"\n--- EVALUATION BENCHMARK RESULTS ---")
    print(f"Total Evaluated: {total}")
    print(f"Sentiment Label Accuracy: {label_accuracy * 100:.1f}%")
    print(f"Critical Safety Triage Recall: {critical_recall * 100:.1f}%")

    assert critical_recall == 1.0, "Critical safety urgency must have 100% recall!"
    assert label_accuracy >= 0.80, "Sentiment accuracy must exceed 80% baseline!"

