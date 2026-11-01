"""Tests for benchmark evaluation metrics."""

import pytest
from src.evaluation import MetricCalculator


def test_exact_match():
    calc = MetricCalculator()
    assert calc.compute_exact_match("Chicago", "chicago") == 1.0
    assert calc.compute_exact_match("The Grand Prix", "Grand Prix") == 1.0
    assert calc.compute_exact_match("New York", "Chicago") == 0.0
    # Test alias support
    assert calc.compute_exact_match("NYC", "New York City", aliases=["NYC", "New York"]) == 1.0


def test_token_f1():
    calc = MetricCalculator()
    # Identical
    assert calc.compute_token_f1("Grand Prix Spécial", "Grand Prix Spécial") == 1.0
    # Partial overlap: "John Doe" vs "John"
    # precision = 1/2, recall = 1/1, f1 = 2 * 0.5 * 1.0 / 1.5 = 2/3 = 0.666...
    f1 = calc.compute_token_f1("John Doe", "John")
    assert round(f1, 2) == 0.67
    # No overlap
    assert calc.compute_token_f1("Paris", "London") == 0.0
    # Alias support
    assert calc.compute_token_f1("Hillage", "Steve Hillage", aliases=["Hillage"]) == 1.0


def test_multi_hop_recall():
    calc = MetricCalculator()
    retrieved = ["c1", "c2", "c3"]
    # All gold chunks present
    assert calc.compute_multi_hop_recall(retrieved, ["c1", "c2"]) == 1.0
    # Missing one gold chunk
    assert calc.compute_multi_hop_recall(retrieved, ["c1", "c4"]) == 0.0


def test_mrr_at_k():
    calc = MetricCalculator()
    retrieved = ["c3", "c1", "c2"]
    # First gold chunk is c1 at rank 2
    assert calc.compute_mrr_at_k(retrieved, ["c1", "c2"], k=5) == 0.5
    # Gold chunk not in top 2
    assert calc.compute_mrr_at_k(retrieved, ["c4"], k=5) == 0.0


def test_faithfulness():
    calc = MetricCalculator()
    context = "John Doe was born in Chicago, Illinois in 1950."
    pred_grounded = "John Doe was born in Chicago."
    assert calc.compute_faithfulness(pred_grounded, context) == 1.0

    pred_hallucinated = "The Martian spaceship arrived at Jupiter."
    assert calc.compute_faithfulness(pred_hallucinated, context) == 0.0
