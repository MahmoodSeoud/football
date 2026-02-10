"""
Tests for receiver prediction baselines.

Phase 3 of RECEIVER_GNN_FIX_PLAN.md: Complete Baselines
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import existing functions
from models.receiver_model import (
    majority_class_baseline,
    nearest_player_baseline,
)

# Import new functions (may not exist yet - TDD Red phase)
try:
    from models.receiver_model import frequency_weighted_baseline
except ImportError:
    frequency_weighted_baseline = None

try:
    from models.receiver_model import baseline_topk_accuracy
except ImportError:
    baseline_topk_accuracy = None


@pytest.mark.skipif(frequency_weighted_baseline is None,
                    reason="frequency_weighted_baseline not implemented yet")
class TestFrequencyWeightedBaseline:
    """Tests for frequency_weighted_baseline function."""

    def test_returns_correct_shape(self):
        """Output should have n_samples elements."""
        train_labels = np.array([0, 0, 0, 1, 1, 2])
        n_samples = 100
        preds = frequency_weighted_baseline(train_labels, n_samples, seed=42)
        assert preds.shape == (n_samples,)

    def test_predictions_within_valid_range(self):
        """All predictions should be valid class indices 0-22."""
        train_labels = np.array([0, 5, 10, 15, 20, 22])
        preds = frequency_weighted_baseline(train_labels, 1000, seed=42)
        assert preds.min() >= 0
        assert preds.max() <= 22

    def test_respects_class_frequency(self):
        """More frequent classes should be predicted more often."""
        # Class 5 appears 90% of the time in training
        train_labels = np.array([5] * 900 + [10] * 100)
        preds = frequency_weighted_baseline(train_labels, 10000, seed=42)

        class_5_ratio = (preds == 5).sum() / len(preds)
        class_10_ratio = (preds == 10).sum() / len(preds)

        # Should approximately match training distribution
        assert 0.85 < class_5_ratio < 0.95
        assert 0.05 < class_10_ratio < 0.15

    def test_deterministic_with_seed(self):
        """Same seed should produce same predictions."""
        train_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        preds1 = frequency_weighted_baseline(train_labels, 100, seed=42)
        preds2 = frequency_weighted_baseline(train_labels, 100, seed=42)
        np.testing.assert_array_equal(preds1, preds2)

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different predictions."""
        train_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        preds1 = frequency_weighted_baseline(train_labels, 100, seed=42)
        preds2 = frequency_weighted_baseline(train_labels, 100, seed=123)
        assert not np.array_equal(preds1, preds2)


@pytest.mark.skipif(baseline_topk_accuracy is None,
                    reason="baseline_topk_accuracy not implemented yet")
class TestBaselineTopkAccuracy:
    """Tests for baseline_topk_accuracy function."""

    def test_top1_majority_class(self):
        """Top-1 accuracy for majority class baseline."""
        train_labels = np.array([5, 5, 5, 5, 5, 10, 10])  # 5 is majority
        test_labels = np.array([5, 5, 5, 10, 10, 15])

        # Top-1: only class 5 is predicted
        # Test has 3 class-5 labels out of 6 total
        acc = baseline_topk_accuracy(train_labels, test_labels, k=1)
        assert acc == pytest.approx(3/6)

    def test_top3_includes_more_classes(self):
        """Top-3 should include the 3 most frequent classes."""
        # Training: class 0 (4x), class 1 (3x), class 2 (2x), class 3 (1x)
        train_labels = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
        # Test: classes 0, 1, 2 should be hits, class 3 and 10 should be misses
        test_labels = np.array([0, 1, 2, 3, 10])

        # Top-3 classes are 0, 1, 2 (most frequent in training)
        # Test: 3 hits (0, 1, 2), 2 misses (3, 10)
        acc = baseline_topk_accuracy(train_labels, test_labels, k=3)
        assert acc == pytest.approx(3/5)

    def test_top5_accuracy(self):
        """Top-5 should include the 5 most frequent classes."""
        # Training: classes 0-4 each appear once
        train_labels = np.array([0, 1, 2, 3, 4])
        # Test: classes 0-4 are hits, class 10 is a miss
        test_labels = np.array([0, 1, 2, 3, 4, 10])

        acc = baseline_topk_accuracy(train_labels, test_labels, k=5)
        assert acc == pytest.approx(5/6)

    def test_perfect_accuracy_when_all_in_topk(self):
        """Perfect accuracy when all test labels are in top-k."""
        train_labels = np.array([0, 0, 1, 1, 2, 2])
        test_labels = np.array([0, 1, 2, 0, 1, 2])

        acc = baseline_topk_accuracy(train_labels, test_labels, k=3)
        assert acc == 1.0

    def test_zero_accuracy_when_none_in_topk(self):
        """Zero accuracy when no test labels are in top-k."""
        train_labels = np.array([0, 0, 1, 1, 2, 2])  # Top-3 are 0, 1, 2
        test_labels = np.array([10, 11, 12, 15, 20])  # None in top-3

        acc = baseline_topk_accuracy(train_labels, test_labels, k=3)
        assert acc == 0.0

    def test_handles_23_classes(self):
        """Should handle full 23-class receiver prediction."""
        train_labels = np.array(list(range(23)) * 10)  # Equal frequency
        test_labels = np.array([0, 5, 10, 15, 22])

        # With equal frequency, top-5 could be any 5 classes
        # This test just ensures it doesn't crash
        acc = baseline_topk_accuracy(train_labels, test_labels, k=5)
        assert 0.0 <= acc <= 1.0


class TestExistingBaselines:
    """Tests for existing baseline functions (regression tests)."""

    def test_majority_class_baseline(self):
        """Majority class baseline returns correct majority."""
        train_labels = np.array([2, 2, 2, 5, 10])  # 2 is majority
        preds = majority_class_baseline(train_labels, 10)

        assert len(preds) == 10
        assert (preds == 2).all()

    def test_nearest_player_baseline(self):
        """Nearest player baseline finds closest player to ball."""
        # 3 samples, 23 nodes (22 players + 1 ball), 8 features
        # Ball is node 22, positions are features 0:2
        node_features = np.zeros((3, 23, 8))

        # Place all players far away by default
        for sample in range(3):
            for player in range(22):
                node_features[sample, player, 0:2] = [10.0, 10.0]

        # Sample 0: Player 0 closest to ball
        node_features[0, 0, 0:2] = [0.1, 0.1]
        node_features[0, 22, 0:2] = [0.0, 0.0]  # Ball position

        # Sample 1: Player 5 closest to ball
        node_features[1, 5, 0:2] = [0.5, 0.5]
        node_features[1, 22, 0:2] = [0.5, 0.6]  # Ball position

        # Sample 2: Player 21 closest to ball
        node_features[2, 21, 0:2] = [1.0, 1.0]
        node_features[2, 22, 0:2] = [0.9, 0.9]  # Ball position

        preds = nearest_player_baseline(node_features)

        assert preds[0] == 0
        assert preds[1] == 5
        assert preds[2] == 21
