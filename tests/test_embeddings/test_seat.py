"""Tests for SEAT (Sentence Encoder Association Test)."""

import numpy as np
import pytest
import torch

from bias_scope.embeddings_based import SEAT


class TestSEAT:
    """Test Sentence Encoder Association Test."""

    def test_basic_functionality(self):
        """Test SEAT with simple inputs."""
        target1 = np.random.randn(4, 768)
        target2 = np.random.randn(4, 768)
        attr1 = np.random.randn(3, 768)
        attr2 = np.random.randn(3, 768)

        # Test OO API
        seat_instance = SEAT()
        score = seat_instance.evaluate((target1, target2), (attr1, attr2))

        assert isinstance(score, float)
        assert not np.isnan(score)

    def test_calls_weat(self):
        """Test SEAT produces same result as WEAT."""
        from bias_scope.embeddings_based import WEAT

        target1 = np.random.randn(4, 768)
        target2 = np.random.randn(4, 768)
        attr1 = np.random.randn(3, 768)
        attr2 = np.random.randn(3, 768)

        weat = WEAT()
        seat = SEAT()

        weat_score = weat.evaluate((target1, target2), (attr1, attr2))
        seat_score = seat.evaluate((target1, target2), (attr1, attr2))

        assert weat_score == seat_score

    def test_sentence_embeddings(self):
        """Test SEAT with typical sentence embedding dimensions."""
        seat = SEAT()

        target1 = np.random.randn(5, 768)
        target2 = np.random.randn(5, 768)
        attr1 = np.random.randn(3, 768)
        attr2 = np.random.randn(3, 768)

        score = seat.evaluate((target1, target2), (attr1, attr2))
        assert isinstance(score, float)
        assert not np.isnan(score)

    def test_seat_very_long_sentences(self):
        """Test SEAT with long sentence embeddings."""
        seat = SEAT()

        target1 = np.random.randn(5, 768)
        target2 = np.random.randn(5, 768)
        attr1 = np.random.randn(3, 768)
        attr2 = np.random.randn(3, 768)

        score = seat.evaluate((target1, target2), (attr1, attr2))
        assert isinstance(score, float)

    def test_seat_different_group_sizes(self):
        """Test SEAT with unbalanced groups."""
        seat = SEAT()

        target1 = np.random.randn(3, 768)
        target2 = np.random.randn(10, 768)
        attr1 = np.random.randn(5, 768)
        attr2 = np.random.randn(2, 768)

        score = seat.evaluate((target1, target2), (attr1, attr2))
        assert isinstance(score, float)

    def test_with_torch_tensors(self):
        """Test SEAT handles PyTorch tensors."""
        seat = SEAT()

        target1 = torch.randn(4, 768)
        target2 = torch.randn(4, 768)
        attr1 = torch.randn(3, 768)
        attr2 = torch.randn(3, 768)

        score = seat.evaluate((target1, target2), (attr1, attr2))
        assert isinstance(score, float)
