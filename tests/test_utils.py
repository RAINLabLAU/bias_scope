"""Tests for bias_scope.utils."""

import hashlib
import json
import os
import random

import numpy as np
import pytest

from bias_scope.utils import cosine_similarity, protocol_hash, seed_everything, to_numpy


class TestSeedEverything:
    """seed_everything makes random, numpy (and torch, when present) reproducible."""

    def test_returns_the_seed_it_applied(self):
        assert seed_everything(7) == 7

    def test_default_seed_is_42(self):
        assert seed_everything() == 42

    def test_sets_pythonhashseed(self):
        seed_everything(123)
        assert os.environ["PYTHONHASHSEED"] == "123"

    def test_random_is_reproducible(self):
        seed_everything(0)
        first = [random.random() for _ in range(5)]
        seed_everything(0)
        assert [random.random() for _ in range(5)] == first

    def test_numpy_is_reproducible(self):
        seed_everything(0)
        first = np.random.rand(5)
        seed_everything(0)
        assert np.array_equal(np.random.rand(5), first)

    def test_different_seeds_give_different_draws(self):
        seed_everything(0)
        a = np.random.rand(5)
        seed_everything(1)
        assert not np.array_equal(np.random.rand(5), a)

    def test_torch_is_reproducible_when_installed(self):
        torch = pytest.importorskip("torch")
        seed_everything(0)
        first = torch.randn(5)
        seed_everything(0)
        assert torch.equal(torch.randn(5), first)

    @pytest.mark.parametrize("bad", ["42", 4.2, None, True])
    def test_non_int_seed_raises_naming_the_argument(self, bad):
        with pytest.raises(ValueError, match="seed"):
            seed_everything(bad)


class TestProtocolHash:
    """protocol_hash = first 12 hex chars of sha256(canonical JSON)."""

    def test_known_answer(self):
        # Hand-derived: canonical JSON of this dict is exactly
        # '{"metric":"WEAT","model_id":"gpt2","seed":42}' (keys sorted, no
        # whitespace); sha256 of those bytes starts with c9c1562f13e7.
        protocol = {"seed": 42, "metric": "WEAT", "model_id": "gpt2"}
        canonical = '{"metric":"WEAT","model_id":"gpt2","seed":42}'
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
        assert expected == "c9c1562f13e7"
        assert protocol_hash(protocol) == "c9c1562f13e7"

    def test_is_twelve_lowercase_hex_chars(self):
        digest = protocol_hash({"a": 1})
        assert len(digest) == 12
        assert all(c in "0123456789abcdef" for c in digest)

    def test_key_order_does_not_matter(self):
        assert protocol_hash({"a": 1, "b": 2}) == protocol_hash({"b": 2, "a": 1})

    def test_a_changed_value_changes_the_hash(self):
        assert protocol_hash({"seed": 42}) != protocol_hash({"seed": 43})

    def test_nested_structures_are_canonicalised(self):
        left = {"decoding": {"top_p": 0.95, "k": 5}}
        right = {"decoding": {"k": 5, "top_p": 0.95}}
        assert protocol_hash(left) == protocol_hash(right)

    def test_non_serialisable_values_fall_back_to_str(self):
        # dtype objects appear in real protocol blocks; they must not raise.
        digest = protocol_hash({"dtype": np.float32})
        assert len(digest) == 12

    def test_empty_protocol_is_the_hash_of_empty_json(self):
        expected = hashlib.sha256(b"{}").hexdigest()[:12]
        assert protocol_hash({}) == expected

    @pytest.mark.parametrize("bad", [None, [], "{}", 42])
    def test_non_dict_raises_naming_the_argument(self, bad):
        with pytest.raises(ValueError, match="protocol"):
            protocol_hash(bad)

    def test_matches_a_from_scratch_json_dump(self):
        protocol = {"metric": "HONEST", "seed": 42, "resources": ["hurtlex-1.2"]}
        canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
        assert protocol_hash(protocol) == hashlib.sha256(canonical.encode()).hexdigest()[:12]


class TestToNumpy:
    """to_numpy accepts lists, arrays, and torch tensors."""

    def test_list_becomes_array(self):
        assert np.array_equal(to_numpy([1.0, 2.0]), np.array([1.0, 2.0]))

    def test_array_passes_through(self):
        arr = np.array([1.0, 2.0])
        assert to_numpy(arr) is arr

    def test_torch_tensor_is_detached_when_installed(self):
        torch = pytest.importorskip("torch")
        tensor = torch.ones(3, requires_grad=True)
        assert np.array_equal(to_numpy(tensor), np.ones(3))


class TestCosineSimilarity:
    """cos(theta) = (A . B) / (||A|| ||B||)."""

    def test_identical_vectors_give_one(self):
        assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0

    def test_orthogonal_vectors_give_zero(self):
        assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0

    def test_opposite_vectors_give_minus_one(self):
        assert cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == -1.0

    def test_zero_vector_gives_zero_rather_than_nan(self):
        assert cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0
