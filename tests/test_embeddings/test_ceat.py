"""Mathematical tests for canonical CEAT."""

import math

import numpy as np
import pytest

from bias_scope.embeddings_based import CEAT
from bias_scope.embeddings_based._helpers import _ceat_random_effects, _weat_effect_components
from bias_scope.stats import Z_95


def _groups(n_contexts=4, dimension=3):
    """Distinct, non-degenerate contextual token embeddings per stimulus."""
    rng = np.random.default_rng(101)
    return tuple(
        {f"{prefix}{i}": rng.normal(loc=i, size=(n_contexts, dimension))
         for i in range(count)}
        for prefix, count in (("x", 3), ("y", 3), ("a", 2), ("b", 2))
    )


def test_every_stimulus_is_retained_once_per_iteration():
    x, y, a, b = _groups()
    result = CEAT().evaluate((x, y), (a, b), n_samples=3, random_seed=9, return_details=True)

    selected = result["sampled_context_indices"]
    assert set(selected["X"]) == set(x)
    assert set(selected["Y"]) == set(y)
    assert set(selected["A"]) == set(a)
    assert set(selected["B"]) == set(b)
    assert all(len(indices) == 3 for group in selected.values() for indices in group.values())


def test_selection_is_stimulus_aligned_and_uses_replacement_only_when_needed():
    x, y, a, b = _groups(n_contexts=5)
    result = CEAT().evaluate((x, y), (a, b), n_samples=5, random_seed=2, return_details=True)
    selected = result["sampled_context_indices"]
    assert all(len(set(indices)) == 5 for group in selected.values() for indices in group.values())

    x["x0"] = x["x0"][:2]
    with_replacement = CEAT().evaluate(
        (x, y), (a, b), n_samples=5, random_seed=2, return_details=True
    )
    assert len(with_replacement["sampled_context_indices"]["X"]["x0"]) == 5
    assert len(set(with_replacement["sampled_context_indices"]["X"]["x0"])) < 5


def test_sample_variance_is_square_of_weat_pooled_sd_not_standardized_variance():
    x, y, a, b = _groups(n_contexts=1, dimension=4)
    result = CEAT().evaluate((x, y), (a, b), n_samples=1, return_details=True)
    matrices = [np.vstack(list(group.values())) for group in (x, y, a, b)]
    _, _, pooled_sd, effect_size = _weat_effect_components(*matrices)
    expected_variance = pooled_sd**2
    old_formula = 2 / len(x) + effect_size**2 / (4 * len(x) - 4)
    assert result["sample_variances"][0] == pytest.approx(expected_variance)
    assert expected_variance != pytest.approx(old_formula)


def test_random_effects_matches_independent_oracle_and_fixed_effect_limit():
    effects = np.array([0.2, 1.5, -0.4])
    variances = np.array([0.4, 0.7, 0.2])
    actual = _ceat_random_effects(effects, variances)
    w = 1 / variances
    fixed = np.sum(w * effects) / np.sum(w)
    q = np.sum(w * (effects - fixed) ** 2)
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau = max(0, (q - 2) / c)
    v = 1 / (variances + tau)
    ces = np.sum(v * effects) / np.sum(v)
    se = math.sqrt(1 / np.sum(v))
    assert actual["fixed_effect_mean"] == pytest.approx(fixed)
    assert actual["Q"] == pytest.approx(q)
    assert actual["between_context_variance"] == pytest.approx(tau)
    assert actual["effect_size"] == pytest.approx(ces)
    assert actual["standard_error"] == pytest.approx(se)

    limit = _ceat_random_effects(np.array([1.0, 1.01]), np.array([0.5, 0.5]))
    assert limit["between_context_variance"] == 0
    assert limit["effect_size"] == pytest.approx(1.005)


def test_sign_symmetry_reproducibility_and_two_sided_p_value():
    x, y, a, b = _groups(n_contexts=6)
    original = CEAT().evaluate((x, y), (a, b), n_samples=4, random_seed=7, return_details=True)
    swapped = CEAT().evaluate((y, x), (a, b), n_samples=4, random_seed=7, return_details=True)
    attributes_swapped = CEAT().evaluate(
        (x, y), (b, a), n_samples=4, random_seed=7, return_details=True
    )
    again = CEAT().evaluate((x, y), (a, b), n_samples=4, random_seed=7, return_details=True)
    assert swapped["effect_size"] == pytest.approx(-original["effect_size"])
    assert swapped["p_value"] == pytest.approx(original["p_value"])
    assert swapped["sampled_context_indices"]["Y"] == original["sampled_context_indices"]["X"]
    assert swapped["sampled_context_indices"]["X"] == original["sampled_context_indices"]["Y"]
    assert attributes_swapped["effect_size"] == pytest.approx(-original["effect_size"])
    assert attributes_swapped["p_value"] == pytest.approx(original["p_value"])
    assert attributes_swapped["sampled_context_indices"]["B"] == original["sampled_context_indices"]["A"]  # noqa: E501
    assert attributes_swapped["sampled_context_indices"]["A"] == original["sampled_context_indices"]["B"]  # noqa: E501
    assert again["sampled_context_indices"] == original["sampled_context_indices"]
    assert again["sample_effect_sizes"] == original["sample_effect_sizes"]
    assert 0 <= original["p_value"] <= 1


def test_different_seeds_change_contextual_selection_when_contexts_allow_it():
    x, y, a, b = _groups(n_contexts=6)
    first = CEAT().evaluate((x, y), (a, b), n_samples=4, random_seed=1, return_details=True)
    second = CEAT().evaluate((x, y), (a, b), n_samples=4, random_seed=2, return_details=True)
    assert first["sampled_context_indices"] != second["sampled_context_indices"]


@pytest.mark.parametrize(
    "targets, attributes, message",
    [
        (({}, {"y": np.ones((1, 2))}), ({"a": np.ones((1, 2))}, {"b": np.ones((1, 2))}), "empty"),
        (({"x": np.ones((0, 2))}, {"y": np.ones((1, 2))}), ({"a": np.ones((1, 2))}, {"b": np.ones((1, 2))}), "zero contextual"),  # noqa: E501
        (({"x": np.ones((1, 2))}, {"y": np.ones((1, 3))}), ({"a": np.ones((1, 2))}, {"b": np.ones((1, 2))}), "same dimension"),  # noqa: E501
        (({"x": np.array([[np.nan, 1]])}, {"y": np.ones((1, 2))}), ({"a": np.ones((1, 2))}, {"b": np.ones((1, 2))}), "NaN or Inf"),  # noqa: E501
        (({"x": np.array([[np.inf, 1]])}, {"y": np.ones((1, 2))}), ({"a": np.ones((1, 2))}, {"b": np.ones((1, 2))}), "NaN or Inf"),  # noqa: E501
        (({"x": np.array(1.0)}, {"y": np.ones((1, 2))}), ({"a": np.ones((1, 2))}, {"b": np.ones((1, 2))}), "shape"),  # noqa: E501
        (({"x": np.ones((1, 2))}, {"y": np.ones((1, 2))}), ({"a": np.ones((1, 2))}, {"b": np.ones((1, 2))}), None),  # noqa: E501
    ],
)
def test_malformed_inputs_are_rejected(targets, attributes, message):
    if message is None:
        with pytest.raises(ValueError, match="equal numbers"):
            CEAT().evaluate((targets[0], {"y1": np.ones((1, 2)), "y2": np.ones((1, 2))}), attributes)  # noqa: E501
    else:
        with pytest.raises(ValueError, match=message):
            CEAT().evaluate(targets, attributes)


class TestCeatRun:
    """PLAN.md 5.3: run() must be reproducible under `seed` and report CEAT's
    own uncertainty, not a Hedges-Olkin interval built from stimulus counts."""

    def test_run_threads_and_records_the_random_seed(self):
        """Without threading, ``run(seed=...)`` would not reach ``random_seed``
        and every call would resample from OS entropy (as WEAT.run/SEAT.run
        already guard against for their own RNGs)."""
        x, y, a, b = _groups(n_contexts=40, dimension=6)
        result = CEAT().run((x, y), (a, b), seed=3, n_samples=50)
        direct = CEAT().evaluate(
            (x, y), (a, b), random_seed=3, n_samples=50, return_details=True
        )
        assert result.protocol["random_seed"] == 3
        assert result.score == pytest.approx(direct["effect_size"])
        assert result.p_value == pytest.approx(direct["p_value"])

    def test_run_is_reproducible_with_a_fixed_seed(self):
        x, y, a, b = _groups(n_contexts=40, dimension=6)
        first = CEAT().run((x, y), (a, b), seed=7, n_samples=50)
        second = CEAT().run((x, y), (a, b), seed=7, n_samples=50)
        assert first.score == second.score
        assert first.p_value == second.p_value

    def test_different_run_seeds_change_the_score(self):
        x, y, a, b = _groups(n_contexts=40, dimension=6)
        first = CEAT().run((x, y), (a, b), seed=1, n_samples=50)
        second = CEAT().run((x, y), (a, b), seed=2, n_samples=50)
        assert first.score != second.score

    def test_explicit_random_seed_overrides_run_seed(self):
        x, y, a, b = _groups(n_contexts=40, dimension=6)
        first = CEAT().run((x, y), (a, b), seed=1, random_seed=9, n_samples=50)
        second = CEAT().run((x, y), (a, b), seed=2, random_seed=9, n_samples=50)
        assert first.protocol["random_seed"] == second.protocol["random_seed"] == 9
        assert first.score == second.score

    def test_run_reports_the_random_effects_interval_not_hedges_olkin(self):
        """CEAT's uncertainty is SE(CES) from the random-effects model (paper
        Appendix 'Random-Effects Model Details'), not a Hedges-Olkin interval
        on |X|,|Y|. Before this fix a 3-vs-3-stimulus run reported a CI ~13x
        wider than the true SE(CES)."""
        x, y, a, b = _groups(n_contexts=40, dimension=6)
        result = CEAT().run((x, y), (a, b), seed=5, n_samples=200)
        se = result.details["standard_error"]
        assert result.ci_method == "random_effects"
        assert result.ci == pytest.approx(
            (result.score - Z_95 * se, result.score + Z_95 * se)
        )

    def test_run_n_is_the_number_of_sampled_contexts(self):
        """CEAT's 'n' is the meta-analysis sample size, not the stimulus count."""
        x, y, a, b = _groups(n_contexts=40, dimension=6)
        result = CEAT().run((x, y), (a, b), seed=5, n_samples=123)
        assert result.n == 123


def test_flat_arrays_raw_strings_and_sample_size_are_rejected():
    arrays = (np.ones((2, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError, match="mapping"):
        CEAT().evaluate(arrays, arrays)
    with pytest.raises(ValueError, match="mapping"):
        CEAT().evaluate((["x"], ["y"]), (["a"], ["b"]))
    x, y, a, b = _groups()
    with pytest.raises(ValueError, match="sample_size"):
        CEAT().evaluate((x, y), (a, b), sample_size=2)


class TestCeatIsReachableThroughRun:
    """RL-048: CEAT completed `evaluate()` but never `run()`.

    `run()` raised "n must be positive, got 0" for every input, because
    `_count_items` recognised no key in CEAT's details. CEAT reports
    `n_samples`, and the open question was whether that is an items-scored
    count or merely a permutation budget - a question about Guo & Caliskan
    (2021), not about plumbing, so it was left labelled rather than guessed.

    The authors' own code answers it. `ceat.py:205 ceat_meta(..., N=10000)`
    draws N samples, each yielding one effect size and one variance
    (`e_lst`, `v_lst`), and then computes the Q statistic with
    `df = N - 1` (`ceat.py:243`). N is the number of observations the
    random-effects meta-analysis pools - the degrees of freedom say so
    explicitly - so it is the count `n` is meant to carry.
    """

    def _result(self):
        rng = np.random.default_rng(0)

        def group(*names):
            return {name: rng.normal(size=(5, 16)) for name in names}

        return CEAT().run(
            target_embeddings=(group("x1", "x2", "x3"), group("y1", "y2", "y3")),
            attribute_embeddings=(group("a1", "a2"), group("b1", "b2")),
            n_samples=25,
            random_seed=42,
        )

    def test_run_completes_and_reports_the_sample_count_as_n(self):
        assert self._result().n == 25

    def test_the_score_is_finite(self):
        import math

        assert math.isfinite(self._result().score)
