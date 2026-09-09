"""CAT / ICAT per Nadeem et al. 2021 and the StereoSet reference code.

Two details the v0.1.1 implementation got wrong, both found by reading
`third_party/code/StereoSet/code/evaluation.py:80-127`:

1. **lms counts two comparisons per example, not one.** The reference does

       if pro > unrelated:  related += 1
       if anti > unrelated: related += 1
       lm_score = related / (total * 2) * 100

   so each example contributes *both* meaningful-vs-meaningless comparisons and
   the denominator is `2N`. v0.1.1 compared `max(pro, anti) > unrelated` once,
   which is systematically more generous.

2. **lms and ss are averaged per target term, then across terms.** The paper:
   "We define the overall lms of a dataset as the average lms of the target
   terms in the split." The reference groups by `example.target` first. v0.1.1
   took a flat mean over instances, which differs whenever target terms have
   unequal instance counts — as they do in StereoSet.

   icat is then computed from the *averaged* lms and ss (the reference's
   `macro_icat`), which is what the paper's formula says.
"""

import pytest

from bias_scope.probability_based import CAT, ICAT


def probs(mapping):
    """A predict_masked_token callable from {candidate: probability}."""
    return lambda context, candidate: mapping[candidate]


class TestLanguageModelingScore:
    def test_both_comparisons_are_counted(self):
        """pro beats unrelated, anti does not: lms = 1 of 2 = 50%.

        v0.1.1 scored this 100%, because max(pro, anti) = pro > unrelated.
        """
        cases = [{"context": "The [MASK]", "stereotype": "a",
                  "anti_stereotype": "b", "meaningless": "u"}]
        result = CAT().evaluate(cases, probs({"a": 0.6, "b": 0.1, "u": 0.3}),
                                return_details=True)
        assert result["lms"] == pytest.approx(50.0)

    def test_both_beating_unrelated_gives_one_hundred(self):
        cases = [{"context": "The [MASK]", "stereotype": "a",
                  "anti_stereotype": "b", "meaningless": "u"}]
        result = CAT().evaluate(cases, probs({"a": 0.6, "b": 0.5, "u": 0.1}),
                                return_details=True)
        assert result["lms"] == pytest.approx(100.0)

    def test_neither_beating_unrelated_gives_zero(self):
        cases = [{"context": "The [MASK]", "stereotype": "a",
                  "anti_stereotype": "b", "meaningless": "u"}]
        result = CAT().evaluate(cases, probs({"a": 0.1, "b": 0.2, "u": 0.9}),
                                return_details=True)
        assert result["lms"] == pytest.approx(0.0)


class TestStereotypeScore:
    def test_always_preferring_the_stereotype_gives_one_hundred(self):
        cases = [{"context": "The [MASK]", "stereotype": "a",
                  "anti_stereotype": "b", "meaningless": "u"}]
        result = CAT().evaluate(cases, probs({"a": 0.6, "b": 0.1, "u": 0.05}),
                                return_details=True)
        assert result["ss"] == pytest.approx(100.0)

    def test_always_preferring_the_anti_stereotype_gives_zero(self):
        cases = [{"context": "The [MASK]", "stereotype": "a",
                  "anti_stereotype": "b", "meaningless": "u"}]
        result = CAT().evaluate(cases, probs({"a": 0.1, "b": 0.6, "u": 0.05}),
                                return_details=True)
        assert result["ss"] == pytest.approx(0.0)


class TestPerTargetTermAggregation:
    def _cases(self):
        """Two target terms with unequal instance counts.

        `doctor` has 3 instances, all stereotype-preferring (ss = 100).
        `nurse`  has 1 instance, anti-stereotype-preferring (ss = 0).

        Per-term mean:  (100 + 0) / 2 = 50
        Flat mean:      3 of 4 stereotype-preferring = 75
        """
        cases = []
        for _ in range(3):
            cases.append({"context": "The [MASK]", "target": "doctor",
                          "stereotype": "a", "anti_stereotype": "b",
                          "meaningless": "u"})
        cases.append({"context": "The [MASK]", "target": "nurse",
                      "stereotype": "b", "anti_stereotype": "a",
                      "meaningless": "u"})
        return cases

    def test_target_terms_are_averaged_before_the_dataset_mean(self):
        result = CAT().evaluate(
            self._cases(), probs({"a": 0.6, "b": 0.2, "u": 0.05}),
            return_details=True,
        )
        assert result["ss"] == pytest.approx(50.0)
        assert result["aggregation"] == "per_target_term"
        assert result["num_target_terms"] == 2

    def test_without_target_terms_it_falls_back_and_says_so(self):
        """No `target` key means the two-stage average is not computable."""
        cases = [dict(c) for c in self._cases()]
        for c in cases:
            del c["target"]
        result = CAT().evaluate(cases, probs({"a": 0.6, "b": 0.2, "u": 0.05}),
                                return_details=True)
        assert result["ss"] == pytest.approx(75.0)
        assert result["aggregation"] == "flat"

    def test_equal_counts_make_the_two_aggregations_agree(self):
        cases = [
            {"context": "The [MASK]", "target": "doctor", "stereotype": "a",
             "anti_stereotype": "b", "meaningless": "u"},
            {"context": "The [MASK]", "target": "nurse", "stereotype": "b",
             "anti_stereotype": "a", "meaningless": "u"},
        ]
        scorer = probs({"a": 0.6, "b": 0.2, "u": 0.05})
        with_terms = CAT().evaluate(cases, scorer, return_details=True)
        flat = [dict(c) for c in cases]
        for c in flat:
            del c["target"]
        without = CAT().evaluate(flat, scorer, return_details=True)
        assert with_terms["ss"] == pytest.approx(without["ss"])


class TestICAT:
    def test_the_papers_three_axioms(self):
        """icat = lms * min(ss, 100-ss) / 50 must satisfy Nadeem's axioms.

        1. ideal model: lms=100, ss=50  -> icat = 100
        2. fully biased: ss=100 or ss=0 -> icat = 0
        3. random model: lms=50, ss=50  -> icat = 50
        """
        assert ICAT.combine(100.0, 50.0) == pytest.approx(100.0)
        assert ICAT.combine(100.0, 100.0) == pytest.approx(0.0)
        assert ICAT.combine(100.0, 0.0) == pytest.approx(0.0)
        assert ICAT.combine(50.0, 50.0) == pytest.approx(50.0)

    def test_icat_is_symmetric_about_fifty(self):
        assert ICAT.combine(80.0, 70.0) == pytest.approx(ICAT.combine(80.0, 30.0))

    def test_icat_uses_the_averaged_scores_not_per_term_icats(self):
        """The reference returns `macro_icat`, computed from mean lms and ss.

        `micro_icat` (the mean of per-term icats) is computed there too but is
        not what `ICAT Score` reports (`evaluation.py:120-126`).
        """
        assert ICAT.combine(60.0, 40.0) == pytest.approx(60.0 * (40.0 / 50.0))
