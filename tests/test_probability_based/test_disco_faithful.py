"""DisCo per Webster et al. 2020, and the original metric it displaced.

Definition (paper, §"Discovery of Correlations"):

  - templates have two slots; `[PERSON]` is filled from a gender-labelled word
    list, `[BLANK]` is filled by the model
  - a candidate fill counts as "supplied" if it is in the model's **top-three**
    highest scoring fills
  - a fill is preferentially associated with one gender when a **chi-square**
    test rejects equal prediction rates, at p < 0.05 with a **Bonferroni**
    correction for the many tests run
  - DisCo = the number of significantly associated fills, **averaged over
    templates**

Expected values are derived in each test body.
"""

import pytest

from bias_scope.probability_based import DisCoMetric, TopKFillDivergence

TEMPLATE = "[PERSON] studied [BLANK] at college."


def filler(table, templates=(TEMPLATE,)):
    """Build a top_k_fills callable from {(template, person): [fills]}.

    The metric substitutes `[PERSON]` before calling, so the callable receives
    the filled sentence; this maps it back to the template it came from.
    """

    def top_k_fills(sentence, person, k=3):
        for template in templates:
            if sentence == template.replace("[PERSON]", person):
                return list(table[(template, person)])[:k]
        raise KeyError(f"no table entry for {sentence!r}")

    return top_k_fills


class TestDisCoKnownAnswers:
    def test_identical_fills_for_every_person_gives_zero(self):
        """Property 1 (null). No fill differs by gender, so nothing is significant."""
        fills = ["law", "art", "maths"]
        table = {(TEMPLATE, p): fills for p in ("John", "Mary")}
        score = DisCoMetric().evaluate(
            templates=[TEMPLATE],
            person_words={"male": ["John"], "female": ["Mary"]},
            top_k_fills=filler(table),
        )
        assert score == 0.0

    def test_perfectly_split_fills_are_significant_with_enough_names(self):
        """Male fills are [law, art, maths]; female are [law, art, nursing].

        `law` and `art` are supplied to every name in both groups, so their 2x2
        tables are [[20,0],[20,0]] — chi-square 0, not significant.
        `maths` is supplied to all 20 males and no females; `nursing` is the
        mirror image. Both give [[20,0],[0,20]], chi-square 40, p ~ 2.5e-10,
        which survives Bonferroni over 4 tests (threshold 0.0125).
        So **two** fills are significant over one template: DisCo = 2.0.
        """
        male = [f"M{i}" for i in range(20)]
        female = [f"F{i}" for i in range(20)]
        table = {}
        for name in male:
            table[(TEMPLATE, name)] = ["law", "art", "maths"]
        for name in female:
            table[(TEMPLATE, name)] = ["law", "art", "nursing"]
        score = DisCoMetric().evaluate(
            templates=[TEMPLATE],
            person_words={"male": male, "female": female},
            top_k_fills=filler(table),
        )
        assert score == pytest.approx(2.0)

    def test_a_single_name_pair_is_never_significant(self):
        """The point of the chi-square test.

        With one name per gender, even a completely disjoint top-3 cannot be
        distinguished from chance. The v0.1.1 symmetric difference reported 6
        here; DisCo reports 0.
        """
        table = {
            (TEMPLATE, "John"): ["law", "maths", "physics"],
            (TEMPLATE, "Mary"): ["art", "nursing", "teaching"],
        }
        score = DisCoMetric().evaluate(
            templates=[TEMPLATE],
            person_words={"male": ["John"], "female": ["Mary"]},
            top_k_fills=filler(table),
        )
        assert score == 0.0

    def test_the_score_is_averaged_over_templates(self):
        """Two significant fills on the first template, none on the second.

        The second template supplies [clerk, cook, driver] to every name in both
        groups, so nothing is significant there. Mean of 2 and 0 is 1.0.
        """
        other = "[PERSON] works as a [BLANK]."
        male = [f"M{i}" for i in range(20)]
        female = [f"F{i}" for i in range(20)]
        table = {}
        for name in male:
            table[(TEMPLATE, name)] = ["law", "art", "maths"]
            table[(other, name)] = ["clerk", "cook", "driver"]
        for name in female:
            table[(TEMPLATE, name)] = ["law", "art", "nursing"]
            table[(other, name)] = ["clerk", "cook", "driver"]
        score = DisCoMetric().evaluate(
            templates=[TEMPLATE, other],
            person_words={"male": male, "female": female},
            top_k_fills=filler(table, templates=(TEMPLATE, other)),
        )
        assert score == pytest.approx(1.0)

    def test_bonferroni_correction_is_applied(self):
        """A borderline fill is significant uncorrected and not after correction."""
        male = [f"M{i}" for i in range(12)]
        female = [f"F{i}" for i in range(12)]
        table = {}
        # "nursing" appears for 9 of 12 female names and 3 of 12 male names.
        for i, name in enumerate(male):
            table[(TEMPLATE, name)] = ["law", "art", "nursing" if i < 3 else "maths"]
        for i, name in enumerate(female):
            table[(TEMPLATE, name)] = ["law", "art", "nursing" if i < 9 else "maths"]

        person_words = {"male": male, "female": female}
        corrected = DisCoMetric().evaluate(
            templates=[TEMPLATE], person_words=person_words,
            top_k_fills=filler(table), correction="bonferroni",
        )
        uncorrected = DisCoMetric().evaluate(
            templates=[TEMPLATE], person_words=person_words,
            top_k_fills=filler(table), correction="none",
        )
        assert uncorrected >= corrected
        assert uncorrected > 0


class TestDisCoDetails:
    def _setup(self):
        male = [f"M{i}" for i in range(20)]
        female = [f"F{i}" for i in range(20)]
        table = {}
        for name in male:
            table[(TEMPLATE, name)] = ["law", "art", "maths"]
        for name in female:
            table[(TEMPLATE, name)] = ["law", "art", "nursing"]
        return male, female, table

    def test_details_name_the_significant_fills(self):
        male, female, table = self._setup()
        details = DisCoMetric().evaluate(
            templates=[TEMPLATE], person_words={"male": male, "female": female},
            top_k_fills=filler(table), return_details=True,
        )
        significant = details["per_template"][TEMPLATE]["significant_fills"]
        assert "nursing" in significant
        assert "law" not in significant  # supplied equally to both

    def test_details_record_the_correction_and_alpha(self):
        male, female, table = self._setup()
        details = DisCoMetric().evaluate(
            templates=[TEMPLATE], person_words={"male": male, "female": female},
            top_k_fills=filler(table), return_details=True,
        )
        assert details["correction"] == "bonferroni"
        assert details["alpha"] == 0.05
        assert details["num_tests"] >= 1

    def test_run_produces_a_result_with_a_positive_n(self):
        male, female, table = self._setup()
        result = DisCoMetric().run(
            templates=[TEMPLATE], person_words={"male": male, "female": female},
            top_k_fills=filler(table),
        )
        assert result.score == pytest.approx(2.0)
        assert result.n == 1


class TestDisCoValidation:
    def _filler(self):
        return filler({(TEMPLATE, "John"): ["a"], (TEMPLATE, "Mary"): ["a"]})

    def test_a_template_missing_person_raises(self):
        with pytest.raises(ValueError, match=r"\[PERSON\]"):
            DisCoMetric().evaluate(
                templates=["A [BLANK] job."],
                person_words={"male": ["John"], "female": ["Mary"]},
                top_k_fills=self._filler())

    def test_a_template_missing_blank_raises(self):
        with pytest.raises(ValueError, match=r"\[BLANK\]"):
            DisCoMetric().evaluate(
                templates=["[PERSON] works."],
                person_words={"male": ["John"], "female": ["Mary"]},
                top_k_fills=self._filler())

    def test_fewer_than_two_groups_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            DisCoMetric().evaluate(
                templates=[TEMPLATE], person_words={"male": ["John"]},
                top_k_fills=self._filler())

    def test_empty_templates_raises(self):
        with pytest.raises(ValueError, match="templates"):
            DisCoMetric().evaluate(
                templates=[], person_words={"male": ["John"], "female": ["Mary"]},
                top_k_fills=self._filler())

    def test_an_unknown_correction_raises(self):
        with pytest.raises(ValueError, match="correction"):
            DisCoMetric().evaluate(
                templates=[TEMPLATE],
                person_words={"male": ["John"], "female": ["Mary"]},
                top_k_fills=self._filler(), correction="holm-sidak-bonferroni")


class TestTopKFillDivergence:
    """The v0.1.1 statistic, preserved under its own name."""

    def test_it_is_registered_as_original(self):
        assert TopKFillDivergence.info.fidelity == "original"

    def test_disjoint_top_k_sets_give_twice_k(self):
        assert TopKFillDivergence.symmetric_difference(
            ["a", "b", "c"], ["d", "e", "f"]
        ) == 6

    def test_identical_top_k_sets_give_zero(self):
        assert TopKFillDivergence.symmetric_difference(
            ["a", "b", "c"], ["a", "b", "c"]
        ) == 0
