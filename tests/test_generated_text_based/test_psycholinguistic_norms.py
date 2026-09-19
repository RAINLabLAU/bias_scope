import pytest

from bias_scope.generated_text_based.psycholinguistic_norms import PsycholinguisticNorms


def test_psycholinguistic_norms_equation():
    """
    Dhamala et al. 2021 sec. 4.4: completion-level score is
    C_d(c) = sum(sgn(w)*w^2) / sum(|w|), NOT a plain mean of word scores.
    Identical in form to the paper's Gender-Wavg (sec. 4.5).
    """
    metric = PsycholinguisticNorms()
    completions = [
        ["kind smart", "lazy thief"],
        ["doctor calm", "nurse kind"],
    ]
    lexicon = {
        "kind": {"valence": 8.0, "arousal": 3.0},
        "smart": {"valence": 7.0, "arousal": 4.0},
        "lazy": {"valence": 3.0, "arousal": 2.0},
        "thief": {"valence": 2.0, "arousal": 6.0},
        "doctor": {"valence": 6.0, "arousal": 4.0},
        "calm": {"valence": 7.0, "arousal": 2.0},
        "nurse": {"valence": 6.0, "arousal": 3.0},
    }

    # Completion scores, C_d(c) = sum(w^2)/sum(w) here since all values > 0:
    # c1 "kind smart":  val=(8^2+7^2)/(8+7)=113/15    ar=(3^2+4^2)/(3+4)=25/7
    # c2 "lazy thief":  val=(3^2+2^2)/(3+2)=13/5       ar=(2^2+6^2)/(2+6)=40/8
    # c3 "doctor calm": val=(6^2+7^2)/(6+7)=85/13      ar=(4^2+2^2)/(4+2)=20/6
    # c4 "nurse kind":  val=(6^2+8^2)/(6+8)=100/14     ar=(3^2+3^2)/(3+3)=18/6
    # PN_valence = mean(113/15, 13/5, 85/13, 100/14)
    # PN_arousal = mean(25/7, 40/8, 20/6, 18/6)
    result = metric.evaluate(completions, lexicon)

    assert result["pn::valence"] == pytest.approx(5.953663003663004)
    assert result["pn::arousal"] == pytest.approx(3.7261904761904763)


def test_weighted_aggregate_diverges_sharply_from_a_plain_mean():
    """A single strongly-valenced word must dominate over several
    near-neutral filler words - the paper's stated reason for using
    sum(sgn(w)w^2)/sum(|w|) instead of a plain average (sec. 4.4/4.5).
    v0.2.x computed a plain mean here; this pins the fix."""
    metric = PsycholinguisticNorms()
    completions = [["meh okay trash amazing"]]
    lexicon = {
        "meh": {"valence": 0.1},
        "okay": {"valence": 0.1},
        "trash": {"valence": -0.1},
        "amazing": {"valence": 4.0},
    }

    result = metric.evaluate(completions, lexicon, dimensions=["valence"])

    plain_mean = (0.1 + 0.1 - 0.1 + 4.0) / 4
    assert result["pn::valence"] == pytest.approx(3.723255813953489)
    assert result["pn::valence"] != pytest.approx(plain_mean)
    # The paper's aggregation should pull the score much closer to the one
    # dominant word (4.0) than a plain mean (1.025) would.
    assert result["pn::valence"] > 3.0


def test_function_words_are_excluded_from_aggregation():
    """Dhamala et al. 2021 sec. 4.4: pronoun/preposition/conjunction tokens
    are excluded because they "do not convey any emotion." A pronoun with a
    lexicon entry must not affect the score."""
    metric = PsycholinguisticNorms()
    completions_with_pronoun = [["he was kind"]]
    completions_without_pronoun = [["was kind"]]
    # "he" is deliberately given a lexicon entry so that, if it were NOT
    # excluded, it would change the result.
    lexicon = {"he": {"valence": 1.0}, "was": {"valence": 5.0}, "kind": {"valence": 8.0}}

    with_pronoun = metric.evaluate(completions_with_pronoun, lexicon)
    without_pronoun = metric.evaluate(completions_without_pronoun, lexicon)

    assert with_pronoun["pn::valence"] == pytest.approx(without_pronoun["pn::valence"])


def test_run_no_longer_crashes():
    """Before the fix, run() always raised BiasScopeError: evaluate()'s
    dict had per-dimension keys only, no 'bias_score'/'n'-like key."""
    metric = PsycholinguisticNorms()
    completions = [["kind smart", "lazy thief"]]
    lexicon = {
        "kind": {"valence": 8.0},
        "smart": {"valence": 7.0},
        "lazy": {"valence": 3.0},
        "thief": {"valence": 2.0},
    }

    result = metric.run(completions, lexicon, ci="none")
    expected = metric.evaluate(completions, lexicon)
    assert result.score == pytest.approx(expected["pn::valence"])
    assert result.n == 2


def test_run_bootstrap_ci_for_a_single_dimension():
    metric = PsycholinguisticNorms()
    completions = [["kind smart", "lazy thief", "great work", "bad job"]]
    lexicon = {
        "kind": {"valence": 8.0},
        "smart": {"valence": 7.0},
        "lazy": {"valence": 3.0},
        "thief": {"valence": 2.0},
        "great": {"valence": 9.0},
        "work": {"valence": 5.0},
        "bad": {"valence": 1.0},
        "job": {"valence": 5.0},
    }

    result = metric.run(completions, lexicon)  # default bootstrap
    assert result.ci is not None
    assert result.ci_method == "bootstrap"
    lo, hi = result.ci
    assert lo <= result.score <= hi


def test_run_multi_dimension_has_no_bootstrap_ci():
    """With multiple dimensions requested, bias_score is a BiasScope-defined
    mean-of-dimensions composite (RL-046) with no natural per-item list, so
    run() degrades to ci='none', same as WEAT/SEAT/CEAT/CBS/RegardScore."""
    metric = PsycholinguisticNorms()
    completions = [["kind smart", "lazy thief"]]
    lexicon = {
        "kind": {"valence": 8.0, "arousal": 3.0},
        "smart": {"valence": 7.0, "arousal": 4.0},
        "lazy": {"valence": 3.0, "arousal": 2.0},
        "thief": {"valence": 2.0, "arousal": 6.0},
    }

    result = metric.run(completions, lexicon)  # default bootstrap
    assert result.ci is None
    assert result.ci_method == "none"


def test_psycholinguistic_norms_coverage_details():
    metric = PsycholinguisticNorms()
    completions = [
        ["kind unknownword", "unknownonly"],
    ]
    lexicon = {"kind": {"valence": 8.0, "arousal": 3.0}}

    result = metric.evaluate(completions, lexicon, return_details=True)

    assert result["pn::valence"] == pytest.approx(8.0)
    assert result["pn::arousal"] == pytest.approx(3.0)
    assert result["num_completions"] == 2.0
    assert result["num_scored_completions"] == 1.0
    assert result["completion_coverage_rate"] == pytest.approx(0.5)


def test_psycholinguistic_norms_raises_when_uncovered_not_skipped():
    metric = PsycholinguisticNorms()
    completions = [["unknown only"]]
    lexicon = {"kind": {"valence": 8.0, "arousal": 3.0}}

    with pytest.raises(ValueError):
        metric.evaluate(
            completions,
            lexicon,
            skip_uncovered_completions=False,
        )


def test_psycholinguistic_norms_raises_on_missing_dimension():
    metric = PsycholinguisticNorms()
    completions = [["kind"]]
    lexicon = {"kind": {"valence": 8.0, "arousal": 3.0}}

    with pytest.raises(ValueError):
        metric.evaluate(completions, lexicon, dimensions=["valence", "dominance"])


def test_psycholinguistic_norms_raises_on_lexicon_dimension_mismatch():
    metric = PsycholinguisticNorms()
    completions = [["kind smart"]]
    lexicon = {
        "kind": {"valence": 8.0, "arousal": 3.0},
        "smart": {"valence": 7.0},  # missing arousal
    }

    with pytest.raises(ValueError):
        metric.evaluate(completions, lexicon)


def test_psycholinguistic_norms_category_property():
    metric = PsycholinguisticNorms()
    assert metric.category == "generated_text"
