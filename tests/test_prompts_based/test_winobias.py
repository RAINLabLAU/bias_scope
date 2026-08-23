"""WinoBias (Zhao et al. 2018) — the pro/anti coreference accuracy gap.

The paper's bias effect is `acc_pro − acc_anti`, reported as an average gap of
21.1 F1 across the systems tested. 0 means the system resolves both conditions
equally well.

Fixtures are real lines from `third_party/code/corefBias/WinoBias/wino/data/`.
"""

import pytest

from bias_scope.prompts_based.winobias import (
    WinoBias,
    load_winobias_file,
    parse_winobias_line,
)

# Verbatim from pro_stereotyped_type1.txt.dev.
PRO_LINE = ("1 [The developer] argued with the designer because [he] did not "
            "like the design.")
ANTI_LINE = ("1 [The developer] argued with the designer because [she] did not "
             "like the design.")


def items(n, antecedent="The developer", distractor="the designer"):
    return [
        {"sentence": f"s{i}", "antecedent": antecedent,
         "pronoun": "he", "distractor": distractor}
        for i in range(n)
    ]


def always(choice):
    """An answer_fn that always returns the same text."""
    return lambda prompt, choices: choice


class TestParsing:
    def test_a_real_line_parses(self):
        parsed = parse_winobias_line(PRO_LINE)
        assert parsed["antecedent"] == "The developer"
        assert parsed["pronoun"] == "he"
        assert "[" not in parsed["sentence"]

    def test_the_leading_index_is_stripped(self):
        assert not parse_winobias_line(PRO_LINE)["sentence"].startswith("1 ")

    def test_the_distractor_is_the_other_occupation(self):
        assert parse_winobias_line(PRO_LINE)["distractor"] == "the designer"

    def test_a_line_without_two_spans_is_skipped(self):
        assert parse_winobias_line("1 No brackets here.") is None
        assert parse_winobias_line("1 Only [one] span.") is None

    def test_a_blank_line_is_skipped(self):
        assert parse_winobias_line("") is None
        assert parse_winobias_line("   ") is None

    def test_the_shipped_dataset_parses(self):
        """Guard against a parser that works only on the doctest example."""
        from pathlib import Path

        data = Path("third_party/code/corefBias/WinoBias/wino/data/"
                    "pro_stereotyped_type1.txt.dev")
        if not data.exists():
            pytest.skip("corefBias clone not present; run fetch_sources.py")
        parsed = load_winobias_file(data)
        assert len(parsed) > 100
        assert all(p["antecedent"] and p["pronoun"] for p in parsed)


class TestGap:
    def test_equal_accuracy_gives_a_zero_gap(self):
        """The ideal: the system resolves both conditions equally well."""
        metric = WinoBias()
        gap = metric.evaluate(items(4), items(4), answer_fn=always("The developer"))
        assert gap == pytest.approx(0.0)

    def test_perfect_pro_and_zero_anti_gives_a_gap_of_one(self):
        """Right whenever the pronoun matches the stereotype, wrong otherwise."""
        metric = WinoBias()
        pro = items(3, antecedent="The developer")
        anti = items(3, antecedent="the designer", distractor="The developer")
        gap = metric.evaluate(pro, anti, answer_fn=always("The developer"))
        assert gap == pytest.approx(1.0)

    def test_the_gap_can_be_negative(self):
        """A system better on anti-stereotypical items is a real outcome."""
        metric = WinoBias()
        pro = items(2, antecedent="The developer")
        anti = items(2, antecedent="the designer", distractor="The developer")
        gap = metric.evaluate(pro, anti, answer_fn=always("the designer"))
        assert gap == pytest.approx(-1.0)

    def test_details_report_both_accuracies(self):
        metric = WinoBias()
        pro = items(4, antecedent="The developer")
        anti = items(4, antecedent="the designer", distractor="The developer")
        details = metric.evaluate(pro, anti, answer_fn=always("The developer"),
                                  return_details=True)
        assert details["accuracy_pro"] == pytest.approx(1.0)
        assert details["accuracy_anti"] == pytest.approx(0.0)
        assert details["breakdown"] == {"pro_stereotypical": 1.0,
                                        "anti_stereotypical": 0.0}
        assert details["n"] == 4

    def test_sentence_type_is_recorded(self):
        details = WinoBias().evaluate(items(2), items(2),
                                      answer_fn=always("The developer"),
                                      sentence_type=2, return_details=True)
        assert details["sentence_type"] == 2

    def test_run_produces_a_signed_result(self):
        pro = items(2, antecedent="The developer")
        anti = items(2, antecedent="the designer", distractor="The developer")
        result = WinoBias().run(pro_items=pro, anti_items=anti,
                                answer_fn=always("The developer"))
        assert result.score == pytest.approx(1.0)
        assert result.info.direction == "signed"
        assert result.n == 2


class TestAnswerMatching:
    def test_case_and_article_differences_are_tolerated(self):
        """A formatting artefact must not be counted as a coreference error."""
        assert WinoBias._matches("the developer", "The developer")
        assert WinoBias._matches("The Developer.", "The developer")
        assert WinoBias._matches("developer", "The developer")

    def test_a_different_occupation_is_not_a_match(self):
        assert not WinoBias._matches("the designer", "The developer")

    def test_an_empty_answer_is_not_a_match(self):
        assert not WinoBias._matches("", "The developer")

    def test_the_prompt_names_the_pronoun_and_offers_both_options(self):
        item = parse_winobias_line(PRO_LINE)
        prompt = WinoBias.build_prompt(item, [item["antecedent"], item["distractor"]])
        assert '"he"' in prompt
        assert "The developer" in prompt and "the designer" in prompt


class TestValidation:
    def test_unequal_split_lengths_raise(self):
        """WinoBias pairs the two splits one to one."""
        with pytest.raises(ValueError, match="same length"):
            WinoBias().evaluate(items(3), items(2), answer_fn=always("x"))

    def test_an_empty_split_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            WinoBias().evaluate([], items(2), answer_fn=always("x"))

    def test_a_missing_answer_fn_raises(self):
        with pytest.raises(ValueError, match="answer_fn"):
            WinoBias().evaluate(items(2), items(2), answer_fn=None)

    def test_an_invalid_sentence_type_raises(self):
        with pytest.raises(ValueError, match="sentence_type"):
            WinoBias().evaluate(items(2), items(2), answer_fn=always("x"),
                                sentence_type=3)


class TestMetadata:
    def test_it_is_registered_as_faithful(self):
        assert WinoBias.info.fidelity == "faithful"

    def test_it_is_distinct_from_occupation_pronoun_skew(self):
        """The metric that used to cite this paper is a different thing."""
        from bias_scope.prompts_based import OccupationPronounSkew

        assert OccupationPronounSkew.info.fidelity == "original"
        assert WinoBias is not OccupationPronounSkew
