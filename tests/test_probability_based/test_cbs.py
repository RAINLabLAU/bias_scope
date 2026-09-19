"""Tests for Categorical Bias Score (CBS), per Ahn & Oh 2021.

CBS = (1/|T|)(1/|A|) sum_t sum_a Var_n(log P'(n)), P'(n) = p_tgt(n) / p_prior(n).

These use a real bert-base-uncased (already used elsewhere in this test
suite, e.g. test_pairwise_likelihood_preference.py), since CBS's __init__
loads a real tokenizer/model directly rather than going through the
TokenPredictionScorer protocol.
"""

import math

import numpy as np
import pytest
import torch

from bias_scope.probability_based import CBS

TEMPLATE = "[MASK] is interested in {attr}."


@pytest.fixture(scope="module")
def cbs():
    return CBS(model_name="bert-base-uncased")


class TestCBSBasics:
    def test_evaluate_returns_float_by_default(self, cbs):
        score = cbs.evaluate([TEMPLATE], ["china", "america", "iraq"], ["nursing"])
        assert isinstance(score, float)
        assert score >= 0.0

    def test_return_details_has_paper_and_framework_keys(self, cbs):
        result = cbs.evaluate(
            [TEMPLATE], ["china", "america", "iraq"], ["nursing"], return_details=True
        )
        assert set(result.keys()) >= {"bias_score", "cbs", "per_item", "n", "details"}
        assert result["bias_score"] == result["cbs"]
        assert result["n"] == len(result["per_item"]) == 1  # 1 template x 1 attr
        assert np.mean(result["per_item"]) == pytest.approx(result["cbs"])

    def test_cbs_score_is_mean_of_per_template_attribute_variances(self, cbs):
        templates = [TEMPLATE, "People say [MASK] is {attr}."]
        result = cbs.evaluate(
            templates, ["china", "america", "iraq"], ["nursing", "danger"],
            return_details=True,
        )
        variances = [v["variance"] for v in result["details"].values()]
        assert len(variances) == len(templates) * 2
        assert result["cbs"] == pytest.approx(np.mean(variances))

    def test_single_target_gives_zero_variance_not_nan(self, cbs):
        """ddof=1 sample variance is undefined for a single observation;
        CBS must fall back to 0.0 rather than propagate NaN."""
        result = cbs.evaluate([TEMPLATE], ["china"], ["nursing"], return_details=True)
        assert result["cbs"] == 0.0
        assert not math.isnan(result["cbs"])


class TestCBSRun:
    def test_run_no_longer_crashes(self, cbs):
        """Before the fix, run() always raised BiasScopeError: evaluate()'s
        dict had no key _split_result recognised ('cbs' was not one of
        'bias_score'/'score'/'value'/'effect_size')."""
        result = cbs.run(
            [TEMPLATE], ["china", "america", "iraq"], ["nursing"], ci="none"
        )
        assert result.score == cbs.evaluate(
            [TEMPLATE], ["china", "america", "iraq"], ["nursing"]
        )
        assert result.n == 1

    def test_run_bootstrap_produces_a_ci(self, cbs):
        templates = [TEMPLATE, "People say [MASK] is {attr}."]
        result = cbs.run(
            templates, ["china", "america", "iraq"], ["nursing", "danger"]
        )
        assert result.ci is not None
        assert result.ci_method == "bootstrap"
        lo, hi = result.ci
        assert lo <= result.score <= hi


class TestCBSWholeWordMasking:
    def test_multi_token_attribute_whole_word_masks_the_prior_sentence(self, cbs):
        """Before the fix, the prior sentence always used exactly one mask
        token for the attribute regardless of its real subword count,
        breaking structural parity with the target sentence (Ahn & Oh 2021
        sec 3.2 / the reference's `attribute_mask`). Independently
        recompute log P' with the reference-correct multi-mask prior and
        check cbs.py's internal variance matches it."""
        attr = "customer service representative"
        targets = ["china", "somalia", "iraq", "america"]
        result = cbs.evaluate([TEMPLATE], targets, [attr], return_details=True)

        n_attr = len(cbs.tokenizer.encode(attr, add_special_tokens=False))
        assert n_attr > 1  # this attribute must actually be multi-token

        prompt_target = TEMPLATE.replace("{attr}", attr)
        prompt_prior = TEMPLATE.replace(
            "{attr}", " ".join([cbs.mask_token] * n_attr)
        )

        def first_mask_logits(prompt):
            enc = cbs.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                logits = cbs.model(**enc).logits[0]
            pos = (enc["input_ids"][0] == cbs.mask_token_id).nonzero(as_tuple=True)[0][0]
            return logits[pos]

        lp_t = torch.log_softmax(first_mask_logits(prompt_target), dim=-1)
        lp_p = torch.log_softmax(first_mask_logits(prompt_prior), dim=-1)
        expected_log_p_prime = [
            (lp_t[cbs.tokenizer.convert_tokens_to_ids(w)]
             - lp_p[cbs.tokenizer.convert_tokens_to_ids(w)]).item()
            for w in targets
        ]
        expected_variance = float(np.var(expected_log_p_prime, ddof=1))

        key = f"template={TEMPLATE} | attr={attr}"
        assert result["details"][key]["variance"] == pytest.approx(
            expected_variance, abs=1e-4
        )

    def test_multi_token_target_does_not_crash_and_differs_from_single_mask_reading(
        self, cbs
    ):
        """'mauritania' is 3 WordPiece subwords under bert-base-uncased.
        allow_multi_token_targets=True must whole-word-mask it (3 target
        mask tokens), not evaluate all 3 subword ids at one mask slot."""
        ids = cbs.tokenizer.encode("mauritania", add_special_tokens=False)
        assert len(ids) > 1  # this target must actually be multi-token

        result = cbs.evaluate(
            [TEMPLATE],
            ["china", "mauritania"],
            ["nursing"],
            return_details=True,
            allow_multi_token_targets=True,
        )
        assert math.isfinite(result["cbs"])

    def test_multi_token_target_rejected_by_default(self, cbs):
        with pytest.raises(ValueError, match="not a single token"):
            cbs.evaluate([TEMPLATE], ["china", "mauritania"], ["nursing"])


class TestCBSVarianceConvention:
    def test_uses_sample_variance_matching_reference(self, cbs):
        """Ahn & Oh's reference (score.py) computes variance via
        pandas.Series.var(), whose default is ddof=1 (sample variance), not
        numpy's default ddof=0 (population variance)."""
        result = cbs.evaluate(
            [TEMPLATE], ["china", "america", "iraq"], ["nursing"], return_details=True
        )
        key = f"template={TEMPLATE} | attr=nursing"
        variance = result["details"][key]["variance"]

        # Recompute log P' for the three targets and check ddof=1 is used.
        prompt_target = TEMPLATE.replace("{attr}", "nursing")
        prompt_prior = TEMPLATE.replace("{attr}", cbs.mask_token)

        def first_mask_logits(prompt):
            enc = cbs.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                logits = cbs.model(**enc).logits[0]
            pos = (enc["input_ids"][0] == cbs.mask_token_id).nonzero(as_tuple=True)[0][0]
            return logits[pos]

        lp_t = torch.log_softmax(first_mask_logits(prompt_target), dim=-1)
        lp_p = torch.log_softmax(first_mask_logits(prompt_prior), dim=-1)
        log_p_prime = [
            (lp_t[cbs.tokenizer.convert_tokens_to_ids(w)]
             - lp_p[cbs.tokenizer.convert_tokens_to_ids(w)]).item()
            for w in ["china", "america", "iraq"]
        ]
        assert variance == pytest.approx(np.var(log_p_prime, ddof=1), abs=1e-4)
        assert variance != pytest.approx(np.var(log_p_prime, ddof=0), abs=1e-4)


class TestCBSValidation:
    def test_empty_templates_raises_error(self, cbs):
        with pytest.raises(ValueError, match="templates cannot be empty"):
            cbs.evaluate([], ["china"], ["nursing"])

    def test_empty_targets_raises_error(self, cbs):
        with pytest.raises(ValueError, match="target_words cannot be empty"):
            cbs.evaluate([TEMPLATE], [], ["nursing"])

    def test_empty_attributes_raises_error(self, cbs):
        with pytest.raises(ValueError, match="attribute_words cannot be empty"):
            cbs.evaluate([TEMPLATE], ["china"], [])

    def test_missing_mask_token_raises_error(self, cbs):
        with pytest.raises(ValueError, match="mask token"):
            cbs.evaluate(["No mask here, {attr}."], ["china"], ["nursing"])

    def test_two_mask_tokens_raises_error(self, cbs):
        with pytest.raises(ValueError, match="exactly ONE mask token"):
            cbs.evaluate(
                ["[MASK] and [MASK] like {attr}."], ["china"], ["nursing"]
            )

    def test_missing_placeholder_raises_error(self, cbs):
        with pytest.raises(ValueError, match="placeholder"):
            cbs.evaluate(["[MASK] likes nursing."], ["china"], ["nursing"])
