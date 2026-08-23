"""Tier 2: `iat_bias_score` against Bai et al.'s released per-response scores.

`data/result_implicit.csv` in `github.com/baixuechunzi/llm-implicit-bias` ships
33,600 raw model responses **with the authors' own `iat_bias` value for each**.
That makes a per-item equivalence check possible, not just a comparison of
published means — every disagreement is one identifiable response.

Scope of the claim. The check covers responses whose words all come from a
single stereotype's stimulus set in `stimuli/iat_stimuli.csv`, the one shipped
file where each dataset is one attribute set with the positive words first.
Most of the 33,600 prompts used automatically generated variations whose word
sets are not in the repository at all (paper, Sec. 2.1: "we use a language model
to automatically generate new sets Xa and Xb"), so they cannot be scored by
anyone from the released files, us included.

Run with:  BIASSCOPE_RUN_EQUIVALENCE=1 pytest tests/equivalence -m equivalence
"""

import os
from pathlib import Path

import pytest

from bias_scope.prompts_based.implicit_association import (
    iat_bias_score,
    load_iat_stimuli,
    parse_iat_response,
)

REPO = Path("third_party/code/llm-implicit-bias")
STIMULI = REPO / "stimuli" / "iat_stimuli.csv"
RESPONSES = REPO / "data" / "result_implicit.csv"

#: Every scored response must match to this many decimals. The reference is
#: float arithmetic on small integer counts, so agreement should be exact;
#: the tolerance only absorbs CSV round-tripping of the published column.
TOLERANCE = 1e-9

#: Fixed in advance from the authors' data, not tuned to the result. Below this
#: the check has too few responses to mean anything.
MIN_SCORED = 500

pytestmark = [
    pytest.mark.equivalence,
    pytest.mark.skipif(
        not os.environ.get("BIASSCOPE_RUN_EQUIVALENCE"),
        reason="Tier-2 equivalence; set BIASSCOPE_RUN_EQUIVALENCE=1",
    ),
    pytest.mark.skipif(
        not RESPONSES.exists(),
        reason=f"{RESPONSES} not present; run scripts/sources/fetch_sources.py",
    ),
]


def _rows():
    import csv

    with open(RESPONSES, newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _score_against_its_stereotype(text, stimuli):
    """Score `text` with the one stimulus set that covers every word it used.

    Returns None when no shipped set covers the response — an automatically
    generated variation, or a refusal. Guessing a set would fabricate a score.
    """
    pairs = parse_iat_response(text)
    if not pairs:
        return None
    groups = {g for _, g in pairs}
    attributes = {a for a, _ in pairs}
    for spec in stimuli.values():
        known_groups = {w.lower() for w in spec["target_group"] + spec["reference_group"]}
        known_attributes = {
            w.lower()
            for w in spec["target_attributes"] + spec["reference_attributes"]
        }
        if groups <= known_groups and attributes <= known_attributes:
            return iat_bias_score(
                pairs, spec["target_group"], spec["reference_group"],
                spec["target_attributes"], spec["reference_attributes"],
            )
    return None


@pytest.fixture(scope="module")
def comparison():
    stimuli = load_iat_stimuli(str(STIMULI))
    scored, matches, disagreements = 0, 0, []
    for row in _rows():
        ours = _score_against_its_stereotype(row["formatted_iat"], stimuli)
        if ours is None:
            continue
        scored += 1
        theirs = float(row["iat_bias"])
        if abs(ours - theirs) <= TOLERANCE:
            matches += 1
        else:
            disagreements.append((row["formatted_iat"], ours, theirs))
    return scored, matches, disagreements


class TestPerResponseEquivalence:
    def test_enough_responses_were_scored_for_the_check_to_mean_anything(
        self, comparison
    ):
        scored, _, _ = comparison
        assert scored >= MIN_SCORED, (
            f"only {scored} of the released responses could be scored from the "
            "shipped stimulus file; the equivalence claim would be vacuous"
        )

    def test_every_scored_response_matches_the_authors_value(self, comparison):
        """Exact agreement, per response — not agreement of the means."""
        scored, matches, disagreements = comparison
        assert matches == scored, (
            f"{len(disagreements)} of {scored} responses disagree with the "
            f"authors' published iat_bias. First three: "
            f"{[(o, t) for _, o, t in disagreements[:3]]}"
        )

    def test_the_reference_epsilon_is_what_makes_it_match(self, comparison):
        """Dropping the code's 0.01 breaks the agreement, so it is not cosmetic.

        This is the evidence for `deviation_note`: the published column was
        produced with the epsilon, and the paper's formula without it gives
        different numbers on the very same responses — and on some of them no
        number at all, because a group that received no words leaves 0/0.
        """
        stimuli = load_iat_stimuli(str(STIMULI))
        exact = 0
        checked = 0
        for row in _rows():
            pairs = parse_iat_response(row["formatted_iat"])
            if not pairs:
                continue
            for spec in stimuli.values():
                groups = {w.lower() for w in
                          spec["target_group"] + spec["reference_group"]}
                attributes = {w.lower() for w in
                              spec["target_attributes"] + spec["reference_attributes"]}
                if {g for _, g in pairs} <= groups and {a for a, _ in pairs} <= attributes:
                    without = iat_bias_score(
                        pairs, spec["target_group"], spec["reference_group"],
                        spec["target_attributes"], spec["reference_attributes"],
                        epsilon=0.0,
                    )
                    checked += 1
                    if without is not None and abs(
                        without - float(row["iat_bias"])
                    ) <= TOLERANCE:
                        exact += 1
                    break
            if checked >= 2000:
                break
        assert checked > 0
        assert exact < checked, (
            "epsilon=0.0 matched the published column on every response, so "
            "the documented deviation is not real and the note should be removed"
        )
