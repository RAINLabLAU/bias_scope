"""Every metric has coherent MetricInfo, and fidelity tracks the audit.

PLAN.md 5.1 requires a test over all metric classes checking range/neutral
consistency and that `deviation_note` is present when fidelity is not faithful.

This file adds one thing the plan does not: it pins `fidelity` to
`sources/SOURCES.yaml`. Section 4.0 forbids assigning a fidelity status before
the metric's sources have been read, so a metric whose entry is still `pending`
must be `unaudited`, and one marked `read` must have a real verdict and a
fidelity note. That makes the audit gap impossible to paper over.
"""

from pathlib import Path

import pytest
import yaml

import bias_scope
from bias_scope._metric_info import METRIC_INFO
from bias_scope.base import BiasMetric
from bias_scope.metadata import (
    FIDELITIES,
    FIDELITIES_NEEDING_NOTE,
    MetricInfo,
    list_metrics,
    normalized_deviation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "sources" / "SOURCES.yaml"

EXPECTED_METRIC_COUNT = 56  # 43 + WinoBias, DiscrimEval, PoliticalEvenHandedness,
#                             ImplicitAssociationTest, LLMDecisionBias (PLAN.md 7.2)


def _manifest_entries():
    text = MANIFEST.read_text()
    document = yaml.safe_load("metrics:" + text.partition("metrics:")[2])
    return {e["metric"]: e for e in document["metrics"]}


MANIFEST_ENTRIES = _manifest_entries()
ALL_NAMES = sorted(METRIC_INFO)


def test_the_library_has_the_expected_number_of_metrics():
    """A metric added or removed without updating the table is a bug."""
    assert len(METRIC_INFO) == EXPECTED_METRIC_COUNT


def test_every_metric_class_is_registered_on_import():
    """`import bias_scope` attaches MetricInfo to every metric class."""
    # Touch the lazily-imported prompt family so it registers too.
    for name in ALL_NAMES:
        getattr(bias_scope, name, None)
    assert set(list_metrics()) == set(METRIC_INFO)


@pytest.mark.parametrize("name", ALL_NAMES)
class TestEveryMetric:
    def test_info_is_valid(self, name):
        problems = METRIC_INFO[name].validate()
        assert not problems, f"{name}: {problems}"

    def test_class_carries_its_info(self, name):
        cls = getattr(bias_scope, name, None)
        if cls is None or not isinstance(cls, type) or not issubclass(cls, BiasMetric):
            pytest.skip(f"{name} is not importable in this environment")
        assert cls.info is METRIC_INFO[name]

    def test_neutral_value_lies_inside_value_range(self, name):
        info = METRIC_INFO[name]
        low, high = info.value_range
        assert low <= info.neutral_value <= high

    def test_deviation_note_present_when_not_faithful(self, name):
        info = METRIC_INFO[name]
        if info.fidelity in FIDELITIES_NEEDING_NOTE:
            assert info.deviation_note.strip(), (
                f"{name} is {info.fidelity} and must say what differs and why"
            )

    def test_has_a_manifest_entry(self, name):
        """PLAN.md 4.0: no metric without a recorded source entry."""
        assert name in MANIFEST_ENTRIES, f"{name} has no sources/SOURCES.yaml entry"

    def test_fidelity_matches_the_audit_state(self, name):
        """`unaudited` iff the manifest entry is still `pending`."""
        info = METRIC_INFO[name]
        status = MANIFEST_ENTRIES[name]["status"]
        if status == "pending":
            assert info.fidelity == "unaudited", (
                f"{name} claims fidelity {info.fidelity!r} but its sources have not "
                "been read (SOURCES.yaml status is `pending`). Section 4.0 forbids "
                "assigning a status before reading the paper and the reference code."
            )
        else:
            assert info.fidelity != "unaudited", (
                f"{name} is marked `read` in SOURCES.yaml but is still `unaudited`; "
                "record the verdict from its docs/fidelity/ note"
            )

    def test_audited_metrics_point_at_their_fidelity_note(self, name):
        info = METRIC_INFO[name]
        if info.fidelity == "unaudited":
            return
        assert info.fidelity_note, f"{name} is audited but records no fidelity note"
        assert (REPO_ROOT / info.fidelity_note).exists(), (
            f"{name}: fidelity_note {info.fidelity_note} does not exist"
        )

    def test_family_matches_the_manifest(self, name):
        assert METRIC_INFO[name].family == MANIFEST_ENTRIES[name]["family"]


class TestNormalizedDeviation:
    """Zero is neutral; positive is more biased, whatever the metric's direction."""

    def _info(self, **overrides):
        base = dict(
            name="X", family="probability", access=("logits",), neutral_value=0.5,
            direction="higher_more_biased", value_range=(0.0, 1.0),
            fidelity="faithful", reference="ref",
        )
        base.update(overrides)
        return MetricInfo(**base)

    def test_neutral_score_gives_zero(self):
        assert normalized_deviation(0.5, self._info()) == 0.0

    def test_higher_more_biased_is_positive_above_neutral(self):
        assert normalized_deviation(0.75, self._info()) == pytest.approx(0.5)

    def test_lower_more_biased_flips_the_sign(self):
        info = self._info(direction="lower_more_biased")
        # 0.75 is *less* biased for this direction, so the deviation is negative.
        assert normalized_deviation(0.75, info) == pytest.approx(-0.5)
        assert normalized_deviation(0.25, info) == pytest.approx(0.5)

    def test_asymmetric_range_uses_the_larger_half(self):
        # neutral 0.5 in [0, 100] -> scale = max(0.5, 99.5) = 99.5
        info = self._info(value_range=(0.0, 100.0))
        assert normalized_deviation(100.0, info) == pytest.approx(99.5 / 99.5)

    def test_unbounded_range_returns_the_raw_deviation(self):
        info = self._info(neutral_value=0.0, direction="signed",
                          value_range=(float("-inf"), float("inf")))
        assert normalized_deviation(1.81, info) == pytest.approx(1.81)

    def test_icat_style_metric_where_higher_is_better(self):
        # ICAT: neutral (best) is 100, lower is worse.
        info = self._info(name="ICAT", neutral_value=100.0,
                          direction="lower_more_biased", value_range=(0.0, 100.0))
        assert normalized_deviation(100.0, info) == 0.0
        assert normalized_deviation(0.0, info) == pytest.approx(1.0)

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_neutral_value_maps_to_zero_for_every_real_metric(self, name):
        info = METRIC_INFO[name]
        assert normalized_deviation(info.neutral_value, info) == pytest.approx(0.0)


class TestRegistryQueries:
    def test_filtering_by_family_returns_only_that_family(self):
        for family in ("embedding", "probability", "generated_text", "prompt"):
            selected = list_metrics(family=family)
            assert selected
            assert all(i.family == family for i in selected.values())

    def test_family_counts_match_the_plan(self):
        """PLAN.md Section 2's map, plus PairwiseLikelihoodPreference in probability."""
        counts = {f: len(list_metrics(family=f)) for f in
                  ("embedding", "probability", "generated_text", "prompt")}
        assert counts == {"embedding": 4, "probability": 11,
                          "generated_text": 17, "prompt": 24}

    def test_filtering_by_fidelity_works(self):
        assert len(list_metrics(fidelity="mismatch")) == len(
            [i for i in METRIC_INFO.values() if i.fidelity == "mismatch"]
        )

    def test_an_unknown_family_raises(self):
        with pytest.raises(ValueError, match="family"):
            list_metrics(family="not-a-family")

    def test_an_unknown_fidelity_raises(self):
        with pytest.raises(ValueError, match="fidelity"):
            list_metrics(fidelity="pretty-good")

    def test_counts_cover_every_status_and_sum_to_the_total(self):
        counts = bias_scope.fidelity_counts()
        assert set(counts) == set(FIDELITIES)
        assert sum(counts.values()) == len(METRIC_INFO)


class TestReleaseGates:
    """Section 13's definition of done, as executable checks.

    These are expected to fail until the audit and the reimplementations are
    done, so they are `xfail(strict=True)`: when they start passing, that is a
    signal to promote them to hard assertions and tick the Section 13 boxes.
    """

    def test_no_metric_is_unaudited(self):
        assert not list_metrics(fidelity="unaudited")

    @pytest.mark.xfail(
        strict=True,
        reason="FGB and PGB are blocked on HolisticBias's unpublished 217-class "
               "style classifier (REVIEW_LATER RL-013). Every other mismatch is fixed.",
    )
    def test_no_metric_is_a_mismatch(self):
        assert not list_metrics(fidelity="mismatch")

    def test_the_only_remaining_mismatches_are_the_blocked_ones(self):
        """Every mismatch that *can* be fixed has been.

        FGB and PGB are defined over a 217-class style classifier that
        HolisticBias never published (RL-013), so they cannot be made faithful
        by any amount of work here. This test pins that: if a new mismatch
        appears, or one of these is resolved, it fails and someone must decide
        what the release gate now says.
        """
        assert set(list_metrics(fidelity="mismatch")) == {"FGB", "PGB"}
