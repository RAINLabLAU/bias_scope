"""The framework layer: backends, recommend, suite, report (PLAN.md 5.4).

Includes the acceptance checks 5.4 lists, each as a named test:

- the selected metric set equals exactly the metrics whose access ⊆ backend
  access and whose languages ∋ language
- a second `suite.run()` on the same model makes zero generation calls
- `compare` deltas equal the differences of the individual scores
- `correlate` on synthetic reports with a known correlation returns that matrix
- every `BiasResult` in a report appears in all three output formats
- the HTML parses with `html.parser` and contains one section per family and
  the protocol table
- `recommend_metrics` never returns a metric the given access cannot run
"""

from html.parser import HTMLParser
from pathlib import Path

import pytest

from bias_scope.backends import (
    HuggingFaceBackend,
    LiteLLMBackend,
    StubBackend,
    cached_generate,
    load_model,
)
from bias_scope.metadata import MetricInfo, list_metrics
from bias_scope.recommend import explain_exclusions, recommend_metrics
from bias_scope.report import (
    Report,
    compare,
    correlate,
    save_json,
    to_html,
    to_markdown,
)
from bias_scope.result import BiasResult, make_protocol
from bias_scope.suite import BiasSuite


def make_info(name, family="probability", access=("logits",), fidelity="faithful",
              neutral=0.5, low=0.0, high=1.0):
    return MetricInfo(
        name=name, family=family, access=tuple(access), neutral_value=neutral,
        direction="higher_more_biased", value_range=(low, high),
        fidelity=fidelity, reference="test fixture",
        deviation_note="fixture" if fidelity != "faithful" else "",
    )


def make_result(name, score, ci=(0.0, 1.0), family="probability", n=10,
                fidelity="faithful", breakdown=None):
    return BiasResult(
        metric=name, score=score, n=n, ci=ci, ci_method="bootstrap",
        per_item=[score] * n, breakdown=breakdown or {}, details={},
        protocol=make_protocol(name, model_id="m", seed=42, dtype="bf16"),
        info=make_info(name, family=family, fidelity=fidelity),
    )


def make_report(model_id, scores, **kw):
    return Report(
        model_id=model_id,
        results=[make_result(n, s, **kw) for n, s in scores.items()],
        protocol=make_protocol("BiasSuite", model_id=model_id, seed=42, dtype="bf16"),
    )


class TestBackends:
    def test_encoder_access_excludes_completions(self):
        backend = HuggingFaceBackend("m", kind="encoder")
        assert set(backend.access) == {"embeddings", "logits"}

    def test_causal_access_includes_completions(self):
        backend = HuggingFaceBackend("m", kind="causal")
        assert "completions" in backend.access

    def test_no_hf_backend_claims_chat(self):
        """A transformers model is not a chat API, whatever it was tuned for."""
        for kind in ("causal", "encoder"):
            assert "chat" not in HuggingFaceBackend("m", kind=kind).access

    def test_litellm_does_not_claim_logits(self):
        """Chat APIs expose at best top-k logprobs, not a full distribution."""
        assert "logits" not in LiteLLMBackend("openai/gpt-4o").access

    def test_supports_is_a_subset_check(self):
        backend = StubBackend(access=("completions", "chat"))
        assert backend.supports(["chat"])
        assert not backend.supports(["logits"])

    def test_an_encoder_refuses_to_generate(self):
        with pytest.raises(ValueError, match="cannot generate"):
            HuggingFaceBackend("m", kind="encoder").generate(["hi"])

    @pytest.mark.parametrize("bad", ["int4", "int8", "float64"])
    def test_unknown_dtype_raises(self, bad):
        with pytest.raises(ValueError, match="dtype"):
            HuggingFaceBackend("m", dtype=bad)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            HuggingFaceBackend("m", kind="seq2seq")

    def test_load_model_dispatches(self):
        assert isinstance(load_model("m", backend="hf"), HuggingFaceBackend)
        assert isinstance(load_model("m", backend="litellm"), LiteLLMBackend)
        with pytest.raises(ValueError, match="backend"):
            load_model("m", backend="vllm")

    def test_protocol_fields_carry_model_and_dtype(self):
        fields = HuggingFaceBackend("gpt2", dtype="fp32").protocol_fields()
        assert fields == {"model_id": "gpt2", "dtype": "fp32"}


class TestGenerationCache:
    def test_a_second_run_makes_zero_generation_calls(self, tmp_path):
        """PLAN.md 5.4 acceptance check."""
        backend = StubBackend(answers=["a", "b"], access=("chat",))
        prompts = ["p1", "p2"]

        first = cached_generate(backend, prompts, cache_dir=tmp_path)
        calls_after_first = len(backend.calls)
        second = cached_generate(backend, prompts, cache_dir=tmp_path)

        assert first == second
        assert len(backend.calls) == calls_after_first, "cache hit still generated"

    def test_changing_decoding_is_a_cache_miss(self, tmp_path):
        """Reusing generations from a different protocol would be worse than no cache."""
        backend = StubBackend(answers=["a"], access=("chat",))
        cached_generate(backend, ["p"], {"temperature": 0.0}, cache_dir=tmp_path)
        before = len(backend.calls)
        cached_generate(backend, ["p"], {"temperature": 1.0}, cache_dir=tmp_path)
        assert len(backend.calls) > before

    def test_changing_prompts_is_a_cache_miss(self, tmp_path):
        backend = StubBackend(answers=["a"], access=("chat",))
        cached_generate(backend, ["p1"], cache_dir=tmp_path)
        before = len(backend.calls)
        cached_generate(backend, ["p2"], cache_dir=tmp_path)
        assert len(backend.calls) > before

    def test_without_a_cache_dir_it_just_generates(self):
        backend = StubBackend(answers=["a"], access=("chat",))
        assert cached_generate(backend, ["p"]) == ["a"]


class TestRecommend:
    def test_never_returns_a_metric_the_access_cannot_run(self):
        """PLAN.md 5.4 acceptance check."""
        for access in (("embeddings",), ("logits",), ("chat",),
                       ("completions", "chat"), ("embeddings", "logits")):
            for rec in recommend_metrics(access):
                assert set(rec.info.access) <= set(access), rec.metric

    def test_the_selected_set_is_exactly_the_eligible_set(self):
        """PLAN.md 5.4 acceptance check: equality, not merely a subset."""
        access = ("completions", "chat")
        expected = {
            name
            for name, info in list_metrics().items()
            if set(info.access) <= set(access)
            and "en" in info.languages
            and info.fidelity != "mismatch"
            and name not in {"TruthfulQA", "TofNof"}
        }
        assert {r.metric for r in recommend_metrics(access)} == expected

    def test_mismatch_metrics_are_excluded_by_default(self):
        names = {r.metric for r in recommend_metrics(("completions",))}
        assert not any(list_metrics()[n].fidelity == "mismatch" for n in names)

    def test_mismatch_metrics_can_be_opted_into(self):
        with_them = recommend_metrics(("completions",), include_mismatch=True)
        without = recommend_metrics(("completions",))
        assert len(with_them) >= len(without)

    def test_faithful_metrics_are_offered_first(self):
        order = {"faithful": 0, "adaptation": 1, "original": 2,
                 "unaudited": 3, "mismatch": 4}
        ranks = [order[r.info.fidelity]
                 for r in recommend_metrics(("embeddings", "logits", "completions"))]
        assert ranks == sorted(ranks)

    def test_every_recommendation_states_its_fidelity(self):
        for rec in recommend_metrics(("chat",), include_unaudited=True):
            assert rec.info.fidelity in rec.reason.lower() or \
                   rec.info.fidelity.upper() in rec.reason

    def test_an_unsupported_language_yields_nothing_lexicon_bound(self):
        for rec in recommend_metrics(("completions",), language="fr"):
            assert "fr" in rec.info.languages

    def test_truthfulqa_is_excluded_as_not_social_bias(self):
        names = {r.metric for r in recommend_metrics(("logits",))}
        assert "TruthfulQA" not in names
        opted_in = {r.metric for r in recommend_metrics(("logits",), include_non_bias=True)}
        assert "TruthfulQA" in opted_in

    def test_empty_access_raises(self):
        with pytest.raises(ValueError, match="access"):
            recommend_metrics(())

    def test_unknown_access_mode_raises(self):
        with pytest.raises(ValueError, match="unknown access"):
            recommend_metrics(("telepathy",))

    def test_exclusions_are_explained(self):
        reasons = explain_exclusions(("chat",))
        assert "WEAT" in reasons
        assert "embeddings" in reasons["WEAT"]


class TestSuite:
    def test_plan_respects_backend_access(self):
        suite = BiasSuite(StubBackend(access=("chat",)))
        for _, info in suite.plan():
            assert set(info.access) <= {"chat"}

    def test_requesting_an_unsupported_metric_raises(self):
        suite = BiasSuite(StubBackend(access=("chat",)), metrics=["WEAT"])
        with pytest.raises(ValueError, match="needs embeddings"):
            suite.plan()

    def test_requesting_an_unknown_metric_raises(self):
        suite = BiasSuite(StubBackend(), metrics=["NotAMetric"])
        with pytest.raises(ValueError, match="unknown metric"):
            suite.plan()

    def test_metrics_without_inputs_are_skipped_not_zeroed(self):
        """A missing number is information; a fabricated one is a defect."""
        report = BiasSuite(StubBackend(access=("chat",))).run()
        assert report.results == []
        assert report.skipped
        assert all("requires caller-supplied" in v for v in report.skipped.values())

    def test_a_metric_with_inputs_runs(self):
        import numpy as np

        backend = StubBackend(access=("embeddings",), model_id="stub/emb")
        rng = np.random.default_rng(42)
        sets = [rng.standard_normal((8, 16)) for _ in range(4)]
        suite = BiasSuite(backend, metrics=["WEAT"])
        report = suite.run(inputs={"WEAT": {
            "target_embeddings": (sets[0], sets[1]),
            "attribute_embeddings": (sets[2], sets[3]),
        }})
        assert len(report.results) == 1
        assert report.results[0].metric == "WEAT"

    def test_a_failing_metric_is_recorded_not_swallowed(self):
        backend = StubBackend(access=("embeddings",))
        suite = BiasSuite(backend, metrics=["WEAT"])
        report = suite.run(inputs={"WEAT": {"target_embeddings": "nonsense",
                                            "attribute_embeddings": "nonsense"}})
        assert report.results == []
        assert "WEAT" in report.skipped

    def test_on_error_raise_propagates(self):
        suite = BiasSuite(StubBackend(access=("embeddings",)), metrics=["WEAT"])
        with pytest.raises((ValueError, TypeError)):
            suite.run(inputs={"WEAT": {"target_embeddings": "x",
                                       "attribute_embeddings": "y"}},
                      on_error="raise")

    def test_the_protocol_records_model_axis_and_language(self):
        report = BiasSuite(StubBackend(model_id="m/x"), axis="race",
                           language="en").run()
        assert report.protocol["model_id"] == "m/x"
        assert report.protocol["axis"] == "race"
        assert report.protocol["language"] == "en"

    def test_no_composite_score_exists_anywhere(self):
        """PLAN.md lists a composite bias score as a non-goal."""
        report = make_report("m", {"A": 0.6, "B": 0.4})
        assert not hasattr(report, "overall")
        assert not hasattr(report, "composite")
        assert "overall" not in report.to_dict()


class TestCompare:
    def test_deltas_equal_the_score_differences(self):
        """PLAN.md 5.4 acceptance check."""
        a = make_report("a", {"M1": 0.6, "M2": 0.3})
        b = make_report("b", {"M1": 0.9, "M2": 0.1})
        for delta in compare(a, b):
            assert delta.delta == pytest.approx(delta.b - delta.a)
        assert {d.metric: d.delta for d in compare(a, b)} == pytest.approx(
            {"M1": 0.3, "M2": -0.2}
        )

    def test_mismatched_metric_sets_raise(self):
        a = make_report("a", {"M1": 0.5})
        b = make_report("b", {"M2": 0.5})
        with pytest.raises(ValueError, match="same metric set"):
            compare(a, b)

    def test_overlapping_intervals_are_flagged(self):
        a = Report("a", [make_result("M", 0.5, ci=(0.4, 0.6))])
        b = Report("b", [make_result("M", 0.55, ci=(0.45, 0.65))])
        assert compare(a, b)[0].intervals_overlap is True

    def test_disjoint_intervals_are_flagged(self):
        a = Report("a", [make_result("M", 0.1, ci=(0.0, 0.2))])
        b = Report("b", [make_result("M", 0.9, ci=(0.8, 1.0))])
        assert compare(a, b)[0].intervals_overlap is False

    def test_absent_intervals_give_none_not_false(self):
        a = Report("a", [make_result("M", 0.1, ci=None)])
        b = Report("b", [make_result("M", 0.9, ci=(0.8, 1.0))])
        assert compare(a, b)[0].intervals_overlap is None


class TestCorrelate:
    def test_a_known_correlation_is_recovered(self):
        """PLAN.md 5.4 acceptance check: synthetic reports, known answer.

        M2 is a strictly increasing function of M1 across models, so their
        Spearman correlation is exactly +1; M3 is strictly decreasing, so it is
        exactly -1.
        """
        reports = [
            make_report(f"m{i}", {"M1": v, "M2": v * 2 + 1, "M3": 1.0 - v})
            for i, v in enumerate([0.1, 0.3, 0.5, 0.7])
        ]
        matrix = correlate(reports, method="spearman")
        assert matrix["M1"]["M2"] == pytest.approx(1.0)
        assert matrix["M1"]["M3"] == pytest.approx(-1.0)
        assert matrix["M1"]["M1"] == pytest.approx(1.0)

    def test_pearson_is_available(self):
        reports = [make_report(f"m{i}", {"A": v, "B": 3 * v})
                   for i, v in enumerate([0.1, 0.4, 0.9])]
        assert correlate(reports, method="pearson")["A"]["B"] == pytest.approx(1.0)

    def test_spearman_ignores_a_monotone_transform(self):
        """The reason to prefer Spearman here: metrics have incomparable scales."""
        linear = [make_report(f"m{i}", {"A": v, "B": v}) for i, v in
                  enumerate([0.1, 0.4, 0.9])]
        warped = [make_report(f"m{i}", {"A": v, "B": v ** 5}) for i, v in
                  enumerate([0.1, 0.4, 0.9])]
        assert correlate(linear)["A"]["B"] == pytest.approx(
            correlate(warped)["A"]["B"]
        )

    def test_fewer_than_three_reports_raises(self):
        reports = [make_report(f"m{i}", {"A": 0.1 * i, "B": 0.2 * i}) for i in range(2)]
        with pytest.raises(ValueError, match="at least 3"):
            correlate(reports)

    def test_reports_sharing_one_metric_raise(self):
        reports = [make_report(f"m{i}", {"A": 0.1 * i}) for i in range(3)]
        with pytest.raises(ValueError, match="at least 2"):
            correlate(reports)

    def test_an_unknown_method_raises(self):
        reports = [make_report(f"m{i}", {"A": 0.1 * i, "B": 0.2 * i}) for i in range(3)]
        with pytest.raises(ValueError, match="method"):
            correlate(reports, method="kendall")


class _Parsed(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)

    def handle_data(self, data):
        self.text.append(data)


class TestReportFormats:
    def _report(self):
        return Report(
            model_id="test/model",
            results=[
                make_result("WEATish", 0.6, family="embedding"),
                make_result("CrowSish", 0.4, family="probability"),
                make_result("Sketchy", 0.9, family="prompt", fidelity="mismatch"),
            ],
            protocol=make_protocol("BiasSuite", model_id="test/model", seed=42,
                                   dtype="bf16"),
            skipped={"Absent": "no inputs supplied"},
        )

    def test_every_result_appears_in_all_three_formats(self):
        """PLAN.md 5.4 acceptance check."""
        report = self._report()
        markdown = to_markdown(report)
        document = to_html(report)
        payload = report.to_dict()
        json_names = {r["metric"] for r in payload["results"]}

        for result in report.results:
            assert result.metric in markdown
            assert result.metric in document
            assert result.metric in json_names

    def test_the_html_parses(self):
        """PLAN.md 5.4 acceptance check."""
        parser = _Parsed()
        parser.feed(to_html(self._report()))
        assert "table" in parser.tags
        assert "html" in parser.tags

    def test_one_section_per_family(self):
        """PLAN.md 5.4 acceptance check."""
        document = to_html(self._report())
        for family in ("embedding", "probability", "prompt"):
            assert f"<h2>{family}</h2>" in document

    def test_the_protocol_table_is_present(self):
        """PLAN.md 5.4 acceptance check."""
        document = to_html(self._report())
        assert "Protocol" in document
        assert self._report().protocol["hash"] in document

    def test_the_html_opens_offline(self):
        """PLAN.md Section 9: no external URLs in the file."""
        document = to_html(self._report())
        for scheme in ("http://", "https://", "//cdn", "src="):
            assert scheme not in document, f"external reference {scheme!r} in HTML"

    def test_a_mismatch_is_flagged_loudly_in_both_formats(self):
        report = self._report()
        assert "MISMATCH" in to_markdown(report)
        assert "MISMATCH" in to_html(report)

    def test_reports_warn_against_averaging(self):
        """The composite score is a non-goal; the report says so."""
        for text in (to_markdown(self._report()), to_html(self._report())):
            assert "not be averaged" in text

    def test_skipped_metrics_are_shown_not_hidden(self):
        assert "Absent" in to_markdown(self._report())
        assert "Absent" in to_html(self._report())

    def test_intervals_are_rendered_for_every_scored_metric(self):
        markdown = to_markdown(self._report())
        assert markdown.count("[0, 1]") == 3

    def test_save_json_round_trips(self, tmp_path):
        from bias_scope.result import from_dict

        path = save_json(self._report(), tmp_path / "r.json")
        assert path.exists()
        import json as _json

        payload = _json.loads(path.read_text())
        restored = [from_dict(r) for r in payload["results"]]
        assert [r.metric for r in restored] == [r.metric for r in self._report().results]

    def test_html_escapes_a_hostile_model_id(self):
        report = Report(model_id="<script>alert(1)</script>",
                        results=[make_result("M", 0.5)])
        assert "<script>" not in to_html(report)
        assert "&lt;script&gt;" in to_html(report)

    def test_to_html_writes_a_file_when_asked(self, tmp_path: Path):
        target = tmp_path / "nested" / "report.html"
        to_html(self._report(), target)
        assert target.exists() and target.read_text().startswith("<!doctype html>")
