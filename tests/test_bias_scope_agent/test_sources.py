"""On-demand retrieval of the vendored sources a dataset loader reads.

third_party/ is git-ignored, so on a fresh clone every loader's file is
missing and the agent used to stop and tell a human to run fetch_sources.py.
These tests pin the behaviour that replaces that: the harness runs the same
command itself, for the one manifest entry whose file is missing.

No test here reaches the network. `ensure_metric_sources` only ever acts on a
path under this repository's own third_party/code, and every test either
injects a fake runner or points at a temporary directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bias_scope_agent import sources
from bias_scope_agent.datasets_common import _require


@pytest.fixture(autouse=True)
def forget_attempts():
    sources.reset_attempts()
    yield
    sources.reset_attempts()


class _Runner:
    """A stand-in for fetch_sources.py that records what it was asked for."""

    def __init__(self, creates: Path = None):
        self.calls = []
        self._creates = creates

    def __call__(self, metric: str) -> bool:
        self.calls.append(metric)
        if self._creates is not None:
            self._creates.parent.mkdir(parents=True, exist_ok=True)
            self._creates.write_text("fetched", encoding="utf-8")
        return True


class TestEnsureMetricSources:
    def test_a_missing_vendored_file_triggers_the_fetch_for_its_metric(self, tmp_path):
        missing = sources.VENDORED_ROOT / "crows-pairs" / "data" / "nothing-here.csv"
        runner = _Runner()
        assert sources.ensure_metric_sources("CrowSPairs", missing, runner=runner) is True
        assert runner.calls == ["CrowSPairs"]

    def test_a_path_outside_this_repo_is_never_fetched(self, tmp_path):
        runner = _Runner()
        outside = tmp_path / "crows-pairs" / "data" / "crows_pairs_anonymized.csv"
        assert sources.ensure_metric_sources("CrowSPairs", outside, runner=runner) is False
        assert runner.calls == []

    def test_a_file_already_on_disk_is_not_refetched(self, tmp_path):
        present = sources.VENDORED_ROOT / "already-there.txt"
        present.parent.mkdir(parents=True, exist_ok=True)
        present.write_text("here", encoding="utf-8")
        runner = _Runner()
        try:
            assert sources.ensure_metric_sources("CrowSPairs", present, runner=runner) is False
            assert runner.calls == []
        finally:
            present.unlink()

    def test_one_attempt_per_metric_per_process(self):
        missing = sources.VENDORED_ROOT / "gone" / "a.csv"
        runner = _Runner()
        sources.ensure_metric_sources("CrowSPairs", missing, runner=runner)
        sources.ensure_metric_sources("CrowSPairs", missing, runner=runner)
        assert runner.calls == ["CrowSPairs"], "a failed fetch must not be retried all run"

    def test_the_env_var_turns_automatic_fetching_off(self, monkeypatch):
        monkeypatch.setenv("BIASSCOPE_AGENT_AUTO_FETCH", "0")
        runner = _Runner()
        missing = sources.VENDORED_ROOT / "gone" / "b.csv"
        assert sources.ensure_metric_sources("CrowSPairs", missing, runner=runner) is False
        assert runner.calls == []


class TestRequireFetchesBeforeGivingUp:
    def test_a_fetch_that_produces_the_file_lets_the_loader_continue(self, monkeypatch):
        target = sources.VENDORED_ROOT / "fetched-by-test" / "dev.json"
        runner = _Runner(creates=target)
        monkeypatch.setattr(sources, "_fetch_metric", runner)
        try:
            assert _require(target, "CAT") == target
            assert runner.calls == ["CAT"]
        finally:
            if target.exists():
                target.unlink()

    def test_a_fetch_that_does_not_produce_it_still_names_the_command(self, monkeypatch):
        runner = _Runner()
        monkeypatch.setattr(sources, "_fetch_metric", runner)
        with pytest.raises(ValueError, match="fetch_sources"):
            _require(sources.VENDORED_ROOT / "still-missing" / "dev.json", "CAT")
        assert runner.calls == ["CAT"]

    def test_a_temporary_tree_never_reaches_the_fetcher(self, tmp_path, monkeypatch):
        runner = _Runner()
        monkeypatch.setattr(sources, "_fetch_metric", runner)
        with pytest.raises(ValueError, match="fetch_sources"):
            _require(tmp_path / "sent-bias" / "tests" / "weat3.jsonl", "WEAT")
        assert runner.calls == []


class TestPreflight:
    def test_it_fetches_every_manifest_entry_the_loaders_read_from(self):
        runner = _Runner()
        done = sources.ensure_dataset_sources(runner=runner)
        assert set(done) == set(sources.DATASET_SOURCE_METRICS)

    def test_every_metric_it_names_is_an_entry_in_the_manifest(self):
        import yaml

        manifest = yaml.safe_load((sources.REPO_ROOT / "sources" / "SOURCES.yaml").read_text())
        known = {entry.get("metric") for entry in manifest.get("metrics") or []}
        assert set(sources.DATASET_SOURCE_METRICS) <= known

    def test_it_names_every_metric_the_loaders_actually_require(self):
        """A loader gaining a new source must not silently escape the preflight."""
        import re

        hints = set()
        for name in ("datasets", "datasets_common", "datasets_generated", "datasets_prompt"):
            text = (sources.REPO_ROOT / "src" / "bias_scope_agent" / f"{name}.py").read_text()
            hints |= set(re.findall(r'_require\([^)]*,\s*"([A-Za-z]+)"\s*\)', text))
        assert hints <= set(sources.DATASET_SOURCE_METRICS), sorted(
            hints - set(sources.DATASET_SOURCE_METRICS)
        )
