"""Smoke tests for examples/metric_usage_examples.py."""

import runpy


def test_metric_usage_examples_smoke() -> None:
    """The example file loads and runs non-network examples without errors."""
    module_globals = runpy.run_path("examples/metric_usage_examples.py")
    module_globals["main"]()
