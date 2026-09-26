"""The offline examples that the documentation pages include must run.

Each API page for a v0.2 metric shows one of these files, so a broken example is
a broken page. They use stub callables in place of a model, so they need no
network, no key and no optional extra.
"""

import doctest
import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

OFFLINE_EXAMPLES = [
    "prompts_based/decodingtrust.py",
    "prompts_based/discrim_eval.py",
    "prompts_based/first_person_fairness.py",
    "prompts_based/implicit_association.py",
    "prompts_based/political_even_handedness.py",
    "prompts_based/trustllm.py",
    "prompts_based/winobias.py",
    "probability_based/pairwise_likelihood_preference.py",
    "generated_text_based/stereotype_rule_hit_rate.py",
    "framework/recommend.py",
    "framework/suite.py",
]


@pytest.mark.parametrize("relative", OFFLINE_EXAMPLES)
def test_the_example_runs_offline(relative, capsys):
    runpy.run_path(str(EXAMPLES / relative), run_name="__main__")
    assert capsys.readouterr().out.strip(), f"{relative} printed nothing"


def test_the_stereotype_rule_hit_rate_docstring_example_runs():
    """Its example once named the class it was renamed from."""
    from bias_scope.generated_text_based import stereotype_rule_hit_rate as module

    results = doctest.testmod(
        module, optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS,
    )
    assert results.failed == 0
