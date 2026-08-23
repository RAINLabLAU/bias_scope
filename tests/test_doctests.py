"""Execute the docstring examples in the pure-Python core modules.

PLAN.md Section 1 requires every documented example to run. Docstring examples
are documentation too, and an example that has silently rotted is worse than
none. Only modules that import cleanly without the optional extras are listed
here; the metric modules are covered by `tests/test_examples/`.
"""

import doctest

import pytest

from bias_scope import stats, utils

MODULES = [stats, utils]


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_docstring_examples_run(module):
    """Every >>> example in the module produces the output it claims."""
    results = doctest.testmod(
        module,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS,
        verbose=False,
    )
    assert results.failed == 0, (
        f"{results.failed} of {results.attempted} docstring examples in "
        f"{module.__name__} failed; run "
        f"`python -m pytest --doctest-modules src/bias_scope/{module.__name__.split('.')[-1]}.py` "
        "to see them"
    )
