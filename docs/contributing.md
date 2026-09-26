# How to Contribute

<div class="contrib-hero">
  <p class="contrib-kicker">Bias Scope Contributions</p>
  <h2>Help us expand, test, and improve bias evaluation tooling.</h2>
  <p>Whether you are fixing a bug, proposing a new metric, or improving documentation, this page outlines the workflow we use to keep contributions easy to review and maintain.</p>
</div>

<div class="contrib-grid">
  <div class="contrib-card">
    <h3>Report</h3>
    <p>Open a focused issue with enough detail to reproduce the problem quickly.</p>
  </div>
  <div class="contrib-card">
    <h3>Discuss</h3>
    <p>For new metrics or larger changes, align on scope early before implementation.</p>
  </div>
  <div class="contrib-card">
    <h3>Submit</h3>
    <p>Use a small branch, include tests, and send a clear pull request.</p>
  </div>
</div>

## Quick Workflow

| Step | What to do |
| --- | --- |
| 1. Open an issue | Report a bug or propose a feature before starting substantial work. |
| 2. Create a branch | Keep each change isolated in its own branch. |
| 3. Implement carefully | Match project conventions and keep changes scoped. |
| 4. Add tests and docs | Update documentation and validate behavior before opening a PR. |
| 5. Submit a PR | Explain the motivation, changes, and any review notes. |

<div class="contrib-note">
  <strong>Before you start:</strong> Check existing issues and pull requests first so work is not duplicated.
</div>

## Reporting Issues and Bugs

If you find a bug, open an issue in the repository with enough context for maintainers to reproduce it quickly.

<div class="contrib-section" markdown="1">

### Include in the issue

- A clear summary of the problem
- Steps to reproduce the issue
- Expected behavior and actual behavior
- Environment details such as Python version, operating system, and relevant package versions
- A minimal code sample, traceback, or dataset snippet when applicable

</div>

<div class="contrib-note">
  <strong>Good bug reports save review time:</strong> concise reproduction steps and a minimal failing example usually lead to faster fixes.
</div>

## Suggesting New Bias Metrics

New metric proposals are welcome, especially when they extend coverage across model types, tasks, or demographic dimensions.

<div class="contrib-section" markdown="1">

### Include in the proposal

- The metric name and its original paper or source
- The bias dimension or evaluation setting it targets
- Why it should be added to <code>bias-scope</code>
- Expected inputs, outputs, and dependencies
- Any implementation constraints, licensing concerns, or dataset requirements

</div>

<div class="contrib-note">
  <strong>Planning to implement it yourself?</strong> Mention that in the issue so maintainers can align on scope, naming, dependencies, and review expectations early.
</div>

## Adding a Metric in Six Steps

Once a proposal is agreed, a metric is added by copying one existing module in the same
family and following these steps. The [architecture](architecture.md) page shows where each
piece lives.

1. **Read the sources first.** Read the paper and the authors' code before writing any
   code, and record the sections and files you read in `sources/SOURCES.yaml`
   (`scripts/sources/fetch_sources.py` downloads them). A metric is never implemented
   from memory.
2. **Copy a module.** Copy one existing module in `src/bias_scope/<family>_based/`, keep
   functions short, and state the formula in one or two lines of the class docstring.
3. **Fill `MetricInfo` and add the registry entry.** In `_metric_info.py` declare the
   access the metric needs, its neutral value, direction and range, its reference, and its
   fidelity. Any status other than `faithful` needs a deviation note that says what
   differs and why. Then add one entry to `validation/registry.yaml` naming the published
   value to check against, or `no_published_reference`.
4. **Write the known-answer test first.** Reproduce a worked example from the paper as a
   test that fails, then make it pass. Never change a protocol to make a result match.
5. **Write the audit note.** Add `docs/fidelity/<metric>.md` from the template, then
   regenerate the index with `python scripts/verification/render_fidelity_index.py`.
6. **Add an example and the docs page.** Add a runnable example under `examples/`, add a
   `NewPage` entry in `scripts/docs/render_api_pages.py`, and run that script to create
   the page and its metric card. Finish with `ruff check src tests`,
   `python -m pytest -q --cov=bias_scope`, and `mkdocs build --strict`.

The tests in `tests/test_docs.py` fail if a metric has no API page, if a page is missing
from the nav, or if the fidelity index no longer matches the registry.

## Forking and Branching

Use a standard fork-and-branch workflow for contributions.

<div class="contrib-section" markdown="1">

### Recommended flow

1. Fork the repository to your GitHub account.
2. Clone your fork locally.
3. Create a dedicated branch from <code>main</code> for each change.
4. Keep each branch focused on a single fix, feature, or documentation update.

</div>

Use descriptive branch names such as <code>fix/disco-threshold-bug</code> or <code>feat/add-new-metric</code>.

## Pull Request Guidelines

Small, focused pull requests are easier to review and merge than large mixed changes.

<div class="contrib-section" markdown="1">

### Before opening a PR

- Sync your branch with the latest <code>main</code> branch
- Keep the change scoped to one feature, fix, or documentation improvement
- Update documentation when behavior, APIs, or usage expectations change
- Add or update tests that cover the change

### In the PR description include

- A short summary of what changed
- The motivation for the change
- Links to related issues
- Notes on any new dependencies, datasets, or limitations

</div>

## Code Style and Testing Expectations

Contributions should match the existing project structure and style conventions.

<div class="contrib-section contrib-checklist" markdown="1">

### Expectations

- Write clear, readable Python code with consistent naming and formatting
- Prefer small, composable functions over large monolithic implementations
- Add docstrings or inline documentation when behavior is not obvious
- Preserve backward compatibility unless a breaking change is explicitly discussed
- Add tests for new functionality and regression tests for bug fixes
- Run the relevant test suite before submitting a pull request

</div>

<div class="contrib-note contrib-note-strong">
  <strong>Strong contributions are usually small and clear:</strong> a focused scope, good tests, and a readable PR description matter more than a large mixed change.
</div>

