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

<div class="contrib-section">

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

<div class="contrib-section">

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

## Forking and Branching

Use a standard fork-and-branch workflow for contributions.

<div class="contrib-section">

### Recommended flow

1. Fork the repository to your GitHub account.
2. Clone your fork locally.
3. Create a dedicated branch from <code>main</code> for each change.
4. Keep each branch focused on a single fix, feature, or documentation update.

</div>

Use descriptive branch names such as <code>fix/disco-threshold-bug</code> or <code>feat/add-new-metric</code>.

## Pull Request Guidelines

Small, focused pull requests are easier to review and merge than large mixed changes.

<div class="contrib-section">

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

<div class="contrib-section contrib-checklist">

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
