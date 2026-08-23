# BiasScope
Read PLAN.md before starting any task; update its checkboxes as you go.
Never stop to ask a question. Decide (ultrathink when it is not obvious), apply
the most reversible defensible option, log it in REVIEW_LATER.md, and continue.
Before auditing, fixing, or adding any metric: fetch and read the original paper
and the authors' code (PLAN.md Section 4.0); never implement a metric from memory.
Never invent published values; never change a protocol to make a result match;
never delete a failing test.
Write the failing test first. Run `ruff check src tests` and
`python -m pytest -q --cov=bias_scope` before every commit.
Keep code plain: no metaclasses, no operator overloading, functions under 40 lines.
Append a dated entry to PROGRESS.md at the end of each session.
