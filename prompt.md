Using your Web Search feature (or by typing `@Web` / `@arXiv`), look up the original research papers cited in our codebase's docstrings for the specific bias metrics listed below. 

I need you to write a comprehensive, rigorous section for a Markdown file (`metrics_deep_dive.md`) documenting these metrics. For each metric, you must provide the exact theoretical methodology and mathematical formulations directly from the authors' original papers, combined with our specific code architecture.

Do not summarize loosely, do not say "similar to the above," and do not omit equations. Use standard LaTeX format for all mathematical formulas.

For each metric in the list, follow this exact structure:

#### [Metric Class Name]
* **Module Path:** [e.g., `bias_scope.probability_based`]
* **Original Paper Reference:** [Extract the formal paper citation—Authors, Year, Venue—written in our code's docstring.]
* **Theoretical Foundation & Mathematical Rationale:** - What was the core problem the original authors were trying to solve?
  - What is the underlying psychological, social, or statistical hypothesis behind their test?
  - Rationale for the approach according to the original paper.
* **Exact Mathematical Formulations:** - Provide the complete mathematical equations exactly as defined in the original paper using LaTeX.
  - Define every single variable, indicator function, set, and normalization factor cleanly.
* **Code Implementation & Data Architecture:** - Explain how our codebase specifically translates this math into Python.
  - Detail the expected inputs (including data structures, shapes, required callables, or boolean masks).
  - List the internal algorithmic steps (such as loop sequences, data casting, outlier removal rules, or layer attention-weight extraction).
  - Detail the exact output formats (such as specific dictionary structures or float boundaries) and clarify how to interpret the numerical results.
* **Self-Contained Code Example:** - Provide a short, clean, fully working Python example initializing the metric class and evaluating it using mock data or a brief dataset snippet modeled after our `examples/` directory.


