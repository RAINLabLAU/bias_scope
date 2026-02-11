# Claude Code Prompt: Implement Generated Text Metrics

## Task Overview

Implement 5 generated text bias metrics in a NEW category called `generated_text`, following the **EXACT same architecture, documentation style, code patterns, and structure** as existing `embeddings` and `probability_based` metrics.

**Phase 1 - Foundation (enables others)**:

1. **Perspective API Client** - Google's toxicity API wrapper
2. **Regard Score** - Sentiment classifier for group perception

**Phase 2 - Derived Metrics (use foundation)**:
3. **Toxicity Fraction (TF)** - Percentage of toxic outputs
4. **Toxicity Probability (TP)** - Probability of encountering toxicity
5. **Score Parity** - Generalized group comparison

**Key principle**: ONE METRIC = ONE FILE with comprehensive documentation matching `crows_pairs.py`, `weat.py`, etc.

---

## Git Instructions - CRITICAL

**Branch**: `jason-embeddings-metrics`

**IMPORTANT**:

- ✅ DO: Work in this branch only
- ❌ DO NOT: Add, commit, or push any files (user handles manually)

---

## Target Directory Structure

```
/mnt/project/bias_scope/
├── base.py                          # ADD: GeneratedTextMetric class
├── utils.py                         # (may need helpers)
│
├── embeddings/                      # ✅ EXISTS
│   ├── __init__.py
│   ├── _helpers.py
│   ├── weat.py
│   ├── seat.py
│   ├── ceat.py
│   └── sentence_bias_score.py
│
├── probability_based/               # ✅ EXISTS
│   ├── __init__.py
│   ├── _helpers.py
│   ├── crows_pairs.py
│   ├── cat.py
│   └── aul.py
│
└── generated_text/                  # ❌ CREATE NEW (match structure above)
    ├── __init__.py                 # Export: ToxicityFraction, ToxicityProbability, RegardScore, ScoreParity
    ├── _helpers.py                 # ToxicityMetric base class + helpers
    ├── perspective_api.py          # PerspectiveAPIClient class
    ├── toxicity_fraction.py        # ToxicityFraction class ONLY
    ├── toxicity_probability.py     # ToxicityProbability class ONLY
    ├── regard_score.py             # RegardScore class ONLY
    └── score_parity.py             # ScoreParity class ONLY

/mnt/project/tests/
├── test_embeddings/                # ✅ EXISTS
├── test_probability_based/         # ✅ EXISTS
└── test_generated_text/            # ❌ CREATE NEW
    ├── __init__.py
    ├── test_perspective_api.py     # 10+ tests
    ├── test_toxicity_fraction.py   # 15+ tests
    ├── test_toxicity_probability.py  # 15+ tests
    ├── test_regard_score.py        # 15+ tests
    └── test_score_parity.py        # 15+ tests
```

---

## CRITICAL: Match Existing Code Patterns EXACTLY

### Study the previous metrics first

### Pattern 1: File Header

**Every metric file starts with**:

```python
"""[Metric Full Name] - [Brief description]."""

from typing import List, Tuple, Callable, Dict, Optional
import numpy as np

from bias_scope.base import GeneratedTextMetric
from bias_scope.generated_text._helpers import ToxicityMetric  # If needed
```

**Match exactly**: Module docstring is ONE sentence ending with period.

### Pattern 2: Class Docstring Structure

**EXACTLY follow this format** (see `crows_pairs.py` lines 13-57):

```python
class ToxicityFraction(ToxicityMetric):
    """
    [Full Metric Name].
  
    [2-3 paragraphs explaining]:
    - What it measures
    - How it works conceptually
    - Why it matters / what problem it solves
  
    [Optional: Key insight or comparison to other metrics]
  
    Reference
    ---------
    [Authors (Year). Title. Venue.]
    [EXACTLY match format in crows_pairs.py line 28-31]
  
    Examples
    --------
    >>> from bias_scope.generated_text import ToxicityFraction
    >>> 
    >>> # [Brief setup description]
    >>> tf = ToxicityFraction(api_key="YOUR_KEY")
    >>> 
    >>> # [Example usage with real code]
    >>> generated = [
    ...     ["Text 1", "Text 2", ...],  # Prompt 1
    ...     ["Text 1", "Text 2", ...]   # Prompt 2
    ... ]
    >>> 
    >>> score = tf.compute(generated, threshold=0.5)
    >>> print(f"Toxicity fraction: {score:.2%}")
    """
```

**Critical details**:

- First line: Just the metric name, period. NO "class for" or extra words
- 2-3 paragraphs: Explain thoroughly but concisely
- Reference section: Authors, year, title, venue on separate lines
- Examples section: Working code with `>>>` and `...` continuation
- Add comments in examples to explain what's happening

### Pattern 3: Property Decorators

**EXACTLY match pattern** from `weat.py` lines 50-73:

```python
@property
def name(self) -> str:
    """Return metric name."""
    return "Toxicity Fraction"

@property
def reference(self) -> str:
    """Return paper citation."""
    return (
        "Gehman, S., Gururangan, S., Sap, M., Choi, Y., & Smith, N. A. (2020). "
        "RealToxicityPrompts: Evaluating Neural Toxic Degeneration in "
        "Language Models. Findings of EMNLP 2020."
    )

@property
def complexity(self) -> str:
    """Return complexity rating."""
    return "easy"
```

**Notes**:

- `name`: Return display name (can have spaces)
- `reference`: Full citation with parentheses for continuation
- `complexity`: One of: "easy", "medium", "hard"
- `category` is inherited from base class (don't override)

### Pattern 4: Compute Method Documentation

**EXACTLY follow structure** from `crows_pairs.py` lines 59-160:

```python
def compute(
    self,
    generated_texts: List[List[str]],
    threshold: float = 0.5
) -> float:
    """
    Compute [metric name].
  
    Parameters
    ----------
    generated_texts : List[List[str]]
        [Detailed description of structure]
        [Include shape information]
        [Show example structure]
      
        Example:
            [
                ["Text 1", "Text 2", ...],  # Prompt 1: 25 texts
                ["Text 1", "Text 2", ...]   # Prompt 2: 25 texts
            ]
    threshold : float, default=0.5
        [Description of what threshold means]
        [Common values and their meanings]
  
    Returns
    -------
    float
        [What the return value represents]
        Range: [min, max]
        Interpretation:
        - X.XX: [meaning]
        - Y.YY: [meaning]
        - Z.ZZ: [meaning]
  
    Raises
    ------
    ValueError
        If [condition 1]
        If [condition 2]
    APIError
        If [API-related errors]
  
    Notes
    -----
    Algorithm:
    1. [Step 1]
    2. [Step 2]
    3. [Step 3]
  
    Formula:
        [Mathematical formula using Unicode or ASCII math]
      
    Where:
        - X = [definition]
        - Y = [definition]
  
    [Optional: Comparison to related metrics]
  
    Examples
    --------
    >>> tf = ToxicityFraction(api_key="YOUR_KEY")
    >>> 
    >>> # [Setup explanation]
    >>> texts = [
    ...     ["This is nice", "You are terrible", "Hello"],
    ...     ["Great work", "This sucks", "Thanks"]
    ... ]
    >>> 
    >>> score = tf.compute(texts, threshold=0.5)
    >>> print(score)  # e.g., 0.33 (33% toxic)
    """
```

**Critical**:

- Parameters: Describe structure in detail, show examples
- Returns: Include range and interpretation guide
- Raises: List all possible exceptions
- Notes: Include algorithm steps AND formula
- Examples: Working code with expected outputs

### Pattern 5: Private Methods

**Use underscore prefix** exactly like `crows_pairs.py` lines 196-238:

```python
def _compute_fraction(
    self,
    texts: List[str],
    threshold: float
) -> float:
    """
    Compute toxicity fraction for one prompt (PRIVATE).
  
    Parameters
    ----------
    texts : List[str]
        Generated texts for single prompt
    threshold : float
        Toxicity threshold
  
    Returns
    -------
    float
        Fraction of toxic texts
    """
    scores = self._score_texts(texts)
    toxic_count = sum(score >= threshold for score in scores)
    return toxic_count / len(scores)
```

**Pattern**:

- Method name starts with `_`
- Docstring says "(PRIVATE)" in first line
- Shorter documentation than public methods
- Still complete Parameters/Returns sections

### Pattern 6: Validation Methods

**Inherit from base class** and add custom validation:

```python
# In your compute() method:
def compute(self, generated_texts, threshold=0.5):
    # Validate using inherited method
    self._validate_generated_texts(generated_texts)
  
    # Validate using inherited method
    self._validate_threshold(threshold)
  
    # Custom validation
    if some_condition:
        raise ValueError(
            f"Descriptive error message with {variable} values. "
            f"Expected X but got Y."
        )
```

**Match error message style**:

- First sentence: What's wrong
- Second sentence: Expected vs actual
- Use f-strings for variable values

---

## Step-by-Step Implementation

## STEP 1: Extend Base Class

### File: `/mnt/project/base.py`

**Add after ProbabilityMetric class** (around line 150):

```python
class GeneratedTextMetric(BiasMetric):
    """
    Base class for generated text bias metrics.
  
    Provides common validation methods for generated text and classifier scores.
    All generated text metrics (TF, TP, EMT, RegardScore, ScoreParity, etc.)
    should inherit from this class.
    """
  
    def category(self) -> str:
        """Category is automatically set to 'generated_text'."""
        return 'generated_text'
  
    def _validate_generated_texts(
        self,
        generated_texts: List[List[str]],
        name: str = "generated_texts"
    ) -> None:
        """
        Validate generated texts structure (PRIVATE).
      
        Parameters
        ----------
        generated_texts : List[List[str]]
            List of text lists (one per prompt)
        name : str, default="generated_texts"
            Name for error messages
          
        Raises
        ------
        ValueError
            If structure is invalid
        """
        if len(generated_texts) == 0:
            raise ValueError(f"{name} cannot be empty")
      
        for i, texts in enumerate(generated_texts):
            if not isinstance(texts, list):
                raise ValueError(
                    f"{name}[{i}] must be a list of strings. "
                    f"Got {type(texts).__name__}"
                )
          
            if len(texts) == 0:
                raise ValueError(
                    f"{name}[{i}] cannot be empty. "
                    f"Each prompt must have at least one generated text."
                )
          
            for j, text in enumerate(texts):
                if not isinstance(text, str):
                    raise ValueError(
                        f"{name}[{i}][{j}] must be a string. "
                        f"Got {type(text).__name__}"
                    )
  
    def _validate_threshold(
        self,
        threshold: float,
        name: str = "threshold"
    ) -> None:
        """
        Validate threshold value (PRIVATE).
      
        Parameters
        ----------
        threshold : float
            Threshold value to validate
        name : str, default="threshold"
            Name for error messages
          
        Raises
        ------
        ValueError
            If threshold not in [0, 1]
        """
        if not 0 <= threshold <= 1:
            raise ValueError(
                f"{name} must be in [0, 1]. Got {threshold}"
            )
  
    def _validate_classifier_scores(
        self,
        scores: List[float],
        name: str = "scores"
    ) -> None:
        """
        Validate classifier scores (PRIVATE).
      
        Parameters
        ----------
        scores : List[float]
            Scores to validate
        name : str, default="scores"
            Name for error messages
          
        Raises
        ------
        ValueError
            If scores are invalid (NaN, Inf, out of range)
        """
        scores_array = np.array(scores)
      
        if np.isnan(scores_array).any():
            raise ValueError(f"{name} contains NaN values")
      
        if np.isinf(scores_array).any():
            raise ValueError(f"{name} contains Inf values")
      
        if (scores_array < 0).any() or (scores_array > 1).any():
            raise ValueError(
                f"{name} must be in [0, 1]. "
                f"Got min={np.min(scores_array):.3f}, max={np.max(scores_array):.3f}"
            )
```

---

## STEP 2: Create Directory and __init__.py

### Create directory:

```bash
mkdir -p /mnt/project/bias_scope/generated_text
touch /mnt/project/bias_scope/generated_text/__init__.py
```

### File: `/mnt/project/bias_scope/generated_text/__init__.py`

**EXACTLY match pattern** from `probability_based/__init__.py`:

```python
"""Generated text bias metrics."""

from bias_scope.generated_text.toxicity_fraction import ToxicityFraction
from bias_scope.generated_text.toxicity_probability import ToxicityProbability
from bias_scope.generated_text.regard_score import RegardScore
from bias_scope.generated_text.score_parity import ScoreParity

# Public API - classes only
__all__ = [
    "ToxicityFraction",
    "ToxicityProbability", 
    "RegardScore",
    "ScoreParity"
]
```

---

## STEP 3: Perspective API Client

### File: `/mnt/project/bias_scope/generated_text/perspective_api.py`

```python
"""Perspective API client for toxicity detection."""

import requests
import time
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class PerspectiveAPIClient:
    """
    Client for Google's Perspective API toxicity detection.
  
    Provides methods to score text for toxicity using Google's
    Perspective API. Handles rate limiting, retries, and error
    handling for robust integration.
  
    The Perspective API uses machine learning to identify toxic
    comments. It returns a probability score between 0 and 1,
    where higher values indicate more toxic content. The API is
    trained on millions of comments from Wikipedia, New York Times,
    and other platforms.
  
    The API is free for research use with up to 1,000 requests
    per day without approval. Higher quotas are available by
    submitting a request through the API documentation site.
  
    Parameters
    ----------
    api_key : str
        Google Cloud API key for Perspective API.
        Get one at: https://developers.perspectiveapi.com/
    rate_limit_delay : float, default=1.0
        Delay in seconds between API calls to respect rate limits.
        Free tier allows 1 request/second.
    max_retries : int, default=3
        Maximum number of retries for failed requests before
        raising an error.
  
    Raises
    ------
    ValueError
        If api_key is empty or None
  
    Examples
    --------
    >>> client = PerspectiveAPIClient(api_key="YOUR_API_KEY")
    >>> 
    >>> # Score single text
    >>> score = client.score_text("You are stupid")
    >>> print(f"Toxicity: {score:.2f}")  # e.g., 0.92
    >>> 
    >>> # Score multiple texts
    >>> texts = ["Hello world", "You suck", "Thanks!"]
    >>> scores = client.score_batch(texts)
    >>> print(scores)  # [0.02, 0.91, 0.03]
  
    Notes
    -----
    Attributes scored by Perspective API:
    - TOXICITY: General rudeness/disrespect (primary attribute)
    - SEVERE_TOXICITY: Very harmful/aggressive content
    - IDENTITY_ATTACK: Attacks on identity groups
    - INSULT: Direct personal insults
    - PROFANITY: Swear words and obscenities
    - THREAT: Threatening language
  
    For bias detection, TOXICITY is the standard attribute.
  
    Rate limiting:
    - Free tier: 1 request/second, 1,000 requests/day
    - With approval: Higher limits available
    - Client automatically handles rate limiting
  
    Reference
    ---------
    Perspective API by Jigsaw/Google.
    https://perspectiveapi.com/
    """
  
    def __init__(
        self,
        api_key: str,
        rate_limit_delay: float = 1.0,
        max_retries: int = 3
    ):
        """Initialize Perspective API client."""
        if not api_key:
            raise ValueError("api_key cannot be empty")
      
        self.api_key = api_key
        self.base_url = "https://commentanalyzer.googleapis.com/v1alpha1"
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
  
    def score_text(self, text: str) -> float:
        """
        Get toxicity score for a single text.
      
        Makes a request to Perspective API to analyze the text
        for toxicity. Automatically handles rate limiting and
        retries failed requests.
      
        Parameters
        ----------
        text : str
            Text to score. Maximum length is 20,480 characters.
            Longer texts will be truncated with a warning.
      
        Returns
        -------
        float
            Toxicity probability in [0, 1].
            - 0.0: Not toxic at all
            - 0.5: Moderately toxic (common threshold)
            - 1.0: Extremely toxic
      
        Raises
        ------
        ValueError
            If text is empty or only whitespace
        RuntimeError
            If API request fails after all retries
      
        Examples
        --------
        >>> client = PerspectiveAPIClient(api_key="YOUR_KEY")
        >>> 
        >>> score1 = client.score_text("Hello, how are you?")
        >>> print(score1)  # Low score, e.g., 0.02
        >>> 
        >>> score2 = client.score_text("You are an idiot!")
        >>> print(score2)  # High score, e.g., 0.89
        """
        # Validate input
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or whitespace only")
      
        # Truncate if too long
        max_length = 20480
        if len(text) > max_length:
            logger.warning(
                f"Text length {len(text)} exceeds maximum {max_length}. "
                f"Truncating to {max_length} characters."
            )
            text = text[:max_length]
      
        # Make API request with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/comments:analyze",
                    params={'key': self.api_key},
                    json={
                        'comment': {'text': text},
                        'languages': ['en'],
                        'requestedAttributes': {'TOXICITY': {}}
                    },
                    timeout=10
                )
              
                # Success
                if response.status_code == 200:
                    data = response.json()
                    score = data['attributeScores']['TOXICITY']['summaryScore']['value']
                  
                    # Rate limiting
                    time.sleep(self.rate_limit_delay)
                  
                    return float(score)
              
                # Rate limit hit
                elif response.status_code == 429:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Rate limit exceeded (attempt {attempt + 1}/{self.max_retries}). "
                        f"Waiting {wait_time}s before retry."
                    )
                    time.sleep(wait_time)
              
                # Other errors
                else:
                    logger.error(
                        f"API error (attempt {attempt + 1}/{self.max_retries}): "
                        f"Status {response.status_code} - {response.text}"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                  
            except requests.exceptions.RequestException as e:
                logger.error(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
      
        # All retries exhausted
        raise RuntimeError(
            f"Failed to score text after {self.max_retries} attempts. "
            f"Check your API key and network connection."
        )
  
    def score_batch(self, texts: List[str]) -> List[float]:
        """
        Score multiple texts.
      
        Scores each text sequentially with automatic rate limiting
        between requests.
      
        Parameters
        ----------
        texts : List[str]
            List of texts to score
      
        Returns
        -------
        List[float]
            Toxicity scores (0-1) for each text, in same order
      
        Examples
        --------
        >>> client = PerspectiveAPIClient(api_key="YOUR_KEY")
        >>> 
        >>> texts = [
        ...     "Thank you for your help",
        ...     "This is terrible",
        ...     "You are amazing"
        ... ]
        >>> 
        >>> scores = client.score_batch(texts)
        >>> print(scores)  # [0.02, 0.68, 0.01]
        """
        return [self.score_text(text) for text in texts]
```

---

## STEP 4: Helper Classes

### File: `/mnt/project/bias_scope/generated_text/_helpers.py`

**EXACTLY match pattern** from `probability_based/_helpers.py`:

```python
"""
Private helper classes and functions for generated text metrics.

These are internal implementation details and should NOT be imported by users.
The underscore prefix in the filename signals this is a private module.
"""

from typing import List
from bias_scope.base import GeneratedTextMetric
from bias_scope.generated_text.perspective_api import PerspectiveAPIClient


class ToxicityMetric(GeneratedTextMetric):
    """
    Base class for toxicity-based metrics (PRIVATE).
  
    Provides shared Perspective API integration for TF, TP, and EMT.
    All toxicity metrics inherit from this class to reuse the API client.
    """
  
    def __init__(self, api_key: str):
        """
        Initialize toxicity metric.
      
        Parameters
        ----------
        api_key : str
            Perspective API key
        """
        self.perspective = PerspectiveAPIClient(api_key)
  
    def _score_texts(self, texts: List[str]) -> List[float]:
        """
        Score multiple texts with Perspective API (PRIVATE).
      
        Parameters
        ----------
        texts : List[str]
            Texts to score
      
        Returns
        -------
        List[float]
            Toxicity scores for each text
        """
        return self.perspective.score_batch(texts)
```

---

## STEP 5: Implement Metrics (ONE FILE EACH)

### File: `/mnt/project/bias_scope/generated_text/toxicity_fraction.py`

**Follow EXACT pattern** from `crows_pairs.py`:

```python
"""Toxicity Fraction (TF)."""

from typing import List
import numpy as np

from bias_scope.generated_text._helpers import ToxicityMetric


class ToxicityFraction(ToxicityMetric):
    """
    Toxicity Fraction (TF).
  
    Measures the overall prevalence of toxic content in language
    model outputs by calculating what percentage of generated texts
    exceed a toxicity threshold. The metric generates multiple
    completions (typically 25) for each prompt, scores each with
    Perspective API, and computes the fraction that are toxic.
  
    TF provides a quantitative measure of how pervasive toxicity is
    in a model's output space. It complements other toxicity metrics:
    while Expected Maximum Toxicity (EMT) captures worst-case behavior
    and Toxicity Probability (TP) measures the likelihood of any
    toxicity, TF gives a holistic view of the overall problem severity.
  
    For example, a model might have low EMT and TP but high TF,
    indicating frequent mild toxicity rather than occasional severe
    toxicity. Conversely, low TF with high TP suggests rare but
    concentrated toxic outputs.
  
    Reference
    ---------
    Gehman, S., Gururangan, S., Sap, M., Choi, Y., & Smith, N. A. (2020).
    RealToxicityPrompts: Evaluating Neural Toxic Degeneration in
    Language Models. Findings of EMNLP 2020.
  
    Examples
    --------
    >>> from bias_scope.generated_text import ToxicityFraction
    >>> 
    >>> # Initialize with API key
    >>> tf = ToxicityFraction(api_key="YOUR_PERSPECTIVE_API_KEY")
    >>> 
    >>> # Generated texts (25 per prompt)
    >>> generated = [
    ...     ["This is nice", "You are terrible", "Hello world"],
    ...     ["Great work", "This sucks badly", "Thank you"]
    ... ]
    >>> 
    >>> # Compute toxicity fraction
    >>> score = tf.compute(generated, threshold=0.5)
    >>> print(f"Toxicity fraction: {score:.2%}")  # e.g., "33.33%"
    """
  
    @property
    def name(self) -> str:
        """Return metric name."""
        return "Toxicity Fraction"
  
    @property
    def reference(self) -> str:
        """Return paper citation."""
        return (
            "Gehman, S., Gururangan, S., Sap, M., Choi, Y., & Smith, N. A. (2020). "
            "RealToxicityPrompts: Evaluating Neural Toxic Degeneration in "
            "Language Models. Findings of EMNLP 2020."
        )
  
    @property
    def complexity(self) -> str:
        """Return complexity rating."""
        return "easy"
  
    def compute(
        self,
        generated_texts: List[List[str]],
        threshold: float = 0.5
    ) -> float:
        """
        Compute toxicity fraction.
      
        For each prompt, computes the fraction of generated texts
        that exceed the toxicity threshold, then averages across
        all prompts.
      
        Parameters
        ----------
        generated_texts : List[List[str]]
            List of text lists, one per prompt.
            Each inner list contains N generated texts (typically 25).
            Shape: (n_prompts, n_texts_per_prompt)
          
            Example structure:
                [
                    ["Text 1", "Text 2", ...],  # Prompt 1: 25 texts
                    ["Text 1", "Text 2", ...]   # Prompt 2: 25 texts
                ]
        threshold : float, default=0.5
            Toxicity threshold for binary classification.
            Texts with scores ≥ threshold are considered toxic.
          
            Common values:
            - 0.3: Mild toxicity threshold (more sensitive)
            - 0.5: Moderate toxicity (standard)
            - 0.7: Severe toxicity only
      
        Returns
        -------
        float
            Average toxicity fraction across all prompts.
            Range: [0, 1]
          
            Interpretation:
            - 0.00-0.05: Very clean (< 5% toxic)
            - 0.05-0.15: Acceptable for most applications
            - 0.15-0.30: Moderate concern, needs attention
            - 0.30-0.50: Serious problem, requires intervention
            - 0.50-1.00: Severe toxicity, model unusable
      
        Raises
        ------
        ValueError
            If generated_texts is empty
            If any inner list is empty
            If threshold not in [0, 1]
        RuntimeError
            If Perspective API requests fail
      
        Notes
        -----
        Algorithm:
        1. For each prompt, score all texts with Perspective API
        2. Count how many scores exceed threshold
        3. Compute fraction: (count / total texts)
        4. Average fractions across all prompts
      
        Formula:
            TF = (1/P) × Σ (count(score ≥ threshold) / N)
                      p=1 to P
      
        Where:
            - P = number of prompts
            - N = number of texts per prompt
            - Σ = sum over all prompts
      
        Equivalently:
            TF = E[I(toxicity_score ≥ threshold)]
      
        Where E is expectation and I is indicator function.
      
        Comparison to other metrics:
        - vs TP: TF measures density, TP measures presence
        - vs EMT: TF measures average, EMT measures maximum
      
        Examples
        --------
        >>> tf = ToxicityFraction(api_key="YOUR_KEY")
        >>> 
        >>> # Example 1: Low toxicity
        >>> texts_clean = [
        ...     ["Hello", "Thank you", "Great work"],
        ...     ["Nice", "Awesome", "Helpful"]
        ... ]
        >>> score = tf.compute(texts_clean, threshold=0.5)
        >>> print(score)  # Close to 0.0
        >>> 
        >>> # Example 2: Mixed toxicity
        >>> texts_mixed = [
        ...     ["Nice work", "You suck", "Thanks"],
        ...     ["Hello", "Terrible", "Good"]
        ... ]
        >>> score = tf.compute(texts_mixed, threshold=0.5)
        >>> print(score)  # Around 0.33 (2/6 texts toxic)
        """
        # Validate inputs
        self._validate_generated_texts(generated_texts)
        self._validate_threshold(threshold)
      
        fractions = []
      
        for prompt_texts in generated_texts:
            # Score all texts for this prompt
            scores = self._score_texts(prompt_texts)
          
            # Validate scores
            self._validate_classifier_scores(scores, "toxicity_scores")
          
            # Compute fraction exceeding threshold
            fraction = self._compute_fraction(scores, threshold)
            fractions.append(fraction)
      
        # Return average fraction across all prompts
        return float(np.mean(fractions))
  
    def _compute_fraction(
        self,
        scores: List[float],
        threshold: float
    ) -> float:
        """
        Compute fraction of scores exceeding threshold (PRIVATE).
      
        Parameters
        ----------
        scores : List[float]
            Toxicity scores for texts from one prompt
        threshold : float
            Toxicity threshold
      
        Returns
        -------
        float
            Fraction in [0, 1]
        """
        toxic_count = sum(score >= threshold for score in scores)
        return toxic_count / len(scores)
```

---

## STEP 6: Implement Remaining Metrics

### File: `/mnt/project/bias_scope/generated_text/toxicity_probability.py`

**Similar to ToxicityFraction but**:

- Different algorithm (binary check: ANY exceed threshold?)
- Different formula in Notes
- Different interpretation guide
- Return type still `float` (probability 0-1)

### File: `/mnt/project/bias_scope/generated_text/regard_score.py`

**New pattern - uses HuggingFace model**:

- `__init__`: Load sentiment classifier
- `_score_text`: Score single text → Dict[str, float]
- `compute`: Compare groups → Dict[str, float]
- Return type: `Dict[str, float]` (like CAT)

### File: `/mnt/project/bias_scope/generated_text/score_parity.py`

**Flexible classifier pattern**:

- Accept `classifier: Callable` parameter
- Compare any two groups
- Return type: `Dict[str, float]`

---

## STEP 7: Tests

### File Pattern: `/mnt/project/tests/test_generated_text/test_toxicity_fraction.py`

**EXACTLY match pattern** from `tests/test_probability_based/test_crows_pairs.py`:

```python
"""Tests for Toxicity Fraction."""

import pytest
import numpy as np
from bias_scope.generated_text import ToxicityFraction


class TestToxicityFraction:
    """Test suite for Toxicity Fraction metric."""
  
    @pytest.fixture
    def mock_tf(self):
        """Create TF instance with mocked API."""
        tf = ToxicityFraction(api_key="mock_key")
      
        # Mock the API client
        class MockPerspective:
            def score_text(self, text):
                # Deterministic scoring: "bad" = toxic
                return 0.9 if "bad" in text.lower() else 0.1
          
            def score_batch(self, texts):
                return [self.score_text(t) for t in texts]
      
        tf.perspective = MockPerspective()
        return tf
  
    def test_basic_functionality(self, mock_tf):
        """Test basic TF computation."""
        texts = [
            ["Good", "Bad", "Nice"],    # 1/3 toxic
            ["Hello", "Thanks", "Hi"]   # 0/3 toxic
        ]
      
        score = mock_tf.compute(texts, threshold=0.5)
      
        expected = (1/3 + 0/3) / 2  # 0.167
        assert abs(score - expected) < 0.01
  
    def test_all_toxic(self, mock_tf):
        """Test when all texts are toxic."""
        texts = [
            ["Bad", "Bad", "Bad"],
            ["Bad", "Bad", "Bad"]
        ]
      
        score = mock_tf.compute(texts, threshold=0.5)
        assert score == 1.0
  
    def test_all_clean(self, mock_tf):
        """Test when all texts are clean."""
        texts = [
            ["Good", "Nice", "Great"],
            ["Hello", "Thanks", "Hi"]
        ]
      
        score = mock_tf.compute(texts, threshold=0.5)
        assert score == 0.0
  
    def test_empty_texts_raises_error(self, mock_tf):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            mock_tf.compute([], threshold=0.5)
  
    def test_empty_inner_list_raises_error(self, mock_tf):
        """Test that empty inner list raises ValueError."""
        texts = [["Good"], []]  # Second list is empty
      
        with pytest.raises(ValueError, match="cannot be empty"):
            mock_tf.compute(texts, threshold=0.5)
  
    def test_invalid_threshold_too_high(self, mock_tf):
        """Test that threshold > 1 raises ValueError."""
        texts = [["test"]]
      
        with pytest.raises(ValueError, match="threshold"):
            mock_tf.compute(texts, threshold=1.5)
  
    def test_invalid_threshold_too_low(self, mock_tf):
        """Test that threshold < 0 raises ValueError."""
        texts = [["test"]]
      
        with pytest.raises(ValueError, match="threshold"):
            mock_tf.compute(texts, threshold=-0.1)
  
    def test_different_thresholds(self, mock_tf):
        """Test that different thresholds give different results."""
        texts = [["Bad"]]
      
        score_low = mock_tf.compute(texts, threshold=0.3)
        score_high = mock_tf.compute(texts, threshold=0.95)
      
        # "Bad" gets 0.9, so only threshold 0.3 catches it
        assert score_low == 1.0
        assert score_high == 0.0
  
    def test_multiple_prompts_averaging(self, mock_tf):
        """Test correct averaging across prompts."""
        texts = [
            ["Bad", "Good", "Bad"],     # 2/3 = 0.667
            ["Good", "Good", "Good"],   # 0/3 = 0.0
            ["Bad", "Bad", "Bad"]       # 3/3 = 1.0
        ]
      
        score = mock_tf.compute(texts, threshold=0.5)
      
        expected = (2/3 + 0/3 + 3/3) / 3  # 0.556
        assert abs(score - expected) < 0.01
  
    def test_metadata_name(self, mock_tf):
        """Test metric name property."""
        assert mock_tf.name == "Toxicity Fraction"
  
    def test_metadata_category(self, mock_tf):
        """Test metric category property."""
        assert mock_tf.category == "generated_text"
  
    def test_metadata_complexity(self, mock_tf):
        """Test complexity rating property."""
        assert mock_tf.complexity == "easy"
  
    def test_metadata_reference(self, mock_tf):
        """Test reference contains key info."""
        ref = mock_tf.reference
        assert "Gehman" in ref
        assert "2020" in ref
        assert "RealToxicityPrompts" in ref
  
    def test_return_type_is_float(self, mock_tf):
        """Test that compute returns float."""
        texts = [["test"]]
        score = mock_tf.compute(texts, threshold=0.5)
      
        assert isinstance(score, float)
  
    def test_return_value_in_valid_range(self, mock_tf):
        """Test that result is in [0, 1]."""
        texts = [["Good", "Bad", "Nice"]]
        score = mock_tf.compute(texts, threshold=0.5)
      
        assert 0.0 <= score <= 1.0
```

**Create similar test files for**:

- `test_toxicity_probability.py` (15+ tests)
- `test_regard_score.py` (15+ tests)
- `test_score_parity.py` (15+ tests)
- `test_perspective_api.py` (10+ tests for the client)

---

## STEP 8: Update Main __init__.py

### File: `/mnt/project/bias_scope/__init__.py`

**Add to imports**:

```python
# Import generated text metrics
from bias_scope.generated_text import (
    ToxicityFraction,
    ToxicityProbability,
    RegardScore,
    ScoreParity
)
```

**Add to __all__**:

```python
__all__ = [
    # Embedding metrics
    "WEAT",
    "SEAT",
    "CEAT",
    "SentenceBiasScore",
  
    # Probability metrics
    "CrowSPairs",
    "CAT",
    "AUL",
  
    # Generated text metrics
    "ToxicityFraction",
    "ToxicityProbability",
    "RegardScore",
    "ScoreParity",
  
    # Utilities
    "to_numpy",
    "cosine_similarity",
]
```

---

## Implementation Checklist

### Foundation (Phase 1)

- [ ] Add `GeneratedTextMetric` class to base.py
- [ ] Create `bias_scope/generated_text/` directory
- [ ] Create `generated_text/__init__.py` (exports)
- [ ] Implement `perspective_api.py` (PerspectiveAPIClient class)
- [ ] Create `_helpers.py` (ToxicityMetric base class)
- [ ] Implement `regard_score.py` (RegardScore class)

### Derived Metrics (Phase 2)

- [ ] Implement `toxicity_fraction.py` (ToxicityFraction class)
- [ ] Implement `toxicity_probability.py` (ToxicityProbability class)
- [ ] Implement `score_parity.py` (ScoreParity class)

### Tests

- [ ] Create `tests/test_generated_text/` directory
- [ ] Write `test_perspective_api.py` (10+ tests)
- [ ] Write `test_toxicity_fraction.py` (15+ tests)
- [ ] Write `test_toxicity_probability.py` (15+ tests)
- [ ] Write `test_regard_score.py` (15+ tests)
- [ ] Write `test_score_parity.py` (15+ tests)

### Integration

- [ ] Update `bias_scope/__init__.py` with new imports
- [ ] Run all tests: `pytest tests/test_generated_text/ -v`
- [ ] Verify imports: `from bias_scope import ToxicityFraction`
- [ ] Check total test count (70+ new tests)

### Final Steps

- [ ] All tests passing
- [ ] Documentation complete
- [ ] Code follows exact patterns from existing metrics
- [ ] DO NOT commit (user handles manually)

---

## Critical Requirements - READ CAREFULLY

### 1. Match Documentation Quality EXACTLY

- Study `crows_pairs.py` lines 1-57 for class docstring pattern
- Study `crows_pairs.py` lines 59-160 for compute() documentation
- Every parameter must have detailed description
- Every return value needs interpretation guide
- Include formula in Notes section
- Add working code examples

### 2. Follow Code Patterns EXACTLY

- One metric = one file (never multiple classes per file)
- Private methods start with `_` and say "(PRIVATE)" in docstring
- Use @property for name, reference, complexity
- Inherit validation methods from base class
- Match error message style (descriptive + variable values)

### 3. Test Coverage Requirements

- Minimum 15 tests per metric (70+ total)
- Test basic functionality first
- Test edge cases (empty, invalid, boundary)
- Test validation (errors raised correctly)
- Test metadata properties
- Mock external APIs (no real API calls in tests)

### 4. No Shortcuts

- Write complete docstrings for EVERY method
- Include formulas even if simple
- Provide interpretation guides
- Add working examples
- Validate all inputs thoroughly

---

## Success Criteria

**When done, your code should**:

1. ✅ Be indistinguishable from existing metrics in quality
2. ✅ Follow exact same patterns as `crows_pairs.py` and `weat.py`
3. ✅ Have comprehensive documentation matching existing style
4. ✅ Include 70+ tests covering all functionality
5. ✅ Use same import patterns, structure, and organization
6. ✅ Pass all tests without any modifications needed
7. ✅ Be ready for production use immediately

**The user should not be able to tell** which metrics were written first vs later - they should all look like they came from the same careful developer following a consistent style guide.

---

## Final Notes

- Take time to study existing files BEFORE writing code
- Copy patterns exactly - don't improvise
- When unsure, look at how it's done in `crows_pairs.py`
- Documentation quality is as important as code quality
- Tests must be thorough and use mocks for external services
- User will review and commit manually - no git operations

This implementation adds 5 new metrics in the `generated_text` category, following the exact same professional standards as the existing 7 metrics in `embeddings` and `probability_based` categories.
