"""Perspective API client for toxicity detection."""

import logging
import time
from typing import List, Optional

import requests

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
        self, api_key: str, rate_limit_delay: float = 1.0, max_retries: int = 3
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

        Args:
            text (str): Text to score. Maximum length is 20,480 characters.
                Longer texts will be truncated with a warning.

        Returns:
            float: Toxicity probability in [0, 1].
                - 0.0: Not toxic at all
                - 0.5: Moderately toxic (common threshold)
                - 1.0: Extremely toxic

        Raises:
            ValueError: If text is empty or only whitespace
            RuntimeError: If API request fails after all retries

        Examples:
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
                    params={"key": self.api_key},
                    json={
                        "comment": {"text": text},
                        "languages": ["en"],
                        "requestedAttributes": {"TOXICITY": {}},
                    },
                    timeout=10,
                )

                # Success
                if response.status_code == 200:
                    data = response.json()
                    score = data["attributeScores"]["TOXICITY"]["summaryScore"]["value"]

                    # Rate limiting
                    time.sleep(self.rate_limit_delay)

                    return float(score)

                # Rate limit hit
                elif response.status_code == 429:
                    wait_time = 2**attempt  # Exponential backoff
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
                        time.sleep(2**attempt)

            except requests.exceptions.RequestException as e:
                logger.error(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)

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

        Args:
            texts (List[str]): List of texts to score

        Returns:
            List[float]: Toxicity scores (0-1) for each text, in same order

        Examples:
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
