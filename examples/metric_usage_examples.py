"""Small non-network usage examples for bias-scope metrics."""

from __future__ import annotations

import math

import numpy as np

from bias_scope.embeddings_based import WEAT
from bias_scope.generated_text_based import CounterfactualSentimentBias, EMT
from bias_scope.probability_based import CrowSPairs


class TinyProbabilityScorer:
    """Minimal custom scorer object compatible with TokenPredictionScorer."""

    def token_probability(self, tokens: list[str], position: int) -> float:
        return 0.8 if tokens[0] == "Women" else 0.2

    def token_probability_with_attention(self, tokens: list[str], position: int) -> dict[str, object]:
        attention = np.ones(len(tokens), dtype=float) / len(tokens)
        return {
            "prob": self.token_probability(tokens, position),
            "attention": attention,
        }

    def masked_token_probability(self, context: list[str], candidate: str) -> float:
        if candidate in {"Women", "Girls"}:
            return 0.8
        if candidate in {"Men", "Boys"}:
            return 0.2
        return 0.5

    def logprob(self, tokens: list[str], batch_size: int = 16) -> float:
        _ = batch_size
        return sum(math.log(self.token_probability(tokens, i)) for i in range(len(tokens)))


def run_embedding_example() -> float:
    """Run WEAT with precomputed toy embeddings."""
    targets = (
        np.array([[1.0, 0.0], [0.9, 0.1]]),
        np.array([[0.0, 1.0], [0.1, 0.9]]),
    )
    attributes = (
        np.array([[1.0, 0.0], [0.95, 0.05]]),
        np.array([[0.0, 1.0], [0.05, 0.95]]),
    )
    return WEAT().evaluate(targets, attributes)


def run_generated_text_example() -> dict[str, float]:
    """Run generated-text metrics with precomputed scores."""
    completions = [["A calm answer", "A harsh answer"]]
    toxicity_scores = [[0.1, 0.4]]
    emt_score = EMT().evaluate(completions, toxicity_scores)

    csb_score = CounterfactualSentimentBias().evaluate(
        [["Group A response"]],
        [["Group B response"]],
        [[0.3]],
        [[0.1]],
    )
    return {"emt": emt_score, "csb": csb_score}


def run_probability_example() -> float:
    """Run CrowS-Pairs with a tiny custom scorer object."""
    scorer = TinyProbabilityScorer()
    pairs = [(["Women", "are", "kind"], ["Men", "are", "kind"])]
    return CrowSPairs().evaluate(sentence_pairs=pairs, predict_masked_token=scorer)


def main() -> dict[str, object]:
    """Execute non-network examples and return their outputs."""
    return {
        "embedding": run_embedding_example(),
        "generated_text": run_generated_text_example(),
        "probability": run_probability_example(),
    }


if __name__ == "__main__":
    print(main())
