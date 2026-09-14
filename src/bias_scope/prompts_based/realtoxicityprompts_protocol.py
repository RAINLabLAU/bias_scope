"""Private protocol constants for RealToxicityPrompts paper reconstruction."""

from __future__ import annotations

PAPER_EVIDENCE_COMMIT = "3beff74a01f6c880f84b1d9d355d9d02be62fc94"
RTP_PROTOCOL_VERSION = "rtp-paper-reconstruction-v1"

PAPER_CITATION = (
    "Gehman et al. (2020), RealToxicityPrompts: Evaluating Neural Toxic "
    "Degeneration in Language Models, Findings of EMNLP 2020"
)
OFFICIAL_REPOSITORY = "https://github.com/allenai/real-toxicity-prompts"

PERSPECTIVE_ATTRIBUTES = (
    "TOXICITY",
    "SEVERE_TOXICITY",
    "IDENTITY_ATTACK",
    "INSULT",
    "THREAT",
    "PROFANITY",
    "SEXUALLY_EXPLICIT",
    "FLIRTATION",
)

# The released custom GPT-2 generator, rather than a chat-template protocol.
GPT2_PAPER_GENERATION = {
    "checkpoint": "gpt2",
    "samples_per_prompt": 25,
    "generation_batch_size": 32,
    "max_new_tokens": 20,
    "do_sample": True,
    "top_p": 0.9,
    "top_k": 0,
    "temperature": 1.0,
    "repetition_penalty": 1.0,
    "seed": 42,
    "eos_stopping": True,
    "pad_token": "eos",
    "prompt_format": "raw_causal",
    "chat_template": False,
    "model_eval": True,
    "torch_no_grad": True,
    "dtype": "fp32",
}

SCORING_MODES = {
    "historical_score_parity",
    "current_perspective_reconstruction",
    "substitute_scorer",
}
