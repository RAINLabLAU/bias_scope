#!/usr/bin/env python3
"""Opt-in BOLD paper-protocol scoring over external artifacts; no model downloads on import."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bias_scope.prompts_based._bold_reproduction import (
    embedding_gender_polarity,
    gender_unigram,
    load_official_prompt_file,
    psycholinguistic_norms,
    regard_from_labels,
    sentiment_from_compounds,
    toxicity_from_labels,
    validate_paper_counts,
)


def _write(value: dict, output: Path | None) -> None:
    text = json.dumps(value, indent=2)
    if output:
        output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    validate = sub.add_parser("validate-prompts")
    validate.add_argument("--domain-file", action="append", required=True, metavar="DOMAIN=PATH")
    sentiment = sub.add_parser("score-sentiment")
    toxicity = sub.add_parser("score-toxicity-labels")
    regard = sub.add_parser("score-regard-labels")
    norms = sub.add_parser("score-norms")
    unigram = sub.add_parser("score-gender-unigram")
    embedding = sub.add_parser("score-gender-embedding")
    for command in (sentiment, toxicity, norms, unigram, embedding):
        command.add_argument("--input-file", type=Path, required=True)
    regard.add_argument("--input-file", type=Path, required=True)
    regard.add_argument("--domain", required=True)
    regard.add_argument("--group", required=True)
    embedding.add_argument("--vectors-file", type=Path, required=True)
    for command in (validate, sentiment, toxicity, regard, norms, unigram, embedding):
        command.add_argument("--output-file", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "validate-prompts":
        artifacts = {}
        for item in args.domain_file:
            if "=" not in item:
                raise ValueError("--domain-file must be DOMAIN=PATH")
            domain, path = item.split("=", 1)
            artifacts[domain] = load_official_prompt_file(path, domain)["groups"]
        result = validate_paper_counts(artifacts)
    else:
        payload = json.loads(args.input_file.read_text(encoding="utf-8"))
        if args.mode == "score-sentiment":
            result = sentiment_from_compounds(payload)
        elif args.mode == "score-toxicity-labels":
            result = toxicity_from_labels(payload)
        elif args.mode == "score-regard-labels":
            result = regard_from_labels(payload, domain=args.domain, group=args.group)
        elif args.mode == "score-norms":
            result = psycholinguistic_norms(payload["tokens"], payload["word_norms"])
        elif args.mode == "score-gender-unigram":
            result = gender_unigram(payload)
        else:
            vectors = json.loads(args.vectors_file.read_text(encoding="utf-8"))
            result = embedding_gender_polarity(payload, vectors)
    _write(result, args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
