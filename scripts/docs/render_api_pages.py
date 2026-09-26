#!/usr/bin/env python3
"""Keep docs/api in step with the metric registry (PLAN.md 11.1 and 11.2).

Three jobs, all derived from `METRIC_INFO` so the docs cannot claim something the
code does not:

1. **Create a page** for each metric that has none (the `NEW_PAGES` table below:
   the v0.2 metrics). A page is a title, a short introduction, the mkdocstrings
   block, and a runnable example included from `examples/` so the docs show the
   file the test suite runs.

2. **Write a metric card** into every API page: family, model access, neutral
   value, direction, range, fidelity status with the first sentence of its
   deviation note, and the source. The card sits between two HTML comment markers
   and is rewritten in place, so hand-written prose around it is never touched.

3. **Rewrite generated regions** elsewhere: the per-family tables in
   `api/overview.md`, the category table on the home page, and the readiness
   table in `framework/multilingual.md`. Each is a `<!-- NAME:start -->` ...
   `<!-- NAME:end -->` region.

`tests/test_docs.py` fails when any of this is stale.

    python scripts/docs/render_api_pages.py          # write
    python scripts/docs/render_api_pages.py --check  # exit 1 if anything would change
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bias_scope._metric_info import METRIC_INFO  # noqa: E402

DOCS = REPO_ROOT / "docs"
API = DOCS / "api"
INDEX = DOCS / "fidelity" / "INDEX.md"

START = "<!-- metric-card:start -->"
END = "<!-- metric-card:end -->"
AUTODOC = re.compile(r"^:::\s+([\w.]+)\s*$", re.MULTILINE)
INDEX_ROW = re.compile(r"^\| `(\w+)` \| \w+ \| \*\*\w+\*\* \| \[[^\]]*\]\(([^)]+)\) \|$",
                       re.MULTILINE)

STATUS = {
    "faithful": "same formula and protocol as the cited paper",
    "adaptation": "same comparison as the cited paper, but a different access mode "
                  "or scoring path that can change the numbers",
    "original": "BiasScope's own metric, inspired by a cited idea",
    "mismatch": "does not implement the statistic its cited name suggests; do not use",
    "unaudited": "sources not yet read, so no fidelity claim is made",
}
DIRECTION = {
    "higher_more_biased": "higher means more biased",
    "lower_more_biased": "lower means more biased",
    "signed": "signed; 0 is neutral",
}


@dataclass(frozen=True)
class NewPage:
    """A page to create for a metric that has none."""

    path: str
    title: str
    classes: Tuple[str, ...]
    intro: str
    example: Optional[str] = None
    usage: Optional[str] = None


_PB = "bias_scope.prompts_based"
NEW_PAGES: Tuple[NewPage, ...] = (
    NewPage(
        "probability/pairwise_likelihood_preference.md", "PairwiseLikelihoodPreference",
        ("bias_scope.probability_based.pairwise_likelihood_preference.PairwiseLikelihoodPreference",),
        "The share of (stereotype, anti-stereotype) sentence pairs where the model gives "
        "the stereotype the higher log-probability. It shipped as `LPBS` through v0.1.1; "
        "[`LPBS`](lpbs.md) is now Kurita et al.'s metric.",
        example="examples/probability_based/pairwise_likelihood_preference.py",
    ),
    NewPage(
        "probability/topk_fill_divergence.md", "TopKFillDivergence",
        ("bias_scope.probability_based.topk_fill_divergence.TopKFillDivergence",),
        "The size of the symmetric difference between the top-k mask fills of two prompts "
        "that differ only in a sensitive attribute. It shipped as `DisCoMetric` through "
        "v0.1.1; [`DisCoMetric`](disco.md) is now Webster et al.'s significance-tested metric.",
        usage=(
            "from bias_scope.probability_based import TopKFillDivergence\n\n"
            'metric = TopKFillDivergence(model_name="bert-base-uncased")\n'
            'result = metric.evaluate("The {attr} works as a [MASK].", "man", "woman", k=3)'
        ),
    ),
    NewPage(
        "generated_text/stereotype_rule_hit_rate.md", "StereotypeRuleHitRate",
        ("bias_scope.generated_text_based.stereotype_rule_hit_rate.StereotypeRuleHitRate",),
        "Counts how often a generation places rule-defined group terms and attribute terms "
        "within a token window of each other, per 1,000 generations. It shipped as "
        "`StereotypicalAssociations` through v0.1.1; "
        "[`StereotypicalAssociations`](stereotypical_associations.md) is now HELM's metric.",
        example="examples/generated_text_based/stereotype_rule_hit_rate.py",
    ),
    NewPage(
        "prompts/winobias.md", "WinoBias", (f"{_PB}.winobias.WinoBias",),
        "The gap in coreference accuracy between pro- and anti-stereotypical versions of "
        "the same sentences (Zhao et al., 2018).",
        example="examples/prompts_based/winobias.py",
    ),
    NewPage(
        "prompts/decodingtrust.md", "DecodingTrust",
        (f"{_PB}.decodingtrust.DecodingTrustStereotype",
         f"{_PB}.decodingtrust.DecodingTrustFairness"),
        "Two metrics from the DecodingTrust benchmark (Wang et al., 2023): how often a model "
        "agrees with stereotype statements, and the demographic-parity gap of a model used "
        "as a classifier on tabular records.",
        example="examples/prompts_based/decodingtrust.py",
    ),
    NewPage(
        "prompts/discrim_eval.md", "DiscrimEval", (f"{_PB}.discrim_eval.DiscrimEval",),
        "How a model's probability of answering \"yes\" to the same decision question shifts "
        "when only the demographic details change (Tamkin et al., 2023).",
        example="examples/prompts_based/discrim_eval.py",
    ),
    NewPage(
        "prompts/first_person_fairness.md", "FirstPersonFairness",
        (f"{_PB}.first_person_fairness.FirstPersonFairness",),
        "Whether a chatbot's responses carry a harmful stereotype when only the user's name "
        "changes (Eloundou et al., 2024).",
        example="examples/prompts_based/first_person_fairness.py",
    ),
    NewPage(
        "prompts/implicit_association.md", "Implicit association and decision bias",
        (f"{_PB}.implicit_association.ImplicitAssociationTest",
         f"{_PB}.implicit_association.LLMDecisionBias"),
        "Two psychology-inspired probes from Bai et al. (PNAS 2025): a word-association task "
        "and a paired decision task.",
        example="examples/prompts_based/implicit_association.py",
    ),
    NewPage(
        "prompts/political_even_handedness.md", "PoliticalEvenHandedness",
        (f"{_PB}.political_even_handedness.PoliticalEvenHandedness",),
        "Whether a model treats paired prompts with opposing political stances the same way "
        "(Anthropic, 2025). Read the three reported rates together.",
        example="examples/prompts_based/political_even_handedness.py",
    ),
    NewPage(
        "prompts/trustllm.md", "TrustLLM fairness",
        (f"{_PB}.trustllm.TrustLLMStereotypeRecognition",
         f"{_PB}.trustllm.TrustLLMStereotypeAgreement",
         f"{_PB}.trustllm.TrustLLMDisparagement",
         f"{_PB}.trustllm.TrustLLMPreference"),
        "Four fairness metrics from the TrustLLM benchmark (Huang et al., 2024). The neutral "
        "value differs between them, so check the card for each before reading a score.",
        example="examples/prompts_based/trustllm.py",
    ),
)


def note_links() -> Dict[str, str]:
    """metric name -> fidelity note file name, read from the generated index."""
    text = INDEX.read_text(encoding="utf-8")
    return {name: target for name, target in INDEX_ROW.findall(text)
            if not target.startswith("..")}


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _range(low: float, high: float) -> str:
    """'0 to 1', '0 or more', '-1 or less', or 'unbounded'."""
    inf = float("inf")
    if low == -inf and high == inf:
        return "unbounded"
    if high == inf:
        return f"{low:g} or more"
    if low == -inf:
        return f"{high:g} or less"
    return f"{low:g} to {high:g}"


MAX_NOTE = 320


def first_sentence(note: str) -> str:
    """The lead sentence of a deviation note.

    The registry's notes are written for maintainers and run to audit history;
    the card shows what differs and the linked audit note holds the rest.
    """
    text = " ".join(note.split())
    lead = re.split(r"(?<=[.!?])\s+(?=[A-Z`'\"(])", text, maxsplit=1)[0]
    if len(lead) > MAX_NOTE:
        lead = lead[:MAX_NOTE].rsplit(" ", 1)[0] + " ..."
    return lead


def metric_card(name: str, notes: Dict[str, str], depth: int) -> str:
    """One metric's card. `depth` is how many directories the page is below docs/."""
    info = METRIC_INFO[name]
    fidelity = f"**{info.fidelity}**: {STATUS[info.fidelity]}."
    if info.deviation_note:
        fidelity += f" {first_sentence(info.deviation_note)}"
    if name in notes:
        fidelity += f" [Audit note]({'../' * depth}fidelity/{notes[name]})."
    rows = [
        ("Family", info.family),
        ("Model access", ", ".join(f"`{a}`" for a in info.access)),
        ("Neutral value", f"{info.neutral_value:g}"),
        ("Direction", DIRECTION[info.direction]),
        ("Range", _range(*info.value_range)),
        ("Languages", ", ".join(info.languages)),
        ("Fidelity", fidelity),
        ("Source", info.reference),
    ]
    if info.reference_impl:
        rows.append(("Reference code", info.reference_impl))
    body = "\n".join(f"| {label} | {_cell(value)} |" for label, value in rows)
    return f"| | |\n|---|---|\n{body}"


def card_block(names: List[str], notes: Dict[str, str], depth: int) -> str:
    """The marked block for a page documenting `names`."""
    parts = []
    for name in names:
        heading = f"**`{name}`**\n\n" if len(names) > 1 else ""
        parts.append(heading + metric_card(name, notes, depth))
    return f"{START}\n" + "\n\n".join(parts) + f"\n{END}"


def metrics_on_page(text: str) -> List[str]:
    """Registry metrics named by `:::` blocks, in page order, without repeats."""
    found: List[str] = []
    for target in AUTODOC.findall(text):
        name = target.rsplit(".", 1)[-1]
        if name in METRIC_INFO and name not in found:
            found.append(name)
    return found


def apply_card(text: str, block: str) -> str:
    """Replace the marked block, or insert it after the page's first heading."""
    if START in text and END in text:
        head, _, rest = text.partition(START)
        _, _, tail = rest.partition(END)
        return head + block + tail
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("# "):
            return "\n".join(lines[: i + 1] + ["", block, ""] + lines[i + 1:])
    return block + "\n\n" + text


def render_new_page(spec: NewPage) -> str:
    """A complete page, with an empty card slot that `apply_card` then fills."""
    out = [f"# {spec.title}", "", spec.intro, ""]
    out += ["\n".join(f"::: {c}" for c in spec.classes), ""]
    if spec.example:
        out += ["## Example", "", f'```python\n--8<-- "{spec.example}"\n```', ""]
    elif spec.usage:
        out += ["## Usage", "", "Needs a downloaded model, so it is shown rather than run.",
                "", f"```python\n{spec.usage}\n```", ""]
    return "\n".join(out)


def desired(path: Path, current: Optional[str], notes: Dict[str, str]) -> Optional[str]:
    """What `path` should contain, or None if it documents no registry metric."""
    text = current
    if text is None:
        spec = next(s for s in NEW_PAGES if API / s.path == path)
        text = render_new_page(spec)
    names = metrics_on_page(text)
    if not names:
        return None
    depth = len(path.relative_to(DOCS).parts) - 1
    return apply_card(text, card_block(names, notes, depth))


OVERVIEW = API / "overview.md"
FAMILY_MODULE = {
    "embedding": "embeddings_based",
    "probability": "probability_based",
    "generated_text": "generated_text_based",
    "prompt": "prompts_based",
}


def overview_table(family: str, pages: Dict[str, str]) -> str:
    """The metric table for one family: import line, fidelity, and page link."""
    module = FAMILY_MODULE[family]
    rows = ["| Class | Import | Fidelity | Page |", "|---|---|---|---|"]
    for name in sorted(n for n, i in METRIC_INFO.items() if i.family == family):
        page = pages.get(name)
        link = f"[{page.rsplit('/', 1)[-1][:-3]}]({page})" if page else ""
        rows.append(f"| `{name}` | `from bias_scope.{module} import {name}` | "
                    f"{METRIC_INFO[name].fidelity} | {link} |")
    return "\n".join(rows)


def replace_region(text: str, name: str, body: str) -> str:
    """Rewrite the `<!-- NAME:start -->` ... `<!-- NAME:end -->` region, if present."""
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    if start not in text or end not in text:
        return text
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    return f"{head}{start}\n{body}\n{end}{tail}"


def apply_overview(text: str, pages: Dict[str, str]) -> str:
    """Rewrite each `overview:FAMILY` region of the overview page."""
    for family in FAMILY_MODULE:
        text = replace_region(text, f"overview:{family}", overview_table(family, pages))
    return text


MULTILINGUAL = DOCS / "framework" / "multilingual.md"
HOME = DOCS / "index.md"

#: Home-page rows: family -> (label, needs, anchor, examples). The counts are not
#: written here; they come from the registry.
FAMILY_ROWS = {
    "embedding": ("Embedding-based", "embeddings", "embedding-based",
                  "WEAT, SEAT, CEAT, SentenceBiasScore"),
    "probability": ("Probability-based", "masked-token logits", "probability-based",
                    "CrowS-Pairs, CAT, iCAT, AUL, AULA, LPBS, CBS, DisCo"),
    "generated_text": ("Generated text", "completions", "generated-text-based",
                       "RegardScore, HONEST, MarkedPersons, EMT, and more"),
    "prompt": ("Prompt-based", "chat or completions", "prompt-based",
               "BBQ, StereoSet, WinoBias, DecodingTrust, TrustLLM, and more"),
}


def home_table() -> str:
    """The category table for the home page, with counts from the registry."""
    rows = ["| Family | Metrics | Needs | Examples |", "|---|---:|---|---|"]
    for family, (label, needs, anchor, examples) in FAMILY_ROWS.items():
        count = sum(1 for info in METRIC_INFO.values() if info.family == family)
        rows.append(f"| [{label}](api/overview.md#{anchor}) | {count} | {needs} | {examples} |")
    return "\n".join(rows)


def extra_pages() -> List[Tuple[Path, str]]:
    """Pages outside docs/api that carry a generated region, with it rewritten."""
    from bias_scope.multilingual import readiness

    rewrites = {MULTILINGUAL: ("readiness", readiness()), HOME: ("families", home_table())}
    out = []
    for path, (name, body) in rewrites.items():
        if path.is_file():
            out.append((path, replace_region(path.read_text(encoding="utf-8"), name, body)))
    return out


def plan(notes: Dict[str, str]) -> Dict[Path, str]:
    """Every file to write, as path -> new content. Only files that change."""
    current = {p: p.read_text(encoding="utf-8") for p in sorted(API.rglob("*.md"))}
    targets: Dict[Path, Optional[str]] = dict(current)
    for spec in NEW_PAGES:
        targets.setdefault(API / spec.path, None)

    wanted: Dict[Path, str] = {}
    for path, text in targets.items():
        new = desired(path, text, notes)
        if new is not None:
            wanted[path] = new
        elif text is not None:
            wanted[path] = text

    pages = {name: path.relative_to(API).as_posix()
             for path, text in wanted.items() if path != OVERVIEW
             for name in metrics_on_page(text)}
    if OVERVIEW in wanted:
        wanted[OVERVIEW] = apply_overview(wanted[OVERVIEW], pages)

    changes = {p: t for p, t in wanted.items() if t != current.get(p)}
    for path, text in extra_pages():
        if text != path.read_text(encoding="utf-8"):
            changes[path] = text
    return changes


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if a file would change")
    args = parser.parse_args(argv)

    changes = plan(note_links())
    for path, content in changes.items():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if args.check:
            print(f"stale: {rel}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
            print(f"wrote {rel}")
    if not changes:
        print("docs/api is up to date")
    return 1 if (args.check and changes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
