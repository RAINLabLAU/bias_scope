"""The documentation must agree with the code.

The Read the Docs site is built from `mkdocs.yml`. Between v0.1 and v0.2 it
drifted in ways nothing caught: `docs/api/generated_text/score_parity.md`
documented a class renamed in 0.2.0 (a hard build error), 39 pages existed but
were unreachable from the nav, and the fidelity index counted a different
library from the one that shipped.

These tests read the docs as text and compare them with `METRIC_INFO`, the
static registry, so they run on a minimal install (no torch, no network). The
last test runs the real `mkdocs build --strict` and is `slow`.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest

from bias_scope._metric_info import METRIC_INFO

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SRC = ROOT / "src"

#: The source distribution ships src/, tests/ and examples/ but not the docs or
#: the scripts that generate them, so these tests only make sense in a checkout.
pytestmark = pytest.mark.skipif(
    not (DOCS / "index.md").is_file(), reason="docs/ is not part of this distribution",
)

AUTODOC = re.compile(r"^:::\s+([\w.]+)\s*$", re.MULTILINE)


def _pages() -> List[Path]:
    return sorted(DOCS.rglob("*.md"))


def _rel(path: Path) -> str:
    return path.relative_to(DOCS).as_posix()


def _flatten_nav(node) -> List[str]:
    """Every .md path named anywhere in the mkdocs nav."""
    if isinstance(node, str):
        return [node] if node.endswith(".md") else []
    if isinstance(node, dict):
        return [p for value in node.values() for p in _flatten_nav(value)]
    if isinstance(node, list):
        return [p for item in node for p in _flatten_nav(item)]
    return []


def _nav_pages() -> Set[str]:
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    return set(_flatten_nav(config["nav"]))


def _defined_names(tree: ast.AST) -> Dict[str, ast.AST]:
    """Top-level names a module defines or re-exports."""
    names: Dict[str, ast.AST] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names[target.id] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names[(alias.asname or alias.name).split(".")[0]] = node
    return names


def _module_file(parts: List[str]) -> Tuple[Optional[Path], List[str]]:
    """Longest prefix of `parts` that is a module under src/, and the rest."""
    for cut in range(len(parts), 0, -1):
        base = SRC.joinpath(*parts[:cut])
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
            if candidate.is_file():
                return candidate, parts[cut:]
    return None, parts


def resolves(dotted: str) -> bool:
    """Would mkdocstrings find `dotted`? Checked statically, without importing."""
    path, rest = _module_file(dotted.split("."))
    if path is None:
        return False
    node: ast.AST = ast.parse(path.read_text(encoding="utf-8"))
    for name in rest:
        node = _defined_names(node).get(name)
        if node is None:
            return False
    return True


def _autodoc_targets() -> List[Tuple[str, str]]:
    found = []
    for page in _pages():
        for target in AUTODOC.findall(page.read_text(encoding="utf-8")):
            found.append((_rel(page), target))
    return found


def test_every_autodoc_target_resolves():
    """A `:::` line naming a renamed or deleted class breaks the whole build."""
    broken = [f"{page}: {target}" for page, target in _autodoc_targets()
              if not resolves(target)]
    assert not broken, "docs name code that does not exist:\n  " + "\n  ".join(broken)


def test_every_page_is_reachable_from_the_nav():
    """A page missing from the nav is invisible on Read the Docs."""
    orphans = sorted({_rel(p) for p in _pages()} - _nav_pages())
    assert not orphans, "pages not in mkdocs.yml nav:\n  " + "\n  ".join(orphans)


def test_the_nav_names_only_pages_that_exist():
    missing = sorted(p for p in _nav_pages() if not (DOCS / p).is_file())
    assert not missing, "mkdocs.yml nav points at missing files:\n  " + "\n  ".join(missing)


def _api_pages() -> Dict[str, str]:
    """metric class name -> the page whose `:::` block documents it."""
    documented: Dict[str, str] = {}
    for page, target in _autodoc_targets():
        if page.startswith("api/"):
            documented.setdefault(target.rsplit(".", 1)[-1], page)
    return documented


def test_every_registered_metric_has_an_api_page():
    documented = _api_pages()
    missing = sorted(set(METRIC_INFO) - set(documented))
    assert not missing, (
        f"{len(missing)} metrics in the registry have no API page: {missing}"
    )


def test_the_api_overview_lists_every_registered_metric():
    overview = (DOCS / "api" / "overview.md").read_text(encoding="utf-8")
    missing = sorted(name for name in METRIC_INFO if f"`{name}`" not in overview)
    assert not missing, f"docs/api/overview.md omits: {missing}"


INDEX_ROW = re.compile(r"^\| `(\w+)` \| (\w+) \| \*\*(\w+)\*\* \| (.+) \|$", re.MULTILINE)
INDEX_COUNT = re.compile(r"^\| (faithful|adaptation|original|mismatch|unaudited) \| (\d+) \|$",
                         re.MULTILINE)


def _index_text() -> str:
    return (DOCS / "fidelity" / "INDEX.md").read_text(encoding="utf-8")


def test_the_fidelity_index_lists_the_registry_with_its_statuses():
    """INDEX.md is generated; a stale copy claims a fidelity the code does not."""
    listed = {name: status for name, _fam, status, _note in INDEX_ROW.findall(_index_text())}
    actual = {name: info.fidelity for name, info in METRIC_INFO.items()}
    assert listed == actual, (
        "docs/fidelity/INDEX.md is stale; run "
        "`python scripts/verification/render_fidelity_index.py`"
    )


def test_the_fidelity_index_counts_match_the_registry():
    text = _index_text()
    counts = {status: int(n) for status, n in INDEX_COUNT.findall(text)}
    expected = {}
    for info in METRIC_INFO.values():
        expected[info.fidelity] = expected.get(info.fidelity, 0) + 1
    assert {k: v for k, v in counts.items() if v} == expected
    assert f"Metrics in the library: **{len(METRIC_INFO)}**" in text


def test_every_fidelity_index_link_points_at_a_file():
    dead = []
    for name, _fam, _status, note in INDEX_ROW.findall(_index_text()):
        for target in re.findall(r"\]\(([^)#]+)", note):
            if not (DOCS / "fidelity" / target).resolve().is_file():
                dead.append(f"{name} -> {target}")
    assert not dead, "fidelity index links to missing notes:\n  " + "\n  ".join(dead)


def test_the_generated_docs_are_up_to_date():
    """Metric cards, overview tables and the readiness table come from the registry."""
    from scripts.docs.render_api_pages import note_links, plan

    stale = sorted(path.relative_to(ROOT).as_posix() for path in plan(note_links()))
    assert not stale, (
        "generated docs are stale; run `python scripts/docs/render_api_pages.py`:\n  "
        + "\n  ".join(stale)
    )


@pytest.mark.slow
def test_mkdocs_builds_without_warnings(tmp_path):
    """The check PLAN.md 11.1 requires: `mkdocs build --strict`."""
    pytest.importorskip("mkdocs")
    pytest.importorskip("mkdocstrings")
    result = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "-d", str(tmp_path / "site")],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]
