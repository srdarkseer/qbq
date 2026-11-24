"""Documentation and release hygiene: generated tables, links, CLI coverage, licensing."""

import importlib.util
import re
import tomllib
from pathlib import Path

from cubic.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "docs/ARCHITECTURE.md", ROOT / "benchmarks/README.md",
        ROOT / "CONTRIBUTING.md", ROOT / "SECURITY.md"]


def _summarize():
    spec = importlib.util.spec_from_file_location("summarize", ROOT / "benchmarks/summarize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_is_generated_from_current_snapshots():
    assert _summarize().main(["--check"]) == 0


def test_readme_documents_every_cli_command():
    readme = (ROOT / "README.md").read_text()
    commands = next(a for a in build_parser()._actions if a.dest == "command").choices
    for command in commands:
        assert f"cubic {command}" in readme, command


def test_relative_links_resolve():
    for doc in DOCS:
        for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", doc.read_text()):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            assert (doc.parent / target).exists(), f"{doc.name}: broken link {target}"


def test_dual_license_metadata():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert meta["license"] == "MIT OR Apache-2.0"
    for name in meta["license-files"]:
        assert (ROOT / name).is_file()
    assert "Apache License" in (ROOT / "LICENSE-APACHE").read_text()[:200]
    assert "MIT License" in (ROOT / "LICENSE-MIT").read_text()[:50]


def test_headline_numbers_match_snapshots():
    import json

    remed = json.loads((ROOT / "benchmarks/results/remediation_bench.json").read_text())
    readme = (ROOT / "README.md").read_text()
    for app in remed["apps"].values():
        det = app["detection"]["grounded"]
        assert det["recall_at_1"] == 1.0 and det["false_positives"] == 0
        assert f"0% (0/{det['negatives']})" in readme
        totals = app["remediation"]["totals"]
        assert f"{totals['verified_patches']}/{totals['findings']}" in readme
