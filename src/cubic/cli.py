"""Command-line entry point for Cubic."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from cubic import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cubic", description="QbQ (Cubic) QA & security engine")
    parser.add_argument("--version", action="version", version=f"cubic {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
