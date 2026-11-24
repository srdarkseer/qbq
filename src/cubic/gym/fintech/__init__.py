"""Reference fintech testbed: wallets, transfers, and loans with planted defects."""

from cubic.gym.fintech.app import BUG_IDS, create_app
from cubic.gym.fintech.graph import build_graph
from cubic.gym.fintech.manifest import BUGS, baseline_candidates, environment

__all__ = ["BUGS", "BUG_IDS", "baseline_candidates", "build_graph", "create_app", "environment"]
