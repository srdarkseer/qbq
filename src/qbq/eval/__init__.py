"""Evaluation harness: Recall@1 and False Positive Rate against ground truth."""

from qbq.eval.autonomous import autonomous_environment, plan_for
from qbq.eval.harness import (
    CaseResult,
    EvalReport,
    Metrics,
    arun_eval,
    compute_metrics,
    format_report,
    run_eval,
)

__all__ = [
    "CaseResult",
    "EvalReport",
    "Metrics",
    "arun_eval",
    "autonomous_environment",
    "compute_metrics",
    "format_report",
    "plan_for",
    "run_eval",
]
