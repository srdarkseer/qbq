"""Closed-loop remediation: verified patches and permanent regression tests."""

from qbq.remediate.exporter import export_regression_test, render_regression_test, regression_file_name
from qbq.remediate.models import (
    ConfirmedFinding,
    FixVerdict,
    FixVerification,
    FunctionalCheck,
    PatchCandidate,
    SourceRef,
)
from qbq.remediate.patcher import (
    LLMPatcher,
    Patcher,
    PatchError,
    PatchModel,
    PatchPrompt,
    RulePatcher,
    SourceContext,
    apply_unified_diff,
    build_patch_prompt,
    load_context,
    propose_patches,
)
from qbq.remediate.pipeline import (
    FindingRemediation,
    RegressionCheck,
    RemediationReport,
    aremediate,
    remediate,
)
from qbq.remediate.verify_fix import averify_fix, patched_module, verify_fix

__all__ = [
    "ConfirmedFinding",
    "FindingRemediation",
    "FixVerdict",
    "FixVerification",
    "FunctionalCheck",
    "LLMPatcher",
    "PatchCandidate",
    "PatchError",
    "PatchModel",
    "PatchPrompt",
    "RegressionCheck",
    "RemediationReport",
    "Patcher",
    "RulePatcher",
    "SourceContext",
    "SourceRef",
    "apply_unified_diff",
    "aremediate",
    "averify_fix",
    "build_patch_prompt",
    "export_regression_test",
    "load_context",
    "patched_module",
    "propose_patches",
    "remediate",
    "render_regression_test",
    "regression_file_name",
    "verify_fix",
]
