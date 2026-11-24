"""Closed-loop remediation: verified patches and permanent regression tests."""

from cubic.remediate.exporter import export_regression_test, render_regression_test, regression_file_name
from cubic.remediate.models import (
    ConfirmedFinding,
    FixVerdict,
    FixVerification,
    FunctionalCheck,
    PatchCandidate,
    SourceRef,
)
from cubic.remediate.patcher import (
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
from cubic.remediate.verify_fix import averify_fix, patched_module, verify_fix

__all__ = [
    "ConfirmedFinding",
    "FixVerdict",
    "FixVerification",
    "FunctionalCheck",
    "LLMPatcher",
    "PatchCandidate",
    "PatchError",
    "PatchModel",
    "PatchPrompt",
    "Patcher",
    "RulePatcher",
    "SourceContext",
    "SourceRef",
    "apply_unified_diff",
    "averify_fix",
    "build_patch_prompt",
    "export_regression_test",
    "load_context",
    "patched_module",
    "propose_patches",
    "render_regression_test",
    "regression_file_name",
    "verify_fix",
]
