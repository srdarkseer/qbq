"""Gray-box application compiler: source code + runtime -> Behavioral State Graph."""

from cubic.compiler.dynamic import DynamicObservation, DynamicRoute, aintrospect, introspect
from cubic.compiler.fidelity import FidelityReport, compare_graphs
from cubic.compiler.invariants import synthesize_invariants
from cubic.compiler.merge import CompileReport, CompileResult, acompile_app, build_graph, compile_app
from cubic.compiler.static import (
    LedgerSpec,
    RequestModel,
    StaticAnalysis,
    StaticRoute,
    analyze_files,
    analyze_sources,
    extract_candidates,
)
from cubic.compiler.target import TargetError, ensure_cwd_importable, load_target

__all__ = [
    "CompileReport",
    "CompileResult",
    "DynamicObservation",
    "DynamicRoute",
    "FidelityReport",
    "LedgerSpec",
    "RequestModel",
    "StaticAnalysis",
    "StaticRoute",
    "TargetError",
    "acompile_app",
    "aintrospect",
    "analyze_files",
    "analyze_sources",
    "build_graph",
    "compare_graphs",
    "compile_app",
    "ensure_cwd_importable",
    "extract_candidates",
    "introspect",
    "load_target",
    "synthesize_invariants",
]
