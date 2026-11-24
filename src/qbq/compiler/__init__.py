"""Gray-box application compiler: source code + runtime -> Behavioral State Graph."""

from qbq.compiler.dynamic import DynamicObservation, DynamicRoute, aintrospect, introspect
from qbq.compiler.fidelity import FidelityReport, compare_graphs
from qbq.compiler.invariants import synthesize_invariants
from qbq.compiler.merge import CompileReport, CompileResult, acompile_app, build_graph, compile_app
from qbq.compiler.static import (
    LedgerSpec,
    RequestModel,
    StaticAnalysis,
    StaticRoute,
    analyze_files,
    analyze_sources,
    extract_candidates,
)
from qbq.compiler.target import TargetError, ensure_cwd_importable, load_target

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
