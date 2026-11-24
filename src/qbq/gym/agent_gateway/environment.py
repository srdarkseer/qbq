"""Agent-gateway Bug Gym environment, driven entirely by its compiled graph.

Like the SaaS gym, it has no hand-written graph and no probes. The only app-specific
data is persona seeds and ground truth.
"""

from __future__ import annotations

from functools import cache

from qbq.compiler import compile_app
from qbq.graph import ActorPersona, StateGraph
from qbq.gym.agent_gateway import app as testbed
from qbq.gym.base import AppSource, Environment, GroundTruthBug


def _owner(pid: str, name: str, token: str, tenant: str, suffix: str, tool: str) -> ActorPersona:
    return ActorPersona(
        id=pid, display_name=name, role="USER", tenant_id=f"u_{suffix}", auth_token=token,
        owned_resources={
            "tenants": frozenset({tenant}), "agents": frozenset({f"agent_{suffix}"}),
            "conversations": frozenset({f"conv_{suffix}"}), "actions": frozenset({f"act_{suffix}"}),
            "tools": frozenset({tool}),
        },
    )


PERSONAS: tuple[ActorPersona, ...] = (
    ActorPersona(id="anonymous", display_name="Unauthenticated", role="ANONYMOUS"),
    _owner("alpha_dev", "Alpha tenant developer", "tok_alpha_dev", "t_alpha", "alpha", "tool_alpha_crm"),
    _owner("beta_dev", "Beta tenant developer", "tok_beta_dev", "t_beta", "beta", "tool_beta_billing"),
    ActorPersona(id="platform_admin", display_name="Platform operator", role="ADMIN",
                 tenant_id="u_platform", auth_token="tok_platform"),
)

BUGS: tuple[GroundTruthBug, ...] = (
    GroundTruthBug("cross_tenant_conversation_read", "inv.tenant.read_conversation",
                   "Agent conversation history readable across tenants", "CWE-639"),
    GroundTruthBug("cross_tenant_tool_invocation", "inv.tool_authz.create_tool_call",
                   "Agent can invoke another tenant's private tool", "CWE-863"),
    GroundTruthBug("unbounded_max_tokens", "inv.ai_bounds.create_completion.max_tokens",
                   "Regression: per-agent max_tokens cap dropped", "CWE-770"),
    GroundTruthBug("approval_skip", "inv.lifecycle.execute_action",
                   "Tool action executes without human approval", "CWE-841"),
)


@cache
def compiled_graph() -> StateGraph:
    return compile_app(lambda: testbed.create_app(), personas=PERSONAS).graph


def environment() -> Environment:
    return Environment(
        name="agent-gateway-compiled",
        graph=compiled_graph(),
        bugs=BUGS,
        build_app=lambda active: testbed.create_app(active),
        snapshot=testbed.snapshot,
        app_source=AppSource(module="qbq.gym.agent_gateway.app"),
    )
