"""SaaS Bug Gym environment, driven entirely by its compiled graph.

Unlike the fintech testbed, this app has no hand-written State Graph and no
hand-written probes. The graph comes from ``cubic.compiler`` plus the persona
seeds below, and the probes come from ``cubic.search``. The only app-specific
data here is ground truth: which synthesized invariant each planted bug breaks.
The harness uses it for scoring and nothing else.
"""

from __future__ import annotations

from functools import cache

from cubic.compiler import compile_app
from cubic.graph import ActorPersona, StateGraph
from cubic.gym.base import AppSource, Environment, GroundTruthBug
from cubic.gym.saas import app as testbed

PERSONAS: tuple[ActorPersona, ...] = (
    ActorPersona(id="anonymous", display_name="Unauthenticated", role="ANONYMOUS"),
    ActorPersona(
        id="alpha_owner", display_name="Alpha Corp owner", role="USER", tenant_id="u_alpha",
        auth_token="tok_alpha",
        owned_resources={"workspaces": frozenset({"ws_alpha"}), "subscriptions": frozenset({"sub_alpha"}),
                         "api_keys": frozenset({"key_alpha"})},
    ),
    ActorPersona(
        id="beta_owner", display_name="Beta LLC owner", role="USER", tenant_id="u_beta",
        auth_token="tok_beta",
        owned_resources={"workspaces": frozenset({"ws_beta"}), "subscriptions": frozenset({"sub_beta"}),
                         "api_keys": frozenset({"key_beta"})},
    ),
    ActorPersona(id="support_admin", display_name="Support Admin", role="ADMIN", tenant_id="u_support",
                 auth_token="tok_support"),
)

BUGS: tuple[GroundTruthBug, ...] = (
    GroundTruthBug("cross_tenant_workspace_read", "inv.tenant.read_workspace",
                   "Workspace read lacks tenant check (cross-tenant IDOR)", "CWE-639"),
    GroundTruthBug("seat_limit_race", "inv.ledger.conservation",
                   "Member invite TOCTOU overruns the seat limit", "CWE-362"),
    GroundTruthBug("tier_state_skip", "inv.lifecycle.upgrade_subscription",
                   "Subscription upgrade skips the ACTIVE state", "CWE-841"),
    GroundTruthBug("negative_seat_quantity", "inv.input.buy_seats.quantity",
                   "Regression: negative seat purchases accepted", "CWE-20"),
)


@cache
def compiled_graph() -> StateGraph:
    return compile_app(lambda: testbed.create_app(), personas=PERSONAS).graph


def environment() -> Environment:
    return Environment(
        name="saas-compiled",
        graph=compiled_graph(),
        bugs=BUGS,
        build_app=lambda active: testbed.create_app(active),
        snapshot=testbed.snapshot,
        app_source=AppSource(module="cubic.gym.saas.app"),
    )
