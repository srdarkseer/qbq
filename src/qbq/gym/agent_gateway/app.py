"""Reference multi-tenant AI agent / MCP tool gateway ("Bug Gym" environment #3).

An AI-native backend: tenants buy token credits; each tenant runs agents that call
an LLM (simulated, deterministic) and invoke tools in the MCP style. Tools are
private to a tenant or public. Side-effecting tool actions need human approval
before they execute (``PENDING_APPROVAL -> APPROVED -> EXECUTED``). Token usage
is metered like money: ``tenants[*].token_balance`` must equal the signed sum of
``usage_ledger`` and must never go negative. Each planted defect can be toggled
on its own. The app runs only in-process, for local and authorized CI/CD testing.

Planted defects (ids are the toggles):

* ``cross_tenant_conversation_read`` - any user can read any tenant's agent
                                       conversation history (OWASP LLM02 / API1, CWE-639).
* ``cross_tenant_tool_invocation``   - the gateway checks the caller owns the agent
                                       but not that the requested tool is granted to the
                                       agent's tenant, so one tenant's agent can drive
                                       another tenant's private tool (OWASP LLM06 / API5,
                                       CWE-863).
* ``unbounded_max_tokens``           - regression: the per-agent ``max_tokens`` cap was
                                       dropped (OWASP LLM10 / API4, CWE-770).
* ``approval_skip``                  - executing a tool action does not require human
                                       approval first (OWASP LLM06 excessive agency, CWE-841).
"""

# NOTE: no `from __future__ import annotations`: FastAPI resolves the local `User` alias at runtime.

import asyncio
import copy
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from qbq.verifier.snapshot import DbSnapshot

BUG_IDS = frozenset({
    "cross_tenant_conversation_read", "cross_tenant_tool_invocation", "unbounded_max_tokens", "approval_skip",
})

DEFAULT_IO_DELAY_S = 0.005

SEED_TOKENS = {"tok_alpha_dev": "u_alpha", "tok_beta_dev": "u_beta", "tok_platform": "u_platform"}


def _seed_state() -> dict[str, Any]:
    users = {
        "u_alpha": {"id": "u_alpha", "name": "Alpha developer", "role": "USER"},
        "u_beta": {"id": "u_beta", "name": "Beta developer", "role": "USER"},
        "u_platform": {"id": "u_platform", "name": "Platform operator", "role": "ADMIN"},
    }
    tenants = {
        "t_alpha": {"id": "t_alpha", "owner_id": "u_alpha", "plan": "team", "token_balance": 50_000},
        "t_beta": {"id": "t_beta", "owner_id": "u_beta", "plan": "starter", "token_balance": 20_000},
    }
    agents = {
        "agent_alpha": {"id": "agent_alpha", "tenant_id": "t_alpha", "model": "local-small", "max_tokens_limit": 4096},
        "agent_beta": {"id": "agent_beta", "tenant_id": "t_beta", "model": "local-small", "max_tokens_limit": 2048},
    }
    tools = {
        "tool_alpha_crm": {"id": "tool_alpha_crm", "tenant_id": "t_alpha", "name": "crm.lookup"},
        "tool_beta_billing": {"id": "tool_beta_billing", "tenant_id": "t_beta", "name": "billing.refund"},
        "tool_web_search": {"id": "tool_web_search", "tenant_id": None, "name": "web.search"},
    }
    conversations = {
        "conv_alpha": {"id": "conv_alpha", "agent_id": "agent_alpha", "title": "Q3 pipeline review",
                       "transcript": "user: summarise Q3 deals; assistant: ..."},
        "conv_beta": {"id": "conv_beta", "agent_id": "agent_beta", "title": "Refund triage",
                      "transcript": "user: list pending refunds; assistant: ..."},
    }
    actions = {
        "act_alpha": {"id": "act_alpha", "agent_id": "agent_alpha", "tool_id": "tool_alpha_crm",
                      "input": "export all contacts", "status": "PENDING_APPROVAL"},
        "act_beta": {"id": "act_beta", "agent_id": "agent_beta", "tool_id": "tool_beta_billing",
                     "input": "refund order 42", "status": "APPROVED"},
    }
    usage_ledger = [
        {"tenant_id": tid, "amount": t["token_balance"], "kind": "credit_purchase", "ref": "seed"}
        for tid, t in tenants.items()
    ]
    return {"users": users, "tenants": tenants, "agents": agents, "tools": tools,
            "conversations": conversations, "actions": actions, "usage_ledger": usage_ledger}


@dataclass
class GatewayStore:
    users: dict[str, dict[str, Any]]
    tenants: dict[str, dict[str, Any]]
    agents: dict[str, dict[str, Any]]
    tools: dict[str, dict[str, Any]]
    conversations: dict[str, dict[str, Any]]
    actions: dict[str, dict[str, Any]]
    usage_ledger: list[dict[str, Any]]
    tokens: dict[str, str] = field(default_factory=lambda: dict(SEED_TOKENS))
    locks: defaultdict[str, asyncio.Lock] = field(default_factory=lambda: defaultdict(asyncio.Lock))
    _seq: int = 0

    @classmethod
    def seeded(cls) -> "GatewayStore":
        return cls(**_seed_state())

    def next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}_{self._seq:04d}"

    def post(self, tenant_id: str, amount: int, kind: str, ref: str) -> None:
        self.usage_ledger.append({"tenant_id": tenant_id, "amount": amount, "kind": kind, "ref": ref})

    def snapshot(self) -> DbSnapshot:
        return DbSnapshot(
            entities=copy.deepcopy({
                "users": self.users, "tenants": self.tenants, "agents": self.agents, "tools": self.tools,
                "conversations": self.conversations, "actions": self.actions,
            }),
            ledger=tuple(copy.deepcopy(self.usage_ledger)),
        )


class CompletionIn(BaseModel):
    prompt: str
    max_tokens: int = Field(gt=0)  # the per-agent cap is enforced in the handler (togglable regression)


class ToolCallIn(BaseModel):
    tool_id: str
    input: str


class ActionIn(BaseModel):
    agent_id: str
    tool_id: str
    input: str


def create_app(
    active_bugs: Iterable[str] = (), *, io_delay_s: float = DEFAULT_IO_DELAY_S
) -> FastAPI:
    bugs = frozenset(active_bugs)
    unknown = bugs - BUG_IDS
    if unknown:
        raise ValueError(f"unknown bug toggles: {sorted(unknown)}")

    store = GatewayStore.seeded()
    app = FastAPI(title="QbQ Agent Gateway Testbed", version="1.0")
    app.state.store = store
    app.state.active_bugs = bugs

    async def io() -> None:
        """Simulated model / database round-trip; the await is where interleaving happens."""
        await asyncio.sleep(io_delay_s)

    def current_user(authorization: Annotated[str | None, Header()] = None) -> dict[str, Any]:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing bearer token")
        user_id = store.tokens.get(authorization.removeprefix("Bearer "))
        if user_id is None:
            raise HTTPException(401, "invalid token")
        return store.users[user_id]

    User = Annotated[dict[str, Any], Depends(current_user)]

    def owned_tenant(tenant_id: str, user: dict[str, Any]) -> dict[str, Any]:
        tenant = store.tenants.get(tenant_id)
        if tenant is None:
            raise HTTPException(404, "tenant not found")
        if tenant["owner_id"] != user["id"]:
            raise HTTPException(403, "not your tenant")
        return tenant

    def owned_agent(agent_id: str, user: dict[str, Any]) -> dict[str, Any]:
        agent = store.agents.get(agent_id)
        if agent is None:
            raise HTTPException(404, "agent not found")
        owned_tenant(agent["tenant_id"], user)
        return agent

    def owned_action(action_id: str, user: dict[str, Any]) -> dict[str, Any]:
        action = store.actions.get(action_id)
        if action is None:
            raise HTTPException(404, "action not found")
        owned_agent(action["agent_id"], user)
        return action

    def lookup_tool(tool_id: str) -> dict[str, Any]:
        tool = store.tools.get(tool_id)
        if tool is None:
            raise HTTPException(404, "tool not found")
        return tool

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/tenants/{tenant_id}")
    async def read_tenant(tenant_id: str, user: User) -> dict[str, Any]:
        return owned_tenant(tenant_id, user)

    @app.get("/agents/{agent_id}")
    async def read_agent(agent_id: str, user: User) -> dict[str, Any]:
        return owned_agent(agent_id, user)

    @app.post("/agents/{agent_id}/completions")
    async def create_completion(agent_id: str, body: CompletionIn, user: User) -> dict[str, Any]:
        agent = owned_agent(agent_id, user)
        if "unbounded_max_tokens" not in bugs and body.max_tokens > agent["max_tokens_limit"]:
            raise HTTPException(422, "max_tokens exceeds the agent's limit")
        usage = min(body.max_tokens, 16 + len(body.prompt.split()))  # deterministic stand-in for a model
        tenant_id = agent["tenant_id"]
        async with store.locks[tenant_id]:
            tenant = store.tenants[tenant_id]
            await io()
            if tenant["token_balance"] < usage:
                raise HTTPException(402, "insufficient token credits")
            tenant["token_balance"] -= usage
            store.post(tenant_id, -usage, "completion", store.next_id("cmpl"))
        return {"agent_id": agent_id, "model": agent["model"], "text": "ok", "usage_tokens": usage}

    @app.post("/agents/{agent_id}/tool_calls")
    async def create_tool_call(agent_id: str, body: ToolCallIn, user: User) -> dict[str, Any]:
        agent = owned_agent(agent_id, user)
        tool = lookup_tool(body.tool_id)
        if "cross_tenant_tool_invocation" not in bugs and tool["tenant_id"] not in (None, agent["tenant_id"]):
            raise HTTPException(403, "tool not granted to this tenant")
        return {"agent_id": agent_id, "tool_id": tool["id"], "output": f"{tool['name']}: ok"}

    @app.get("/conversations/{conversation_id}")
    async def read_conversation(conversation_id: str, user: User) -> dict[str, Any]:
        conversation = store.conversations.get(conversation_id)
        if conversation is None:
            raise HTTPException(404, "conversation not found")
        if "cross_tenant_conversation_read" not in bugs:
            owner = store.tenants[store.agents[conversation["agent_id"]]["tenant_id"]]
            if owner["owner_id"] != user["id"] and user["role"] != "ADMIN":
                raise HTTPException(404, "conversation not found")  # do not leak existence
        return conversation

    @app.post("/actions", status_code=201)
    async def create_action(body: ActionIn, user: User) -> dict[str, Any]:
        agent = owned_agent(body.agent_id, user)
        tool = lookup_tool(body.tool_id)
        if tool["tenant_id"] not in (None, agent["tenant_id"]):
            raise HTTPException(403, "tool not granted to this tenant")
        action_id = store.next_id("act")
        store.actions[action_id] = {"id": action_id, "agent_id": agent["id"], "tool_id": tool["id"],
                                    "input": body.input, "status": "PENDING_APPROVAL"}
        return store.actions[action_id]

    @app.get("/actions/{action_id}")
    async def read_action(action_id: str, user: User) -> dict[str, Any]:
        return owned_action(action_id, user)

    @app.post("/actions/{action_id}/approve")
    async def approve_action(action_id: str, user: User) -> dict[str, Any]:
        action = owned_action(action_id, user)
        if action["status"] != "PENDING_APPROVAL":
            raise HTTPException(409, f"cannot approve a {action['status']} action")
        action["status"] = "APPROVED"
        return action

    @app.post("/actions/{action_id}/execute")
    async def execute_action(action_id: str, user: User) -> dict[str, Any]:
        action = owned_action(action_id, user)
        if "approval_skip" not in bugs and action["status"] != "APPROVED":
            raise HTTPException(409, f"cannot execute a {action['status']} action without approval")
        action["status"] = "EXECUTED"
        return action

    return app


def snapshot(app: FastAPI) -> DbSnapshot:
    return app.state.store.snapshot()
