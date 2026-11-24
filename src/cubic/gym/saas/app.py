"""Reference multi-tenant B2B SaaS testbed ("Bug Gym" environment #2).

A workspace & billing API: workspaces with a seat allocation, members, API keys,
and subscriptions with a ``TRIALING -> ACTIVE`` lifecycle. Each planted defect
can be toggled on its own, so one codebase yields a fully patched build and the
vulnerable builds. It runs only in-process, for local and authorized CI/CD testing.

Seats are accounted like money: ``workspaces[*].seats_available`` must always
equal the signed sum of the append-only ``seat_ledger`` (grants are positive,
allocations negative) and must never go negative.

Planted defects (ids are the toggles):

* ``cross_tenant_workspace_read`` - any authenticated user can read any workspace,
                                    including its billing profile (CWE-639).
* ``seat_limit_race``             - adding a member checks free seats, awaits I/O,
                                    then writes a stale count: concurrent invites
                                    overrun the seat limit (CWE-362).
* ``tier_state_skip``             - upgrading a subscription does not require it to
                                    be ACTIVE, so a trial jumps straight to a paid
                                    tier without activation (CWE-841).
* ``negative_seat_quantity``      - regression: the positivity check on seat
                                    purchases was dropped, so a negative quantity
                                    removes seats (CWE-20).
"""

# NOTE: no `from __future__ import annotations`: FastAPI resolves the local `User` alias at runtime.

import asyncio
import copy
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from cubic.verifier.snapshot import DbSnapshot

BUG_IDS = frozenset({
    "cross_tenant_workspace_read", "seat_limit_race", "tier_state_skip", "negative_seat_quantity",
})

DEFAULT_IO_DELAY_S = 0.005

SEED_TOKENS = {
    "tok_alpha": "u_alpha",
    "tok_beta": "u_beta",
    "tok_support": "u_support",
}


def _seed_state() -> dict[str, Any]:
    users = {
        "u_alpha": {"id": "u_alpha", "name": "Alpha Owner", "role": "USER"},
        "u_beta": {"id": "u_beta", "name": "Beta Owner", "role": "USER"},
        "u_support": {"id": "u_support", "name": "Support Agent", "role": "ADMIN"},
    }
    workspaces = {
        # Alpha is on an active paid plan with every seat in use and billing still unverified.
        "ws_alpha": {"id": "ws_alpha", "name": "Alpha Corp", "owner_id": "u_alpha",
                     "seats_available": 0, "billing_status": "PENDING"},
        # Beta is trialing with free seats and verified billing.
        "ws_beta": {"id": "ws_beta", "name": "Beta LLC", "owner_id": "u_beta",
                    "seats_available": 5, "billing_status": "VERIFIED"},
    }
    members = {
        f"m_alpha_{i}": {"id": f"m_alpha_{i}", "workspace_id": "ws_alpha", "email": f"user{i}@alpha.test"}
        for i in range(1, 4)
    }
    api_keys = {
        "key_alpha": {"id": "key_alpha", "workspace_id": "ws_alpha", "name": "ci", "revoked": False},
        "key_beta": {"id": "key_beta", "workspace_id": "ws_beta", "name": "ci", "revoked": False},
    }
    subscriptions = {
        "sub_alpha": {"id": "sub_alpha", "workspace_id": "ws_alpha", "status": "ACTIVE", "tier": "PRO"},
        "sub_beta": {"id": "sub_beta", "workspace_id": "ws_beta", "status": "TRIALING", "tier": "STARTER"},
    }
    seat_ledger = [
        {"workspace_id": "ws_alpha", "amount": 3, "kind": "grant", "ref": "seed"},
        *({"workspace_id": "ws_alpha", "amount": -1, "kind": "allocate", "ref": m} for m in members),
        {"workspace_id": "ws_beta", "amount": 5, "kind": "grant", "ref": "seed"},
    ]
    return {"users": users, "workspaces": workspaces, "members": members, "api_keys": api_keys,
            "subscriptions": subscriptions, "seat_ledger": seat_ledger}


@dataclass
class SaasStore:
    users: dict[str, dict[str, Any]]
    workspaces: dict[str, dict[str, Any]]
    members: dict[str, dict[str, Any]]
    api_keys: dict[str, dict[str, Any]]
    subscriptions: dict[str, dict[str, Any]]
    seat_ledger: list[dict[str, Any]]
    tokens: dict[str, str] = field(default_factory=lambda: dict(SEED_TOKENS))
    locks: defaultdict[str, asyncio.Lock] = field(default_factory=lambda: defaultdict(asyncio.Lock))
    _seq: int = 0

    @classmethod
    def seeded(cls) -> "SaasStore":
        return cls(**_seed_state())

    def next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}_{self._seq:04d}"

    def post(self, workspace_id: str, amount: int, kind: str, ref: str) -> None:
        self.seat_ledger.append({"workspace_id": workspace_id, "amount": amount, "kind": kind, "ref": ref})

    def snapshot(self) -> DbSnapshot:
        return DbSnapshot(
            entities=copy.deepcopy({
                "users": self.users, "workspaces": self.workspaces, "members": self.members,
                "api_keys": self.api_keys, "subscriptions": self.subscriptions,
            }),
            ledger=tuple(copy.deepcopy(self.seat_ledger)),
        )


class MemberIn(BaseModel):
    email: str


class SeatsIn(BaseModel):
    quantity: int  # positivity is enforced in the handler so it can be toggled as a regression


class ApiKeyIn(BaseModel):
    name: str


class SubscriptionIn(BaseModel):
    workspace_id: str
    tier: Literal["STARTER", "PRO", "ENTERPRISE"]


class UpgradeIn(BaseModel):
    tier: Literal["PRO", "ENTERPRISE"]


def create_app(
    active_bugs: Iterable[str] = (), *, io_delay_s: float = DEFAULT_IO_DELAY_S
) -> FastAPI:
    bugs = frozenset(active_bugs)
    unknown = bugs - BUG_IDS
    if unknown:
        raise ValueError(f"unknown bug toggles: {sorted(unknown)}")

    store = SaasStore.seeded()
    app = FastAPI(title="Cubic SaaS Testbed", version="1.0")
    app.state.store = store
    app.state.active_bugs = bugs

    async def io() -> None:
        """Simulated database round-trip; the await is where interleaving happens."""
        await asyncio.sleep(io_delay_s)

    def current_user(authorization: Annotated[str | None, Header()] = None) -> dict[str, Any]:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing bearer token")
        user_id = store.tokens.get(authorization.removeprefix("Bearer "))
        if user_id is None:
            raise HTTPException(401, "invalid token")
        return store.users[user_id]

    User = Annotated[dict[str, Any], Depends(current_user)]

    def owned_workspace(workspace_id: str, user: dict[str, Any]) -> dict[str, Any]:
        workspace = store.workspaces.get(workspace_id)
        if workspace is None:
            raise HTTPException(404, "workspace not found")
        if workspace["owner_id"] != user["id"]:
            raise HTTPException(403, "not your workspace")
        return workspace

    def owned_subscription(subscription_id: str, user: dict[str, Any]) -> dict[str, Any]:
        subscription = store.subscriptions.get(subscription_id)
        if subscription is None:
            raise HTTPException(404, "subscription not found")
        owned_workspace(subscription["workspace_id"], user)
        return subscription

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/workspaces/{workspace_id}")
    async def read_workspace(workspace_id: str, user: User) -> dict[str, Any]:
        workspace = store.workspaces.get(workspace_id)
        if workspace is None:
            raise HTTPException(404, "workspace not found")
        if "cross_tenant_workspace_read" not in bugs:
            if workspace["owner_id"] != user["id"] and user["role"] != "ADMIN":
                raise HTTPException(404, "workspace not found")  # do not leak existence
        return workspace

    @app.post("/workspaces/{workspace_id}/members", status_code=201)
    async def add_member(workspace_id: str, body: MemberIn, user: User) -> dict[str, Any]:
        owned_workspace(workspace_id, user)
        workspace = store.workspaces[workspace_id]
        member_id = store.next_id("m")
        if "seat_limit_race" in bugs:
            available = workspace["seats_available"]  # time-of-check
            if available < 1:
                raise HTTPException(409, "no free seats")
            await io()
            workspace["seats_available"] = available - 1  # time-of-use: stale write
            store.post(workspace_id, -1, "allocate", member_id)
        else:
            async with store.locks[workspace_id]:
                await io()
                if workspace["seats_available"] < 1:
                    raise HTTPException(409, "no free seats")
                workspace["seats_available"] -= 1
                store.post(workspace_id, -1, "allocate", member_id)
        store.members[member_id] = {"id": member_id, "workspace_id": workspace_id, "email": body.email}
        return store.members[member_id]

    @app.post("/workspaces/{workspace_id}/seats")
    async def buy_seats(workspace_id: str, body: SeatsIn, user: User) -> dict[str, Any]:
        if "negative_seat_quantity" not in bugs and body.quantity <= 0:
            raise HTTPException(422, "quantity must be positive")
        owned_workspace(workspace_id, user)
        async with store.locks[workspace_id]:
            workspace = store.workspaces[workspace_id]
            await io()
            workspace["seats_available"] += body.quantity
            store.post(workspace_id, body.quantity, "grant", store.next_id("purchase"))
        return {"workspace_id": workspace_id, "seats_available": store.workspaces[workspace_id]["seats_available"]}

    @app.post("/workspaces/{workspace_id}/api_keys", status_code=201)
    async def create_api_key(workspace_id: str, body: ApiKeyIn, user: User) -> dict[str, Any]:
        owned_workspace(workspace_id, user)
        key_id = store.next_id("key")
        store.api_keys[key_id] = {"id": key_id, "workspace_id": workspace_id, "name": body.name, "revoked": False}
        return store.api_keys[key_id]

    @app.delete("/api_keys/{api_key_id}")
    async def revoke_api_key(api_key_id: str, user: User) -> dict[str, Any]:
        key = store.api_keys.get(api_key_id)
        if key is None:
            raise HTTPException(404, "api key not found")
        owned_workspace(key["workspace_id"], user)
        key["revoked"] = True
        return key

    @app.post("/subscriptions", status_code=201)
    async def create_subscription(body: SubscriptionIn, user: User) -> dict[str, Any]:
        owned_workspace(body.workspace_id, user)
        subscription_id = store.next_id("sub")
        store.subscriptions[subscription_id] = {
            "id": subscription_id, "workspace_id": body.workspace_id, "status": "TRIALING", "tier": body.tier,
        }
        return store.subscriptions[subscription_id]

    @app.get("/subscriptions/{subscription_id}")
    async def read_subscription(subscription_id: str, user: User) -> dict[str, Any]:
        return owned_subscription(subscription_id, user)

    @app.post("/subscriptions/{subscription_id}/activate")
    async def activate_subscription(subscription_id: str, user: User) -> dict[str, Any]:
        subscription = owned_subscription(subscription_id, user)
        if subscription["status"] != "TRIALING":
            raise HTTPException(409, f"cannot activate a {subscription['status']} subscription")
        workspace = store.workspaces[subscription["workspace_id"]]
        if workspace["billing_status"] != "VERIFIED":
            raise HTTPException(409, "billing profile not verified")
        subscription["status"] = "ACTIVE"
        return subscription

    @app.post("/subscriptions/{subscription_id}/upgrade")
    async def upgrade_subscription(subscription_id: str, body: UpgradeIn, user: User) -> dict[str, Any]:
        subscription = owned_subscription(subscription_id, user)
        if "tier_state_skip" not in bugs and subscription["status"] != "ACTIVE":
            raise HTTPException(409, f"cannot upgrade a {subscription['status']} subscription")
        subscription["tier"] = body.tier
        return subscription

    return app


def snapshot(app: FastAPI) -> DbSnapshot:
    return app.state.store.snapshot()
