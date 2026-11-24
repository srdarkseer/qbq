"""Reference fintech testbed ("Bug Gym" environment #1).

A small wallet + lending API with seeded roles. Each planted defect can be toggled
independently, so the same code yields a fully patched build (for false-positive
measurement) and vulnerable builds (for recall measurement). It runs only
in-process for local, authorized CI/CD testing.

Planted defects (ids are the toggles):

* ``idor_wallet_read``  - any authenticated user can read any wallet (CWE-639).
* ``race_withdraw``     - withdraw checks balance, awaits I/O, then writes a stale
                          value: a TOCTOU double-spend (CWE-362).
* ``loan_state_skip``   - disbursement does not verify the loan was approved, so a
                          borrower can skip KYC review (CWE-841).
* ``negative_transfer`` - regression: transfer amount positivity check was dropped,
                          letting a user pull funds from another wallet (CWE-20).
"""

# NOTE: no `from __future__ import annotations` here: FastAPI must resolve the
# locally defined `User` dependency alias at runtime.

import asyncio
import copy
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from cubic.verifier.snapshot import DbSnapshot

BUG_IDS = frozenset({"idor_wallet_read", "race_withdraw", "loan_state_skip", "negative_transfer"})

DEFAULT_IO_DELAY_S = 0.005

# token -> user id; these double as the actor tokens in the State Graph.
SEED_TOKENS = {
    "tok_alice": "u_alice",
    "tok_bob": "u_bob",
    "tok_carol": "u_carol",
    "tok_admin": "u_admin",
}


def _seed_state() -> dict[str, Any]:
    users = {
        "u_alice": {"id": "u_alice", "name": "Alice", "role": "USER", "kyc_status": "APPROVED"},
        "u_bob": {"id": "u_bob", "name": "Bob", "role": "USER", "kyc_status": "APPROVED"},
        "u_carol": {"id": "u_carol", "name": "Carol", "role": "USER", "kyc_status": "PENDING"},
        "u_admin": {"id": "u_admin", "name": "Merchant Admin", "role": "ADMIN", "kyc_status": "APPROVED"},
    }
    wallets = {
        "w_alice": {"id": "w_alice", "owner_id": "u_alice", "balance": 10_000, "currency": "NPR"},
        "w_bob": {"id": "w_bob", "owner_id": "u_bob", "balance": 5_000, "currency": "NPR"},
        "w_carol": {"id": "w_carol", "owner_id": "u_carol", "balance": 0, "currency": "NPR"},
    }
    loans = {
        "loan_carol": {
            "id": "loan_carol", "borrower_id": "u_carol", "wallet_id": "w_carol",
            "amount": 20_000, "status": "APPLIED", "approved_by": None,
        },
    }
    ledger = [
        {"wallet_id": wid, "amount": w["balance"], "kind": "seed", "ref": "seed"}
        for wid, w in wallets.items()
        if w["balance"]
    ]
    return {"users": users, "wallets": wallets, "loans": loans, "ledger": ledger}


@dataclass
class FintechStore:
    users: dict[str, dict[str, Any]]
    wallets: dict[str, dict[str, Any]]
    loans: dict[str, dict[str, Any]]
    ledger: list[dict[str, Any]]
    tokens: dict[str, str] = field(default_factory=lambda: dict(SEED_TOKENS))
    locks: defaultdict[str, asyncio.Lock] = field(default_factory=lambda: defaultdict(asyncio.Lock))
    _seq: int = 0

    @classmethod
    def seeded(cls) -> "FintechStore":
        return cls(**_seed_state())

    def next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}_{self._seq:04d}"

    def post(self, wallet_id: str, amount: int, kind: str, ref: str) -> None:
        self.ledger.append({"wallet_id": wallet_id, "amount": amount, "kind": kind, "ref": ref})

    def snapshot(self) -> DbSnapshot:
        return DbSnapshot(
            entities=copy.deepcopy({"users": self.users, "wallets": self.wallets, "loans": self.loans}),
            ledger=tuple(copy.deepcopy(self.ledger)),
        )


class AmountIn(BaseModel):
    amount: int = Field(gt=0)


class TransferIn(BaseModel):
    from_wallet_id: str
    to_wallet_id: str
    amount: int  # positivity is enforced in the handler so it can be toggled as a regression


class LoanIn(BaseModel):
    amount: int = Field(gt=0, le=1_000_000)


def create_app(
    active_bugs: Iterable[str] = (), *, io_delay_s: float = DEFAULT_IO_DELAY_S
) -> FastAPI:
    bugs = frozenset(active_bugs)
    unknown = bugs - BUG_IDS
    if unknown:
        raise ValueError(f"unknown bug toggles: {sorted(unknown)}")

    store = FintechStore.seeded()
    app = FastAPI(title="Cubic Fintech Testbed", version="1.0")
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

    def owned_wallet(wallet_id: str, user: dict[str, Any]) -> dict[str, Any]:
        wallet = store.wallets.get(wallet_id)
        if wallet is None:
            raise HTTPException(404, "wallet not found")
        if wallet["owner_id"] != user["id"]:
            raise HTTPException(403, "not your wallet")
        return wallet

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/wallets/{wallet_id}")
    async def read_wallet(wallet_id: str, user: User) -> dict[str, Any]:
        wallet = store.wallets.get(wallet_id)
        if wallet is None:
            raise HTTPException(404, "wallet not found")
        if "idor_wallet_read" not in bugs:
            if wallet["owner_id"] != user["id"] and user["role"] != "ADMIN":
                raise HTTPException(404, "wallet not found")  # do not leak existence
        return wallet

    @app.post("/wallets/{wallet_id}/withdraw")
    async def withdraw(wallet_id: str, body: AmountIn, user: User) -> dict[str, Any]:
        wallet = owned_wallet(wallet_id, user)
        ref = store.next_id("wd")
        if "race_withdraw" in bugs:
            balance = wallet["balance"]  # time-of-check
            if balance < body.amount:
                raise HTTPException(400, "insufficient funds")
            await io()
            wallet["balance"] = balance - body.amount  # time-of-use: stale write
            store.post(wallet_id, -body.amount, "withdrawal", ref)
        else:
            async with store.locks[wallet_id]:
                await io()
                if wallet["balance"] < body.amount:
                    raise HTTPException(400, "insufficient funds")
                wallet["balance"] -= body.amount
                store.post(wallet_id, -body.amount, "withdrawal", ref)
        return {"id": ref, "wallet_id": wallet_id, "amount": body.amount, "balance": wallet["balance"]}

    @app.post("/transfers")
    async def transfer(body: TransferIn, user: User) -> dict[str, Any]:
        if "negative_transfer" not in bugs and body.amount <= 0:
            raise HTTPException(422, "amount must be positive")
        source = owned_wallet(body.from_wallet_id, user)
        target = store.wallets.get(body.to_wallet_id)
        if target is None:
            raise HTTPException(404, "destination wallet not found")
        if target["id"] == source["id"]:
            raise HTTPException(400, "cannot transfer to the same wallet")
        first, second = sorted((source["id"], target["id"]))
        async with store.locks[first], store.locks[second]:
            await io()
            if source["balance"] < body.amount:
                raise HTTPException(400, "insufficient funds")
            ref = store.next_id("tx")
            source["balance"] -= body.amount
            target["balance"] += body.amount
            store.post(source["id"], -body.amount, "transfer_out", ref)
            store.post(target["id"], body.amount, "transfer_in", ref)
        return {
            "id": ref, "from_wallet_id": source["id"], "to_wallet_id": target["id"],
            "amount": body.amount, "status": "COMPLETED",
        }

    @app.post("/loans", status_code=201)
    async def apply_loan(body: LoanIn, user: User) -> dict[str, Any]:
        wallet = next((w for w in store.wallets.values() if w["owner_id"] == user["id"]), None)
        if wallet is None:
            raise HTTPException(409, "borrower has no wallet")
        loan_id = store.next_id("loan")
        store.loans[loan_id] = {
            "id": loan_id, "borrower_id": user["id"], "wallet_id": wallet["id"],
            "amount": body.amount, "status": "APPLIED", "approved_by": None,
        }
        return store.loans[loan_id]

    def get_loan(loan_id: str) -> dict[str, Any]:
        loan = store.loans.get(loan_id)
        if loan is None:
            raise HTTPException(404, "loan not found")
        return loan

    @app.post("/loans/{loan_id}/approve")
    async def approve_loan(loan_id: str, user: User) -> dict[str, Any]:
        if user["role"] != "ADMIN":
            raise HTTPException(403, "admin only")
        loan = get_loan(loan_id)
        if loan["status"] != "APPLIED":
            raise HTTPException(409, f"cannot approve a {loan['status']} loan")
        if store.users[loan["borrower_id"]]["kyc_status"] != "APPROVED":
            raise HTTPException(409, "borrower KYC not approved")
        loan.update(status="APPROVED", approved_by=user["id"])
        return loan

    @app.post("/loans/{loan_id}/disburse")
    async def disburse_loan(loan_id: str, user: User) -> dict[str, Any]:
        loan = get_loan(loan_id)
        if loan["borrower_id"] != user["id"]:
            raise HTTPException(403, "not your loan")
        if "loan_state_skip" not in bugs and loan["status"] != "APPROVED":
            raise HTTPException(409, f"cannot disburse a {loan['status']} loan")
        if loan["status"] == "DISBURSED":
            raise HTTPException(409, "already disbursed")
        async with store.locks[loan["wallet_id"]]:
            wallet = store.wallets[loan["wallet_id"]]
            wallet["balance"] += loan["amount"]
            store.post(wallet["id"], loan["amount"], "loan_disbursal", loan_id)
            loan["status"] = "DISBURSED"
        return loan

    return app


def snapshot(app: FastAPI) -> DbSnapshot:
    return app.state.store.snapshot()
