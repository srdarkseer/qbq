"""In-memory persistence layer. Each app instance gets its own freshly seeded store."""

import copy
from dataclasses import dataclass, field
from typing import Any

from qbq.verifier.snapshot import DbSnapshot


def _seed() -> dict[str, Any]:
    return {
        "users": {
            "u_ann": {"id": "u_ann", "email": "ann@example.test", "role": "CUSTOMER"},
            "u_ben": {"id": "u_ben", "email": "ben@example.test", "role": "CUSTOMER"},
            "u_ops": {"id": "u_ops", "email": "ops@example.test", "role": "ADMIN"},
        },
        "products": {
            "p_mug": {"id": "p_mug", "name": "Mug", "price_cents": 1200, "stock": 40},
            "p_pen": {"id": "p_pen", "name": "Pen", "price_cents": 300, "stock": 500},
        },
        "orders": {
            "o_ann": {"id": "o_ann", "customer_id": "u_ann", "product_id": "p_mug", "quantity": 2,
                      "total_cents": 2400, "status": "PLACED", "shipping_address": "1 Ann St"},
            "o_ben": {"id": "o_ben", "customer_id": "u_ben", "product_id": "p_pen", "quantity": 10,
                      "total_cents": 3000, "status": "SHIPPED", "shipping_address": "2 Ben Rd"},
        },
    }


@dataclass
class Store:
    users: dict[str, dict[str, Any]]
    products: dict[str, dict[str, Any]]
    orders: dict[str, dict[str, Any]]
    sessions: dict[str, str] = field(default_factory=lambda: {
        "tok_ann": "u_ann", "tok_ben": "u_ben", "tok_ops": "u_ops"})
    _seq: int = 0

    @classmethod
    def seeded(cls) -> "Store":
        return cls(**_seed())

    def next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}_{self._seq:04d}"

    def snapshot(self) -> DbSnapshot:
        return DbSnapshot(entities=copy.deepcopy(
            {"users": self.users, "products": self.products, "orders": self.orders}))
