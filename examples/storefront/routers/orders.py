from typing import Any

from fastapi import APIRouter, HTTPException

from storefront.deps import CurrentUser, OwnedOrder, StoreDep
from storefront.schemas import InvoiceOut, OrderCreate, OrderOut

router = APIRouter(prefix="/orders", tags=["orders"])


@router.post("", status_code=201, response_model=OrderOut)
async def create_order(body: OrderCreate, user: CurrentUser, store: StoreDep) -> dict[str, Any]:
    product = store.products.get(body.product_id)
    if product is None:
        raise HTTPException(404, "product not found")
    if product["stock"] < body.quantity:
        raise HTTPException(409, "insufficient stock")
    order_id = store.next_id("o")
    product["stock"] -= body.quantity
    store.orders[order_id] = {
        "id": order_id, "customer_id": user["id"], "product_id": product["id"], "quantity": body.quantity,
        "total_cents": product["price_cents"] * body.quantity, "status": "PLACED",
        "shipping_address": body.shipping_address,
    }
    return store.orders[order_id]


@router.get("/{order_id}", response_model=OrderOut)
async def read_order(order: OwnedOrder) -> dict[str, Any]:
    return order


@router.get("/{order_id}/invoice", response_model=InvoiceOut)
async def read_invoice(order_id: str, user: CurrentUser, store: StoreDep) -> dict[str, Any]:
    order = store.orders.get(order_id)
    if order is None:
        raise HTTPException(404, "order not found")
    customer = store.users[order["customer_id"]]
    return {"order_id": order["id"], "customer_email": customer["email"],
            "shipping_address": order["shipping_address"], "total_cents": order["total_cents"]}


@router.post("/{order_id}/cancel", response_model=OrderOut)
async def cancel_order(order: OwnedOrder) -> dict[str, Any]:
    if order["status"] != "PLACED":
        raise HTTPException(409, f"cannot cancel a {order['status']} order")
    order["status"] = "CANCELLED"
    return order
