from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from storefront.deps import StoreDep, require_role
from storefront.schemas import OrderOut, RefundIn

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_role("ADMIN"))])


@router.post("/orders/{order_id}/refund", response_model=OrderOut)
async def refund_order(order_id: str, body: RefundIn, store: StoreDep) -> dict[str, Any]:
    order = store.orders.get(order_id)
    if order is None:
        raise HTTPException(404, "order not found")
    if order["status"] != "SHIPPED":
        raise HTTPException(409, f"cannot refund a {order['status']} order")
    order["status"] = "REFUNDED"
    return order
