"""Request dependencies: store access, authentication, roles, and resource ownership."""

from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, Request

from storefront.models import Store


def get_store(request: Request) -> Store:
    return request.app.state.store


StoreDep = Annotated[Store, Depends(get_store)]


def get_current_user(store: StoreDep, authorization: Annotated[str | None, Header()] = None) -> dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    user_id = store.sessions.get(authorization.removeprefix("Bearer "))
    if user_id is None:
        raise HTTPException(401, "invalid or expired session")
    return store.users[user_id]


CurrentUser = Annotated[dict[str, Any], Depends(get_current_user)]


def require_role(*roles: str):
    def checker(user: CurrentUser) -> dict[str, Any]:
        if user["role"] not in roles:
            raise HTTPException(403, "insufficient role")
        return user

    return checker


def get_owned_order(order_id: str, user: CurrentUser, store: StoreDep) -> dict[str, Any]:
    order = store.orders.get(order_id)
    if order is None or order["customer_id"] != user["id"]:
        raise HTTPException(404, "order not found")
    return order


OwnedOrder = Annotated[dict[str, Any], Depends(get_owned_order)]
