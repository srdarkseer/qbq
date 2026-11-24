from typing import Any

from fastapi import APIRouter, HTTPException

from storefront.deps import StoreDep

router = APIRouter(prefix="/products", tags=["products"])


@router.get("/{product_id}")
async def read_product(product_id: str, store: StoreDep) -> dict[str, Any]:
    product = store.products.get(product_id)
    if product is None:
        raise HTTPException(404, "product not found")
    return product
