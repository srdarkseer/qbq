"""App factory: every call returns a fresh app with a freshly seeded store."""

from fastapi import FastAPI

from storefront.models import Store
from storefront.qbq_hooks import snapshot  # noqa: F401  (re-exported for `--snapshot storefront.main:snapshot`)
from storefront.routers import admin, orders, products


def create_app() -> FastAPI:
    app = FastAPI(title="Storefront")
    app.state.store = Store.seeded()
    api = "/api/v1"
    app.include_router(products.router, prefix=api)
    app.include_router(orders.router, prefix=api)
    app.include_router(admin.router, prefix=api)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
