# Examples

## `storefront/`: a multi-file FastAPI service

A small but realistically laid-out shop API:

```
storefront/
  main.py            app factory; mounts routers under /api/v1; re-exports the QbQ snapshot hook
  deps.py            get_store -> get_current_user (CurrentUser), require_role(...) factory, get_owned_order (OwnedOrder)
  schemas.py         Pydantic v2 models (Annotated[int, Field(gt=0, le=100)], Field(min_length=...))
  models.py          in-memory store, freshly seeded per app
  qbq_hooks.py       snapshot(app) -> DbSnapshot for state-based invariants
  routers/           orders.py, admin.py (router-level Depends(require_role("ADMIN"))), products.py
  personas.json      seeded test personas for QbQ
```

It contains one ordinary bug, with no feature flag or toggle: `GET /api/v1/orders/{order_id}/invoice`
authenticates the caller but forgot the `OwnedOrder` dependency its siblings use, so any
customer can read another customer's invoice (email and shipping address).

```bash
cd examples
uv run qbq compile --app storefront.main:create_app --personas storefront/personas.json --out graph.json
uv run qbq remediate --app storefront.main:create_app --personas storefront/personas.json \
    --snapshot storefront.main:snapshot --out-dir regression_tests
# with a live model to propose the fix that the deterministic rule patcher cannot:
GEMINI_API_KEY=... uv run qbq remediate --app storefront.main:create_app --personas storefront/personas.json \
    --snapshot storefront.main:snapshot --llm-provider gemini --patch-dir patches --require-verified
```

QbQ resolves all seven prefixed routes, the `Depends` chains in `deps.py` and the schemas
in `schemas.py`. It confirms exactly one finding (`inv.tenant.read_invoice`) and no false
positives. The rule patcher proposes nothing, because there is no bypassed guard to
re-enable. A model-proposed diff is accepted only if the exploit stops reproducing (0/3)
and the owner's own invoice, and every other functional flow, still works.
