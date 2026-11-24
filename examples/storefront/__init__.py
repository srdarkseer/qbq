"""Storefront: a small multi-file FastAPI service used to validate QbQ on real project layouts.

Layout: ``main.py`` (app factory, router mounts), ``deps.py`` (auth / role / ownership
dependencies), ``schemas.py`` (Pydantic v2 request and response models), ``models.py`` (the
in-memory store) and ``routers/`` (one ``APIRouter`` per area). It contains one ordinary,
unscripted bug, with no feature flag or toggle: an endpoint that forgot its ownership dependency.
"""
