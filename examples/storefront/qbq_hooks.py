"""QbQ snapshot hook: exposes the store's state to the deterministic verifier."""

from qbq.verifier.snapshot import DbSnapshot


def snapshot(app) -> DbSnapshot:
    return app.state.store.snapshot()
