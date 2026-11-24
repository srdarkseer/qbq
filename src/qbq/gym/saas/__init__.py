"""Reference multi-tenant SaaS testbed: workspaces, seats, API keys and subscriptions."""

from qbq.gym.saas.app import BUG_IDS, create_app, snapshot

__all__ = ["BUG_IDS", "create_app", "snapshot"]
