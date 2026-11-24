"""The Bug Gym: benchmark environments with ground-truth defects."""

from collections.abc import Callable

from cubic.gym.base import AppSource, Environment, GroundTruthBug, Variant
from cubic.gym.fintech import environment as fintech_environment


def _fintech_compiled() -> Environment:
    from cubic.gym.fintech.compiled import environment  # imports the compiler lazily

    return environment()


def _fintech_autonomous() -> Environment:
    from cubic.eval.autonomous import autonomous_environment

    return autonomous_environment(_fintech_compiled(), name="fintech-autonomous")


def _saas_compiled() -> Environment:
    from cubic.gym.saas.environment import environment

    return environment()


def _saas_autonomous() -> Environment:
    from cubic.eval.autonomous import autonomous_environment

    return autonomous_environment(_saas_compiled(), name="saas-autonomous")


def _gateway_compiled() -> Environment:
    from cubic.gym.agent_gateway.environment import environment

    return environment()


def _gateway_autonomous() -> Environment:
    from cubic.eval.autonomous import autonomous_environment

    return autonomous_environment(_gateway_compiled(), name="agent-gateway-autonomous")


ENVIRONMENTS: dict[str, Callable[[], Environment]] = {
    "fintech": fintech_environment,
    "fintech-compiled": _fintech_compiled,
    "fintech-autonomous": _fintech_autonomous,
    "saas-compiled": _saas_compiled,
    "saas-autonomous": _saas_autonomous,
    "agent-gateway-compiled": _gateway_compiled,
    "agent-gateway-autonomous": _gateway_autonomous,
}


def load_environment(name: str) -> Environment:
    try:
        return ENVIRONMENTS[name]()
    except KeyError:
        raise KeyError(f"unknown environment {name!r}; available: {sorted(ENVIRONMENTS)}") from None


__all__ = ["ENVIRONMENTS", "AppSource", "Environment", "GroundTruthBug", "Variant", "load_environment"]
