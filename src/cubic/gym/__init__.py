"""The Bug Gym: benchmark environments with ground-truth defects."""

from collections.abc import Callable

from cubic.gym.base import Environment, GroundTruthBug, Variant
from cubic.gym.fintech import environment as fintech_environment



def _fintech_compiled() -> Environment:
    from cubic.gym.fintech.compiled import environment  # imports the compiler lazily

    return environment()


ENVIRONMENTS: dict[str, Callable[[], Environment]] = {
    "fintech": fintech_environment,
    "fintech-compiled": _fintech_compiled,
}


def load_environment(name: str) -> Environment:
    try:
        return ENVIRONMENTS[name]()
    except KeyError:
        raise KeyError(f"unknown environment {name!r}; available: {sorted(ENVIRONMENTS)}") from None


__all__ = ["ENVIRONMENTS", "Environment", "GroundTruthBug", "Variant", "load_environment"]
