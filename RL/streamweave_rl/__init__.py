"""StreamWeave RL integration package."""

try:
    from . import advantage as _advantage  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name not in {"ray", "verl", "verl.utils"}:
        raise
    _advantage = None

__all__ = ["_advantage"]
