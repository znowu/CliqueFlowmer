"""Model exports, imported on demand to keep inference dependencies small."""

from importlib import import_module

__all__ = ["CliqueFlowmer", "Transformer", "Flow"]

_MODEL_MODULES = {
    "CliqueFlowmer": "models.cliqueflowmer",
    "Transformer": "models.transformer",
    "Flow": "models.flow",
}


def __getattr__(name):
    if name not in _MODEL_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_MODEL_MODULES[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
