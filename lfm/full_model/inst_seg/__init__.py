"""Instance segmentation helpers for full-model experiments."""

from importlib import import_module

__all__ = [
    "GfftInstanceModelAdapter",
    "GrahaInstanceModelAdapter",
]


def __getattr__(name: str):
    modules = {
        "GfftInstanceModelAdapter": ".instance_gfft_model_adapter",
        "GrahaInstanceModelAdapter": ".instance_model_adapter",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
